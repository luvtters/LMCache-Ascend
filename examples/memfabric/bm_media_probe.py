#!/usr/bin/env python3
"""Two-rank raw BM medium probe. Defaults to a CPU-only plan.

Push writes rank 0's patterned NPU buffer to rank 1's pool, then rank 1 verifies.
Pull-only initializes rank 1's own pool locally, then rank 0 reads and verifies.
Every byte is compared to an independent expectation; HTTP is metadata only.
The caller owns device masks, a fresh store/group, both process waits and raw
stdout/stderr (including actual native ExportInner segType records).
"""
import argparse
import json
import os
from pathlib import Path
import socket
import time
import traceback
from urllib.parse import urlparse

from direct_coordination import publish, wait_event

PAYLOAD_BYTES = 1 << 20
POOL_BYTES = 1 << 30
# Exact SDK 1.2.1 smem_bm_def.h; allocation variant, not a copy flag.
DRAM_MAP_HOST_VA_FLAG = 1 << 9


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--device', type=int, required=True,
                        help='Logical NPU id after the caller applies its device mask')
    parser.add_argument('--world-size', type=int, choices=(2,), default=2)
    parser.add_argument('--media', choices=('host', 'device'), required=True)
    parser.add_argument('--direction', choices=('push', 'pull-only'), default='push')
    parser.add_argument('--data-op-type', choices=('sdma', 'device_rdma'), default='sdma')
    parser.add_argument('--device-nic', default='tcp://127.0.0.1:10005',
                        help='SDK NIC URL; DEVICE_RDMA selects its own NPU IP using this port')
    parser.add_argument('--enable-56bits-gva', action='store_true',
                        help='Explicit SDK address mode; permits RDMA allocator HAL fallback')
    parser.add_argument('--dram-map-host-va', action='store_true',
                        help='Isolated HOST/SDMA allocation diagnostic; default remains flags=0')
    parser.add_argument('--store-url', required=True)
    parser.add_argument('--coord-url', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output', type=Path, required=True, help='New rank-local JSON result file')
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--init-timeout', type=int, default=300)
    parser.add_argument('--allow-npu', action='store_true')
    args = parser.parse_args(argv)
    if args.dram_map_host_va and (args.media != 'host' or args.data_op_type != 'sdma'):
        parser.error('--dram-map-host-va is only a HOST/SDMA allocation variant')
    if args.device < 0 or args.timeout <= 0 or args.init_timeout <= 0:
        parser.error('device and timeouts must be valid')
    if (not args.run_id or not args.run_id.isascii() or
            any(not (c.isalnum() or c in '-_') for c in args.run_id)):
        parser.error('run-id must contain ASCII letters, digits, dash or underscore')
    if not args.store_url.startswith('tcp://') or not args.coord_url.startswith(('http://', 'https://')):
        parser.error('store-url must be tcp:// and coord-url must be HTTP(S)')
    nic = urlparse(args.device_nic)
    if nic.scheme != 'tcp' or not nic.hostname or not nic.port or not 0 < nic.port < 65536:
        parser.error('device-nic must be tcp://IPv4:port')
    return args


def build_plan(args):
    hbm = POOL_BYTES if args.media == 'device' else 0
    return dict(status='PLAN_ONLY', rank=args.rank, device=args.device,
        hostname=socket.gethostname(), pid=os.getpid(), world_size=2,
        media=args.media, direction=args.direction, writer_rank=1 if args.direction=='pull-only' else 0,
        reader_rank=0 if args.direction=='pull-only' else 1,
        data_op_type=args.data_op_type, device_nic=args.device_nic,
        run_id=args.run_id, store_url=args.store_url,
        coord_url=args.coord_url, owner_rank=1, payload_bytes=PAYLOAD_BYTES,
        fresh_store_and_process_group_required=True,
        target_memory='DEVICE' if hbm else 'HOST',
        create2=dict(id=0, local_dram_size=POOL_BYTES, max_dram_size=POOL_BYTES,
                     local_hbm_size=hbm, max_hbm_size=hbm,
                     data_op_type='SDMA' if args.data_op_type == 'sdma' else 'DEVICE_RDMA',
                     enable_56bits_gva=args.enable_56bits_gva,
                     flags=DRAM_MAP_HOST_VA_FLAG if args.dram_map_host_va else 0),
        actual_transport='NOT_VERIFIED_READ_NATIVE_TRANSPORT_LOG',
        actual_export_seg_type='NOT_VERIFIED_READ_NATIVE_PROCESS_RAW_LOG',
        payload_via_rpc=False, external_process_wait_required=True,
        comparison_scope='raw_BM_medium_diagnostic_not_LMCache_or_performance')


def pattern_bytes(size):
    return bytes((index * 17 + (index // 251) * 29 + 43) % 256 for index in range(size))


def sdk_data_op(bm, name):
    # A missing SDK enum is an error, never a request to fall back to SDMA.
    if name not in ('sdma', 'device_rdma'):
        raise ValueError('Unsupported data operation')
    return getattr(bm.BmDataOpType, 'SDMA' if name == 'sdma' else 'DEVICE_RDMA')


def sdk_create_options(report, bm):
    return dict(report['create2'], data_op_type=sdk_data_op(bm, report['data_op_type']))


def peer_contract(report):
    return {key: report[key] for key in ('media', 'direction', 'data_op_type', 'device_nic',
        'world_size', 'store_url', 'payload_bytes', 'create2')}


def check_peer_contract(local, peer):
    if local != peer:
        raise ValueError('Ranks disagree on pool geometry, medium or data operation')


def owner_address(handle, bm, media, timeout, emit, data_op_type='sdma'):
    mem_type = bm.BmMemType.DEVICE if media == 'device' else bm.BmMemType.HOST
    deadline = time.monotonic() + timeout
    while True:
        pointer = handle.peer_rank_ptr(1, mem_type)
        if pointer and data_op_type == 'device_rdma':
            # RDMA addresses remote registered memory through transport keys.
            # The SDK deliberately leaves remote LOCAL_DEVICE mappings absent.
            emit('OWNER_GVA_READY', owner_rank=1, target_memory=media,
                 peer_ptr=int(pointer), peer_ptr_hex=hex(pointer),
                 readiness='BM_JOIN_AND_TRANSPORT_REGISTRATION; no SDMA mapping claim')
            return pointer
        local_va = handle.gva_to_va(pointer, bm.BmMemType.LOCAL_DEVICE) if pointer else 0
        if pointer and local_va:
            emit('OWNER_MAPPING_READY', owner_rank=1, target_memory=media,
                 peer_ptr=int(pointer), peer_ptr_hex=hex(pointer),
                 local_device_va=int(local_va), local_device_va_hex=hex(local_va))
            return pointer
        if time.monotonic() >= deadline:
            raise TimeoutError('Owner rank 1 selected pool has no LOCAL_DEVICE mapping')
        time.sleep(.05)


def call_step(name, operation, emit, *, checked=True, **metadata):
    emit('SDK_BEGIN', step=name, **metadata)
    try:
        result = operation()
        emit('SDK_RETURN', step=name, raw_return=result, **metadata)
        if (checked and result != 0) or (not checked and result not in (None, 0)):
            raise RuntimeError(f'{name} returned {result!r}')
        return result
    except BaseException:
        emit('SDK_ERROR', step=name, traceback=traceback.format_exc(), **metadata)
        raise


def cleanup_steps(steps, emit):
    errors = []
    for name, operation, checked in steps:
        try:
            call_step(name, operation, emit, checked=checked)
        except BaseException:
            errors.append(dict(step=name, traceback=traceback.format_exc()))
    return errors


def run(args, report, emit):
    # Device and native SDK imports are intentionally behind --allow-npu.
    import torch
    import torch_npu  # noqa: F401
    import memfabric_hybrid as mf
    bm = mf.bm
    report.update(sdk_path=mf.__file__, bm_native_path=getattr(bm, '__file__', None),
                  visible_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'))
    emit('MODULES_LOADED', **{key: report[key] for key in
         ('sdk_path', 'bm_native_path', 'visible_devices')})
    handle = None
    mf_ready = bm_ready = joined = False
    buffers = []  # Retain source and restore allocations throughout SDK teardown.
    primary_error = None
    primary_traceback = None

    def marker(name, **metadata):
        publish(args.coord_url, args.run_id, name,
                dict(name=name, rank=args.rank, hostname=socket.gethostname(),
                     media=args.media, data_op_type=args.data_op_type, **metadata), min(5, args.timeout))

    def wait(name):
        result = wait_event(args.coord_url, args.run_id, name, args.timeout)
        if (result.get('media') != args.media or
                result.get('data_op_type') != args.data_op_type):
            raise RuntimeError('Peer medium or data operation differs from this attempt')
        return result

    try:
        marker('config-ready-' + str(args.rank), contract=peer_contract(report))
        peer = wait('config-ready-' + str(1 - args.rank))
        check_peer_contract(peer_contract(report), peer['contract'])
        emit('PEER_CONFIG_AGREED', contract=peer_contract(report))
        torch.npu.set_device(args.device)
        emit('DEVICE_SELECTED', logical_device=args.device)
        mf.set_log_level(0)  # Preserve native ExportInner/Import records in caller's raw log.
        call_step('mf.initialize', mf.initialize, emit)
        mf_ready = True
        config = bm.BmConfig()
        config.auto_ranking = False
        config.rank_id = args.rank
        config.start_store = args.rank == 0
        config.init_timeout = args.init_timeout
        config.create_timeout = args.init_timeout
        config.operation_timeout = args.init_timeout
        if args.data_op_type == 'device_rdma':
            config.set_nic(args.device_nic)
            emit('DEVICE_RDMA_NIC_REQUESTED', device_nic=args.device_nic,
                 actual_device_ip='READ_NATIVE_TRANSPORT_LOG')
        call_step('bm.initialize', lambda: bm.initialize(
            args.store_url, 2, args.device, config), emit)
        bm_ready = True
        operation = sdk_data_op(bm, args.data_op_type)
        emit('SDK_DATA_OPERATION_SELECTED', data_op_type=args.data_op_type,
             sdk_enum=str(operation), sdk_enum_value=int(operation))
        options = sdk_create_options(report, bm)
        emit('CREATE_BEGIN', requested=report['create2'],
             actual_export_seg_type='READ_NATIVE_ExportInner_NOT_REQUESTED_MEDIA')
        handle = bm.create2(**options)
        if handle is None:
            raise RuntimeError('bm.create2 returned None')
        emit('CREATE_RETURN', handle_type=str(type(handle)))
        call_step('join', handle.join, emit)
        joined = True
        pointer = owner_address(handle, bm, args.media, args.timeout, emit, args.data_op_type)
        report['owner_peer_ptr'] = int(pointer)
        report['owner_peer_ptr_hex'] = hex(pointer)
        marker('ready-' + str(args.rank), owner_ptr=int(pointer),
               target_memory=report['target_memory'], group_id=report['create2']['id'])
        peer = wait('ready-' + str(1 - args.rank))
        if peer.get('group_id') != report['create2']['id']:
            raise RuntimeError('Peer BM group id differs')
        if peer.get('owner_ptr') != pointer or peer.get('target_memory') != report['target_memory']:
            raise RuntimeError('Ranks disagree on selected owner pool address or memory type')
        if args.rank == report['writer_rank']:
            host_source = bytearray(pattern_bytes(PAYLOAD_BYTES))
            source = torch.frombuffer(host_source, dtype=torch.uint8).to(f'npu:{args.device}')
            buffers.extend([host_source, source])
            torch.npu.synchronize(args.device)
            emit('BUFFER_READY', kind='source_npu', ptr=source.data_ptr(), bytes=PAYLOAD_BYTES)
            call_step('copy_L2G', lambda: handle.copy_data(source.data_ptr(), pointer,
                PAYLOAD_BYTES, bm.BmCopyType.L2G, 0), emit,
                requested_bytes=PAYLOAD_BYTES, src=source.data_ptr(), dst=pointer, flags=0)
            call_step('wait_after_L2G', handle.wait, emit)
            report['completed_write_bytes'] = PAYLOAD_BYTES
            emit('COPY_COMPLETED', direction='L2G', completed_bytes=PAYLOAD_BYTES)
            marker('stored', completed_bytes=PAYLOAD_BYTES)
            verified = wait('verified')
            if verified.get('equal') is not True or verified.get('verified_bytes') != PAYLOAD_BYTES:
                raise AssertionError('Reader did not verify the complete payload')
            report['peer_full_content_equal'] = True
            marker('source-ack')
        else:
            restored = torch.full((PAYLOAD_BYTES,), 165, dtype=torch.uint8,
                                  device=f'npu:{args.device}')
            buffers.append(restored)
            torch.npu.synchronize(args.device)
            stored = wait('stored')
            if stored.get('completed_bytes') != PAYLOAD_BYTES:
                raise AssertionError('Source did not complete the expected byte count')
            call_step('copy_G2L', lambda: handle.copy_data(pointer, restored.data_ptr(),
                PAYLOAD_BYTES, bm.BmCopyType.G2L, 0), emit,
                requested_bytes=PAYLOAD_BYTES, src=pointer, dst=restored.data_ptr(), flags=0)
            call_step('wait_after_G2L', handle.wait, emit)
            torch.npu.synchronize(args.device)
            report['completed_read_bytes'] = PAYLOAD_BYTES
            observed = restored.cpu().numpy().tobytes()
            expected = pattern_bytes(PAYLOAD_BYTES)  # Generated here; never received from rank 0.
            equal = observed == expected
            mismatch = 0 if equal else sum(a != b for a, b in zip(observed, expected))
            report.update(full_content_equal=equal, verified_bytes=len(observed), mismatch_bytes=mismatch)
            emit('CONTENT_CHECK', equal=equal, verified_bytes=len(observed), mismatch_bytes=mismatch)
            if not equal or len(observed) != PAYLOAD_BYTES:
                raise AssertionError('Restored owner buffer differs from independent expected bytes')
            marker('verified', equal=True, verified_bytes=PAYLOAD_BYTES)
            wait('source-ack')
        report['payload_passed'] = True
        marker('cleanup-ready-' + str(args.rank))
        wait('cleanup-ready-' + str(1 - args.rank))
    except BaseException as error:
        primary_error, primary_traceback = error, error.__traceback__
        report['error'] = traceback.format_exc()
        emit('PROBE_ERROR', traceback=report['error'])
        try:
            marker('failed-' + str(args.rank), error=str(error))
        except BaseException:
            report['failure_publish_error'] = traceback.format_exc()
            emit('FAILURE_PUBLISH_ERROR', traceback=report['failure_publish_error'])
    finally:
        steps = []
        if handle is not None:
            steps.append(('close.wait', handle.wait, True))
            if joined:
                steps.append(('close.leave', handle.leave, True))
            steps.append(('close.destroy', handle.destroy, False))
        if bm_ready:
            steps.append(('close.bm.uninitialize', bm.uninitialize, False))
        if mf_ready:
            steps.append(('close.mf.uninitialize', mf.uninitialize, False))
        report['cleanup_errors'] = cleanup_steps(steps, emit)
        report['local_cleanup_passed'] = not report['cleanup_errors']
        report['retained_buffers_through_cleanup'] = len(buffers)
    if primary_error is not None:
        raise primary_error.with_traceback(primary_traceback)
    if report['cleanup_errors']:
        raise RuntimeError('SDK teardown failed; see cleanup_errors')


def main(argv=None):
    args = arguments(argv)
    report = build_plan(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    if not args.allow_npu:
        print(json.dumps(report, sort_keys=True), flush=True)
        return
    event_path = args.output.with_suffix(args.output.suffix + '.events.jsonl')
    with event_path.open('x', encoding='utf-8') as events:
        def emit(event, **metadata):
            record = dict(event=event, unix_ns=time.time_ns(), mono_ns=time.monotonic_ns(),
                hostname=socket.gethostname(), pid=os.getpid(), rank=args.rank,
                device=args.device, media=args.media, run_id=args.run_id, **metadata)
            line = json.dumps(record, sort_keys=True)
            events.write(line + '\n')
            events.flush()
            print(line, flush=True)
        report.update(status='RUNNING', payload_passed=False,
                      completed_write_bytes=0, completed_read_bytes=0)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
        emit('PROBE_BEGIN', requested=report['create2'], payload_bytes=PAYLOAD_BYTES)
        try:
            run(args, report, emit)
            report['status'] = 'PAYLOAD_AND_LOCAL_CLEANUP_PASSED_EXTERNAL_WAIT_REQUIRED'
        except BaseException:
            report['status'] = 'FAILED'
            report.setdefault('error', traceback.format_exc())
            raise
        finally:
            report['ended_unix_ns'] = time.time_ns()
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
            emit('PROBE_RESULT', status=report['status'], payload_passed=report['payload_passed'])


if __name__ == '__main__':
    main()
