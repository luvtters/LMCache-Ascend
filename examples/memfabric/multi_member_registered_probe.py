#!/usr/bin/env python3
"""World3 owner/writer/reader registered DEVICE_RDMA diagnostic; PLAN_ONLY default."""
import argparse
import json
import os
from pathlib import Path
import socket
import time
import traceback
from urllib.parse import urlparse

from bm_media_probe import call_step, cleanup_steps
from direct_coordination import publish, wait_event
from registered_host_rdma_probe import (POOL_BYTES, checked_end, pattern,
    validate_regions, verify_local, write_local)

PROTOCOL = 'world3-registered-host-rdma-v1'
ROLES = {0: 'owner', 1: 'writer', 2: 'reader'}
# Explicit campaign boundaries only; every allocation/registration/copy/check
# keeps the exact requested byte count, including odd lengths (no rounding).
PAYLOADS = (64, 65, 4095, 4096, 4097, 1 << 20, 18 << 20, 64 << 20, 128 << 20)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank', type=int, choices=tuple(ROLES), required=True)
    p.add_argument('--world-size', type=int, choices=(3,), default=3)
    p.add_argument('--device', type=int, choices=(0,), required=True,
                   help='Each caller must mask exactly one authorized physical device as logical 0')
    p.add_argument('--payload-bytes', type=int, choices=PAYLOADS, default=1 << 20)
    p.add_argument('--device-nic', default='tcp://127.0.0.1:10005')
    p.add_argument('--store-url', required=True)
    p.add_argument('--coord-url', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('--init-timeout', type=int, default=180)
    p.add_argument('--allow-npu', action='store_true')
    a = p.parse_args(argv)
    if a.timeout <= 0 or not a.timeout < float('inf') or a.init_timeout <= 0:
        p.error('timeouts must be finite positive values')
    if not a.run_id or not a.run_id.isascii() or any(not (c.isalnum() or c in '-_') for c in a.run_id):
        p.error('run-id requires ASCII letters, digits, dash or underscore')
    for value, schemes in ((a.store_url, ('tcp',)), (a.device_nic, ('tcp',)),
                           (a.coord_url, ('http', 'https'))):
        parsed = urlparse(value)
        try:
            valid = parsed.hostname and parsed.port and 0 < parsed.port < 65536
        except ValueError:
            valid = False
        if not valid or parsed.scheme not in schemes:
            p.error('store/NIC require tcp://host:port; coordination requires HTTP(S) host:port')
    return a


def build_plan(a):
    return dict(status='PLAN_ONLY', protocol=PROTOCOL, rank=a.rank, role=ROLES[a.rank],
        world_size=3, owner_rank=0, writer_rank=1, reader_rank=2, device=a.device,
        run_id=a.run_id, store_url=a.store_url, coord_url=a.coord_url,
        device_nic=a.device_nic, payload_bytes=a.payload_bytes, owner_offset=0,
        swap_space_size='0', payload_via_rpc=False, fallback_allowed=False,
        copy_types=dict(write='L2G', read='G2L'),
        create2=dict(id=0, local_dram_size=POOL_BYTES, max_dram_size=POOL_BYTES,
            local_hbm_size=0, max_hbm_size=0, data_op_type='DEVICE_RDMA',
            enable_56bits_gva=True, flags=0),
        local_device_staging=False, external_process_wait_required=True,
        handle_wait_is_completion_evidence=False,
        timing_basis='causal events and per-process elapsed durations; no cross-host monotonic subtraction',
        actual_transport='NOT_VERIFIED_READ_NATIVE_LOG',
        native_unregister_warning='Public unregister=0 does not erase nativeHalHostUnregisterEx ret8 warnings',
        completion_basis='sync copy return, full owner A/B, separate reader A/B, all three external waits',
        scope='synthetic_three_member_primitive_not_model_TP3_or_production_worker')


def contract(report):
    return {key: report[key] for key in ('protocol', 'world_size', 'owner_rank', 'writer_rank',
        'reader_rank', 'store_url', 'device_nic', 'payload_bytes', 'owner_offset',
        'swap_space_size', 'copy_types', 'create2')}


def validate_event(value, name, rank):
    if (not isinstance(value, dict) or value.get('protocol') != PROTOCOL or
            value.get('name') != name or value.get('rank') != rank):
        raise ValueError('unexpected peer protocol, event name or rank: ' + name)
    return value


def require_content(value, n):
    if value.get('equal') is not True or value.get('verified_bytes') != n:
        raise AssertionError('peer did not verify full payload')


def role_protocol(rank, n, actions, mark, wait):
    """Causal metadata protocol; callbacks perform payload operations locally, never HTTP."""
    if rank == 0:
        if wait('writer-pushed-A', 1).get('completed_bytes') != n:
            raise AssertionError('writer completion byte count mismatch')
        actions['owner_verify']('A')
        mark('owner-verified-A', equal=True, verified_bytes=n)
        wait('writer-destroyed-source', 1)
        require_content(wait('reader-verified-A', 2), n)
        actions['owner_write']('B')
        actions['owner_verify']('B')
        mark('owner-wrote-B', equal=True, verified_bytes=n)
        require_content(wait('reader-verified-B', 2), n)
    elif rank == 1:
        actions['write_A']()
        mark('writer-pushed-A', completed_bytes=n)
        require_content(wait('owner-verified-A', 0), n)
        actions['destroy_source']()
        mark('writer-destroyed-source', verified_bytes=n, fill_byte=51)
        require_content(wait('reader-verified-B', 2), n)
    elif rank == 2:
        require_content(wait('owner-verified-A', 0), n)
        wait('writer-destroyed-source', 1)
        for label in ('A', 'B'):
            if label == 'B':
                require_content(wait('owner-wrote-B', 0), n)
            actions['reset_restore']()
            actions['read'](label)
            actions['reader_verify'](label)
            mark('reader-verified-' + label, equal=True, verified_bytes=n)
    else:
        raise ValueError('unsupported rank')
    mark('payload-done-' + str(rank))
    for peer in ROLES:
        wait('payload-done-' + str(peer), peer)


def run(a, report, emit):
    # Native/device imports and all device operations are behind --allow-npu.
    os.environ['MF_HYBM_RDMA_SWAP_SPACE_SIZE'] = '0'
    import torch
    import torch_npu  # noqa: F401
    import memfabric_hybrid as mf
    bm = mf.bm
    n = a.payload_bytes
    handle = None
    mf_ready = bm_ready = joined = False
    retained = []
    registered = []
    error = error_tb = None

    def mark(name, **data):
        publish(a.coord_url, a.run_id, name, dict(name=name, rank=a.rank, protocol=PROTOCOL,
            hostname=socket.gethostname(), pid=os.getpid(), **data), min(5, a.timeout))
        emit('CONTROL_PUBLISHED', name=name)

    def wait(name, rank):
        value = validate_event(wait_event(a.coord_url, a.run_id, name, a.timeout), name, rank)
        emit('CONTROL_OBSERVED', name=name, peer_rank=rank)
        return value

    def fail_publish(exc):
        try:
            mark('failed-' + str(a.rank), error=str(exc))
        except BaseException:
            report['failure_publish_error'] = traceback.format_exc()
            emit('FAILURE_PUBLISH_ERROR', traceback=report['failure_publish_error'])

    def record_content(result, step):
        emit('CONTENT_CHECK', step=step, **result)
        report[step] = result
        require_content(result, n)

    try:
        own_contract = contract(report)
        mark('config-ready-' + str(a.rank), contract=own_contract)
        for peer in ROLES:
            if wait('config-ready-' + str(peer), peer).get('contract') != own_contract:
                raise ValueError('world3 peer configuration mismatch')
        torch.npu.set_device(a.device)
        report['sdk_path'] = mf.__file__
        report['visible_devices'] = os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
        emit('DEVICE_SELECTED', logical_device=a.device, sdk_path=mf.__file__,
             visible_devices=report['visible_devices'], swap_space_size='0')
        mf.set_log_level(0)
        call_step('mf.initialize', mf.initialize, emit)
        mf_ready = True
        cfg = bm.BmConfig()
        cfg.auto_ranking = False
        cfg.rank_id = a.rank
        cfg.start_store = a.rank == 0
        cfg.init_timeout = cfg.create_timeout = cfg.operation_timeout = a.init_timeout
        cfg.set_nic(a.device_nic)
        call_step('bm.initialize', lambda: bm.initialize(a.store_url, 3, a.device, cfg), emit)
        bm_ready = True
        emit('CREATE_BEGIN', requested=report['create2'])
        handle = bm.create2(**dict(report['create2'], data_op_type=bm.BmDataOpType.DEVICE_RDMA))
        if handle is None:
            raise RuntimeError('create2 returned None')
        call_step('join', handle.join, emit)
        joined = True
        owner_gva = int(handle.peer_rank_ptr(0, bm.BmMemType.HOST))
        checked_end(owner_gva, n)
        report['owner_gva'] = owner_gva
        owner_hva = None
        tensor = None
        if a.rank == 0:
            own_gva = int(handle.peer_rank_ptr(a.rank, bm.BmMemType.HOST))
            if own_gva != owner_gva:
                raise ValueError('owner rank GVA mismatch')
            pool_size = int(handle.local_mem_size(bm.BmMemType.HOST))
            owner_hva = int(handle.gva_to_va(owner_gva, bm.BmMemType.LOCAL_HOST))
            validate_regions(owner_hva, pool_size, [(owner_hva, n)])
            write_local(owner_hva, bytes([165]) * n)
            report['owner_mapping'] = dict(gva=owner_gva, hva=owner_hva, pool_size=pool_size)
            emit('OWNER_LOCAL_MAPPING_VALIDATED', **report['owner_mapping'])
        else:
            if a.rank == 1:
                host_source = bytearray(pattern(n, 'A'))
                tensor = torch.frombuffer(host_source, dtype=torch.uint8).to(f'npu:{a.device}')
                retained.append(host_source)
            else:
                tensor = torch.full((n,), 90, dtype=torch.uint8, device=f'npu:{a.device}')
            retained.append(tensor)
            if tensor.device.type != 'npu' or tensor.dtype != torch.uint8 or not tensor.is_contiguous() or tensor.numel() != n:
                raise ValueError('endpoint requires a contiguous real NPU byte tensor')
            pointer = int(tensor.data_ptr())
            checked_end(pointer, n)
            torch.npu.synchronize(a.device)
            emit('REAL_NPU_TENSOR', pointer=pointer, tensor_device=str(tensor.device), bytes=n,
                 purpose='source' if a.rank == 1 else 'restore')
            call_step('register.real_NPU', lambda: handle.register(pointer, n), emit,
                      pointer=pointer, requested_bytes=n)
            registered.append(pointer)
            report['registered_torch_pointers'] = list(registered)
        mark('address-ready-' + str(a.rank), owner_gva=owner_gva)
        for peer in ROLES:
            if wait('address-ready-' + str(peer), peer).get('owner_gva') != owner_gva:
                raise ValueError('peer owner GVA mismatch')

        def copy(direction, label):
            src, dst, kind = ((tensor.data_ptr(), owner_gva, 'L2G') if direction == 'write'
                              else (owner_gva, tensor.data_ptr(), 'G2L'))
            started = time.perf_counter_ns()
            raw_return = call_step(kind + '_' + label,
                lambda: handle.copy_data(src, dst, n, getattr(bm.BmCopyType, kind), 0),
                emit, requested_bytes=n, src=src, dst=dst, flags=0)
            elapsed = time.perf_counter_ns() - started
            torch.npu.synchronize(a.device)
            report['completed_' + direction + '_bytes'] += n
            report.setdefault('copies', []).append(dict(direction=direction, pattern=label,
                completed_bytes=n, raw_return=raw_return, local_elapsed_ns=elapsed,
                timing='synchronous copy wrapper including logging; not a performance benchmark'))

        def owner_verify(label):
            record_content(verify_local(owner_hva, n, label), 'owner_' + label)

        def verify_fill(value, step):
            torch.npu.synchronize(a.device)
            observed = tensor.cpu().numpy().tobytes()
            result = dict(equal=observed == bytes([value])*n, verified_bytes=len(observed),
                fill_byte=value, verified_from='registered_real_NPU_tensor')
            record_content(result, step)

        def destroy_source():
            tensor.fill_(51)
            verify_fill(51, 'writer_source_destroyed')

        def reset_restore():
            tensor.fill_(90)
            verify_fill(90, 'reader_restore_sentinel')

        def reader_verify(label):
            torch.npu.synchronize(a.device)
            observed = tensor.cpu().numpy().tobytes()
            expected = pattern(n, label)
            record_content(dict(equal=observed == expected, verified_bytes=len(observed),
                pattern=label, mismatch_bytes=sum(x != y for x, y in zip(observed, expected)),
                verified_from='registered_real_NPU_tensor'), 'reader_' + label)

        role_protocol(a.rank, n, dict(owner_verify=owner_verify,
            owner_write=lambda label: write_local(owner_hva, pattern(n, label)),
            write_A=lambda: copy('write', 'A'), destroy_source=destroy_source,
            reset_restore=reset_restore, read=lambda label: copy('read', label),
            reader_verify=reader_verify), mark, wait)
        report['payload_passed'] = True
    except BaseException as exc:
        error, error_tb = exc, exc.__traceback__
        report['error'] = traceback.format_exc()
        emit('PROBE_ERROR', traceback=report['error'])
        fail_publish(exc)
    finally:
        # Success reaches here only after all roles finish all payload access.
        # Failure is fail-closed: publish failure, then attempt local SDK teardown.
        steps = []
        if handle is not None:
            for pointer in reversed(registered):
                steps.append(('close.unregister.' + hex(pointer),
                    lambda pointer=pointer: handle.unregister(pointer), True))
            if joined:
                steps.append(('close.leave', handle.leave, True))
            steps.append(('close.destroy', handle.destroy, False))
        if bm_ready:
            steps.append(('close.bm.uninitialize', bm.uninitialize, False))
        if mf_ready:
            steps.append(('close.mf.uninitialize', mf.uninitialize, False))
        report['cleanup_errors'] = cleanup_steps(steps, emit)
        report['local_cleanup_passed'] = not report['cleanup_errors']
        report['retained_npu_objects_through_cleanup'] = len(retained)
    if error is not None:
        raise error.with_traceback(error_tb)
    if report['cleanup_errors']:
        fail_publish(RuntimeError('local SDK cleanup failed'))
        raise RuntimeError('local SDK cleanup failed')


def main(argv=None):
    a = arguments(argv)
    report = build_plan(a)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as output:
        json.dump(report, output, indent=2, sort_keys=True)
    if not a.allow_npu:
        print(json.dumps(report, sort_keys=True), flush=True)
        return
    with a.output.with_suffix(a.output.suffix + '.events.jsonl').open('x', encoding='utf-8') as events:
        def emit(event, **data):
            line = json.dumps(dict(event=event, unix_ns=time.time_ns(), mono_ns=time.monotonic_ns(),
                pid=os.getpid(), hostname=socket.gethostname(), rank=a.rank, role=ROLES[a.rank],
                device=a.device, run_id=a.run_id, **data), sort_keys=True)
            events.write(line + '\n')
            events.flush()
            print(line, flush=True)
        report.update(status='RUNNING', payload_passed=False, completed_write_bytes=0,
                      completed_read_bytes=0)
        try:
            emit('PROBE_BEGIN', requested=report['create2'])
            run(a, report, emit)
            report['status'] = 'PAYLOAD_AND_LOCAL_CLEANUP_PASSED_EXTERNAL_WAIT_REQUIRED'
        except BaseException:
            report['status'] = 'FAILED'
            report.setdefault('error', traceback.format_exc())
            raise
        finally:
            report['ended_unix_ns'] = time.time_ns()
            a.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
            emit('PROBE_RESULT', status=report['status'], payload_passed=report['payload_passed'])


if __name__ == '__main__':
    main()
