#!/usr/bin/env python3
"""Two-rank external torch CPU allocation registration/H2G/G2H diagnostic; PLAN_ONLY."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
from urllib.parse import urlparse

from bm_media_probe import call_step, cleanup_steps
from direct_coordination import publish, wait_event
from registered_host_rdma_probe import (POOL_BYTES, checked_end, emit_event, pattern,
    validate_regions, validate_rw_range, verify_local, write_local)

PROTOCOL = 'external-torch-cpu-registered-rdma-v1'
PAYLOADS = (64, 1 << 20, 18 << 20)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank', type=int, choices=(0, 1), required=True)
    p.add_argument('--world-size', type=int, choices=(2,), default=2)
    p.add_argument('--device', type=int, choices=(0,), required=True)
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
        p.error('timeouts must be finite and positive')
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
            p.error('store/NIC require TCP host:port; coordination requires HTTP(S) host:port')
    return a


def build_plan(a):
    return dict(status='PLAN_ONLY', protocol=PROTOCOL, rank=a.rank, world_size=2,
        role='external_CPU_writer_reader' if a.rank == 0 else 'HOST_owner',
        owner_rank=1, client_rank=0, device=a.device, run_id=a.run_id,
        store_url=a.store_url, coord_url=a.coord_url, device_nic=a.device_nic,
        payload_bytes=a.payload_bytes, copy_types=dict(write='H2G', read='G2H'),
        allocator='torch.empty uint8 device=cpu pin_memory=False; ordinary allocator alignment',
        page_alignment_requested=False, npu_payload_allocation=False,
        source_restore_actual_pointers_registered=True, swap_space_size='0',
        create2=dict(id=0, local_dram_size=POOL_BYTES, max_dram_size=POOL_BYTES,
            local_hbm_size=0, max_hbm_size=0, data_op_type='DEVICE_RDMA',
            enable_56bits_gva=True, flags=0),
        payload_via_rpc=False, fallback_allowed=False,
        external_process_wait_required=True, handle_wait_is_completion_evidence=False,
        actual_transport='NOT_VERIFIED_READ_NATIVE_LOG',
        actual_hal_host_register='NOT_VERIFIED_READ_NATIVE_REGISTER_LOG',
        scope='external_CPU_allocator_registration_primitive_not_LMCache_MemoryObj_allocator_validation',
        completion_basis='zero sync copy returns, independent owner full A/B, CPU restore full A/B, real waits')


def contract(report):
    return {key: report[key] for key in ('protocol', 'world_size', 'owner_rank', 'client_rank',
        'store_url', 'device_nic', 'payload_bytes', 'copy_types', 'allocator',
        'page_alignment_requested', 'swap_space_size', 'create2')}


def validate_external_ranges(source, restore, n, own_hva, pool_size, maps_text=None):
    """Reject overlapping CPU tensors and any accidental use of SDK HOST payload storage."""
    pool_end = checked_end(own_hva, pool_size)
    end_source, end_restore = checked_end(source, n), checked_end(restore, n)
    if max(source, restore) < min(end_source, end_restore):
        raise ValueError('external CPU source and restore overlap')
    for address, end in ((source, end_source), (restore, end_restore)):
        if max(address, own_hva) < min(end, pool_end):
            raise ValueError('external CPU tensor overlaps SDK HOST pool')
        validate_rw_range(address, n, maps_text)


def pointer_metadata(pointer, n, page_size):
    checked_end(pointer, n)
    if page_size <= 0 or page_size & (page_size - 1):
        raise ValueError('page size must be a positive power of two')
    return dict(pointer=pointer, bytes=n, page_size=page_size,
        page_offset=pointer % page_size, page_aligned=pointer % page_size == 0,
        covering_page_begin=pointer // page_size * page_size,
        covering_page_end=(pointer + n + page_size - 1) // page_size * page_size)


def require_content(result, n):
    if result.get('equal') is not True or result.get('verified_bytes') != n:
        raise AssertionError('full content check failed')


def role_protocol(rank, n, actions, mark, wait):
    if rank == 0:
        actions['write_A']()
        mark('client-pushed-A', completed_bytes=n)
        require_content(wait('owner-verified-A', 1), n)
        actions['destroy_source']()
        for label in ('A', 'B'):
            if label == 'B':
                mark('request-owner-B')
                require_content(wait('owner-wrote-B', 1), n)
            actions['reset_restore']()
            actions['read'](label)
            actions['verify_restore'](label)
        mark('client-verified-B', equal=True, verified_bytes=n)
    elif rank == 1:
        if wait('client-pushed-A', 0).get('completed_bytes') != n:
            raise AssertionError('client write byte count mismatch')
        actions['owner_verify']('A')
        mark('owner-verified-A', equal=True, verified_bytes=n)
        wait('request-owner-B', 0)
        actions['owner_write']('B')
        actions['owner_verify']('B')
        mark('owner-wrote-B', equal=True, verified_bytes=n)
        require_content(wait('client-verified-B', 0), n)
    else:
        raise ValueError('unsupported rank')
    mark('payload-done-' + str(rank))
    wait('payload-done-' + str(1-rank), 1-rank)


def run(a, report, emit):
    # Device context is required by DEVICE_RDMA; payload Tensors remain CPU-only.
    os.environ['MF_HYBM_RDMA_SWAP_SPACE_SIZE'] = '0'
    import torch
    import torch_npu  # noqa: F401
    import memfabric_hybrid as mf
    bm = mf.bm
    handle = None
    mf_ready = bm_ready = joined = False
    retained, registered = [], []
    n = a.payload_bytes
    error = error_tb = None

    def mark(name, **metadata):
        publish(a.coord_url, a.run_id, name,
            dict(protocol=PROTOCOL, name=name, rank=a.rank, **metadata), min(5, a.timeout))
        emit('CONTROL_PUBLISHED', name=name)

    def wait(name, rank):
        result = wait_event(a.coord_url, a.run_id, name, a.timeout)
        if result.get('protocol') != PROTOCOL or result.get('name') != name or result.get('rank') != rank:
            raise ValueError('peer protocol, event or rank mismatch')
        emit('CONTROL_OBSERVED', name=name, peer_rank=rank)
        return result

    def publish_failure(exc):
        try:
            mark('failed-' + str(a.rank), error=str(exc))
        except BaseException:
            report['failure_publish_error'] = traceback.format_exc()
            emit('FAILURE_PUBLISH_ERROR', traceback=report['failure_publish_error'])

    def record_content(result, step):
        report[step] = result
        emit('CONTENT_CHECK', step=step, **result)
        require_content(result, n)

    try:
        own_contract = contract(report)
        mark('config-ready-' + str(a.rank), contract=own_contract)
        if wait('config-ready-' + str(1-a.rank), 1-a.rank).get('contract') != own_contract:
            raise ValueError('external CPU endpoint/creation contract mismatch')
        torch.npu.set_device(a.device)
        report.update(sdk_path=mf.__file__, visible_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'))
        emit('DEVICE_CONTEXT_SELECTED_CPU_PAYLOAD', logical_device=a.device,
             sdk_path=mf.__file__, visible_devices=report['visible_devices'])
        mf.set_log_level(0)
        call_step('mf.initialize', mf.initialize, emit)
        mf_ready = True
        cfg = bm.BmConfig()
        cfg.auto_ranking = False
        cfg.rank_id = a.rank
        cfg.start_store = a.rank == 0
        cfg.init_timeout = cfg.create_timeout = cfg.operation_timeout = a.init_timeout
        cfg.set_nic(a.device_nic)
        call_step('bm.initialize', lambda: bm.initialize(a.store_url, 2, a.device, cfg), emit)
        bm_ready = True
        emit('CREATE_BEGIN', requested=report['create2'])
        handle = bm.create2(**dict(report['create2'], data_op_type=bm.BmDataOpType.DEVICE_RDMA))
        if handle is None:
            raise RuntimeError('create2 returned None')
        call_step('join', handle.join, emit)
        joined = True
        own_gva = int(handle.peer_rank_ptr(a.rank, bm.BmMemType.HOST))
        owner_gva = int(handle.peer_rank_ptr(1, bm.BmMemType.HOST))
        own_hva = int(handle.gva_to_va(own_gva, bm.BmMemType.LOCAL_HOST))
        pool_size = int(handle.local_mem_size(bm.BmMemType.HOST))
        checked_end(owner_gva, n)
        checked_end(own_hva, pool_size)
        report['mapping'] = dict(own_gva=own_gva, own_hva=own_hva, owner_gva=owner_gva,
                                local_pool_size=pool_size)
        emit('SDK_HOST_MAPPING', **report['mapping'])
        source = restore = None
        if a.rank == 0:
            source = torch.empty((n,), dtype=torch.uint8, device='cpu', pin_memory=False)
            restore = torch.empty((n,), dtype=torch.uint8, device='cpu', pin_memory=False)
            retained.extend([source, restore])
            for tensor in retained:
                if (tensor.device.type != 'cpu' or tensor.dtype != torch.uint8 or
                        not tensor.is_contiguous() or tensor.numel() != n or tensor.storage_offset() != 0):
                    raise ValueError('ordinary CPU endpoint tensor requirements failed')
            backing = bytearray(pattern(n, 'A'))
            source.copy_(torch.frombuffer(backing, dtype=torch.uint8))
            restore.fill_(90)
            validate_external_ranges(source.data_ptr(), restore.data_ptr(), n, own_hva, pool_size)
            page_size = os.sysconf('SC_PAGE_SIZE')
            report['external_CPU_tensors'] = dict(allocator=report['allocator'],
                source=pointer_metadata(source.data_ptr(), n, page_size),
                restore=pointer_metadata(restore.data_ptr(), n, page_size), tensor_device='cpu')
            emit('EXTERNAL_CPU_TENSORS', **report['external_CPU_tensors'])
            for tensor, purpose in ((source, 'source'), (restore, 'restore')):
                pointer = int(tensor.data_ptr())
                call_step('register.external_CPU_' + purpose, lambda pointer=pointer: handle.register(pointer, n),
                    emit, pointer=pointer, requested_bytes=n, allocation=report['allocator'],
                    hal_host_register_result='READ_NATIVE_LOG_NOT_INFERRED_FROM_PUBLIC_RETURN')
                registered.append(pointer)
                report['registered_CPU_pointers'] = list(registered)
        else:
            validate_regions(own_hva, pool_size, [(own_hva, n)])
            write_local(own_hva, bytes([165])*n)
        mark('address-ready-' + str(a.rank), owner_gva=owner_gva)
        if wait('address-ready-' + str(1-a.rank), 1-a.rank).get('owner_gva') != owner_gva:
            raise ValueError('peer owner HOST GVA mismatch')

        def copy(direction, label):
            src, dst, kind = ((source.data_ptr(), owner_gva, 'H2G') if direction == 'write'
                              else (owner_gva, restore.data_ptr(), 'G2H'))
            raw_return = call_step(kind + '_' + label,
                lambda: handle.copy_data(src, dst, n, getattr(bm.BmCopyType, kind), 0),
                emit, src=src, dst=dst, requested_bytes=n, flags=0, copy_type=kind)
            report['completed_' + direction + '_bytes'] += n
            report.setdefault('copies', []).append(dict(direction=direction, pattern=label,
                copy_type=kind, raw_return=raw_return, completed_bytes=n))

        def verify_tensor(tensor, expected, step):
            observed = tensor.numpy().tobytes()
            record_content(dict(equal=observed == expected, verified_bytes=len(observed),
                mismatch_bytes=sum(x != y for x, y in zip(observed, expected)),
                verified_from='ordinary_external_torch_CPU_allocation'), step)

        def write_A():
            verify_tensor(source, pattern(n, 'A'), 'client_source_A')
            copy('write', 'A')

        def destroy_source():
            source.fill_(51)
            verify_tensor(source, bytes([51])*n, 'client_source_destroyed')

        def reset_restore():
            restore.fill_(90)
            verify_tensor(restore, bytes([90])*n, 'client_restore_sentinel')

        role_protocol(a.rank, n, dict(write_A=write_A, destroy_source=destroy_source,
            reset_restore=reset_restore, read=lambda label: copy('read', label),
            verify_restore=lambda label: verify_tensor(restore, pattern(n, label), 'client_restore_' + label),
            owner_write=lambda label: write_local(own_hva, pattern(n, label)),
            owner_verify=lambda label: record_content(verify_local(own_hva, n, label), 'owner_' + label)),
            mark, wait)
        report['payload_passed'] = True
    except BaseException as exc:
        error, error_tb = exc, exc.__traceback__
        report['error'] = traceback.format_exc()
        emit('PROBE_ERROR', traceback=report['error'])
        publish_failure(exc)
    finally:
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
        report['retained_CPU_tensors_through_cleanup'] = len(retained)
    if error is not None:
        raise error.with_traceback(error_tb)
    if report['cleanup_errors']:
        publish_failure(RuntimeError('local SDK cleanup failed'))
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
            emit_event(a, events, event, **data)
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
