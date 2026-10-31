#!/usr/bin/env python3
"""Two-rank registered HOST pool DEVICE_RDMA diagnostic; default PLAN_ONLY."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import socket
import time
import traceback
from urllib.parse import urlparse

from bm_media_probe import call_step, cleanup_steps
from direct_coordination import publish, wait_event

POOL_BYTES = 1 << 30
MAX_PAYLOAD = 64 << 20
ADDRESS_LIMIT = 1 << (ctypes.sizeof(ctypes.c_void_p) * 8)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank', type=int, choices=(0, 1), required=True)
    p.add_argument('--world-size', type=int, choices=(2,), default=2)
    p.add_argument('--device', type=int, required=True)
    p.add_argument('--payload-bytes', type=int, default=1 << 20)
    p.add_argument('--device-nic', default='tcp://127.0.0.1:10005')
    p.add_argument('--enable-56bits-gva', action='store_true')
    p.add_argument('--copy-interface', choices=('h2g-g2h', 'g2g', 'l2g-g2l'), default='h2g-g2h',
                   help='g2g uses managed HOST GVAs at both endpoints; local CPU access still uses HVA')
    p.add_argument('--local-endpoint', choices=('host', 'device', 'torch-registered'), default='host',
                   help='device uses registered HBM and real NPU Tensors with explicit local D2D staging')
    p.add_argument('--store-url', required=True)
    p.add_argument('--coord-url', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--init-timeout', type=int, default=120)
    p.add_argument('--allow-npu', action='store_true')
    p.add_argument('--benchmark-iterations', type=int, default=0,
                   help='Explicit torch-registered synchronous copy sampling; 0 preserves functional-only behavior')
    p.add_argument('--benchmark-warmup', type=int, default=3)
    a = p.parse_args(argv)
    if a.local_endpoint == 'device' and a.copy_interface != 'g2g':
        p.error('device local endpoint requires explicit --copy-interface g2g')
    if (a.local_endpoint == 'torch-registered') != (a.copy_interface == 'l2g-g2l'):
        p.error('torch-registered local endpoint requires --copy-interface l2g-g2l and vice versa')
    if not 0 <= a.benchmark_iterations <= 10000 or not 0 <= a.benchmark_warmup <= 1000:
        p.error('benchmark iterations must be 0..10000 and warmup 0..1000')
    if a.benchmark_iterations and (a.local_endpoint, a.copy_interface) not in (
            ('torch-registered', 'l2g-g2l'), ('host', 'g2g')):
        p.error('benchmark requires torch-registered/l2g-g2l or host/g2g')
    if not 0 < a.payload_bytes <= MAX_PAYLOAD or a.payload_bytes % 32:
        p.error('payload-bytes must be a positive multiple of 32, at most 64 MiB')
    if a.device < 0 or a.timeout <= 0 or a.init_timeout <= 0:
        p.error('device and timeouts must be valid')
    if not a.run_id or not a.run_id.isascii() or any(not (c.isalnum() or c in '-_') for c in a.run_id):
        p.error('run-id requires ASCII letters, digits, dash or underscore')
    for value, scheme in ((a.store_url, 'tcp'), (a.device_nic, 'tcp'), (a.coord_url, 'http')):
        parsed = urlparse(value)
        try:
            valid = parsed.hostname and parsed.port and 0 < parsed.port < 65536
        except ValueError:
            valid = False
        if not valid or parsed.scheme not in (('http', 'https') if scheme == 'http' else ('tcp',)):
            p.error('store/NIC require tcp://host:port and coordination requires HTTP(S) host:port')
    return a


def build_plan(a):
    return dict(status='PLAN_ONLY', rank=a.rank, world_size=2, device=a.device,
        run_id=a.run_id, store_url=a.store_url, coord_url=a.coord_url,
        payload_bytes=a.payload_bytes, source_offset=0, restore_offset=2*a.payload_bytes,
        copy_interface=a.copy_interface, local_endpoint=a.local_endpoint,
        benchmark_iterations=a.benchmark_iterations, benchmark_warmup=a.benchmark_warmup,
        local_device_staging=a.local_endpoint == 'device',
        data_op_type='DEVICE_RDMA', device_nic=a.device_nic,
        swap_space_size='0', fallback_allowed=False, payload_via_rpc=False,
        create2=dict(id=0, local_dram_size=POOL_BYTES, max_dram_size=POOL_BYTES,
                     local_hbm_size=POOL_BYTES if a.local_endpoint == 'device' else 0,
                     max_hbm_size=POOL_BYTES if a.local_endpoint == 'device' else 0,
                     data_op_type='DEVICE_RDMA',
                     enable_56bits_gva=a.enable_56bits_gva, flags=0),
        external_process_wait_required=True, handle_wait_is_completion_evidence=False,
        completion_basis='synchronous copy return + full owner/reader contents + external process wait',
        scope=('registered_torch_NPU_to_remote_HOST_RDMA_without_local_staging' if a.local_endpoint == 'torch-registered' else
               'registered_DEVICE_to_remote_HOST_RDMA_with_local_D2D_staging' if a.local_endpoint == 'device'
               else 'independent_registered_HOST_DEVICE_RDMA_primitive_not_SDMA_or_LMCache'),
        actual_transport='NOT_VERIFIED_READ_NATIVE_LOG')


def copy_request(interface, direction, own_gva, remote, hva, restore_hva, size):
    if direction not in ('write', 'read') or interface not in ('h2g-g2h', 'g2g', 'l2g-g2l'):
        raise ValueError('unsupported copy interface or direction')
    if interface == 'g2g':
        return ((own_gva, remote, 'G2G') if direction == 'write' else
                (remote, own_gva + 2*size, 'G2G'))
    if interface == 'l2g-g2l':
        return ((hva, remote, 'L2G') if direction == 'write' else
                (remote, restore_hva, 'G2L'))
    return ((hva, remote, 'H2G') if direction == 'write' else
            (remote, restore_hva, 'G2H'))


def sample_sync_copy(operation, warmup, iterations, size):
    result = dict(warmup_samples=[], samples=[], completed_bytes=0,
                  passed=True, timing='perf_counter_ns around synchronous copy_data API only')
    for index in range(warmup + iterations):
        started = time.perf_counter_ns()
        try:
            raw_return = operation()
            elapsed = time.perf_counter_ns() - started
            item = dict(index=index, raw_return=raw_return, elapsed_ns=elapsed)
        except BaseException:
            elapsed = time.perf_counter_ns() - started
            item = dict(index=index, raw_return=None, elapsed_ns=elapsed, error=traceback.format_exc())
        result['warmup_samples' if index < warmup else 'samples'].append(item)
        if item['raw_return'] != 0:
            result['passed'] = False
            break
        result['completed_bytes'] += size
    return result


def pattern(size, label):
    seed = {'A': 43, 'B': 172}[label]
    return bytes((i * 17 + (i // 251) * 29 + seed) % 256 for i in range(size))


def checked_end(address, size):
    if not isinstance(address, int) or not isinstance(size, int) or address <= 0 or size <= 0:
        raise ValueError('address and length must be positive integers')
    end = address + size
    if end > ADDRESS_LIMIT or address >= ADDRESS_LIMIT:
        raise ValueError('pointer range overflow')
    return end


def validate_device_mapping(gva, pool_size, dva, restore_dva, size):
    checked_end(gva, pool_size)
    checked_end(dva, pool_size)
    checked_end(restore_dva, size)
    if pool_size < 3*size or restore_dva != dva + 2*size:
        raise ValueError('invalid registered DEVICE source/restore pool mapping')


def validate_rw_range(address, size, maps_text=None):
    end = checked_end(address, size)
    if maps_text is None:
        maps_text = Path('/proc/self/maps').read_text(encoding='utf-8')
    cursor = address
    for line in maps_text.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        lo, hi = (int(v, 16) for v in fields[0].split('-'))
        if hi <= cursor:
            continue
        if lo > cursor:
            break
        if not fields[1].startswith('rw'):
            raise ValueError('CPU range lacks read/write permission')
        cursor = min(end, hi)
        if cursor == end:
            return
    raise ValueError('CPU range is not fully covered by /proc/self/maps rw mappings')


def validate_regions(base, pool_size, regions, maps_text=None):
    pool_end = checked_end(base, pool_size)
    intervals = []
    for address, size in regions:
        end = checked_end(address, size)
        if address < base or end > pool_end:
            raise ValueError('local region exceeds SDK HOST pool')
        validate_rw_range(address, size, maps_text)
        intervals.append((address, end))
    intervals.sort()
    if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
        raise ValueError('local source/restore ranges overlap')


def write_local(address, data, maps_text=None):
    validate_rw_range(address, len(data), maps_text)
    ctypes.memmove(address, data, len(data))


def verify_local(address, size, label, maps_text=None):
    validate_rw_range(address, size, maps_text)
    observed = ctypes.string_at(address, size)
    expected = pattern(size, label)
    equal = observed == expected
    return dict(equal=equal, verified_bytes=len(observed), pattern=label,
                mismatch_bytes=0 if equal else sum(a != b for a, b in zip(observed, expected)))


def run(a, report, emit):
    # This function is entered only with explicit --allow-npu.
    os.environ['MF_HYBM_RDMA_SWAP_SPACE_SIZE'] = '0'
    import torch
    import torch_npu  # noqa: F401
    import memfabric_hybrid as mf
    bm = mf.bm
    handle = None
    mf_ready = bm_ready = joined = False
    error = error_tb = None
    n = a.payload_bytes
    retained_tensors = []  # Keep real NPU allocations alive until all SDK teardown is done.
    registered_tensors = []
    contract = {k: report[k] for k in ('world_size', 'store_url', 'payload_bytes',
        'device_nic', 'create2', 'source_offset', 'restore_offset', 'swap_space_size', 'copy_interface', 'local_endpoint',
        'benchmark_iterations', 'benchmark_warmup')}

    def mark(name, **data):
        publish(a.coord_url, a.run_id, name, dict(name=name, rank=a.rank,
            protocol='registered-host-rdma-v1', **data), min(5, a.timeout))

    def wait(name):
        value = wait_event(a.coord_url, a.run_id, name, a.timeout)
        if value.get('protocol') != 'registered-host-rdma-v1':
            raise ValueError('peer protocol mismatch')
        return value

    def record_content(result, step):
        emit('CONTENT_CHECK', step=step, **result)
        report[step] = result
        if not result['equal'] or result['verified_bytes'] != n:
            raise AssertionError('full content mismatch: ' + step)

    def verify(address, label, step):
        record_content(verify_local(address, n, label), step)

    def remote_copy(src, dst, copy_type, step, direction, sample=False):
        native_copy, native_type = handle.copy_data, getattr(bm.BmCopyType, copy_type)
        operation = lambda: native_copy(src, dst, n, native_type, 0)
        counter = 'completed_' + direction + '_bytes'
        if sample and a.benchmark_iterations:
            emit('COPY_SAMPLING_BEGIN', step=step, requested_bytes=n,
                 iterations=a.benchmark_iterations, warmup=a.benchmark_warmup)
            # Public SDK log levels: 0 debug, 1 info, 2 warn, 3 error.
            # Suppress routine per-copy native printing while retaining warnings/errors.
            call_step('sampling.log_level_warn', lambda: mf.set_log_level(2), emit, checked=False)
            try:
                result = sample_sync_copy(operation, a.benchmark_warmup, a.benchmark_iterations, n)
            finally:
                call_step('sampling.restore_log_level_debug', lambda: mf.set_log_level(0), emit, checked=False)
            result['native_log_level_during_samples'] = '2-warn; debug restored outside timing loop'
            report.setdefault('copy_samples', {})[direction] = result
            report[counter] += result['completed_bytes']
            emit('COPY_SAMPLING_END', step=step, passed=result['passed'], completed_bytes=result['completed_bytes'])
            if not result['passed']:
                raise RuntimeError('synchronous copy sampling failed: ' + step)
        else:
            call_step(step, operation, emit, requested_bytes=n, src=src, dst=dst,
                      flags=0, copy_interface=a.copy_interface)
            report[counter] += n

    try:
        mark('config-ready-' + str(a.rank), contract=contract)
        if wait('config-ready-' + str(1-a.rank)).get('contract') != contract:
            raise ValueError('peer configuration mismatch')
        torch.npu.set_device(a.device)
        emit('DEVICE_SELECTED', logical_device=a.device, sdk_path=mf.__file__,
             swap_space_size=os.environ['MF_HYBM_RDMA_SWAP_SPACE_SIZE'])
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
        opts = dict(report['create2'], data_op_type=bm.BmDataOpType.DEVICE_RDMA)
        emit('CREATE_BEGIN', requested=report['create2'])
        handle = bm.create2(**opts)
        if handle is None:
            raise RuntimeError('create2 returned None')
        call_step('join', handle.join, emit)
        joined = True
        own_gva = int(handle.peer_rank_ptr(a.rank, bm.BmMemType.HOST))
        remote = int(handle.peer_rank_ptr(1, bm.BmMemType.HOST))
        checked_end(own_gva, 3*n)
        checked_end(remote, n)
        pool_size = int(handle.local_mem_size(bm.BmMemType.HOST))
        hva = int(handle.gva_to_va(own_gva, bm.BmMemType.LOCAL_HOST))
        restore = int(handle.gva_to_va(own_gva + 2*n, bm.BmMemType.LOCAL_HOST))
        if restore != hva + 2*n:
            raise ValueError('SDK pool mapping is not contiguous at restore offset')
        validate_regions(hva, pool_size, [(hva, n), (restore, n)])
        report['local_mapping'] = dict(gva=own_gva, hva=hva, restore_hva=restore,
            pool_size=pool_size, owner_gva=remote)
        emit('REGISTERED_LOCAL_RANGES_VALIDATED', **report['local_mapping'])
        copy_gva = own_gva
        source_tensor = restored_tensor = None
        if a.rank == 0 and a.local_endpoint in ('device', 'torch-registered'):
            if a.local_endpoint == 'device':
                copy_gva = int(handle.peer_rank_ptr(a.rank, bm.BmMemType.DEVICE))
                device_size = int(handle.local_mem_size(bm.BmMemType.DEVICE))
                dva = int(handle.gva_to_va(copy_gva, bm.BmMemType.LOCAL_DEVICE))
                restore_dva = int(handle.gva_to_va(copy_gva + 2*n, bm.BmMemType.LOCAL_DEVICE))
                validate_device_mapping(copy_gva, device_size, dva, restore_dva, n)
                report['local_device_mapping'] = dict(gva=copy_gva, dva=dva,
                    restore_gva=copy_gva+2*n, restore_dva=restore_dva, pool_size=device_size)
                emit('REGISTERED_DEVICE_RANGES_VALIDATED', **report['local_device_mapping'])
            host_source = bytearray(pattern(n, 'A'))
            source_tensor = torch.frombuffer(host_source, dtype=torch.uint8).to(f'npu:{a.device}')
            restored_tensor = torch.full((n,), 90, dtype=torch.uint8, device=f'npu:{a.device}')
            retained_tensors.extend([host_source, source_tensor, restored_tensor])
            for tensor in (source_tensor, restored_tensor):
                if tensor.device.type != 'npu' or not tensor.is_contiguous() or tensor.numel() != n:
                    raise ValueError('staging requires contiguous real NPU byte tensors')
            report['npu_tensors'] = dict(source_ptr=source_tensor.data_ptr(),
                restore_ptr=restored_tensor.data_ptr(), tensor_device=str(source_tensor.device), bytes=n)
            emit('REAL_NPU_TENSORS', **report['npu_tensors'])
            if a.local_endpoint == 'torch-registered':
                torch.npu.synchronize(a.device)
                ranges = sorted((tensor.data_ptr(), checked_end(tensor.data_ptr(), n))
                                for tensor in (source_tensor, restored_tensor))
                if ranges[0][1] > ranges[1][0]:
                    raise ValueError('real NPU source/restore allocations overlap')
                for tensor in (source_tensor, restored_tensor):
                    pointer = tensor.data_ptr()
                    registered_started = time.perf_counter_ns()
                    call_step('REGISTER_TORCH_NPU', lambda pointer=pointer: handle.register(pointer, n),
                              emit, pointer=pointer, requested_bytes=n)
                    report.setdefault('registration_timings', []).append(dict(pointer=pointer,
                        elapsed_ns=time.perf_counter_ns()-registered_started,
                        timing='registration wrapper including event logging; separate from copy samples'))
                    registered_tensors.append(pointer)
                report['registered_torch_pointers'] = list(registered_tensors)

        def stage_to_pool(tensor, target_gva, step):
            torch.npu.synchronize(a.device)
            call_step(step, lambda: handle.copy_data(tensor.data_ptr(), target_gva, n, bm.BmCopyType.L2G, 0),
                      emit, requested_bytes=n, src=tensor.data_ptr(), dst=target_gva,
                      flags=0, scope='local_DEVICE_TO_DEVICE_staging')
            torch.npu.synchronize(a.device)

        if source_tensor is not None and a.local_endpoint == 'device':
            stage_to_pool(source_tensor, copy_gva, 'LOCAL_D2D_SOURCE_A')
            stage_to_pool(restored_tensor, copy_gva+2*n, 'LOCAL_D2D_RESTORE_SENTINEL')
        write_local(hva, pattern(n, 'A') if a.rank == 0 and source_tensor is None else bytes([165])*n)
        write_local(restore, bytes([90])*n)
        mark('address-ready-' + str(a.rank), own_gva=own_gva, owner_gva=remote, pool_size=pool_size)
        peer = wait('address-ready-' + str(1-a.rank))
        if peer.get('owner_gva') != remote or peer.get('pool_size') != pool_size:
            raise ValueError('owner address or pool size mismatch')
        if a.rank == 0:
            source_pointer = source_tensor.data_ptr() if a.local_endpoint == 'torch-registered' else hva
            restore_pointer = restored_tensor.data_ptr() if a.local_endpoint == 'torch-registered' else restore
            src, dst, copy_type = copy_request(a.copy_interface, 'write', copy_gva, remote, source_pointer, restore_pointer, n)
            remote_copy(src, dst, copy_type, copy_type + '_WRITE_A', 'write', sample=True)
            mark('push-complete', completed_bytes=n)
            owner = wait('owner-verified-A')
            if owner.get('verified_bytes') != n or owner.get('equal') is not True:
                raise AssertionError('owner did not verify full A')
            report['owner_full_content_equal'] = True
            if source_tensor is None:
                write_local(hva, bytes([51])*n)
            else:
                source_tensor.fill_(51)
                torch.npu.synchronize(a.device)
                if a.local_endpoint == 'device':
                    stage_to_pool(source_tensor, copy_gva, 'LOCAL_D2D_DESTROY_SOURCE')
            for label in ('A', 'B'):
                if label == 'B':
                    mark('request-owner-B')
                    wait('owner-wrote-B')
                    if restored_tensor is None:
                        write_local(restore, bytes([90])*n)
                    else:
                        restored_tensor.fill_(90)
                        torch.npu.synchronize(a.device)
                        if a.local_endpoint == 'device':
                            stage_to_pool(restored_tensor, copy_gva+2*n, 'LOCAL_D2D_RESET_RESTORE')
                src, dst, copy_type = copy_request(a.copy_interface, 'read', copy_gva, remote, source_pointer, restore_pointer, n)
                remote_copy(src, dst, copy_type, copy_type + '_READ_' + label, 'read', sample=label == 'A')
                if restored_tensor is None:
                    verify(restore, label, 'reader_' + label)
                else:
                    if a.local_endpoint == 'device':
                        call_step('LOCAL_D2D_TO_TENSOR_' + label,
                            lambda: handle.copy_data(copy_gva+2*n, restored_tensor.data_ptr(), n, bm.BmCopyType.G2L, 0),
                            emit, requested_bytes=n, src=copy_gva+2*n, dst=restored_tensor.data_ptr(),
                            flags=0, scope='local_DEVICE_TO_DEVICE_staging')
                    torch.npu.synchronize(a.device)
                    observed = restored_tensor.cpu().numpy().tobytes()
                    expected = pattern(n, label)
                    record_content(dict(equal=observed == expected, verified_bytes=len(observed),
                        pattern=label, verified_from=('registered_real_NPU_tensor' if a.local_endpoint == 'torch-registered'
                                                     else 'real_NPU_tensor_after_local_D2D'),
                        mismatch_bytes=sum(x != y for x, y in zip(observed, expected))), 'reader_' + label)
            mark('reader-verified-B', verified_bytes=n, equal=True)
        else:
            if wait('push-complete').get('completed_bytes') != n:
                raise AssertionError('unexpected push completed bytes')
            verify(hva, 'A', 'owner_A')
            mark('owner-verified-A', verified_bytes=n, equal=True)
            wait('request-owner-B')
            write_local(hva, pattern(n, 'B'))
            verify(hva, 'B', 'owner_B')
            mark('owner-wrote-B')
            if wait('reader-verified-B').get('verified_bytes') != n:
                raise AssertionError('reader did not verify full B')
        mark('both-done-' + str(a.rank))
        wait('both-done-' + str(1-a.rank))
        report['payload_passed'] = True
    except BaseException as exc:
        error, error_tb = exc, exc.__traceback__
        report['error'] = traceback.format_exc()
        emit('PROBE_ERROR', traceback=report['error'])
        try:
            mark('failed-' + str(a.rank), error=str(exc))
        except BaseException:
            report['failure_publish_error'] = traceback.format_exc()
            emit('FAILURE_PUBLISH_ERROR', traceback=report['failure_publish_error'])
    finally:
        steps = []
        # DEVICE_RDMA wait is a no-op; never label this independent completion.
        if handle is not None:
            for pointer in reversed(registered_tensors):
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
        report['retained_npu_objects_through_cleanup'] = len(retained_tensors)
    if error is not None:
        raise error.with_traceback(error_tb)
    if report['cleanup_errors']:
        raise RuntimeError('SDK cleanup failed')


def emit_event(a, events, event, **data):
    line = json.dumps(dict(event=event, unix_ns=time.time_ns(), mono_ns=time.monotonic_ns(),
        pid=os.getpid(), hostname=socket.gethostname(), rank=a.rank, device=a.device,
        run_id=a.run_id, **data), sort_keys=True)
    events.write(line + '\n')
    events.flush()
    print(line, flush=True)


def main(argv=None):
    a = arguments(argv)
    report = build_plan(a)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, sort_keys=True)
    if not a.allow_npu:
        print(json.dumps(report, sort_keys=True), flush=True)
        return
    with a.output.with_suffix(a.output.suffix + '.events.jsonl').open('x', encoding='utf-8') as events:
        def emit(event, **data):
            emit_event(a, events, event, **data)
        report.update(status='RUNNING', payload_passed=False,
                      completed_write_bytes=0, completed_read_bytes=0)
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
