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
    p.add_argument('--copy-interface', choices=('h2g-g2h', 'g2g'), default='h2g-g2h',
                   help='g2g uses managed HOST GVAs at both endpoints; local CPU access still uses HVA')
    p.add_argument('--store-url', required=True)
    p.add_argument('--coord-url', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--init-timeout', type=int, default=120)
    p.add_argument('--allow-npu', action='store_true')
    a = p.parse_args(argv)
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
        copy_interface=a.copy_interface,
        data_op_type='DEVICE_RDMA', device_nic=a.device_nic,
        swap_space_size='0', fallback_allowed=False, payload_via_rpc=False,
        create2=dict(id=0, local_dram_size=POOL_BYTES, max_dram_size=POOL_BYTES,
                     local_hbm_size=0, max_hbm_size=0, data_op_type='DEVICE_RDMA',
                     enable_56bits_gva=a.enable_56bits_gva, flags=0),
        external_process_wait_required=True, handle_wait_is_completion_evidence=False,
        completion_basis='synchronous copy return + full owner/reader contents + external process wait',
        scope='independent_registered_HOST_DEVICE_RDMA_primitive_not_SDMA_or_LMCache',
        actual_transport='NOT_VERIFIED_READ_NATIVE_LOG')


def copy_request(interface, direction, own_gva, remote, hva, restore_hva, size):
    if direction not in ('write', 'read') or interface not in ('h2g-g2h', 'g2g'):
        raise ValueError('unsupported copy interface or direction')
    if interface == 'g2g':
        return ((own_gva, remote, 'G2G') if direction == 'write' else
                (remote, own_gva + 2*size, 'G2G'))
    return ((hva, remote, 'H2G') if direction == 'write' else
            (remote, restore_hva, 'G2H'))


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
    contract = {k: report[k] for k in ('world_size', 'store_url', 'payload_bytes',
        'device_nic', 'create2', 'source_offset', 'restore_offset', 'swap_space_size', 'copy_interface')}

    def mark(name, **data):
        publish(a.coord_url, a.run_id, name, dict(name=name, rank=a.rank,
            protocol='registered-host-rdma-v1', **data), min(5, a.timeout))

    def wait(name):
        value = wait_event(a.coord_url, a.run_id, name, a.timeout)
        if value.get('protocol') != 'registered-host-rdma-v1':
            raise ValueError('peer protocol mismatch')
        return value

    def verify(address, label, step):
        result = verify_local(address, n, label)
        emit('CONTENT_CHECK', step=step, **result)
        report[step] = result
        if not result['equal'] or result['verified_bytes'] != n:
            raise AssertionError('full content mismatch: ' + step)

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
        write_local(hva, pattern(n, 'A') if a.rank == 0 else bytes([165])*n)
        write_local(restore, bytes([90])*n)
        mark('address-ready-' + str(a.rank), own_gva=own_gva, owner_gva=remote, pool_size=pool_size)
        peer = wait('address-ready-' + str(1-a.rank))
        if peer.get('owner_gva') != remote or peer.get('pool_size') != pool_size:
            raise ValueError('owner address or pool size mismatch')
        if a.rank == 0:
            src, dst, copy_type = copy_request(a.copy_interface, 'write', own_gva, remote, hva, restore, n)
            call_step(copy_type + '_WRITE_A', lambda: handle.copy_data(src, dst, n, getattr(bm.BmCopyType, copy_type), 0),
                      emit, requested_bytes=n, src=src, dst=dst, flags=0, copy_interface=a.copy_interface)
            report['completed_write_bytes'] = n
            mark('push-complete', completed_bytes=n)
            owner = wait('owner-verified-A')
            if owner.get('verified_bytes') != n or owner.get('equal') is not True:
                raise AssertionError('owner did not verify full A')
            report['owner_full_content_equal'] = True
            write_local(hva, bytes([51])*n)  # Destroy source after independent owner verification.
            for label in ('A', 'B'):
                if label == 'B':
                    mark('request-owner-B')
                    wait('owner-wrote-B')
                    write_local(restore, bytes([90])*n)
                src, dst, copy_type = copy_request(a.copy_interface, 'read', own_gva, remote, hva, restore, n)
                call_step(copy_type + '_READ_' + label, lambda: handle.copy_data(src, dst, n, getattr(bm.BmCopyType, copy_type), 0),
                          emit, requested_bytes=n, src=src, dst=dst, flags=0, copy_interface=a.copy_interface)
                report['completed_read_bytes'] += n
                verify(restore, label, 'reader_' + label)
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
            if joined:
                steps.append(('close.leave', handle.leave, True))
            steps.append(('close.destroy', handle.destroy, False))
        if bm_ready:
            steps.append(('close.bm.uninitialize', bm.uninitialize, False))
        if mf_ready:
            steps.append(('close.mf.uninitialize', mf.uninitialize, False))
        report['cleanup_errors'] = cleanup_steps(steps, emit)
        report['local_cleanup_passed'] = not report['cleanup_errors']
    if error is not None:
        raise error.with_traceback(error_tb)
    if report['cleanup_errors']:
        raise RuntimeError('SDK cleanup failed')


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
            line = json.dumps(dict(event=event, unix_ns=time.time_ns(), mono_ns=time.monotonic_ns(),
                pid=os.getpid(), hostname=socket.gethostname(), rank=a.rank, device=a.device,
                run_id=a.run_id, **data), sort_keys=True)
            events.write(line + '\n')
            events.flush()
            print(line, flush=True)
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
