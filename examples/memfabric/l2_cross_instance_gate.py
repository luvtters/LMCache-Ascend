#!/usr/bin/env python3
"""Two physical hosts, one independent L2 adapter per rank. Default: plan only."""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
import uuid

from l2_memoryobj_boundary_gate import load, lookup, store
from pool_coordination import publish, wait_event


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True, help='A=0, B=1')
    parser.add_argument('--device', type=int, default=0, help='local logical NPU')
    parser.add_argument('--storage-rank', type=int, choices=(0, 1), default=0)
    parser.add_argument('--store-url', required=True, help='BM TCP endpoint on rank 0')
    parser.add_argument('--device-nic', required=True, help='this rank\'s device NIC URI')
    parser.add_argument('--join-coord-url', required=True)
    parser.add_argument('--join-group-id', required=True, help='fresh coordinator --run-id')
    parser.add_argument('--data-op-type', choices=('DEVICE_RDMA',), default='DEVICE_RDMA')
    parser.add_argument('--enable-56bits-gva', action='store_true')
    parser.add_argument('--payload-bytes', type=int, default=18 << 20)
    parser.add_argument('--timeout', type=float, default=90, help='seconds per task or peer event')
    parser.add_argument('--worker-timeout', type=float, default=480, help='hard limit for local worker, seconds')
    parser.add_argument('--output', type=Path, required=True, help='new rank-specific JSON report')
    parser.add_argument('--allow-npu', action='store_true')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.device < 0 or not 0 < args.payload_bytes <= 32 << 20 or args.payload_bytes % 4096
            or not all(math.isfinite(v) and v > 0 for v in (args.timeout, args.worker_timeout))):
        parser.error('need nonnegative device, page-multiple payload fitting one 32 MiB slot, positive finite timeouts')
    if not args.join_group_id or any(not (c.isalnum() or c in '-_') for c in args.join_group_id):
        parser.error('join-group-id must contain only alphanumerics, hyphens or underscores')
    return args


def post(args, name, **data):
    return publish(args.join_coord_url, args.join_group_id, name, dict(name=name, **data), min(5, args.timeout))


def receive(args, name):
    return wait_event(args.join_coord_url, args.join_group_id, name, args.timeout)


def failure(args, stage, error):
    try:
        post(args, f'failed-{args.rank}-{stage}', error=str(error), pid=os.getpid())
    except Exception as notification_error:
        print(f'peer failure notification unavailable: {notification_error}', file=sys.stderr, flush=True)


def check(report, case, passed, **data):
    item = dict(case=case, passed=bool(passed), **data)
    report.setdefault('checks', []).append(item)
    print(json.dumps(item), flush=True)
    if not passed:
        raise AssertionError(str(item))


def preflight(args, report):
    seed = os.environ.get('PYTHONHASHSEED', '')
    if not seed.isascii() or not seed.isdecimal() or int(seed) > 4294967295:
        raise ValueError('set PYTHONHASHSEED to the same explicit integer before launching both interpreters')
    boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    uuid.UUID(boot_id)
    shared = {k: v for k, v in report['adapter_params'].items() if k not in ('device_id', 'rank_id', 'device_nic')}
    contract = dict(params=shared, payload_bytes=args.payload_bytes, pythonhashseed=seed)
    local = dict(rank=args.rank, host=socket.gethostname(), boot_id=boot_id, pid=os.getpid(),
                 nonce=uuid.uuid4().hex, contract=contract)
    post(args, f'gate-rank-{args.rank}', **local)
    peer = receive(args, f'gate-rank-{1 - args.rank}')
    report['participants'] = [local, peer]
    check(report, 'two_rank_contract', peer['rank'] == 1 - args.rank and peer['contract'] == contract
          and peer['boot_id'] != boot_id, peer_host=peer['host'])


def exercise(args, report, adapter, key, obj, torch):
    size = args.payload_bytes
    for generation, shift in enumerate((17, 93), start=1):
        phase = f'g{generation}-'
        expected = ((torch.arange(size, dtype=torch.int32) + shift) % 251).to(torch.uint8)
        if args.rank == 0:
            receive(args, phase + 'reader-missed')
            obj.raw_tensor.copy_(expected)
            result = store(adapter, key, obj, args.timeout)
            check(report, phase + 'store', result.is_successful() and result.bytes_transferred() == size,
                  completed_bytes=result.bytes_transferred(), pattern_shift=shift)
            obj.raw_tensor.fill_(255)
            post(args, phase + 'stored')
            receive(args, phase + 'reader-pinned-and-loaded')
            adapter.delete([key])
            # A fresh lookup on A queues behind its delete. B retains its own pin.
            check(report, phase + 'fresh-lookup-after-delete-misses', not lookup(adapter, key, args.timeout))
            post(args, phase + 'deleted')
            receive(args, phase + 'reader-unlocked')
        else:
            check(report, phase + 'miss-before-write', not lookup(adapter, key, args.timeout))
            post(args, phase + 'reader-missed')
            receive(args, phase + 'stored')
            check(report, phase + 'lookup-and-pin', lookup(adapter, key, args.timeout))
            for state in ('visible', 'deleted-while-pinned'):
                if state == 'deleted-while-pinned':
                    receive(args, phase + 'deleted')
                    check(report, phase + 'new-reader-lookup-misses', not lookup(adapter, key, args.timeout))
                obj.raw_tensor.fill_(255)  # Sentinel is absent from the expected pattern.
                before = adapter.report_status()['load_bytes']
                hit = load(adapter, key, obj, args.timeout)
                completed = adapter.report_status()['load_bytes'] - before
                equal = bool(torch.equal(obj.raw_tensor, expected))
                check(report, phase + state, hit and completed == size and equal, hit=bool(hit),
                      completed_bytes=completed, full_equal=equal, compared_bytes=obj.get_size(), pattern_shift=shift)
                if state == 'visible':
                    post(args, phase + 'reader-pinned-and-loaded')
            adapter.submit_unlock([key])
            # Queue a fresh lookup behind the asynchronous unlock on the same I/O owner.
            check(report, phase + 'miss-after-unlock', not lookup(adapter, key, args.timeout))
            status = adapter.report_status()
            check(report, phase + 'released-last-pin', status['total_pins'] == 0 and status['objects'] == 0,
                  total_pins=status['total_pins'], objects=status['objects'])
            post(args, phase + 'reader-unlocked')


def worker(args, report):
    preflight(args, report)
    import torch
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.memory_allocators.host_memory_allocator import HostMemoryAllocator
    from lmcache.v1.memory_management import MemoryFormat
    from lmcache_ascend.v1.distributed.memfabric_l2_adapter import MemFabricL2Adapter

    torch.set_num_threads(1)
    size = args.payload_bytes
    shape = torch.Size((size,))
    allocator = HostMemoryAllocator(size=size, use_paging=True, shapes=[shape], dtypes=[torch.uint8], fmt=MemoryFormat.BINARY)
    obj = allocator.allocate([shape], [torch.uint8], MemoryFormat.BINARY)
    if obj is None or obj.get_size() != size:
        raise RuntimeError('HostMemoryAllocator could not allocate the requested MemoryObj')
    key = ObjectKey(b'X' * 32, 'l2-cross-instance-gate', 0, 0, '')
    report.update(memory_obj=type(obj).__name__, object_key=repr(key))
    adapter = MemFabricL2Adapter(report['adapter_params'])
    try:
        exercise(args, report, adapter, key, obj, torch)
        status = adapter.report_status()
        report['adapter_before_close'] = status
        check(report, 'rank-complete', status['is_healthy'] and obj.get_ref_count() == 1
              and status['store_bytes'] == (2 * size if args.rank == 0 else 0)
              and status['load_bytes'] == (4 * size if args.rank == 1 else 0), refs=obj.get_ref_count(),
              store_bytes=status['store_bytes'], load_bytes=status['load_bytes'])
        # Each side publishes only after its last transfer, unlock and status check.
        post(args, f'io-done-{args.rank}')
        receive(args, f'io-done-{1 - args.rank}')
    except BaseException as error:
        failure(args, 'io', error)
        raise
    finally:
        adapter.close()
    obj.ref_count_down()
    check(report, 'allocator-after-close', allocator.memcheck())
    report['status'] = 'LOCAL_CONTENT_AND_CLOSE_PASSED'


def supervise(args, report, worker_output):
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--_worker'])
    report['worker_pid'] = child.pid
    try:
        rc = child.wait(timeout=args.worker_timeout)
    except BaseException:
        if child.poll() is None:
            child.kill()
        report['worker_exit_code'] = child.wait()
        raise
    report['worker_exit_code'] = rc
    post(args, f'worker-wait-{args.rank}', rank=args.rank, pid=child.pid, exit_code=rc)
    if rc != 0:
        raise RuntimeError(f'rank {args.rank} worker exited {rc}; see {worker_output} and stderr')
    local = json.loads(worker_output.read_text(encoding='utf-8'))
    if local['status'] != 'LOCAL_CONTENT_AND_CLOSE_PASSED':
        raise RuntimeError('worker exited without successful content and close receipt')
    peer = receive(args, f'worker-wait-{1 - args.rank}')
    report['peer_worker_wait'] = peer
    check(report, 'both-native-workers-exited', peer['rank'] == 1 - args.rank and peer['exit_code'] == 0,
          local_exit_code=rc, peer_exit_code=peer['exit_code'])
    report['status'] = 'TWO_ADAPTER_CONTENT_AND_PROCESS_EXIT_PASSED'


def main():
    args = arguments()
    params = dict(rank_id=args.rank, world_size=2, device_id=args.device, pool_gb=1, slot_size_mb=32,
                  shared_directory=True, storage_rank=args.storage_rank, data_op_type=args.data_op_type,
                  store_url=args.store_url, device_nic=args.device_nic, enable_56bits_gva=args.enable_56bits_gva,
                  join_group_id=args.join_group_id, join_coord_url=args.join_coord_url)
    worker_output = args.output.with_name(args.output.name + '.worker.json')
    report = dict(status='PLAN_ONLY', rank=args.rank, adapter_params=params, payload_bytes=args.payload_bytes,
                  worker_report=str(worker_output), reader_access='remote' if args.storage_rank == 0 else 'local',
                  scope='two independent adapters and real MemoryObjs; synthetic bytes, no inference')
    output = worker_output if args._worker else args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        if args.allow_npu:
            report['status'] = 'STARTED'
        json.dump(report, stream, indent=2)
    if args.allow_npu:
        try:
            if args._worker:
                worker(args, report)
            else:
                supervise(args, report, worker_output)
        except BaseException as error:
            report.update(status='FAILED', error=traceback.format_exc())
            failure(args, 'worker' if args._worker else 'supervisor', error)
            raise
        finally:
            report['ended_unix_ns'] = time.time_ns()
            output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
