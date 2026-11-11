#!/usr/bin/env python3
"""Two independent L2 adapters with real MemoryObjs. Default: plan only."""
import argparse
import importlib
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

from l2_memoryobj_boundary_gate import fence, load, lookup, store
from pool_coordination import publish, wait_event


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True, help='A=0, B=1')
    parser.add_argument('--device', type=int, default=0, help='local logical NPU')
    owners = parser.add_mutually_exclusive_group()
    owners.add_argument('--storage-rank', type=int, choices=(0, 1), default=0)
    owners.add_argument('--storage-ranks', type=int, choices=(0, 1), nargs=2)
    parser.add_argument('--pool-gb', type=int, default=1, help='DDR GiB per owner')
    parser.add_argument('--slot-size-mb', type=int, default=32)
    parser.add_argument('--object-count', type=int, default=4, help='distinct objects in the two-owner gate')
    parser.add_argument('--same-host', action='store_true', help='two ranks on distinct devices in one container')
    parser.add_argument('--require-overflow', action='store_true', help='require logical bytes above one owner capacity')
    parser.add_argument('--store-url', required=True, help='BM TCP endpoint on rank 0')
    parser.add_argument('--device-nic', required=True, help='this rank\'s device NIC URI')
    parser.add_argument('--join-coord-url', required=True)
    parser.add_argument('--join-group-id', required=True, help='fresh coordinator --run-id')
    parser.add_argument('--data-op-type', choices=('SDMA', 'DEVICE_RDMA'), default='DEVICE_RDMA')
    parser.add_argument('--adapter-module', default='lmcache_ascend.v1.distributed.memfabric_l2_adapter')
    parser.add_argument('--enable-56bits-gva', action='store_true')
    parser.add_argument('--payload-bytes', type=int, default=18 << 20)
    parser.add_argument('--timeout', type=float, default=90, help='seconds per task or peer event')
    parser.add_argument('--worker-timeout', type=float, default=480, help='hard limit for local worker, seconds')
    parser.add_argument('--output', type=Path, required=True, help='new rank-specific JSON report')
    parser.add_argument('--allow-npu', action='store_true')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    multi_owner = args.storage_ranks is not None
    if (args.device < 0 or args.pool_gb <= 0 or args.slot_size_mb <= 0
            or args.pool_gb * 1024 % args.slot_size_mb or not 0 < args.payload_bytes <= args.slot_size_mb << 20
            or (not multi_owner and args.payload_bytes % 4096)
            or not all(math.isfinite(v) and v > 0 for v in (args.timeout, args.worker_timeout))):
        parser.error('invalid device, pool geometry, payload or timeouts')
    if multi_owner and (args.storage_ranks != [0, 1]
            or not 2 <= args.object_count <= 2 * (args.pool_gb * 1024 // args.slot_size_mb)
            or args.payload_bytes < 8):
        parser.error('need storage ranks 0 1 and at least two distinct objects fitting the two segments')
    if (args.same_host or args.require_overflow) and not multi_owner:
        parser.error('same-host and require-overflow require --storage-ranks 0 1')
    if args.require_overflow and args.object_count * args.payload_bytes <= args.pool_gb << 30:
        parser.error('logical payload must exceed the per-owner DDR capacity')
    if not args.join_group_id or any(not (c.isalnum() or c in '-_') for c in args.join_group_id):
        parser.error('join-group-id must contain only alphanumerics, hyphens or underscores')
    return args


def post(args, name, **data):
    return publish(args.join_coord_url, args.join_group_id, name, dict(name=name, **data), min(5, args.timeout))


def receive(args, name, *, long_phase=False):
    timeout = args.worker_timeout if long_phase else args.timeout
    return wait_event(args.join_coord_url, args.join_group_id, name, timeout)


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
    contract = dict(params=shared, payload_bytes=args.payload_bytes, pythonhashseed=seed,
                    same_host=args.same_host, object_count=args.object_count, require_overflow=args.require_overflow)
    local = dict(rank=args.rank, host=socket.gethostname(), boot_id=boot_id, pid=os.getpid(),
                 device=args.device, nonce=uuid.uuid4().hex, contract=contract)
    post(args, f'gate-rank-{args.rank}', **local)
    peer = receive(args, f'gate-rank-{1 - args.rank}')
    report['participants'] = [local, peer]
    topology_matches = (peer['boot_id'] == boot_id and peer['host'] == local['host']
                        and peer['device'] != args.device) if args.same_host else peer['boot_id'] != boot_id
    check(report, 'two_rank_contract', peer['rank'] == 1 - args.rank and peer['contract'] == contract
          and topology_matches, peer_host=peer['host'])


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


def exercise_multi_owner(args, report, adapter, obj, torch, object_key):
    size = args.payload_bytes
    total = size * args.object_count
    owners = {'0', '1'}
    pattern = torch.arange(256, dtype=torch.uint8).repeat((size + 255) // 256)[:size]
    keys = [object_key(i.to_bytes(32, 'big'), 'l2-two-owner-gate', 0, 0, args.join_group_id)
            for i in range(args.object_count)]
    if args.rank == 1:
        receive(args, 'two-owner-written', long_phase=True)
    for index, key in enumerate(keys):
        pattern[:8] = torch.tensor(list(index.to_bytes(8, 'big')), dtype=torch.uint8)
        if args.rank == 0:
            obj.raw_tensor.copy_(pattern)
            result = store(adapter, key, obj, args.timeout)
            if not result.is_successful() or result.bytes_transferred() != size:
                raise AssertionError(f'object {index}: store did not complete {size} bytes')
        else:
            obj.raw_tensor.zero_()
            if not (lookup(adapter, key, args.timeout) and load(adapter, key, obj, args.timeout)
                    and torch.equal(obj.raw_tensor, pattern)):
                raise AssertionError(f'object {index}: lookup, load or full content comparison failed')
            adapter.submit_unlock([key])
    status = fence(adapter)
    by_owner = status['store_bytes_by_owner' if args.rank == 0 else 'load_bytes_by_owner']
    counts = status['owners']
    check(report, 'two-owner-payload', status['is_healthy'] and status['objects'] == len(keys)
          and status['total_bytes_used'] == total and status['pending_writes'] == status['evicted_objects'] == 0
          and set(counts) == owners and all(data['objects'] > 0 for data in counts.values())
          and set(map(str, by_owner)) == owners and min(by_owner.values()) > 0 and sum(by_owner.values()) == total,
          objects=len(keys), completed_bytes=total, owner_bytes=by_owner, owners=counts,
          full_content_compared=args.rank == 1)
    if args.rank == 0:
        post(args, 'two-owner-written')
        receive(args, 'two-owner-read', long_phase=True)
        adapter.delete(keys[:2])
        remaining = fence(adapter)
        check(report, 'delete-releases-both-owners', remaining['objects'] == len(keys) - 2
              and all(remaining['owners'][rank]['free_slots'] == counts[rank]['free_slots'] + 1 for rank in owners),
              owners=remaining['owners'])
        post(args, 'two-owner-deleted')
    else:
        post(args, 'two-owner-read')
        receive(args, 'two-owner-deleted')
        check(report, 'deleted-keys-miss', all(not lookup(adapter, key, args.timeout) for key in keys[:2]))


def worker(args, report):
    preflight(args, report)
    import torch
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.memory_allocators.host_memory_allocator import HostMemoryAllocator
    from lmcache.v1.memory_management import MemoryFormat
    MemFabricL2Adapter = importlib.import_module(args.adapter_module).MemFabricL2Adapter

    torch.set_num_threads(1)
    size = args.payload_bytes
    shape = torch.Size((size,))
    if args.storage_ranks is not None:
        from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import PagedTensorMemoryAllocator

        backing = torch.empty(size + 4095, dtype=torch.uint8)
        offset = -backing.data_ptr() % 4096
        allocator = PagedTensorMemoryAllocator(backing[offset:offset + size], [shape], [torch.uint8],
                                               MemoryFormat.BINARY)
    else:
        allocator = HostMemoryAllocator(size=size, use_paging=True, shapes=[shape],
                                        dtypes=[torch.uint8], fmt=MemoryFormat.BINARY)
    obj = allocator.allocate([shape], [torch.uint8], MemoryFormat.BINARY)
    if obj is None or obj.get_size() != size:
        raise RuntimeError('HostMemoryAllocator could not allocate the requested MemoryObj')
    key = ObjectKey(b'X' * 32, 'l2-cross-instance-gate', 0, 0, '')
    report.update(memory_obj=type(obj).__name__, object_key=repr(key))
    adapter = MemFabricL2Adapter(report['adapter_params'])
    try:
        if args.storage_ranks is not None:
            exercise_multi_owner(args, report, adapter, obj, torch, ObjectKey)
        else:
            exercise(args, report, adapter, key, obj, torch)
        status = adapter.report_status()
        report['adapter_before_close'] = status
        expected_store = (args.object_count if args.storage_ranks is not None else 2) * size
        expected_load = (args.object_count if args.storage_ranks is not None else 4) * size
        check(report, 'rank-complete', status['is_healthy'] and obj.get_ref_count() == 1
              and status['store_bytes'] == (expected_store if args.rank == 0 else 0)
              and status['load_bytes'] == (expected_load if args.rank == 1 else 0), refs=obj.get_ref_count(),
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
    placement = (dict(storage_ranks=args.storage_ranks) if args.storage_ranks is not None
                 else dict(storage_rank=args.storage_rank))
    params = dict(rank_id=args.rank, world_size=2, device_id=args.device, pool_gb=args.pool_gb,
                  slot_size_mb=args.slot_size_mb, shared_directory=True, **placement, data_op_type=args.data_op_type,
                  store_url=args.store_url, device_nic=args.device_nic, enable_56bits_gva=args.enable_56bits_gva,
                  join_group_id=args.join_group_id, join_coord_url=args.join_coord_url)
    worker_output = args.output.with_name(args.output.name + '.worker.json')
    report = dict(status='PLAN_ONLY', rank=args.rank, adapter_params=params, payload_bytes=args.payload_bytes,
                  worker_report=str(worker_output),
                  reader_access='both-owners' if args.storage_ranks is not None
                  else ('remote' if args.storage_rank == 0 else 'local'),
                  same_host=args.same_host, object_count=args.object_count, require_overflow=args.require_overflow,
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
