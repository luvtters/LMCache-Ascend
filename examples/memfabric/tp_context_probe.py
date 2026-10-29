#!/usr/bin/env python3
"""Opt-in model-free real TP context/RPC probe; default PLAN_ONLY.

Each P/D process owns one real KV shard and BM member. All members register,
then a TP-1-fragment key must miss before a separate complete key can hit.
Only HTTP metadata crosses coordination. The existing production service,
context and scheduler own pool/control/data operations; no production patch.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from types import SimpleNamespace
from urllib.parse import urlparse

from direct_coordination import publish, wait_event

TOKENS = 128
BLOCK_SIZE = 64
SOURCE_BLOCKS = [3, 1]
TARGET_BLOCKS = [4, 2]
MODEL = 'qwen3-synthetic-real-tp-context'
HOSTS = {'137': '192.168.0.223', '12': '192.168.0.50'}
_FAILED_KEEPALIVE = []


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('service', 'p', 'd'), required=True)
    parser.add_argument('--tp-size', type=int, choices=(2, 4, 8), required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--pool-physical-device', type=int, default=0,
                        help='Plan/manifest only; caller applies this service device mask')
    parser.add_argument('--pool-host', choices=tuple(HOSTS), default='12')
    parser.add_argument('--p-host', choices=tuple(HOSTS), default='12')
    parser.add_argument('--d-host', choices=tuple(HOSTS), default='12')
    parser.add_argument('--p-devices', type=lambda value: [int(v) for v in value.split(',')])
    parser.add_argument('--d-devices', type=lambda value: [int(v) for v in value.split(',')])
    parser.add_argument('--pool-memory-type', choices=('host', 'device'), default='host')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ascend-source', type=Path, required=True)
    parser.add_argument('--core-source', type=Path, required=True)
    parser.add_argument('--layers', type=int, choices=(4, 36), default=36)
    parser.add_argument('--slot-size-mb', type=int, default=32)
    parser.add_argument('--store-url')
    parser.add_argument('--server-url')
    parser.add_argument('--service-http-port', type=int, default=28117)
    parser.add_argument('--coord-url')
    parser.add_argument('--timeout', type=float, default=360)
    parser.add_argument('--init-timeout', type=int, default=300)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--allow-npu', action='store_true')
    args = parser.parse_args(argv)
    args.store_url = args.store_url or f'tcp://{HOSTS[args.pool_host]}:28717'
    args.server_url = args.server_url or f'tcp://{HOSTS[args.pool_host]}:25717'
    args.coord_url = args.coord_url or f'http://{HOSTS[args.pool_host]}:29617'
    args.p_devices = args.p_devices or (list(range(8)) if args.tp_size == 8 else [2,3,4,5][:args.tp_size])
    args.d_devices = args.d_devices or [8,9,10,11,12,13,14,15][:args.tp_size]
    if (not 0 <= args.shard < args.tp_size or args.pool_physical_device < 0 or args.timeout <= 0 or
            not 1 <= args.init_timeout <= 3600 or not args.run_id or
            not args.run_id.isascii() or len(args.run_id) > 80 or
            any(not (c.isalnum() or c in '-_') for c in args.run_id)):
        parser.error('invalid shard, timeout, or run-id')
    if args.run and not args.allow_npu:
        parser.error('--run requires --allow-npu')
    if (args.slot_size_mb < 1 or 1024 % args.slot_size_mb or
            1024 // args.slot_size_mb < args.tp_size):
        parser.error('Pool slots must divide 1 GiB and fit the complete TP shard set')
    for devices in (args.p_devices, args.d_devices):
        if (len(devices) != args.tp_size or len(set(devices)) != args.tp_size
                or any(not 0 <= device < 16 for device in devices)):
            parser.error('Each role needs exactly TP distinct physical devices in 0..15')
    placements = [(args.pool_host, [args.pool_physical_device]),
                  (args.p_host, args.p_devices), (args.d_host, args.d_devices)]
    for i, (host, devices) in enumerate(placements):
        for other_host, other_devices in placements[i + 1:]:
            if host == other_host and set(devices) & set(other_devices):
                parser.error('Roles on one host must use disjoint physical devices')
    if (urlparse(args.server_url).scheme != 'tcp' or not urlparse(args.server_url).port
            or not args.store_url.startswith('tcp://')
            or not args.coord_url.startswith(('http://', 'https://'))):
        parser.error('explicit TCP store/server and HTTP coordinator are required')
    return args


def build_plan(args):
    tp = args.tp_size
    rank_base = 1 if args.role == 'p' else 1 + tp
    direct = dict(device_id=args.shard, device_ids=list(range(tp)), tp_size=tp,
        total_kv_heads=8, rank_id=rank_base, world_size=1 + 2 * tp, pool_gb=1,
        store_url=args.store_url, init_timeout=args.init_timeout,
        member_id=args.run_id + '-' + args.role)
    service = dict(device_id=0, rank_id=0, world_size=1 + 2 * tp,
        tp_size=tp, pool_gb=1, slot_size_mb=args.slot_size_mb, store_url=args.store_url,
        init_timeout=args.init_timeout)
    if args.pool_memory_type == 'device':
        direct['pool_memory_type'] = service['pool_memory_type'] = 'device'
    url = urlparse(args.server_url)
    server_argv = [sys.executable, '-u', '-m', 'lmcache.cli.main', 'server',
        '--host', url.hostname, '--port', str(url.port), '--http-host', url.hostname,
        '--http-port', str(args.service_http_port), '--chunk-size', str(TOKENS),
        '--l1-size-gb', '1', '--eviction-policy', 'LRU',
        '--supported-transfer-mode', 'lmcache_driven', '--max-gpu-workers', str(2 * tp),
        '--server-module', json.dumps(dict(module_path=
            'lmcache_ascend.v1.distributed.memfabric_direct_server', config=service))]
    return dict(status='PLAN_ONLY', role=args.role, shard=args.shard, tp_size=tp,
        bm_rank=0 if args.role == 'service' else rank_base + args.shard,
        world_size=1 + 2 * tp, logical_device=0 if args.role == 'service' else args.shard,
        local_kv_heads=8 // tp, total_kv_heads=8, layers=args.layers,
        payload_bytes_per_shard=2 * args.layers * TOKENS * (8 // tp) * 128 * 2,
        run_id=args.run_id, direct_config=direct, service_config=service,
        service_argv=server_argv, source_blocks=SOURCE_BLOCKS, target_blocks=TARGET_BLOCKS,
        source_baseline=('352f3c8/8b0c7e1' if args.pool_memory_type == 'device' else '7d04959'),
        baseline_verified_by_probe=False, pool_memory_type=args.pool_memory_type,
        ascend_source=str(args.ascend_source), core_source=str(args.core_source),
        visible_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        model_weights_loaded=False, model_inference_verified=False,
        rpc_payload=False, physical_plan=dict(service=[args.pool_physical_device],
            p=args.p_devices, d=args.d_devices),
        placement=dict(service=dict(host=args.pool_host, physical_devices=[args.pool_physical_device]),
            p=dict(host=args.p_host, physical_devices=args.p_devices),
            d=dict(host=args.d_host, physical_devices=args.d_devices)),
        selected_pool_boundary=('DEVICE-selected pool; each rank contributes fixed 1GiB DRAM plus 1GiB HBM'
                                if args.pool_memory_type == 'device' else 'HOST-selected DRAM pool'),
        external_real_process_wait_required=True)


def pattern_value(index, layer, component, shard):
    return (index + layer * 37 + component * 53 + shard * 1009) % 127 - 63


def cpu_plane(torch, args, layer, component):
    shape = (6, BLOCK_SIZE, 8 // args.tp_size, 128)
    index = torch.arange(6 * BLOCK_SIZE * (8 // args.tp_size) * 128,
                         dtype=torch.int32, device='cpu')
    return ((index + layer * 37 + component * 53 + args.shard * 1009) % 127 - 63
            ).to(torch.bfloat16).reshape(shape)


def validate_restored_set(receipts, tp_size):
    if (len(receipts) != tp_size or {r.get('shard') for r in receipts} != set(range(tp_size))
            or any(r.get('restored_tokens') != TOKENS or r.get('full_content_equal') is not True
                   for r in receipts)):
        raise AssertionError('Incomplete, duplicate, unequal-length, or incorrect D shard set')
    return [r['restored_tokens'] for r in sorted(receipts, key=lambda r: r['shard'])]


def worker_probe(args, report, record):
    import torch
    import torch_npu  # noqa: F401
    from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
    from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
    from lmcache_ascend.v1.distributed.memfabric_direct import (
        MemFabricDirectTransferContext, MemFabricDirectSchedulerBackend)
    torch.npu.set_device(args.shard)
    report.update(hostname=socket.gethostname(), pid=os.getpid(), actual_device_initialized=True,
        context_module=sys.modules[MemFabricDirectTransferContext.__module__].__file__,
        scheduler_module=sys.modules[MemFabricDirectSchedulerBackend.__module__].__file__)
    pages = {}
    worker = scheduler = client = None
    primary = None
    cleanup_errors = []
    label = args.role + str(args.shard)

    def marker(name, **extra):
        publish(args.coord_url, args.run_id, name, dict(name=name, role=args.role,
            shard=args.shard, tp_size=args.tp_size, **extra), min(args.timeout, 5))

    def wait(name):
        value = wait_event(args.coord_url, args.run_id, name, args.timeout)
        if value.get('tp_size') != args.tp_size:
            raise AssertionError('Coordinator TP geometry differs')
        return value

    def all_events(prefix, role):
        return [wait(prefix + '-' + role + str(shard)) for shard in range(args.tp_size)]

    def key(phase):
        return IPCCacheServerKey.from_token_ids(model_name=MODEL, world_size=args.tp_size,
            worker_id=args.shard, token_ids=list(range(TOKENS)), start=0, end=TOKENS,
            request_id=args.run_id + '-' + phase + '-d', cache_salt=args.run_id + '-' + phase)

    def lookup(phase, expected):
        item = key(phase)
        operation = scheduler.create_lookup(item.no_worker_id_version(), tp_size=args.tp_size,
            tokens_per_chunk=TOKENS, timeout=args.timeout, blocking_lookup=True)
        operation.submit()
        failed = operation.on_scheduler_check()
        hit = operation.get_result()
        record('scheduler_lookup', phase=phase, hit_tokens=hit, failed=failed)
        if failed or hit != expected:
            raise AssertionError(f'{phase} lookup returned {hit}, expected {expected}')
        return hit

    def owner_wait(operation):
        record(operation + '_wait_begin')
        worker._run(worker.session.wait)  # BM wait must execute on its actual owner thread.
        record(operation + '_wait_complete')

    def check_pages(restored):
        results = []
        for layer in range(args.layers):
            for component in (0, 1):
                observed = pages[f'layer.{layer}'][component].cpu()
                expected = torch.full_like(observed, -79)
                if restored:
                    reference = cpu_plane(torch, args, layer, component)
                    for source, target in zip(SOURCE_BLOCKS, TARGET_BLOCKS):
                        expected[target].copy_(reference[source])
                equal = torch.equal(observed, expected)
                results.append(dict(layer=layer, component=component, equal=equal,
                                    checked_elements=observed.numel()))
        if not all(row['equal'] for row in results):
            raise AssertionError('NPU pages differ from independent complete CPU expectation')
        return results

    try:
        for layer in range(args.layers):
            components = []
            for component in (0, 1):
                value = (cpu_plane(torch, args, layer, component) if args.role == 'p' else
                    torch.full((6, BLOCK_SIZE, 8 // args.tp_size, 128), -79,
                               dtype=torch.bfloat16, device='cpu'))
                components.append(value.to(f'npu:{args.shard}'))
            pages[f'layer.{layer}'] = components
        torch.npu.synchronize(args.shard)
        client = RequestClientFactory.create(args.server_url)
        strategy = SimpleNamespace(tp_size=args.tp_size, kv_worker_id=args.shard,
            pp_size=1, dcp_size=1, n_servers=1, vllm_world_size=args.tp_size, mla_only=False)
        worker = MemFabricDirectTransferContext(instance_id=os.getpid(), req_client=client,
            tokens_per_chunk=TOKENS, mq_timeout=args.timeout, parallel_strategy=strategy,
            extra_config={'lmcache.mp.direct': report['direct_config']})
        record('register_begin')
        worker.register(pages, MODEL, args.tp_size, 2, args.timeout,
            layout_hints={'kv_layout': 'NHD'}, lmcache_chunk_tokens=TOKENS)
        report.update(registration=worker.registration, actual_bm_rank=worker.config['rank_id'],
            actual_local_kv_heads=worker.transfer.layout['num_heads'],
            actual_layout=worker.transfer.layout, actual_chunk_bytes=worker.transfer.size,
            actual_pool_memory_type=worker.registration.get('pool_memory_type', 'host'),
            owner_pool_gva=int(worker.base))
        if report['actual_pool_memory_type'] != args.pool_memory_type:
            raise AssertionError('Registered service pool medium differs from requested medium')
        if (worker.config['rank_id'] != report['bm_rank'] or
                worker.transfer.size != report['payload_bytes_per_shard']):
            raise AssertionError('Actual registration differs from TP plan')
        marker('registered-' + label, bm_rank=worker.config['rank_id'], layout=worker.transfer.layout)
        for role in ('p', 'd'):
            all_events('registered', role)
        record('all_members_registered')
        if args.role == 'p':
            if args.shard != args.tp_size - 1:
                if worker.submit_store(args.run_id + '-partial-p', key('partial'), pages,
                        [SOURCE_BLOCKS], None, 2).result(args.timeout) is not True:
                    raise AssertionError('Partial store did not complete')
                owner_wait('partial_store')
            marker('partial-ready-' + label, omitted=args.shard == args.tp_size - 1)
            wait('missing-all-checked')
            if worker.submit_store(args.run_id + '-full-p', key('full'), pages,
                    [SOURCE_BLOCKS], None, 2).result(args.timeout) is not True:
                raise AssertionError('Full store did not complete')
            owner_wait('full_store')
            for components in pages.values():
                for tensor in components:
                    tensor.zero_()
            worker.transfer.buffer.zero_()
            torch.npu.synchronize(args.shard)
            report['source_and_scratch_cleared_after_completion'] = True
            marker('stored-' + label, completed_bytes=worker.transfer.size)
        else:
            all_events('partial-ready', 'p')
            if args.shard == 0:
                scheduler = MemFabricDirectSchedulerBackend(req_clients={args.server_url: client},
                    parallel_strategy=strategy, tokens_per_chunk=TOKENS, mq_timeout=args.timeout)
                report['missing_scheduler_hit_tokens'] = lookup('partial', 0)
                scheduler.end_session(key('partial').request_id)
                marker('missing-scheduler-checked')
            wait('missing-scheduler-checked')
            missing = worker.submit_retrieve(key('partial').request_id, key('partial'), pages,
                [TARGET_BLOCKS], None, 2).result(args.timeout)
            if missing is not False:
                raise AssertionError('Incomplete shard set unexpectedly retrieved')
            report['missing_context_result'] = missing
            report['missing_unchanged_pages'] = check_pages(False)
            marker('missing-checked-' + label)
            if args.shard == 0:
                all_events('missing-checked', 'd')
                status = worker.control.call('status', {})
                if status['total_pins'] != 0:
                    raise AssertionError('Missing lookup left pins')
                report['missing_status_after_release'] = status
                marker('missing-all-checked', total_pins=0)
            all_events('stored', 'p')
            if args.shard == 0:
                report['full_scheduler_hit_tokens'] = lookup('full', TOKENS)
                report['status_after_full_lookup'] = worker.control.call('status', {})
                if report['status_after_full_lookup']['total_pins'] != args.tp_size:
                    raise AssertionError('Scheduler did not pin the complete TP shard set')
                marker('full-lookup-ready', hit_tokens=TOKENS)
            full = wait('full-lookup-ready')
            restored = worker.submit_retrieve(key('full').request_id, key('full'), pages,
                [TARGET_BLOCKS], None, 2).result(args.timeout)
            if restored is not True:
                raise AssertionError('Complete TP object unexpectedly missed')
            owner_wait('full_retrieve')
            report['full_context_result'] = restored
            report['full_page_verification'] = check_pages(True)
            report['scheduler_hit_tokens'] = full['hit_tokens']
            report['verified_restored_tokens'] = TOKENS
            marker('restored-' + label, restored_tokens=TOKENS, full_content_equal=True)
            if args.shard == 0:
                receipts = all_events('restored', 'd')
                report['all_D_restored_lengths'] = validate_restored_set(receipts, args.tp_size)
                scheduler.end_session(key('full').request_id)
                status = worker.control.call('status', {})
                if status['total_pins'] != 0 or status['pending_writes'] != 0:
                    raise AssertionError('Completed TP session still owns pins or writes')
                report['status_after_all_release'] = status
                marker('done', restored_lengths=report['all_D_restored_lengths'], total_pins=0)
        wait('done')
        owner_wait('final')
        marker('drained-' + label)
        for role in ('p', 'd'):
            all_events('drained', role)
        report['content_and_drain_complete'] = True
    except BaseException as error:
        primary = error
        report['traceback'] = traceback.format_exc()
        record('probe_error', error=repr(error))
        try:
            marker('failed-' + label, error=repr(error))
        except Exception:
            report['failure_publish_error'] = traceback.format_exc()
    finally:
        for name, obj in [('scheduler', scheduler), ('context', worker), ('client', client)]:
            if obj is not None:
                try:
                    record('close_' + name + '_begin')
                    obj.close()
                    record('close_' + name + '_complete')
                except BaseException:
                    cleanup_errors.append(dict(step=name, traceback=traceback.format_exc()))
        report['cleanup_errors'] = cleanup_errors
        report['context_and_client_closed_normally'] = not cleanup_errors
    if primary is not None or cleanup_errors:
        _FAILED_KEEPALIVE.append((pages, worker, scheduler, client, primary))
        raise primary if primary is not None else RuntimeError('TP context cleanup failed')
    report['status'] = 'TP_CONTEXT_PASS_EXTERNAL_PROCESS_WAIT_PENDING'


def main(argv=None):
    args = arguments(argv)
    report = build_plan(args)
    if not args.run:
        print(json.dumps(report, indent=2), flush=True)
        return
    sys.path.insert(0, str(args.ascend_source.resolve()))
    sys.path.insert(1, str(args.core_source.resolve()))
    directory = args.output_dir / args.run_id
    directory.mkdir(parents=True, exist_ok=True)
    label = args.role if args.role == 'service' else args.role + str(args.shard)
    path = directory / ('result-' + label + '.json')
    report['status'] = 'SERVICE_EXEC_REQUESTED' if args.role == 'service' else 'RUNNING'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    if args.role == 'service':
        for variable, roots in [('PYTHONPATH', [str(args.ascend_source), str(args.core_source)])]:
            os.environ[variable] = ':'.join(roots + [os.environ.get(variable, '')])
        os.execv(sys.executable, report['service_argv'])
    def record(event, **extra):
        value = dict(event=event, unix_ns=time.time_ns(), mono_ns=time.monotonic_ns(),
            role=args.role, shard=args.shard, tp_size=args.tp_size, pid=os.getpid(), **extra)
        with (directory / ('events-' + label + '.jsonl')).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value) + '\n')
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(value), flush=True)
    failed = False
    try:
        worker_probe(args, report, record)
    except BaseException as error:
        failed = True
        report.update(status='FAIL', error=repr(error))
        report.setdefault('traceback', traceback.format_exc())
        try:
            name = 'failed-' + label
            publish(args.coord_url, args.run_id, name, dict(name=name, role=args.role,
                shard=args.shard, tp_size=args.tp_size, error=repr(error)), min(args.timeout, 5))
        except Exception:
            report.setdefault('failure_publish_error', traceback.format_exc())
    finally:
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
    if failed:
        os._exit(1)  # Unknown device completion must not trigger tensor/native finalizers.


if __name__ == '__main__':
    main()
