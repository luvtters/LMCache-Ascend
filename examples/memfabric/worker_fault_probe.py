#!/usr/bin/env python3
"""Opt-in fixed-world readiness/normal-leave negatives; default CPU PLAN_ONLY."""
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
MODEL = 'synthetic-worker-lifecycle-negative'
_KEEPALIVE = []


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=('missing', 'leave'), required=True)
    p.add_argument('--role', choices=('service', 'p', 'd'), required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--ascend-source', type=Path, required=True)
    p.add_argument('--core-source', type=Path, required=True)
    p.add_argument('--store-url', default='tcp://192.168.0.50:28719')
    p.add_argument('--server-url', default='tcp://192.168.0.50:25719')
    p.add_argument('--coord-url', default='http://192.168.0.50:29619')
    p.add_argument('--http-port', type=int, default=28119)
    p.add_argument('--timeout', type=float, default=90)
    p.add_argument('--run', action='store_true')
    p.add_argument('--allow-npu', action='store_true')
    a = p.parse_args(argv)
    if (not a.run_id or not a.run_id.isascii() or len(a.run_id) > 80 or
            any(not (c.isalnum() or c in '-_') for c in a.run_id) or a.timeout <= 0):
        p.error('invalid run-id/timeout')
    if a.run and not a.allow_npu:
        p.error('--run requires --allow-npu')
    if a.case == 'missing' and a.role == 'd':
        p.error('missing case deliberately never launches rank2/D')
    if (urlparse(a.server_url).scheme != 'tcp' or not urlparse(a.server_url).port or
            not a.store_url.startswith('tcp://') or not a.coord_url.startswith('http://')):
        p.error('TCP store/server and HTTP coordinator required')
    return a


def build_plan(a):
    rank = {'service': 0, 'p': 1, 'd': 2}[a.role]
    config = dict(device_id=0, rank_id=rank, world_size=3, pool_gb=1,
                  store_url=a.store_url, init_timeout=5 if a.case == 'missing' else 90)
    service = dict(config, rank_id=0, slot_size_mb=32)
    u = urlparse(a.server_url)
    command = [sys.executable, '-u', '-m', 'lmcache.cli.main', 'server',
        '--host', u.hostname, '--port', str(u.port), '--http-host', u.hostname,
        '--http-port', str(a.http_port), '--chunk-size', str(TOKENS),
        '--l1-size-gb', '1', '--eviction-policy', 'LRU',
        '--supported-transfer-mode', 'lmcache_driven', '--max-gpu-workers', '2',
        '--server-module', json.dumps(dict(module_path=
            'lmcache_ascend.v1.distributed.memfabric_direct_server', config=service))]
    return dict(status='PLAN_ONLY', case=a.case, role=a.role, run_id=a.run_id,
        rank=rank, world_size=3, logical_device=0, config=config, service_argv=command,
        expected_copy_calls=0, rpc_payload=False, model_loaded=False,
        physical_plan={'service': 6, 'p': 2, 'd': 4},
        native_init_timeout_seconds=config['init_timeout'],
        external_real_wait_required=True, automatic_recompute_verified=False)


def require_peer_left(error):
    text = str(error).lower()
    if 'peer' not in text or 'left' not in text:
        raise AssertionError('Expected explicit peer-left error, got: ' + str(error))
    return str(error)


def require_unhealthy(status):
    require_peer_left(status.get('last_error', ''))
    if (status.get('ready') is not False or status.get('transfer_ready') is not False or
            status.get('total_pins') != 0 or status.get('pending_writes') != 0):
        raise AssertionError('Expected unavailable service with no pins/writes')


def classify_readiness_failure(error, operation):
    if isinstance(error, TimeoutError) and operation == 'absent-rank2-mapping':
        return 'EXPECTED_ABSENT_MEMBER_MAPPING_TIMEOUT_ZERO_COPY'
    return 'OBSERVED_STARTUP_FAILURE_CAUSE_NOT_VERIFIED'


def missing_probe(a, report, event):
    # Raw primitive calls expose the native failing step before cleanup can fail.
    import torch
    import torch_npu  # noqa: F401
    import memfabric_hybrid as mf
    from memfabric_hybrid import bm
    torch.npu.set_device(0)
    initialized = bm_initialized = joined = False
    handle = None
    report['copy_calls'] = 0
    report['cleanup_errors'] = []
    def checked(name, fn):
        report['current_operation'] = name
        event(name + '_begin')
        result = fn()
        event(name + '_return', result=result)
        if result != 0:
            raise RuntimeError(name + ' failed: ' + str(result))
    try:
        checked('mf.initialize', mf.initialize)
        initialized = True
        config = bm.BmConfig()
        config.auto_ranking = False
        config.rank_id = report['rank']
        config.start_store = report['rank'] == 0
        config.init_timeout = config.create_timeout = 5
        config.operation_timeout = 5
        checked('bm.initialize', lambda: bm.initialize(a.store_url, 3, 0, config))
        bm_initialized = True
        report['current_operation'] = 'bm.create2'
        event('bm.create2_begin')
        handle = bm.create2(id=0, local_dram_size=1 << 30, max_dram_size=1 << 30,
                            data_op_type=bm.BmDataOpType.SDMA)
        event('bm.create2_return', nonempty=handle is not None)
        if handle is None:
            raise RuntimeError('bm.create2 returned empty handle')
        checked('BM.join', handle.join)
        joined = True
        report['current_operation'] = 'absent-rank2-mapping'
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            pointer = handle.peer_rank_ptr(2, bm.BmMemType.HOST)
            mapped = handle.gva_to_va(pointer, bm.BmMemType.LOCAL_DEVICE) if pointer else 0
            if mapped:
                raise AssertionError('Absent rank2 unexpectedly mapped')
            time.sleep(.1)
        raise TimeoutError('Absent rank2 mapping readiness timed out (world remains3)')
    except AssertionError:
        raise
    except (RuntimeError, TimeoutError) as error:
        if not initialized:
            raise  # Environment/ACL initialization is not the missing-member negative.
        report['native_readiness_error'] = repr(error)
        report['functional_status'] = classify_readiness_failure(error, report['current_operation'])
        event('observed_readiness_failure', error=repr(error), copy_calls=0,
              classification=report['functional_status'], operation=report['current_operation'])
    finally:
        steps = []
        if handle is not None:
            steps.append(('BM.wait', handle.wait, True))
            if joined:
                steps.append(('BM.leave', handle.leave, True))
            steps.append(('BM.destroy', handle.destroy, False))
        if bm_initialized:
            steps.append(('bm.uninitialize', bm.uninitialize, False))
        if initialized:
            steps.append(('mf.uninitialize', mf.uninitialize, False))
        for name, fn, check in steps:
            try:
                event(name + '_begin')
                value = fn()
                event(name + '_return', result=value)
                if check and value != 0:
                    raise RuntimeError(name + ' failed: ' + str(value))
            except BaseException as error:
                report['cleanup_errors'].append(dict(step=name, error=repr(error)))
                event('cleanup_error', step=name, error=repr(error))
                # Retain the handle if native wait/leave failed; outer executor owns stop.
                _KEEPALIVE.append((handle, mf, bm))
                break


def leave_probe(a, report, event):
    import torch
    import torch_npu  # noqa: F401
    from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
    from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
    from lmcache_ascend.v1.distributed.memfabric_bm import BMSession
    from lmcache_ascend.v1.distributed.memfabric_direct import (
        MemFabricDirectTransferContext, MemFabricDirectSchedulerBackend)
    torch.npu.set_device(0)
    report['copy_calls'] = 0
    report['cleanup_errors'] = []
    worker = scheduler = client = None
    pages = {str(i): [torch.full((6, 64, 8, 128), i + c, dtype=torch.bfloat16,
                                device='npu:0') for c in (0, 1)] for i in range(4)}
    class CountedSession(BMSession):
        def _copy(self, *args, **kwargs):
            report['copy_calls'] += 1
            event('unexpected_copy_attempt', copy_calls=report['copy_calls'])
            return super()._copy(*args, **kwargs)
    def mark(name, **kw):
        publish(a.coord_url, a.run_id, name, dict(role=a.role, **kw))
    def wait(name):
        return wait_event(a.coord_url, a.run_id, name, a.timeout)
    def key(suffix):
        return IPCCacheServerKey.from_token_ids(model_name=MODEL, world_size=1,
            worker_id=0, token_ids=list(range(TOKENS)), start=0, end=TOKENS,
            request_id=a.run_id + '-' + suffix, cache_salt=a.run_id)
    try:
        client = RequestClientFactory.create(a.server_url)
        worker = MemFabricDirectTransferContext(instance_id=os.getpid(), req_client=client,
            tokens_per_chunk=TOKENS, mq_timeout=a.timeout,
            extra_config={'lmcache.mp.direct': report['config']}, session_factory=CountedSession)
        worker.register(pages, MODEL, 1, 2, a.timeout,
            layout_hints={'kv_layout': 'NHD'}, lmcache_chunk_tokens=TOKENS)
        report['registration'] = worker.registration
        mark('registered-' + a.role)
        wait('registered-p'); wait('registered-d')
        report['healthy_status'] = worker.control.call('status', {})
        if report['healthy_status']['total_pins'] != 0 or not report['healthy_status']['ready']:
            raise AssertionError('Full fixed group never became healthy')
        if a.role == 'd':
            wait('allow-idle-leave')
            event('normal_idle_context_close_begin')
            worker.close()
            event('normal_idle_context_close_complete')
            worker = None
            report['functional_status'] = 'IDLE_MEMBER_NORMAL_LEAVE_ZERO_COPY'
            mark('idle-member-left', copy_calls=report['copy_calls'])
        else:
            mark('allow-idle-leave')
            wait('idle-member-left')
            deadline = time.monotonic() + a.timeout
            while True:
                status = worker.control.call('status', {})
                if 'left' in status.get('last_error', '').lower():
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('Service did not observe normal peer leave')
                time.sleep(.1)
            require_unhealthy(status)
            report['unhealthy_status'] = status
            event('service_unhealthy', status=status)
            try:
                worker.submit_store(key('store').request_id, key('store'), pages,
                    [[3, 1]], None, 2).result(a.timeout)
            except Exception as error:
                report['store_error'] = require_peer_left(error)
                event('valid_store_explicitly_rejected', error=repr(error))
            else:
                raise AssertionError('Valid store unexpectedly accepted after peer leave')
            strategy = SimpleNamespace(tp_size=1, pp_size=1, dcp_size=1,
                n_servers=1, vllm_world_size=1, mla_only=False)
            scheduler = MemFabricDirectSchedulerBackend(req_clients={a.server_url: client},
                parallel_strategy=strategy, tokens_per_chunk=TOKENS, mq_timeout=a.timeout)
            operation = scheduler.create_lookup(key('lookup').no_worker_id_version(),
                tp_size=1, tokens_per_chunk=TOKENS, timeout=a.timeout, blocking_lookup=True)
            operation.submit()
            try:
                operation.on_scheduler_check()
            except Exception as error:
                report['lookup_error'] = require_peer_left(error)
                event('valid_lookup_explicitly_rejected', error=repr(error),
                      result=operation.get_result())
            else:
                raise AssertionError('Peer-left lookup became a miss or accepted reply')
            report['final_status'] = worker.control.call('status', {})
            require_unhealthy(report['final_status'])
            if report['copy_calls'] != 0:
                raise AssertionError('Negative operation issued a BM copy')
            report['functional_status'] = 'PEER_LEFT_STORE_LOOKUP_REJECTED_ZERO_COPY_PINS_ZERO'
    finally:
        for name, obj in [('scheduler', scheduler), ('context', worker), ('client', client)]:
            if obj is not None:
                try:
                    event('close_' + name + '_begin')
                    obj.close()
                    event('close_' + name + '_complete')
                except BaseException as error:
                    report['cleanup_errors'].append(dict(step=name, error=repr(error)))
                    event('cleanup_error', step=name, error=repr(error))
        if report['cleanup_errors']:
            _KEEPALIVE.append((pages, worker, scheduler, client))


def main(argv=None):
    a = arguments(argv)
    report = build_plan(a)
    if not a.run:
        print(json.dumps(report, indent=2)); return
    sys.path[:0] = [str(a.ascend_source.resolve()), str(a.core_source.resolve())]
    directory = a.output_dir / a.run_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ('result-' + a.role + '.json')
    report.update(status='RUNNING', hostname=socket.gethostname(), pid=os.getpid(),
                  visible_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'))
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    def event(name, **kw):
        row = dict(event=name, unix_ns=time.time_ns(), mono_ns=time.monotonic_ns(), **kw)
        with (directory / ('events-' + a.role + '.jsonl')).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)
    if a.case == 'leave' and a.role == 'service':
        os.environ['PYTHONPATH'] = ':'.join([str(a.ascend_source), str(a.core_source)])
        os.execv(sys.executable, report['service_argv'])
    failed = False
    try:
        (missing_probe if a.case == 'missing' else leave_probe)(a, report, event)
        report['status'] = 'FUNCTIONAL_EXPECTATION_MET_EXTERNAL_WAIT_PENDING'
        if report.get('functional_status') == 'OBSERVED_STARTUP_FAILURE_CAUSE_NOT_VERIFIED':
            report['status'] = 'NATIVE_STARTUP_FAILURE_CAUSE_NOT_VERIFIED_EXTERNAL_WAIT_PENDING'
        if report['cleanup_errors']:
            report['status'] = 'FUNCTIONAL_EXPECTATION_MET_CLEANUP_ERRORS'
            failed = True
    except BaseException as error:
        failed = True
        report.update(status='UNEXPECTED_FAILURE', error=repr(error), traceback=traceback.format_exc())
        event('unexpected_failure', error=repr(error))
        if a.case == 'leave':
            try:
                publish(a.coord_url, a.run_id, 'failed-' + a.role, dict(error=repr(error)))
            except Exception:
                report['publish_error'] = traceback.format_exc()
    finally:
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)
    if failed:
        os._exit(1)


if __name__ == '__main__':
    main()
