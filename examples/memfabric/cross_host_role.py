#!/usr/bin/env python3
"""Plan or exec ONE container-local role. Caller owns SSH, concurrency and wait()."""
import argparse
import json
import os
from pathlib import PurePosixPath
import sys

HOSTS = {'137': '192.168.0.223', '12': '192.168.0.50'}
TOPOLOGIES = {
    'T0': {'service': ('137', 12), 'p': ('137', 8), 'd': ('137', 10)},
    'T1': {'service': ('12', 8), 'p': ('137', 8), 'd': ('12', 10)},
    'T2': {'service': ('137', 8), 'p': ('137', 10), 'd': ('12', 8)},
    'T3': {'service': ('12', 8), 'p': ('137', 8), 'd': ('137', 10)},
}


def build_plan(args):
    root = PurePosixPath(args.root)
    ascend, core = root / 'src/ascend_direct', root / 'src/core_direct'
    examples = ascend / 'examples/memfabric'
    case = root / 'runs' / args.session_id
    mapping = TOPOLOGIES[args.topology]
    tp_size = args.tp_size
    max_num_seqs = getattr(args, 'max_num_seqs', 1)
    if type(max_num_seqs) is not int or max_num_seqs not in (1, 2):
        raise ValueError('Experimental max_num_seqs must be 1 or 2')
    pool_memory_type = getattr(args, 'pool_memory_type', 'host')
    if pool_memory_type not in ('host', 'device'):
        raise ValueError('pool_memory_type must be host or device')
    if args.stage == 'inference' and tp_size > 1 and args.pool_gb * (1 << 30) < (1373 // 128) * tp_size * (32 << 20):
        raise ValueError('Pool capacity cannot hold this frozen TP prefix with 32 MiB slots')
    pool_host = mapping['service'][0]
    owner = HOSTS[pool_host]
    p_host, d_host = (mapping[role][0] for role in ('p', 'd'))
    gate_host = '137'
    rank = {'service': 0, 'p': 1, 'd': 2}
    host = mapping[args.role][0] if args.role in mapping else (
        pool_host if args.role == 'coordinator' else
        args.role.removeprefix('trace') if args.role.startswith('trace') else gate_host)
    if args.topology == 'T0':
        host = '137'  # Diagnostic positive control; never a cross-host result.
    ip = HOSTS[host]
    device = mapping[args.role][1] if args.role in mapping else None
    p_default = list(range(8, 8 + tp_size))
    d_default = list(range(10, 10 + tp_size))
    if args.topology == 'T2':
        p_default, d_default = d_default, p_default
    if tp_size == 8:
        p_default = list(range(0, 8)) if args.topology in ('T2', 'T3') else list(range(8, 16))
        d_default = list(range(0, 8)) if args.topology == 'T1' else list(range(8, 16))
    elif tp_size == 4 and args.topology == 'T3':
        d_default = list(range(12, 16))
    if tp_size == 4 and args.topology == 'T0':
        p_default, d_default = [2, 3, 4, 5], [8, 9, 10, 11]
    devices = {
        'p': args.p_devices if args.p_devices is not None else p_default,
        'd': args.d_devices if args.d_devices is not None else d_default,
        'service': [args.pool_device if args.pool_device is not None else mapping['service'][1]],
    }
    for role in ('p', 'd'):
        if len(devices[role]) != tp_size or len(set(devices[role])) != tp_size:
            raise ValueError('P/D device lists must contain exactly TP distinct physical ids')
    for left, right in (('p', 'd'), ('p', 'service'), ('d', 'service')):
        if mapping[left][0] == mapping[right][0] and set(devices[left]) & set(devices[right]):
            raise ValueError('Roles on one host must use disjoint physical devices')
    physical_devices = devices.get(args.role, [])
    if physical_devices:
        device = physical_devices[0] if len(physical_devices) == 1 else physical_devices
    if args.role == 'd':
        rank['d'] = 1 + tp_size
    bm = dict(device_id=0, rank_id=rank.get(args.role, 0), world_size=1 + 2 * tp_size,
              pool_gb=args.pool_gb, init_timeout=300, store_url=f'tcp://{owner}:28695')
    if pool_memory_type == 'device':
        bm['pool_memory_type'] = 'device'
    env = dict(PYTHONPATH=f'{examples}:{ascend}:{core}', OMP_NUM_THREADS='1',
               LD_LIBRARY_PATH=str(ascend / 'lmcache_ascend'),
               TOKENIZERS_PARALLELISM='false', VLLM_WORKER_MULTIPROC_METHOD='spawn',
               TORCH_DEVICE_BACKEND_AUTOLOAD='0', LMCACHE_MP_TRANSFER_MODE='auto',
               LMCACHE_REQUEST_TELEMETRY_TYPE='noop', LMCACHE_REQUEST_TELEMETRY_ENDPOINT='')
    if device is not None:
        env['ASCEND_RT_VISIBLE_DEVICES'] = ','.join(map(str, physical_devices))
    python = args.python
    config = None
    if args.role in mapping and args.stage != 'inference' and args.role != 'service':
        argv = [python, '-u', str(examples / 'validate_direct_kv.py'), '--mode', args.stage,
                '--role', args.role, '--device', '0', '--ascend-source', str(ascend),
                '--core-source', str(core), '--coord-dir', str(case / 'kv'),
                '--coord-url', f'http://{owner}:29595', '--run-id', args.session_id,
                '--store-url', bm['store_url'], '--server-url', f'tcp://{owner}:25595',
                '--pool-gb', str(args.pool_gb), '--init-timeout', '300', '--timeout', '360',
                '--source-blocks', '3,1', '--target-blocks', '4,2', '--num-blocks', '6',
                '--block-size', '64', '--skip-blocks', str(args.skip_blocks), '--allow-npu']
    elif args.role == 'service':
        if args.stage == 'primitive':
            argv = [python, '-u', str(examples / 'validate_direct_kv.py'), '--mode', 'primitive',
                    '--role', 'pool', '--device', '0', '--ascend-source', str(ascend),
                    '--core-source', str(core), '--coord-dir', str(case / 'kv'),
                    '--coord-url', f'http://{owner}:29595', '--run-id', args.session_id,
                    '--store-url', bm['store_url'], '--pool-gb', str(args.pool_gb),
                    '--init-timeout', '300', '--timeout', '360', '--allow-npu']
        else:
            config = dict(module_path='lmcache_ascend.v1.distributed.memfabric_direct_server',
                          config=dict(bm, slot_size_mb=32))
            if tp_size > 1:
                config['config']['tp_size'] = tp_size
            argv = [python, '-u', '-m', 'lmcache.cli.main', 'server', '--host', owner,
                    '--port', '25595', '--http-host', owner, '--http-port', '28195',
                    '--chunk-size', '128', '--l1-size-gb', '1', '--eviction-policy', 'LRU',
                    '--supported-transfer-mode', 'lmcache_driven', '--max-gpu-workers', str(2 * tp_size),
                    '--server-module', json.dumps(config)]
    elif args.role in ('p', 'd'):
        if tp_size > 1:
            bm.update(tp_size=tp_size, device_ids=list(range(tp_size)))
        extra = {'lmcache.mp.host': f'tcp://{owner}', 'lmcache.mp.port': 25595,
                 'lmcache.mp.autostart': False, 'lmcache.mp.mq_timeout': 360.0,
                 'lmcache.mp.mp_transfer_mode': 'auto', 'lmcache.mp.direct': bm}
        config = dict(kv_connector='MemFabricDirectConnector',
                      kv_connector_module_path='traced_direct_connector',
                      kv_role='kv_both' if args.role == 'p' else 'kv_consumer',
                      kv_connector_extra_config=extra)
        env.update(MF_DIRECT_TRACE_DIR=str(case / 'trace'),
                   MF_DIRECT_TRACE_ROLE=args.role.upper(), MF_DIRECT_SESSION_ID=args.session_id)
        if args.role == 'p':
            env.update(LMCACHE_REQUEST_TELEMETRY_TYPE='fastapi',
                       LMCACHE_REQUEST_TELEMETRY_ENDPOINT=f'http://{HOSTS[gate_host]}:25809/api/v1/telemetry')
        argv = [python, '-m', 'vllm.entrypoints.cli.main', 'serve', args.model,
                '--served-model-name', 'mf-direct', '--tensor-parallel-size', str(tp_size), '--seed', '1024',
                '--dtype', 'bfloat16', '--max-model-len', '4096', '--max-num-batched-tokens', '1024',
                '--max-num-seqs', str(max_num_seqs), '--block-size', '128', '--gpu-memory-utilization', '0.7',
                '--enforce-eager', '--no-enable-prefix-caching', '--host', ip,
                '--port', '28401' if args.role == 'p' else '28402',
                '--kv-transfer-config', json.dumps(config)]
    elif args.role == 'proxy':
        argv = [python, '-u', str(core / 'examples/disagg_prefill_mp/disagg_proxy_server.py'),
                '--host', ip, '--port', '28400', '--prefiller-host', HOSTS[p_host],
                '--prefiller-port', '28401', '--decoder-host', HOSTS[d_host],
                '--decoder-port', '28402', '--telemetry-port', '25808']
    elif args.role == 'gate':
        context_request = getattr(args, 'context_question_request', None)
        expected_answer = getattr(args, 'expected_answer', None)
        cold_reference = getattr(args, 'cold_reference', False)
        if (expected_answer or cold_reference) and not context_request:
            raise ValueError('Context-question gate requires context_question_request')
        if context_request and (args.stage != 'inference' or not expected_answer):
            raise ValueError('Context-question request requires inference stage and expected_answer')
        gate_script = 'direct_context_question_gate.py' if context_request else 'direct_pd_gate.py'
        argv = [python, '-u', str(examples / gate_script),
                '--service-http', f'http://{owner}:28195',
                '--prefiller', f'http://{HOSTS[p_host]}:28401',
                '--decoder', f'http://{HOSTS[d_host]}:28402', '--proxy', f'http://{ip}:28400',
                '--proxy-telemetry', f'http://{ip}:25808/api/v1/telemetry',
                '--telemetry-bind', ip, '--telemetry-port', '25809',
                '--telemetry-advertise', f'http://{ip}:25809/api/v1/telemetry',
                '--p-trace-url', f'http://{HOSTS[p_host]}:29596/snapshot',
                '--d-trace-url', f'http://{HOSTS[d_host]}:29596/snapshot',
                '--session-id', args.session_id, '--require-direct-trace',
                '--tp-size', str(tp_size),
                '--expected-prompt-tokens', '1373', '--expected-copy-bytes', str(180 << 20),
                '--expected-pool-memory-type', pool_memory_type,
                '--max-tokens', '8', '--timeout', '360', '--output', str(case / 'gate')]
        if context_request:
            for frozen_flag in ('--expected-prompt-tokens', '--expected-copy-bytes'):
                position = argv.index(frozen_flag)
                del argv[position:position + 2]
            argv += ['--request-json', str(context_request), '--expected-answer', expected_answer]
            if cold_reference:
                argv += ['--cold-reference']
    else:
        argv = [python, '-u', str(examples / 'direct_coordination.py'), '--bind', ip,
                '--port', '29595' if args.role == 'coordinator' else '29596',
                '--directory', str(case / ('coord' if args.role == 'coordinator' else 'trace-events')),
                '--run-id', args.session_id, '--trace-dir', str(case / 'trace')]
    if (pool_memory_type == 'device' and args.stage != 'inference'
            and (args.role in ('p', 'd') or (args.role == 'service' and args.stage == 'primitive'))):
        argv += ['--pool-memory-type', 'device']
    return dict(status='PLAN_ONLY', host=host, ip=ip, role=args.role, topology=args.topology,
                comparison_scope='single_host_positive_control' if args.topology == 'T0' else 'cross_host',
                stage=args.stage, session_id=args.session_id, cwd=str(ascend), argv=argv,
                pool_memory_type=pool_memory_type,
                declared_dram_bytes_per_rank=(args.pool_gb if pool_memory_type == 'host' else 1) << 30,
                declared_hbm_bytes_per_rank=(args.pool_gb << 30) if pool_memory_type == 'device' else 0,
                env=env, config=config, physical_device=device, logical_device=0 if device is not None else None,
                physical_devices=physical_devices, tp_size=tp_size,
                logical_devices=(list(range(tp_size)) if args.role in ('p', 'd')
                                 else [0] if physical_devices else []),
                requires_tp_candidate=tp_size > 1,
                service_prestart_required=True, autostart=False,
                expected_core_revision='8bc9edc', ascend_baseline_revision='7e5d40a',
                data_path='synchronous_SDMA_L2G_G2L', evidence_boundary='NOT_VERIFIED_NPU_or_RPC')


def execution_environment(plan, inherited):
    """Prefix local sources/libs while retaining the host's verified ACL/CANN paths."""
    env = dict(inherited, **plan['env'])
    for name in ('PYTHONPATH', 'LD_LIBRARY_PATH'):
        prior = inherited.get(name, '')
        prefix = plan['env'].get(name, '')
        env[name] = ':'.join(value for value in (prefix, prior) if value)
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/workspace/d00896497/crosshost_20261030')
    parser.add_argument('--python', default='/workspace/d00896497/venv/bin/python')
    parser.add_argument('--topology', choices=tuple(TOPOLOGIES), default='T3')
    parser.add_argument('--stage', choices=('primitive', 'context', 'inference'), default='inference')
    parser.add_argument('--role', choices=('service', 'p', 'd', 'proxy', 'gate', 'coordinator', 'trace137', 'trace12'), required=True)
    parser.add_argument('--session-id', required=True)
    parser.add_argument('--model', default='/workspace/d00896497/models/Qwen3-8B')
    parser.add_argument('--pool-gb', type=int, default=1)
    parser.add_argument('--pool-memory-type', choices=('host', 'device'), default='host',
                        help='Selected object pool; device also declares 1 GiB DRAM per rank')
    parser.add_argument('--tp-size', type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument('--max-num-seqs', type=int, choices=(1, 2), default=1,
                        help='Experimental engine scheduling capacity; default remains one')
    parser.add_argument('--p-devices', type=lambda value: [int(v) for v in value.split(',')])
    parser.add_argument('--d-devices', type=lambda value: [int(v) for v in value.split(',')])
    parser.add_argument('--pool-device', type=int, help='Defaults to T0=12, other topologies=8')
    parser.add_argument('--skip-blocks', type=int, choices=(0, 1), default=0)
    parser.add_argument('--context-question-request', help='Optional custom context question; affects gate role only')
    parser.add_argument('--expected-answer', help='Exact expected code for a custom context question')
    parser.add_argument('--cold-reference', action='store_true', help='Direct D cold reference before custom P/pool/D gate')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--allow-npu', action='store_true')
    args = parser.parse_args()
    if args.tp_size > 1 and args.stage != 'inference':
        parser.error('The synthetic primitive/context script is TP1 only; TP candidates require inference stage')
    if args.pool_gb < 1 or any(v < 0 or v > 15 for v in
            (args.p_devices or []) + (args.d_devices or []) +
            ([] if args.pool_device is None else [args.pool_device])):
        parser.error('Invalid capacity or physical devices')
    if not args.session_id or any(not (c.isalnum() or c in '-_') for c in args.session_id):
        parser.error('session-id must be a new simple attempt identifier')
    plan = build_plan(args)
    print(json.dumps(plan, indent=2), flush=True)
    if args.run:
        if plan['physical_device'] is not None and not args.allow_npu:
            parser.error('device roles require --allow-npu')
        env = execution_environment(plan, os.environ)
        os.chdir(plan['cwd'])
        os.execvpe(plan['argv'][0], plan['argv'], env)


if __name__ == '__main__':
    main()
