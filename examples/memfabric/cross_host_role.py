#!/usr/bin/env python3
"""Plan or exec ONE container-local role. Caller owns SSH, concurrency and wait()."""
import argparse
import json
import os
from pathlib import PurePosixPath
import sys

HOSTS = {'137': '192.168.0.223', '12': '192.168.0.50'}
TOPOLOGIES = {
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
    pool_host = mapping['service'][0]
    owner = HOSTS[pool_host]
    p_host, d_host = (mapping[role][0] for role in ('p', 'd'))
    gate_host = '137'
    rank = {'service': 0, 'p': 1, 'd': 2}
    host = mapping[args.role][0] if args.role in mapping else (
        pool_host if args.role == 'coordinator' else
        args.role.removeprefix('trace') if args.role.startswith('trace') else gate_host)
    ip = HOSTS[host]
    device = mapping[args.role][1] if args.role in mapping else None
    bm = dict(device_id=0, rank_id=rank.get(args.role, 0), world_size=3,
              pool_gb=args.pool_gb, init_timeout=300, store_url=f'tcp://{owner}:28695')
    env = dict(PYTHONPATH=f'{examples}:{ascend}:{core}', OMP_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', VLLM_WORKER_MULTIPROC_METHOD='spawn',
               TORCH_DEVICE_BACKEND_AUTOLOAD='0', LMCACHE_MP_TRANSFER_MODE='auto',
               LMCACHE_REQUEST_TELEMETRY_TYPE='noop', LMCACHE_REQUEST_TELEMETRY_ENDPOINT='')
    if device is not None:
        env['ASCEND_RT_VISIBLE_DEVICES'] = str(device)
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
            argv = [python, '-u', '-m', 'lmcache.cli.main', 'server', '--host', owner,
                    '--port', '25595', '--http-host', owner, '--http-port', '28195',
                    '--chunk-size', '128', '--l1-size-gb', '1', '--eviction-policy', 'LRU',
                    '--supported-transfer-mode', 'lmcache_driven', '--max-gpu-workers', '2',
                    '--server-module', json.dumps(config)]
    elif args.role in ('p', 'd'):
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
                '--served-model-name', 'mf-direct', '--tensor-parallel-size', '1', '--seed', '1024',
                '--dtype', 'bfloat16', '--max-model-len', '4096', '--max-num-batched-tokens', '1024',
                '--max-num-seqs', '1', '--block-size', '128', '--gpu-memory-utilization', '0.7',
                '--enforce-eager', '--no-enable-prefix-caching', '--host', ip,
                '--port', '28401' if args.role == 'p' else '28402',
                '--kv-transfer-config', json.dumps(config)]
    elif args.role == 'proxy':
        argv = [python, '-u', str(core / 'examples/disagg_prefill_mp/disagg_proxy_server.py'),
                '--host', ip, '--port', '28400', '--prefiller-host', HOSTS[p_host],
                '--prefiller-port', '28401', '--decoder-host', HOSTS[d_host],
                '--decoder-port', '28402', '--telemetry-port', '25808']
    elif args.role == 'gate':
        argv = [python, '-u', str(examples / 'direct_pd_gate.py'),
                '--service-http', f'http://{owner}:28195',
                '--prefiller', f'http://{HOSTS[p_host]}:28401',
                '--decoder', f'http://{HOSTS[d_host]}:28402', '--proxy', f'http://{ip}:28400',
                '--proxy-telemetry', f'http://{ip}:25808/api/v1/telemetry',
                '--telemetry-bind', ip, '--telemetry-port', '25809',
                '--telemetry-advertise', f'http://{ip}:25809/api/v1/telemetry',
                '--p-trace-url', f'http://{HOSTS[p_host]}:29596/snapshot',
                '--d-trace-url', f'http://{HOSTS[d_host]}:29596/snapshot',
                '--session-id', args.session_id, '--require-direct-trace',
                '--expected-prompt-tokens', '1373', '--expected-copy-bytes', str(180 << 20),
                '--max-tokens', '8', '--timeout', '360', '--output', str(case / 'gate')]
    else:
        argv = [python, '-u', str(examples / 'direct_coordination.py'), '--bind', ip,
                '--port', '29595' if args.role == 'coordinator' else '29596',
                '--directory', str(case / ('coord' if args.role == 'coordinator' else 'trace-events')),
                '--run-id', args.session_id, '--trace-dir', str(case / 'trace')]
    return dict(status='PLAN_ONLY', host=host, ip=ip, role=args.role, topology=args.topology,
                stage=args.stage, session_id=args.session_id, cwd=str(ascend), argv=argv,
                env=env, config=config, physical_device=device, logical_device=0 if device is not None else None,
                service_prestart_required=True, autostart=False,
                expected_core_revision='8bc9edc', ascend_baseline_revision='7e5d40a',
                data_path='synchronous_SDMA_L2G_G2L', evidence_boundary='NOT_VERIFIED_NPU_or_RPC')


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
    parser.add_argument('--skip-blocks', type=int, choices=(0, 1), default=0)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--allow-npu', action='store_true')
    args = parser.parse_args()
    if not args.session_id or any(not (c.isalnum() or c in '-_') for c in args.session_id):
        parser.error('session-id must be a new simple attempt identifier')
    plan = build_plan(args)
    print(json.dumps(plan, indent=2), flush=True)
    if args.run:
        if plan['physical_device'] is not None and not args.allow_npu:
            parser.error('device roles require --allow-npu')
        env = dict(os.environ, **plan['env'])
        os.chdir(plan['cwd'])
        os.execvpe(plan['argv'][0], plan['argv'], env)


if __name__ == '__main__':
    main()
