#!/usr/bin/env python3
"""Narrow CPU contract checks. No torch/MemFabric/native imports or device use."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import cross_host_role as roles
import direct_coordination as coord
import direct_pd_gate as gate


def trace(role, origin, tp_size=1, shard=0):
    records = []
    def emit(kind, phase, offset, **extra):
        records.append(dict(kind=kind, phase=phase, role=role, pid=9 + shard, thread_id=7,
                            mono_ns=origin + offset, host_id='host-' + role,
                            session_id='cpu-case', worker_shard=shard,
                            vllm_worker_id=shard, tp_size=tp_size, **extra))
    info = dict(local_ptr=100, bytes=256 // tp_size, device='npu:' + str(shard))
    emit('trace_installed', 'complete', 1)
    emit('npu_scratch', 'complete', 2, memory_type='npu', **info)
    def stage(direction, start):
        emit('kv_stage', 'begin', start, direction=direction, **info)
        emit('native_transfer', 'begin', start + 1, object_pointers=[100], device=info['device'],
             direction='D2H' if role == 'P' else 'H2D')
        emit('native_transfer', 'returned', start + 2, object_pointers=[100], device=info['device'],
             direction='D2H' if role == 'P' else 'H2D')
        emit('kv_stage', 'complete', start + 3, direction=direction, **info)
    def copy(direction, start):
        for phase, offset in [('begin', start), ('complete', start + 1)]:
            emit('bm_copy', phase, offset, direction=direction, src=100, dst=200,
                 memory_type='npu', bm_rank=(1 if role == 'P' else 1 + tp_size) + shard, **info)
    if role == 'P':
        stage('gather', 11)
        copy('L2G', 15)
    else:
        copy('G2L', 11)
        stage('scatter', 13)
    return records


class Contracts(unittest.TestCase):
    def test_independent_monotonic_domains_and_identity(self):
        # D's clock is millions of ticks behind P: no cross-host ordering is valid.
        records = trace('P', 9000000) + trace('D', 100)
        baselines = {role: dict(host_id='host-' + role, run_id='cpu-case', mono_ns=start)
                     for role, start in [('P', 9000010), ('D', 110)]}
        result = gate.verify_trace(records, baselines)
        self.assertEqual(result['p_l2g_bytes'], 256)
        self.assertEqual(result['d_g2l_bytes'], 256)
        records[-1]['host_id'] = 'wrong-host'
        with self.assertRaisesRegex(AssertionError, 'host changed|identity changed'):
            gate.verify_trace(records, baselines)

    def test_callback_and_actual_output_limit(self):
        with patch.object(sys, 'argv', ['gate', '--output', 'unused', '--max-tokens', '8',
                    '--telemetry-bind', '192.168.0.223', '--telemetry-advertise',
                    'http://192.168.0.223:25809/api/v1/telemetry']):
            args = gate.parse_args()
        self.assertEqual(args.max_tokens, 8)
        self.assertEqual(args.telemetry_bind, '192.168.0.223')
        with patch.object(sys, 'argv', ['gate', '--output', 'unused']):
            self.assertEqual(gate.parse_args().max_tokens, 32)

    def test_tp_rank_counts_and_missing_worker_rejection(self):
        for tp_size in (1, 2, 4, 8):
            records = [record for role, origin in [('P', 9000000), ('D', 100)]
                       for shard in range(tp_size) for record in trace(role, origin, tp_size, shard)]
            baselines = {role: dict(host_id='host-' + role, run_id='cpu-case', mono_ns=start)
                         for role, start in [('P', 9000010), ('D', 110)]}
            result = gate.verify_trace(records, baselines, tp_size, 1, 256)
            self.assertEqual(result['p_l2g_copies'], tp_size)
            self.assertEqual(result['d_g2l_bytes'], 256)
            self.assertEqual(len(result['per_rank']['D']), tp_size)
            missing = [r for r in records if not (r['role'] == 'D' and r['worker_shard'] == tp_size - 1)]
            with self.assertRaisesRegex(AssertionError, 'Both real|Missing real'):
                gate.verify_trace(missing, baselines, tp_size, 1, 256)
            corrupted = [dict(r) for r in records]
            next(r for r in corrupted if r['kind'] == 'bm_copy')['bm_rank'] = 99
            with self.assertRaisesRegex(AssertionError, 'BM rank differs'):
                gate.verify_trace(corrupted, baselines, tp_size, 1, 256)

    def test_tp_receipts_never_release_on_first_or_duplicate_rank(self):
        receipts = {}
        rank0 = dict(event='request_store_finished', request_ids_set=['request'], world_size=2, kv_rank=0)
        self.assertFalse(gate.accept_store_receipt(receipts, rank0, 2))
        self.assertFalse(gate.accept_store_receipt(receipts, dict(rank0), 2))
        with self.assertRaises(RuntimeError):
            gate.accept_store_receipt(receipts, dict(rank0, model_name='conflict'), 2)
        with self.assertRaises(RuntimeError):
            gate.accept_store_receipt(receipts, dict(rank0, kv_rank=1, request_ids_set=['other']), 2)
        with self.assertRaises(ValueError):
            gate.accept_store_receipt(receipts, dict(rank0, kv_rank=2), 2)
        self.assertTrue(gate.accept_store_receipt(receipts, dict(rank0, kv_rank=1), 2))

    def test_three_topology_expansion(self):
        for topology in roles.TOPOLOGIES:
            for role in ('service', 'p', 'd', 'gate'):
                args = argparse.Namespace(root='/private', python='/venv/python',
                    topology=topology, stage='inference', role=role, session_id='cpu-case',
                    pool_gb=1, skip_blocks=0, model='/readonly/model', tp_size=1,
                    p_devices=None, d_devices=None, pool_device=None)
                plan = roles.build_plan(args)
                if role in ('p', 'd'):
                    extra = plan['config']['kv_connector_extra_config']
                    self.assertIs(extra['lmcache.mp.autostart'], False)
                    self.assertEqual(extra['lmcache.mp.direct']['device_id'], 0)
                    self.assertEqual(extra['lmcache.mp.direct']['init_timeout'], 300)
                    self.assertNotIn('127.0.0.1', ' '.join(plan['argv']))
                elif role == 'gate':
                    self.assertEqual(plan['argv'][plan['argv'].index('--max-tokens') + 1], '8')
        self.assertEqual(roles.TOPOLOGIES['T3'],
                         {'service': ('12', 8), 'p': ('137', 8), 'd': ('137', 10)})

    def test_t0_single_host_positive_control_default_and_explicit_devices(self):
        expected = {1: {'p': [8], 'd': [10]},
                    2: {'p': [8, 9], 'd': [10, 11]},
                    4: {'p': [2, 3, 4, 5], 'd': [8, 9, 10, 11]}}
        for tp_size in (1, 2, 4):
            for role in ('service', 'p', 'd', 'proxy', 'gate', 'coordinator', 'trace137', 'trace12'):
                args = argparse.Namespace(root='/private', python='/venv/python',
                    topology='T0', stage='inference', role=role, session_id='cpu-case',
                    pool_gb=2, skip_blocks=0, model='/readonly/model', tp_size=tp_size,
                    p_devices=None, d_devices=None, pool_device=None)
                plan = roles.build_plan(args)
                self.assertEqual(plan['host'], '137')
                self.assertEqual(plan['comparison_scope'], 'single_host_positive_control')
                if role == 'service':
                    self.assertEqual(plan['physical_devices'], [12])
                elif role in ('p', 'd'):
                    self.assertEqual(plan['physical_devices'], expected[tp_size][role])
                elif role == 'gate':
                    self.assertEqual(plan['argv'][plan['argv'].index('--service-http') + 1],
                                     'http://192.168.0.223:28195')
            args.role, args.tp_size, args.pool_device = 'service', 1, 13
            self.assertEqual(roles.build_plan(args)['physical_devices'], [13])

    def test_tp_candidate_role_config_real_device_lists(self):
        for tp_size in (2, 4, 8):
            for role in ('service', 'p', 'd'):
                args = argparse.Namespace(root='/private', python='/venv/python',
                    topology='T3', stage='inference', role=role, session_id='cpu-case',
                    pool_gb=3 if tp_size == 8 else 2, skip_blocks=0, model='/readonly/model', tp_size=tp_size,
                    p_devices=None, d_devices=None, pool_device=8)
                plan = roles.build_plan(args)
                config = plan['config']['config'] if role == 'service' else plan['config']['kv_connector_extra_config']['lmcache.mp.direct']
                self.assertEqual(config['world_size'], 1 + 2 * tp_size)
                self.assertEqual(config['tp_size'], tp_size)
                if role != 'service':
                    self.assertEqual(config['device_ids'], list(range(tp_size)))
                    self.assertEqual(config['rank_id'], 1 if role == 'p' else 1 + tp_size)
                    self.assertEqual(len(plan['physical_devices']), tp_size)
                    self.assertNotIn('total_kv_heads', config)
                    if tp_size == 2:
                        self.assertEqual(plan['physical_devices'], [8, 9] if role == 'p' else [10, 11])

    def test_exec_preserves_verified_acl_and_native_library_paths(self):
        plan = {'env': {'PYTHONPATH': '/example:/ascend:/core',
                        'LD_LIBRARY_PATH': '/ascend/lmcache_ascend', 'OMP_NUM_THREADS': '1'}}
        inherited = {'PYTHONPATH': '/cann/python/site-packages:/sdk/acl',
                     'LD_LIBRARY_PATH': '/driver/lib64:/cann/lib64', 'UNCHANGED': 'value'}
        env = roles.execution_environment(plan, inherited)
        self.assertEqual(env['PYTHONPATH'], '/example:/ascend:/core:/cann/python/site-packages:/sdk/acl')
        self.assertEqual(env['LD_LIBRARY_PATH'], '/ascend/lmcache_ascend:/driver/lib64:/cann/lib64')
        self.assertEqual(env['UNCHANGED'], 'value')
        self.assertEqual(inherited['PYTHONPATH'], '/cann/python/site-packages:/sdk/acl')

    def test_http_immutable_event_failure_timeout_and_raw_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory) / 'traces'
            trace_dir.mkdir()
            record = dict(kind='bm_copy', phase='complete', role='P', direction='L2G',
                          bytes=123, session_id='cpu-case', host_id=coord.host_identity())
            (trace_dir / 'P-1.jsonl').write_text(json.dumps(record) + '\n', encoding='utf-8')
            server = coord.make_server('127.0.0.1', 0, directory, 'cpu-case', trace_dir)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = 'http://127.0.0.1:' + str(server.server_port)
            try:
                value = dict(name='stored', bytes=123)
                self.assertFalse(coord.publish(base, 'cpu-case', 'stored', value)['duplicate'])
                self.assertTrue(coord.publish(base, 'cpu-case', 'stored', value)['duplicate'])
                self.assertEqual(coord.wait_event(base, 'cpu-case', 'stored', 1), value)
                with self.assertRaises(HTTPError) as conflict:
                    coord.publish(base, 'cpu-case', 'stored', dict(name='stored', bytes=999))
                self.assertEqual(conflict.exception.code, 409)
                with self.assertRaises(TimeoutError):
                    coord.wait_event(base, 'cpu-case', 'missing', .05)
                coord.publish(base, 'cpu-case', 'failed-d', dict(name='failed-d', error='CPU injected'))
                with self.assertRaisesRegex(RuntimeError, 'CPU injected'):
                    coord.wait_event(base, 'cpu-case', 'missing', 1)
                snapshot = coord.http_json(base + '/snapshot')
                self.assertEqual(snapshot['records'], [record])
                self.assertEqual(snapshot['trace_derived_counters']['P_shard0_L2G']['completed_bytes'], 123)
                self.assertEqual(json.loads((Path(directory) / 'cpu-case/stored.json').read_text()), value)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == '__main__':
    unittest.main(verbosity=2)
