#!/usr/bin/env python3
"""Narrow TP probe contracts, without torch/ACL/BM imports."""
import contextlib
import io
import json
import sys
import unittest

import tp_context_probe as probe


class TPProbeChecks(unittest.TestCase):
    def args(self, tp, role='p', shard=0):
        return probe.arguments(['--tp-size', str(tp), '--role', role,
            '--shard', str(shard), '--run-id', 'cpu-tp', '--output-dir', 'unused',
            '--ascend-source', '/private/ascend', '--core-source', '/private/core'])

    def test_tp_plans_real_heads_and_complete_fixed_rank_set(self):
        for tp in (2, 4):
            ranks = []
            for role in ('p', 'd'):
                for shard in range(tp):
                    args = self.args(tp, role, shard)
                    plan = probe.build_plan(args)
                    config = plan['direct_config']
                    self.assertEqual(plan['local_kv_heads'], 8 // tp)
                    self.assertEqual(plan['payload_bytes_per_shard'] * tp, 18 << 20)
                    self.assertEqual(config['world_size'], 1 + 2 * tp)
                    self.assertEqual(config['device_ids'], list(range(tp)))
                    self.assertEqual(config['total_kv_heads'], 8)
                    ranks.append(plan['bm_rank'])
            self.assertEqual(sorted(ranks), list(range(1, 1 + 2 * tp)))
            self.assertEqual(probe.build_plan(self.args(tp, 'service'))['service_config']['tp_size'], tp)

    def test_default_main_never_imports_accelerator(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            probe.main(['--tp-size', '2', '--role', 'd', '--shard', '1',
                '--run-id', 'plan', '--output-dir', 'unused',
                '--ascend-source', '/a', '--core-source', '/c'])
        self.assertEqual(json.loads(output.getvalue())['status'], 'PLAN_ONLY')
        self.assertNotIn('torch_npu', sys.modules)
        self.assertNotIn('memfabric_hybrid', sys.modules)

    def test_receipts_require_full_unique_shards_and_equal_restore_lengths(self):
        good = [dict(shard=i, restored_tokens=128, full_content_equal=True) for i in range(4)]
        self.assertEqual(probe.validate_restored_set(good, 4), [128] * 4)
        for bad in (good[:-1], good[:-1] + [good[0]],
                    good[:-1] + [dict(good[-1], restored_tokens=64)],
                    good[:-1] + [dict(good[-1], full_content_equal=False)]):
            with self.assertRaises(AssertionError):
                probe.validate_restored_set(bad, 4)

    def test_different_shards_are_content_distinguishable(self):
        for index in (0, 1, 65536, 200001):
            values = [probe.pattern_value(index, 3, 1, shard) for shard in range(4)]
            self.assertEqual(len(set(values)), 4)
            self.assertTrue(all(-63 <= value <= 63 for value in values))

    def test_pool_physical_plan_can_avoid_competition_without_changing_logical_id(self):
        args = probe.arguments(['--role', 'service', '--tp-size', '4', '--run-id', 'pool6',
            '--output-dir', 'unused', '--ascend-source', '/a', '--core-source', '/c',
            '--pool-physical-device', '6'])
        plan = probe.build_plan(args)
        self.assertEqual(plan['physical_plan']['service'], [6])
        self.assertEqual(plan['service_config']['device_id'], 0)

    def test_exact_group_capacity_and_too_small_rejection(self):
        for tp, slot in [(4, 256), (2, 512)]:
            args = probe.arguments(['--role', 'service', '--tp-size', str(tp),
                '--run-id', 'capacity', '--output-dir', 'unused', '--ascend-source', '/a',
                '--core-source', '/c', '--slot-size-mb', str(slot)])
            plan = probe.build_plan(args)
            self.assertEqual(plan['service_config']['slot_size_mb'], slot)
            self.assertEqual(1024 // slot, tp)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            probe.arguments(['--role', 'service', '--tp-size', '4', '--run-id', 'too-small',
                '--output-dir', 'unused', '--ascend-source', '/a', '--core-source', '/c',
                '--slot-size-mb', '512'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
