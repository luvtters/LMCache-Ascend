#!/usr/bin/env python3
import contextlib
import io
import json
import sys
import unittest
import worker_fault_probe as p


class FaultProbeChecks(unittest.TestCase):
    def args(self, case='leave', role='p', extra=()):
        return p.arguments(['--case', case, '--role', role, '--run-id', 'cpu-fault',
            '--output-dir', 'unused', '--ascend-source', '/a', '--core-source', '/c', *extra])

    def test_missing_fixed_world_and_native_timeout(self):
        for role, rank in [('service', 0), ('p', 1)]:
            plan = p.build_plan(self.args('missing', role))
            self.assertEqual((plan['rank'], plan['world_size']), (rank, 3))
            self.assertEqual(plan['config']['init_timeout'], 5)
            self.assertEqual(plan['expected_copy_calls'], 0)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args('missing', 'd')

    def test_default_plan_does_not_import_accelerators(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            p.main(['--case', 'leave', '--role', 'p', '--run-id', 'cpu-fault',
                '--output-dir', 'unused', '--ascend-source', '/a', '--core-source', '/c'])
        self.assertEqual(json.loads(output.getvalue())['status'], 'PLAN_ONLY')
        self.assertNotIn('torch_npu', sys.modules)
        self.assertNotIn('memfabric_hybrid', sys.modules)

    def test_explicit_opt_in_required(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args(extra=['--run'])

    def test_peer_left_must_be_explicit_not_miss_or_unrelated_failure(self):
        self.assertIn('left', p.require_peer_left(RuntimeError('BM peer rank2 left')))
        for value in ('cache miss', 'timeout', 'invalid model'):
            with self.assertRaises(AssertionError):
                p.require_peer_left(value)

    def test_unhealthy_requires_zero_resources_and_both_readiness_flags(self):
        good = dict(last_error='peer rank2 left', ready=False, transfer_ready=False,
                    total_pins=0, pending_writes=0)
        p.require_unhealthy(good)
        for field, value in [('ready', True), ('transfer_ready', True), ('total_pins', 1),
                             ('pending_writes', 1), ('last_error', 'missing')]:
            with self.assertRaises(AssertionError):
                p.require_unhealthy(dict(good, **{field: value}))


if __name__ == '__main__':
    unittest.main()
