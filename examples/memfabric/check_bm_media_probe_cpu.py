#!/usr/bin/env python3
"""CPU checks for the isolated probe; never imports an accelerator SDK."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import bm_media_probe as probe


class ProbeChecks(unittest.TestCase):
    def test_plan_is_safe_without_sdk(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'plan.json'
            result = subprocess.run([sys.executable, str(Path(probe.__file__)),
                '--rank', '0', '--device', '6', '--media', 'device',
                '--store-url', 'tcp://127.0.0.1:28671', '--coord-url', 'http://127.0.0.1:29571',
                '--run-id', 'cpu-plan', '--output', str(out)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(plan['status'], 'PLAN_ONLY')
            self.assertEqual(plan['payload_bytes'], 1 << 20)
            self.assertEqual(plan['target_memory'], 'DEVICE')
            self.assertEqual(plan['create2']['local_hbm_size'], 1 << 30)
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('memfabric_hybrid', sys.modules)

    def test_explicit_media_pointer_and_mapping(self):
        calls = []
        class Handle:
            def peer_rank_ptr(self, rank, mem_type):
                calls.append(('peer', rank, mem_type))
                return 8192
            def gva_to_va(self, ptr, mem_type):
                calls.append(('mapping', ptr, mem_type))
                return 16384
        bm = SimpleNamespace(BmMemType=SimpleNamespace(DEVICE=2, HOST=3, LOCAL_DEVICE=0))
        for media, expected in [('host', 3), ('device', 2)]:
            calls.clear()
            ptr = probe.owner_address(Handle(), bm, media, .1, lambda *a, **k: None)
            self.assertEqual(ptr, 8192)
            self.assertEqual(calls, [('peer', 1, expected), ('mapping', 8192, 0)])

    def test_full_pattern_is_not_constant_or_short_period(self):
        source = probe.pattern_bytes(1 << 20)
        expected = bytes(((index * 17 + (index // 251) * 29 + 43) % 256)
                         for index in range(1 << 20))
        self.assertEqual(source, expected)
        self.assertNotEqual(source[:4096], source[4096:8192])

    def test_cleanup_keeps_every_error_and_attempts_each_step(self):
        events, calls = [], []
        def step(name, fail=False):
            def operation():
                calls.append(name)
                if fail:
                    raise RuntimeError(name)
                return 0
            return operation
        errors = probe.cleanup_steps([
            ('wait', step('wait', True), True),
            ('leave', step('leave'), True),
            ('destroy', step('destroy', True), False),
            ('bm.uninitialize', step('bm.uninitialize'), False),
            ('mf.uninitialize', step('mf.uninitialize'), False)],
            lambda event, **values: events.append((event, values)))
        self.assertEqual(calls, ['wait', 'leave', 'destroy', 'bm.uninitialize', 'mf.uninitialize'])
        self.assertEqual([error['step'] for error in errors], ['wait', 'destroy'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
