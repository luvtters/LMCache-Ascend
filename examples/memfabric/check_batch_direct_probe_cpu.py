#!/usr/bin/env python3
"""CPU argument/layout/byte-address checks; does not initialize an accelerator."""
import contextlib
import io
import json
import sys
import unittest

import batch_direct_probe as probe
from batch_address_plan import plan_segments


class ProbeCpuChecks(unittest.TestCase):
    def args(self, *extra):
        return probe.arguments(["--role", "p", "--run-id", "cpu-case",
                                "--output-dir", "unused", *extra])

    def test_default_plan_is_cpu_only_and_fixed_geometry(self):
        for layers, payload in ((4, 2 << 20), (36, 18 << 20)):
            args = self.args("--layers", str(layers))
            self.assertFalse(args.run)
            plan = probe.plan(args)
            self.assertEqual(plan["payload_bytes"], payload)
            self.assertEqual(plan["expected_segments"], 4 * layers)
            self.assertFalse(plan["scratch_allocated"])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                probe.main(["--role", "p", "--run-id", "cpu-case", "--output-dir", "unused"])
            self.assertEqual(json.loads(output.getvalue())["status"], "PLAN_ONLY")
        self.assertNotIn("torch_npu", sys.modules)
        self.assertNotIn("memfabric_hybrid", sys.modules)

    def test_run_requires_explicit_device_authorization(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args("--run")
        self.assertTrue(self.args("--run", "--allow-npu").run)

    def test_batch_arrays_use_all_original_pages_and_inverse_pool_layout(self):
        def planes(base):
            return [dict(component=c, layer=l, base_ptr=base + (c * 4 + l) * (1 << 22),
                token_bytes=4, token_stride_bytes=4,
                block_stride_bytes=64 * 4, num_blocks=6)
                for c in (0, 1) for l in range(4)]
        p = plan_segments(planes(10000000), probe.SOURCE_BLOCKS, block_size=64,
            token_start=0, token_count=128, pool_base=100000000, pool_bytes=1 << 30)
        d = plan_segments(planes(200000000), probe.TARGET_BLOCKS, block_size=64,
            token_start=0, token_count=128, pool_base=100000000, pool_bytes=1 << 30)
        p_src, p_dst, p_sizes = probe.batch_arrays(p)
        d_src, d_dst, d_sizes = probe.batch_arrays(d, retrieve=True)
        self.assertEqual(p_dst, d_src)
        self.assertEqual(p_sizes, d_sizes)
        self.assertEqual(len(p_src), len(p_dst))
        self.assertEqual(len(p_src), len(p_sizes))
        self.assertEqual(len(p_src), 16)
        self.assertEqual(len(set(p_src)), 16)
        self.assertEqual(len(set(d_dst)), 16)
        self.assertEqual(sum(p_sizes), 4096)
        # Full synthetic CPU content simulation through the independent page
        # tables checks both batch directions, not only matching byte totals.
        pool, result = {}, {}
        for src, dst, size in zip(p_src, p_dst, p_sizes):
            for offset in range(size):
                pool[dst + offset] = (src + offset) % 251
        for src, dst, size in zip(d_src, d_dst, d_sizes):
            for offset in range(size):
                result[dst + offset] = pool[src + offset]
        for source, target, size in zip(p_src, d_dst, p_sizes):
            self.assertEqual(bytes(result[target + j] for j in range(size)),
                             bytes((source + j) % 251 for j in range(size)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
