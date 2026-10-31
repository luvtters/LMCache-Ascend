#!/usr/bin/env python3
"""CPU-only checks; no SDK, torch, NPU or server connection."""
import ctypes
import contextlib
import io
import json
import sys
import unittest


class Checks(unittest.TestCase):
    def test_actual_event_emitter_tensor_metadata(self):
        import registered_host_rdma_probe as p
        args = p.arguments(['--rank', '0', '--device', '0', '--store-url',
            'tcp://127.0.0.1:19001', '--coord-url', 'http://127.0.0.1:19002',
            '--run-id', 'cpu-emitter', '--output', 'unused.json'])
        events, stdout = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout):
            p.emit_event(args, events, 'REAL_NPU_TENSORS', source_ptr=123,
                         restore_ptr=456, tensor_device='npu:0', bytes=64)
        record = json.loads(events.getvalue())
        self.assertEqual(record['device'], 0)
        self.assertEqual(record['tensor_device'], 'npu:0')
        self.assertEqual(record['event'], 'REAL_NPU_TENSORS')
        self.assertEqual(stdout.getvalue(), events.getvalue())

    def test_device_endpoint_plan_and_mapping(self):
        import registered_host_rdma_probe as p
        base = ['--rank', '0', '--device', '0', '--store-url', 'tcp://127.0.0.1:19001',
            '--coord-url', 'http://127.0.0.1:19002', '--run-id', 'cpu', '--output', 'unused.json']
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            p.arguments(base + ['--local-endpoint', 'device'])
        a = p.arguments(base + ['--local-endpoint', 'device', '--copy-interface', 'g2g'])
        plan = p.build_plan(a)
        self.assertEqual(plan['create2']['local_hbm_size'], 1 << 30)
        self.assertEqual(plan['local_endpoint'], 'device')
        p.validate_device_mapping(0x10000, 4096, 0x80000, 0x80200, 256)
        for values in ((0, 4096, 0x80000, 0x80200, 256),
                       (0x10000, 512, 0x80000, 0x80200, 256),
                       (0x10000, 4096, 0, 512, 256),
                       (0x10000, 4096, 0x80000, 0x80001, 256)):
            with self.assertRaises(ValueError):
                p.validate_device_mapping(*values)

    def test_registered_gva_copy_interface(self):
        import registered_host_rdma_probe as p
        # Both endpoints are managed HOST GVAs; no HVA is passed as a DEVICE pointer.
        own, remote, hva, restore, size = 0x100000000000000, 0x100000040000000, 0x12c1c0000000, 0x12c1c0200000, 1 << 20
        push = p.copy_request('g2g', 'write', own, remote, hva, restore, size)
        pull = p.copy_request('g2g', 'read', own, remote, hva, restore, size)
        self.assertEqual(push, (own, remote, 'G2G'))
        self.assertEqual(pull, (remote, own + 2*size, 'G2G'))
        self.assertEqual(p.copy_request('h2g-g2h', 'write', own, remote, hva, restore, size),
                         (hva, remote, 'H2G'))
        self.assertEqual(p.copy_request('h2g-g2h', 'read', own, remote, hva, restore, size),
                         (remote, restore, 'G2H'))

    def test_invalid_parameters(self):
        import registered_host_rdma_probe as p
        base = ['--rank', '0', '--device', '0', '--store-url', 'tcp://127.0.0.1:19001',
            '--coord-url', 'http://127.0.0.1:19002', '--run-id', 'cpu', '--output', 'unused.json']
        for extra in (['--payload-bytes', '0'], ['--payload-bytes', '33'],
                      ['--payload-bytes', str((64 << 20) + 32)],
                      ['--device', '-1'], ['--timeout', '0'],
                      ['--device-nic', 'tcp://localhost:99999']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                p.arguments(base + extra)

    def test_contract_and_patterns(self):
        import registered_host_rdma_probe as p
        args = p.arguments(['--rank', '0', '--device', '0', '--store-url',
            'tcp://127.0.0.1:19001', '--coord-url', 'http://127.0.0.1:19002',
            '--run-id', 'cpu-check', '--output', 'unused.json'])
        plan = p.build_plan(args)
        self.assertFalse(args.allow_npu)
        self.assertEqual(plan['status'], 'PLAN_ONLY')
        self.assertEqual(plan['create2']['data_op_type'], 'DEVICE_RDMA')
        self.assertEqual(plan['create2']['local_hbm_size'], 0)
        self.assertEqual(plan['create2']['local_dram_size'], 1 << 30)
        self.assertNotEqual(p.pattern(4096, 'A'), p.pattern(4096, 'B'))
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('memfabric_hybrid', sys.modules)

    def test_real_ctypes_buffer_and_range_rejections(self):
        import registered_host_rdma_probe as p
        size = 4096
        allocation = ctypes.create_string_buffer(4 * size)
        base = ctypes.addressof(allocation)
        maps = f'{base:x}-{base + 4 * size:x} rw-p 00000000 00:00 0\n'
        p.validate_regions(base, 4 * size, [(base, size), (base + 2 * size, size)], maps)
        p.write_local(base, p.pattern(size, 'A'), maps)
        self.assertTrue(p.verify_local(base, size, 'A', maps)['equal'])
        self.assertFalse(p.verify_local(base, size, 'B', maps)['equal'])
        cases = [(0, 4 * size, [(0, size)], maps),
                 (base, size, [(base + 2 * size, size)], maps),
                 (base, 4 * size, [(base, size), (base + size - 1, size)], maps),
                 (base, 4 * size, [(base, size)], maps.replace('rw-p', 'r--p')),
                 ((1 << 64) - 1, 4, [((1 << 64) - 1, 4)], maps)]
        for case in cases:
            with self.assertRaises(ValueError):
                p.validate_regions(*case)
        split = f'{base:x}-{base + 20:x} rw-p 0 0:0 0\n{base+20:x}-{base+size:x} rw-p 0 0:0 0\n'
        p.validate_rw_range(base, size, split)
        with self.assertRaises(ValueError):
            p.validate_rw_range(base, size + 1, split)


if __name__ == '__main__':
    unittest.main()
