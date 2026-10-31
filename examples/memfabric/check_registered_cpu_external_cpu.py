#!/usr/bin/env python3
"""CPU-only external-pointer and causal-protocol checks; not SDK/RDMA validation."""
import contextlib
import ctypes
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

import registered_cpu_external_probe as p


def cli(rank=0):
    return ['--rank', str(rank), '--device', '0', '--store-url', 'tcp://127.0.0.1:19701',
        '--coord-url', 'http://127.0.0.1:29701', '--run-id', 'cpu-external', '--output', 'unused.json']


class Checks(unittest.TestCase):
    def test_plan_contract_and_no_native_imports(self):
        a, b = [p.build_plan(p.arguments(cli(rank))) for rank in (0, 1)]
        self.assertEqual(p.contract(a), p.contract(b))
        self.assertEqual(a['copy_types'], dict(write='H2G', read='G2H'))
        self.assertEqual(a['create2']['local_hbm_size'], 0)
        self.assertEqual(a['create2']['local_dram_size'], 1 << 30)
        self.assertTrue(a['create2']['enable_56bits_gva'])
        self.assertFalse(a['page_alignment_requested'])
        self.assertFalse(a['npu_payload_allocation'])
        self.assertIn('pin_memory=False', a['allocator'])
        self.assertFalse(p.arguments(cli()).allow_npu)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'plan.json'
            with contextlib.redirect_stdout(io.StringIO()):
                p.main(cli() + ['--output', str(path)])
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['status'], 'PLAN_ONLY')
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                p.main(cli() + ['--output', str(path)])
        for name in ('torch', 'torch_npu', 'memfabric_hybrid'):
            self.assertNotIn(name, sys.modules)

    def test_cli_payloads_and_invalid_requests(self):
        for n in (64, 1048576, 18874368):
            self.assertEqual(p.arguments(cli() + ['--payload-bytes', str(n)]).payload_bytes, n)
        for extra in (['--payload-bytes', '32'], ['--rank', '2'], ['--world-size', '3'],
                ['--device', '8'], ['--timeout', 'nan'], ['--timeout', 'inf'], ['--timeout', '0'],
                ['--init-timeout', '0'], ['--device-nic', 'tcp://localhost:65536'],
                ['--store-url', 'http://localhost:19001'], ['--run-id', '../bad']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                p.arguments(cli() + extra)

    def test_real_cpu_ranges_external_to_pool_and_nonoverlap(self):
        allocation = ctypes.create_string_buffer(4096)
        base = ctypes.addressof(allocation)
        source, restore, own_pool = base, base + 128, base + 2048
        maps = f'{base:x}-{base+4096:x} rw-p 0 0:0 0\n'
        p.validate_external_ranges(source, restore, 64, own_pool, 1024, maps)
        for src, dst, n, pool, size, text in (
                (source, source + 32, 64, own_pool, 1024, maps),
                (own_pool, restore, 64, own_pool, 1024, maps),
                (source, own_pool - 32, 64, own_pool, 1024, maps),
                (0, restore, 64, own_pool, 1024, maps),
                (source, restore, 64, own_pool, 1024, maps.replace('rw-p', 'r--p'))):
            with self.assertRaises(ValueError):
                p.validate_external_ranges(src, dst, n, pool, size, text)

    def test_unaligned_pointer_report_does_not_realign(self):
        meta = p.pointer_metadata(0x10043, 64, 4096)
        self.assertEqual(meta['pointer'], 0x10043)
        self.assertEqual(meta['page_offset'], 67)
        self.assertFalse(meta['page_aligned'])
        self.assertEqual(meta['covering_page_begin'], 0x10000)
        self.assertEqual(meta['covering_page_end'], 0x11000)
        self.assertEqual(p.pointer_metadata(0x10ff0, 64, 4096)['covering_page_end'], 0x12000)
        with self.assertRaises(ValueError):
            p.pointer_metadata(1234, 64, 12)

    def test_two_role_control_protocol(self):
        # Bytearrays isolate protocol causality from the unverified SDK/native data plane.
        n = 64
        source = bytearray(p.pattern(n, 'A'))
        owner = bytearray([165] * n)
        restore = bytearray([90] * n)
        events, order, errors = {}, [], []
        condition = threading.Condition()

        def verify(buffer, label, label_step):
            self.assertEqual(bytes(buffer), p.pattern(n, label))
            order.append(label_step)

        def write_A():
            owner[:] = source
            order.append('write_A')

        def destroy():
            source[:] = bytes([51]) * n
            order.append('destroy')

        def reset():
            restore[:] = bytes([90]) * n
            order.append('sentinel')

        def read(label):
            self.assertEqual(bytes(source), bytes([51]) * n)
            self.assertEqual(bytes(restore), bytes([90]) * n)
            restore[:] = owner
            order.append('read_' + label)

        def write_B(label):
            self.assertEqual(bytes(restore), p.pattern(n, 'A'))
            owner[:] = p.pattern(n, label)
            order.append('owner_write_B')

        actions = dict(write_A=write_A, destroy_source=destroy, reset_restore=reset, read=read,
            verify_restore=lambda label: verify(restore, label, 'restore_' + label),
            owner_write=write_B, owner_verify=lambda label: verify(owner, label, 'owner_' + label))

        def worker(rank):
            def mark(name, **data):
                with condition:
                    self.assertNotIn(name, events)
                    events[name] = dict(rank=rank, **data)
                    condition.notify_all()
            def wait(name, peer):
                deadline = time.monotonic() + 2
                with condition:
                    while name not in events:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(name)
                        condition.wait(remaining)
                    self.assertEqual(events[name]['rank'], peer)
                    return events[name]
            try:
                p.role_protocol(rank, n, actions, mark, wait)
            except BaseException as exc:
                errors.append(exc)
        workers = [threading.Thread(target=worker, args=(rank,), daemon=True) for rank in (0, 1)]
        for item in workers:
            item.start()
        for item in workers:
            item.join(3)
        self.assertFalse(any(item.is_alive() for item in workers))
        self.assertEqual(errors, [])
        self.assertLess(order.index('owner_A'), order.index('destroy'))
        self.assertLess(order.index('destroy'), order.index('read_A'))
        self.assertLess(order.index('restore_A'), order.index('owner_write_B'))
        self.assertLess(order.index('owner_B'), order.index('read_B'))
        self.assertIn('payload-done-0', events)
        self.assertIn('payload-done-1', events)

    def test_bad_owner_ack_aborts_before_source_destruction(self):
        called = []
        with self.assertRaises(AssertionError):
            p.role_protocol(0, 64, dict(write_A=lambda: called.append('write'),
                destroy_source=lambda: called.append('destroy')), lambda *args, **kwargs: None,
                lambda *args: dict(equal=True, verified_bytes=32))
        self.assertEqual(called, ['write'])


if __name__ == '__main__':
    unittest.main()
