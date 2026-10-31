#!/usr/bin/env python3
"""CPU/control-path checks only; no torch, SDK, SSH, NPU or real RDMA."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

import multi_member_registered_probe as p
from direct_coordination import make_server, publish, wait_event, http_json, event_url


def cli(rank=0):
    return ['--rank', str(rank), '--device', '0', '--store-url', 'tcp://127.0.0.1:19201',
        '--coord-url', 'http://127.0.0.1:19202', '--run-id', 'cpu-world3', '--output', 'unused.json']


class Checks(unittest.TestCase):
    def test_plan_roles_and_contract(self):
        plans = [p.build_plan(p.arguments(cli(rank))) for rank in p.ROLES]
        self.assertEqual([plan['role'] for plan in plans], ['owner', 'writer', 'reader'])
        self.assertEqual([plan['world_size'] for plan in plans], [3, 3, 3])
        self.assertTrue(all(p.contract(plan) == p.contract(plans[0]) for plan in plans))
        self.assertEqual(plans[0]['create2'], dict(id=0, local_dram_size=1 << 30,
            max_dram_size=1 << 30, local_hbm_size=0, max_hbm_size=0,
            data_op_type='DEVICE_RDMA', enable_56bits_gva=True, flags=0))
        self.assertFalse(plans[0]['payload_via_rpc'])
        self.assertTrue(plans[0]['external_process_wait_required'])
        self.assertFalse(p.arguments(cli()).allow_npu)
        changed = dict(plans[0], payload_bytes=64)
        self.assertNotEqual(p.contract(changed), p.contract(plans[0]))

    def test_plan_cli_never_imports_native_modules(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'rank.json'
            with contextlib.redirect_stdout(io.StringIO()):
                p.main(cli(2) + ['--output', str(path)])
            report = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'PLAN_ONLY')
            with self.assertRaises(FileExistsError), contextlib.redirect_stdout(io.StringIO()):
                p.main(cli(2) + ['--output', str(path)])
        for name in ('torch', 'torch_npu', 'memfabric_hybrid'):
            self.assertNotIn(name, sys.modules)

    def test_payload_and_cli_limits(self):
        for n in p.PAYLOADS:
            self.assertEqual(p.arguments(cli() + ['--payload-bytes', str(n)]).payload_bytes, n)
        for options in (['--payload-bytes', '32'], ['--payload-bytes', str(64 << 20)],
                ['--world-size', '4'], ['--rank', '3'], ['--device', '8'],
                ['--timeout', '0'], ['--timeout', 'nan'], ['--timeout', 'inf'],
                ['--init-timeout', '0'], ['--run-id', 'a/b'],
                ['--store-url', 'http://localhost:1000'], ['--coord-url', 'tcp://localhost:1000'],
                ['--device-nic', 'tcp://localhost:99999']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                p.arguments(cli() + options)

    def test_event_identity_and_full_content_rejections(self):
        value = dict(protocol=p.PROTOCOL, name='event-A', rank=2)
        self.assertIs(p.validate_event(value, 'event-A', 2), value)
        for item, name, rank in ((None, 'event-A', 2), (value, 'event-B', 2),
                (value, 'event-A', 1), (dict(value, protocol='wrong'), 'event-A', 2)):
            with self.assertRaises(ValueError):
                p.validate_event(item, name, rank)
        p.require_content(dict(equal=True, verified_bytes=64), 64)
        for item in (dict(equal=False, verified_bytes=64), dict(equal=True, verified_bytes=32), {}):
            with self.assertRaises(AssertionError):
                p.require_content(item, 64)

    def test_world3_causal_protocol_and_metadata_only_http(self):
        # Bytearrays test the causal control protocol, never SDK/data-plane correctness.
        n = 64
        owner = bytearray([165] * n)
        writer = bytearray(p.pattern(n, 'A'))
        reader = bytearray([90] * n)
        order = []
        lock = threading.Lock()
        errors = []
        with tempfile.TemporaryDirectory() as folder:
            server = make_server('127.0.0.1', 0, folder, 'cpu-world3')
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = 'http://127.0.0.1:' + str(server.server_port)

            def record(name):
                with lock:
                    order.append(name)

            def verify(buffer, label, name):
                self.assertEqual(bytes(buffer), p.pattern(n, label))
                record(name)

            def owner_write(label):
                self.assertEqual(bytes(reader), p.pattern(n, 'A'))
                owner[:] = p.pattern(n, label)
                record('owner_write_' + label)

            def write_A():
                owner[:] = writer
                record('writer_push_A')

            def destroy():
                writer[:] = bytes([51]) * n
                record('writer_destroy')

            def reset():
                reader[:] = bytes([90]) * n
                record('reader_sentinel')

            def read(label):
                self.assertEqual(bytes(writer), bytes([51]) * n)
                self.assertEqual(bytes(reader), bytes([90]) * n)
                reader[:] = owner
                record('reader_pull_' + label)

            actions = dict(owner_verify=lambda label: verify(owner, label, 'owner_verify_' + label),
                owner_write=owner_write, write_A=write_A, destroy_source=destroy,
                reset_restore=reset, read=read,
                reader_verify=lambda label: verify(reader, label, 'reader_verify_' + label))

            def launch(rank):
                def mark(name, **data):
                    publish(base, 'cpu-world3', name,
                        dict(protocol=p.PROTOCOL, name=name, rank=rank, **data))
                def wait(name, peer):
                    return p.validate_event(wait_event(base, 'cpu-world3', name, 3), name, peer)
                try:
                    p.role_protocol(rank, n, actions, mark, wait)
                except BaseException as exc:
                    with lock:
                        errors.append(exc)
            workers = [threading.Thread(target=launch, args=(rank,), daemon=True) for rank in p.ROLES]
            try:
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join(5)
                self.assertFalse(any(worker.is_alive() for worker in workers))
                self.assertEqual(errors, [])
                self.assertLess(order.index('owner_verify_A'), order.index('writer_destroy'))
                self.assertLess(order.index('writer_destroy'), order.index('reader_pull_A'))
                self.assertLess(order.index('reader_verify_A'), order.index('owner_write_B'))
                self.assertLess(order.index('owner_verify_B'), order.index('reader_pull_B'))
                events = http_json(event_url(base, 'cpu-world3'))['events']
                self.assertTrue(all('payload-done-' + str(rank) in events for rank in p.ROLES))
                for value in events.values():
                    self.assertFalse(any(isinstance(item, (list, bytes, bytearray)) for item in value.values()))
                self.assertEqual(bytes(owner), p.pattern(n, 'B'))
                self.assertEqual(bytes(reader), p.pattern(n, 'B'))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(2)

    def test_http_failure_and_missing_peer_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            server = make_server('127.0.0.1', 0, folder, 'failure-world3')
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = 'http://127.0.0.1:' + str(server.server_port)
            try:
                with self.assertRaises(TimeoutError):
                    wait_event(base, 'failure-world3', 'missing', .05)
                publish(base, 'failure-world3', 'failed-1',
                    dict(name='failed-1', rank=1, protocol=p.PROTOCOL, error='test-only'))
                with self.assertRaisesRegex(RuntimeError, 'peer failed'):
                    wait_event(base, 'failure-world3', 'missing', .5)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(2)

    def test_role_protocol_rejects_incomplete_writer(self):
        invoked = []
        with self.assertRaises(AssertionError):
            p.role_protocol(0, 64, {}, lambda *args, **kwargs: invoked.append(args),
                            lambda *args: dict(completed_bytes=32))
        self.assertEqual(invoked, [])


if __name__ == '__main__':
    unittest.main()
