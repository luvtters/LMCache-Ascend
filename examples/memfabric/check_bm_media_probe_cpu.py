#!/usr/bin/env python3
"""CPU checks for the isolated probe; never imports an accelerator SDK."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import bm_media_probe as probe


class ProbeChecks(unittest.TestCase):
    def payload_args(self, *extra):
        return ['--rank', '0', '--device', '0', '--media', 'host',
                '--store-url', 'tcp://127.0.0.1:28801',
                '--coord-url', 'http://127.0.0.1:29601',
                '--run-id', 'cpu-payload', '--output', 'unused.json', *extra]

    def test_payload_size_validation_and_default(self):
        self.assertEqual(probe.arguments(self.payload_args()).payload_bytes, 1 << 20)
        for size in (32, 64, 4096, 1 << 20, 18 << 20, 64 << 20):
            with self.subTest(size=size):
                args = probe.arguments(self.payload_args('--payload-bytes', str(size)))
                self.assertEqual(probe.build_plan(args)['payload_bytes'], size)
        for size in ('0', '-32', '1', '63', str((64 << 20) + 32), '1.5'):
            with self.subTest(size=size), self.assertRaises(SystemExit):
                probe.arguments(self.payload_args('--payload-bytes', size))

    def test_small_payload_contract_rejects_different_size(self):
        plans = [probe.build_plan(probe.arguments(self.payload_args(
            '--rank', str(rank), '--payload-bytes', '64'))) for rank in (0, 1)]
        probe.check_peer_contract(probe.peer_contract(plans[0]), probe.peer_contract(plans[1]))
        wrong = dict(probe.peer_contract(plans[1]), payload_bytes=4096)
        with self.assertRaises(ValueError):
            probe.check_peer_contract(probe.peer_contract(plans[0]), wrong)

    def test_small_payload_drives_both_runtime_allocations_copies_and_counts(self):
        self.check_small_buffer_runtime('npu')

    def test_cpu_buffer_kind_is_explicit_restricted_and_in_contract(self):
        original = probe.build_plan(probe.arguments(self.payload_args()))
        self.assertEqual(original['buffer_kind'], 'npu')
        alternate = probe.build_plan(probe.arguments(self.payload_args('--buffer-kind', 'cpu')))
        self.assertEqual(alternate['buffer_kind'], 'cpu')
        self.assertEqual(alternate['copy_types'], dict(write='H2G', read='G2H'))
        self.assertTrue(alternate['npu_context_required'])
        with self.assertRaises(ValueError):
            probe.check_peer_contract(probe.peer_contract(original), probe.peer_contract(alternate))
        for extra in (['--media', 'device'], ['--data-op-type', 'device_rdma']):
            with self.assertRaises(SystemExit):
                probe.arguments(self.payload_args('--buffer-kind', 'cpu', *extra))

    def test_cpu_buffers_copy_types_wait_order_and_full_content(self):
        self.check_small_buffer_runtime('cpu')

    def test_explicit_dram_numa_requests_combine_without_changing_defaults(self):
        original = probe.build_plan(probe.arguments(self.payload_args()))
        self.assertIsNone(original['requested_dram_numa_node'])
        self.assertEqual(original['create2']['flags'], 0)
        mapped = probe.build_plan(probe.arguments(self.payload_args('--dram-map-host-va')))
        self.assertEqual(mapped['create2']['flags'], 512)
        for node, expected in ((0, 128), (4, 132)):
            for extra, map_flag in (([], 0), (['--dram-map-host-va'], 512)):
                with self.subTest(node=node, mapped=bool(extra)):
                    plan = probe.build_plan(probe.arguments(self.payload_args(
                        '--dram-numa-node', str(node), *extra)))
                    self.assertEqual(plan['requested_dram_numa_node'], node)
                    self.assertEqual(plan['actual_dram_numa_node'], 'NOT_VERIFIED')
                    self.assertEqual(plan['create2']['flags'], expected | map_flag)
                    with self.assertRaises(ValueError):
                        probe.check_peer_contract(probe.peer_contract(original), probe.peer_contract(plan))

    def test_dram_numa_rejects_wrong_route_and_invalid_node(self):
        for extra in (['--media', 'device'], ['--data-op-type', 'device_rdma']):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                probe.arguments(self.payload_args('--dram-numa-node', '0', *extra))
        for node in ('-1', '8', '1.5', 'invalid'):
            with self.subTest(node=node), self.assertRaises(SystemExit):
                probe.arguments(self.payload_args('--dram-numa-node', node))

    def check_small_buffer_runtime(self, kind):
        # Fake only unavailable device/SDK boundaries; execute the real run path.
        size = 64
        allocations, copies, publications, steps, devices = [], [], [], [], []
        class Buffer:
            def __init__(self, data):
                self.data = bytes(data)
                allocations.append(len(self.data))
            def to(self, device):
                if kind == 'cpu':
                    raise AssertionError('CPU source must not be transferred to NPU')
                devices[-1] = device
                return self
            def data_ptr(self): return 4096
            def cpu(self): return self
            def numpy(self): return self
            def tobytes(self): return probe.pattern_bytes(len(self.data))
        class Handle:
            def peer_rank_ptr(self, *args): return 8192
            def gva_to_va(self, *args): return 16384
            def copy_data(self, src, dst, count, direction, flags):
                copies.append((count, direction, flags))
                steps.append('copy')
                return 0
            def join(self): return 0
            def wait(self):
                steps.append('wait')
                return 0
            def leave(self): return 0
            def destroy(self): return None
        bm = SimpleNamespace(BmConfig=SimpleNamespace,
            BmMemType=SimpleNamespace(HOST=3, LOCAL_DEVICE=0),
            BmDataOpType=SimpleNamespace(SDMA=1),
            BmCopyType=SimpleNamespace(L2G=1, G2L=2, H2G=3, G2H=4),
            initialize=lambda *a: 0, uninitialize=lambda: None,
            create2=lambda **kw: Handle(), __file__='cpu-fake')
        mf = SimpleNamespace(bm=bm, initialize=lambda: 0, uninitialize=lambda: None,
                             set_log_level=lambda level: None, __file__='cpu-fake')
        def frombuffer(data, **kw):
            devices.append('cpu')
            return Buffer(data)
        def full(shape, fill, **kw):
            devices.append(kw['device'])
            return Buffer(bytes([fill]) * shape[0])
        torch = SimpleNamespace(uint8='uint8', frombuffer=frombuffer, full=full,
            npu=SimpleNamespace(set_device=lambda device: None, synchronize=lambda device: None))
        for rank in (0, 1):
            extra = ['--buffer-kind', 'cpu'] if kind == 'cpu' else []
            args = probe.arguments(self.payload_args('--rank', str(rank), '--payload-bytes', str(size), *extra))
            report = probe.build_plan(args)
            events = []
            def wait_event(url, run_id, name, timeout):
                return dict(media='host', data_op_type='sdma', buffer_kind=kind,
                            contract=probe.peer_contract(report),
                            group_id=0, owner_ptr=8192, target_memory='HOST',
                            completed_bytes=size, equal=True, verified_bytes=size)
            with patch.dict(sys.modules, {'torch': torch, 'torch_npu': SimpleNamespace(),
                                         'memfabric_hybrid': mf}), \
                    patch.object(probe, 'wait_event', wait_event), \
                    patch.object(probe, 'publish', lambda *a: publications.append(a[3])):
                probe.run(args, report, lambda event, **kw: events.append((event, kw)))
            copy_name = ('H2G' if rank == 0 else 'G2H') if kind == 'cpu' else (
                'L2G' if rank == 0 else 'G2L')
            sdk_steps = [metadata['step'] for event, metadata in events if event == 'SDK_BEGIN']
            self.assertIn('copy_' + copy_name, sdk_steps)
            self.assertEqual(sdk_steps[sdk_steps.index('copy_' + copy_name) + 1],
                             'wait_after_' + copy_name)
            self.assertTrue(report['payload_passed'])
            self.assertTrue(report['local_cleanup_passed'])
            self.assertEqual(report['completed_write_bytes' if rank == 0 else 'completed_read_bytes'], size)
            if rank == 1:
                self.assertEqual(report['verified_bytes'], size)
                self.assertTrue(report['full_content_equal'])
        self.assertEqual(allocations, [size, size])
        self.assertEqual(copies, [(size, 3, 0), (size, 4, 0)] if kind == 'cpu'
                         else [(size, 1, 0), (size, 2, 0)])
        self.assertEqual(devices, ['cpu', 'cpu'] if kind == 'cpu' else ['npu:0', 'npu:0'])
        self.assertEqual(steps, ['copy', 'wait', 'wait', 'copy', 'wait', 'wait'])
        self.assertEqual([p['completed_bytes'] for p in publications if 'completed_bytes' in p], [size])
        self.assertEqual([p['verified_bytes'] for p in publications if 'verified_bytes' in p], [size])
        publications.clear()
        with patch.dict(sys.modules, {'torch': torch, 'torch_npu': SimpleNamespace(),
                                     'memfabric_hybrid': mf}), \
                patch.object(probe, 'wait_event', wait_event), \
                patch.object(probe, 'publish', lambda *a: publications.append(a[3])), \
                patch.object(Buffer, 'tobytes', lambda self: probe.pattern_bytes(size - 1) + b'\x00'):
            corrupted = probe.build_plan(args)
            with self.assertRaisesRegex(AssertionError, 'Restored owner buffer'):
                probe.run(args, corrupted, lambda *a, **kw: None)
        self.assertFalse(corrupted['full_content_equal'])
        self.assertEqual(corrupted['mismatch_bytes'], 1)
        self.assertFalse(any('verified_bytes' in p for p in publications))

    def test_dram_mapping_variant_is_explicit_and_in_peer_contract(self):
        common=['--rank','0','--device','0','--media','host',
            '--store-url','tcp://127.0.0.1:28801','--coord-url','http://127.0.0.1:29601',
            '--run-id','cpu-map-mode','--output','unused.json']
        original=probe.build_plan(probe.arguments(common))
        alternate=probe.build_plan(probe.arguments(common+['--dram-map-host-va']))
        self.assertEqual(original['create2']['flags'],0)
        self.assertEqual(alternate['create2']['flags'],1<<9)
        self.assertEqual(alternate['create2']['local_hbm_size'],0)
        bm=SimpleNamespace(BmDataOpType=SimpleNamespace(SDMA=1))
        self.assertEqual(probe.sdk_create_options(alternate,bm)['flags'],1<<9)
        with self.assertRaisesRegex(ValueError,'Ranks disagree'):
            probe.check_peer_contract(probe.peer_contract(original),probe.peer_contract(alternate))
        for extra in (['--media','device'],['--data-op-type','device_rdma']):
            with self.assertRaises(SystemExit):
                probe.arguments(common+extra+['--dram-map-host-va'])

    def test_pull_only_reverses_writer_reader_and_keeps_sdma_default(self):
        plans=[]
        for rank in [0,1]:
            args=probe.arguments(['--rank',str(rank),'--device','0','--media','host',
                '--direction','pull-only','--store-url','tcp://127.0.0.1:28801',
                '--coord-url','http://127.0.0.1:29622','--run-id','cpu-pull','--output','unused'])
            plan=probe.build_plan(args); plans.append(plan)
            self.assertEqual((plan['owner_rank'],plan['writer_rank'],plan['reader_rank']),(1,1,0))
            self.assertEqual(plan['data_op_type'],'sdma')
            self.assertFalse(plan['create2']['enable_56bits_gva'])
        probe.check_peer_contract(probe.peer_contract(plans[0]),probe.peer_contract(plans[1]))
        wrong=dict(probe.peer_contract(plans[1]),direction='push')
        with self.assertRaises(ValueError): probe.check_peer_contract(probe.peer_contract(plans[0]),wrong)

    def test_protocol_enum_and_both_rank_contracts_are_explicit(self):
        fake=SimpleNamespace(BmDataOpType=SimpleNamespace(SDMA=1,DEVICE_RDMA=8))
        self.assertEqual(probe.sdk_data_op(fake,'sdma'),1)
        self.assertEqual(probe.sdk_data_op(fake,'device_rdma'),8)
        with self.assertRaises(AttributeError):
            probe.sdk_data_op(SimpleNamespace(BmDataOpType=SimpleNamespace(SDMA=1)),'device_rdma')
        plans=[]
        for rank in [0,1]:
            args=probe.arguments(['--rank',str(rank),'--device','0','--media','host',
                '--data-op-type','device_rdma','--store-url','tcp://127.0.0.1:28798',
                '--coord-url','http://127.0.0.1:29608','--run-id','cpu-rdma','--output','unused'])
            plans.append(probe.build_plan(args))
        probe.check_peer_contract(probe.peer_contract(plans[0]),probe.peer_contract(plans[1]))
        self.assertEqual(plans[0]['create2']['data_op_type'],'DEVICE_RDMA')
        self.assertFalse(plans[0]['create2']['enable_56bits_gva'])
        args=probe.arguments(['--rank','0','--device','0','--media','host',
            '--data-op-type','device_rdma','--enable-56bits-gva',
            '--store-url','tcp://127.0.0.1:28798','--coord-url','http://127.0.0.1:29608',
            '--run-id','cpu-rdma56','--output','unused'])
        enabled=probe.build_plan(args)
        self.assertTrue(probe.sdk_create_options(enabled,fake)['enable_56bits_gva'])
        self.assertEqual(probe.sdk_create_options(enabled,fake)['data_op_type'],8)
        self.assertFalse(probe.sdk_create_options(plans[0],fake)['enable_56bits_gva'])
        wrong=dict(probe.peer_contract(plans[1]),data_op_type='sdma')
        with self.assertRaises(ValueError): probe.check_peer_contract(probe.peer_contract(plans[0]),wrong)
        wrong=dict(probe.peer_contract(plans[1]))
        wrong['create2']=dict(wrong['create2'],enable_56bits_gva=True)
        with self.assertRaises(ValueError): probe.check_peer_contract(probe.peer_contract(plans[0]),wrong)

    def test_rdma_owner_address_does_not_require_sdma_local_mapping(self):
        class Handle:
            def peer_rank_ptr(self, rank, kind): return 8192
            def gva_to_va(self,*args): raise AssertionError('RDMA must not require SDMA mapping')
        bm=SimpleNamespace(BmMemType=SimpleNamespace(DEVICE=2,HOST=3))
        self.assertEqual(probe.owner_address(Handle(),bm,'host',.1,lambda *a,**k:None,'device_rdma'),8192)

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
