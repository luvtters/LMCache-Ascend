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
