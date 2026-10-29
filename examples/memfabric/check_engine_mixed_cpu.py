"""Narrow CPU checks: engine argv, diagnostic call order, and identity evidence."""
from copy import deepcopy
import importlib.util
import io
import contextlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import cross_host_role as roles
import direct_engine_mixed_probe as probe
import direct_pd_gate as gate
from check_cross_host_cpu import trace


class EngineMixedChecks(unittest.TestCase):
    def args(self, role):
        return SimpleNamespace(root='/private', python='/python', topology='T3',
            stage='inference', session_id='cpu', pool_gb=2, tp_size=4,
            model='/model', skip_blocks=0, pool_memory_type='device',
            p_devices=[2,3,4,5], d_devices=[8,9,10,11], pool_device=14, role=role)

    def test_role_default_is_one_and_two_only_changes_engine_argv(self):
        for role in ('p','d','service','gate','proxy','trace137'):
            args=self.args(role)
            before=roles.build_plan(args)
            args.max_num_seqs=2
            after=roles.build_plan(args)
            if role in ('p','d'):
                pos=after['argv'].index('--max-num-seqs')+1
                self.assertEqual(before['argv'][pos],'1')
                self.assertEqual(after['argv'][pos],'2')
                after['argv'][pos]='1'
            self.assertEqual(before,after)
        bad=self.args('d');bad.max_num_seqs=3
        with self.assertRaises(ValueError): roles.build_plan(bad)

    def test_original_scheduler_call_order_input_and_return_are_preserved(self):
        calls=[];sentinel=object()
        class Base:
            def build_connector_meta(self, output):
                calls.append(('original',output));return sentinel
        name='lmcache_ascend.integration.vllm.memfabric_direct_connector'
        fake=ModuleType(name);fake.MemFabricDirectConnector=Base
        spec=importlib.util.spec_from_file_location('mixed_trace_cpu',
            Path(__file__).with_name('traced_direct_connector.py'))
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{name:fake}):spec.loader.exec_module(module)
        connector=module.MemFabricDirectConnector.__new__(module.MemFabricDirectConnector)
        output=SimpleNamespace(num_scheduled_tokens={'one':1,'two':2,'zero':0})
        original=deepcopy(output.num_scheduled_tokens)
        with patch.object(module,'emit',side_effect=lambda *args,**kw:calls.append(('emit',args,kw))):
            self.assertIs(connector.build_connector_meta(output),sentinel)
            self.assertIs(connector.build_connector_meta(output),sentinel)
        self.assertEqual([row[0] for row in calls],['emit','original','emit','original'])
        self.assertEqual(calls[0][2]['scheduled_request_ids'],['one','two'])
        self.assertEqual(calls[2][2]['scheduler_step'],2)
        self.assertEqual(output.num_scheduled_tokens,original)

    def test_scheduler_pid_is_not_misrepresented_as_tp_worker(self):
        records=[]
        for role in ('P','D'):
            for shard in range(4):records.extend(trace(role,100,4,shard))
        original=gate.worker_roster(records,4)
        records.append(dict(kind='scheduler_step',phase='begin',role='D',pid=999,
            host_id='host-D',session_id='cpu-case',mono_ns=900,
            num_scheduled_tokens={'one':1,'two':2},scheduled_request_ids=['one','two']))
        self.assertEqual(gate.worker_roster(records,4),original)

    def fixture(self):
        before=dict(host_id='host',run_id='run',mono_ns=100)
        row=dict(kind='scheduler_step',phase='begin',role='D',host_id='host',
            session_id='run',mono_ns=110,pid=900,scheduler_step=7,
            num_scheduled_tokens={'chatcmpl-one':1,'chatcmpl-two':2})
        after=dict(host_id='host',run_id='run',records=[row])
        responses=[dict(response_request_id='chatcmpl-'+key,client_request_id=key)
                   for key in ('one','two')]
        return before,after,responses

    def test_same_step_positive_tokens_and_http_identities_are_required(self):
        before,after,responses=self.fixture()
        self.assertTrue(probe.scheduler_evidence(before,after,responses)['proven'])
        after['records'][0]['num_scheduled_tokens']['chatcmpl-two']=0
        self.assertFalse(probe.scheduler_evidence(before,after,responses)['proven'])
        before,after,responses=self.fixture()
        after['records'][0]['num_scheduled_tokens']={'chatcmpl-one':1}
        other=deepcopy(after['records'][0]);other['num_scheduled_tokens']={'chatcmpl-two':1}
        other['scheduler_step']=8;after['records'].append(other)
        self.assertFalse(probe.scheduler_evidence(before,after,responses)['proven'])

    def test_unlinked_or_wrong_session_cannot_claim_mixed_schedule(self):
        before,after,responses=self.fixture()
        responses[1]['response_request_id']='other';responses[1]['client_request_id']='other'
        self.assertFalse(probe.scheduler_evidence(before,after,responses)['proven'])
        before,after,responses=self.fixture();after['records'][0]['session_id']='wrong'
        with self.assertRaises(AssertionError):probe.scheduler_evidence(before,after,responses)

    def test_installed_vllm_exact_api_id_plus_eight_lower_hex_contract(self):
        api='chatcmpl-followup-abcdef0123456789'
        self.assertTrue(probe.scheduler_id_matches_api_id(api,api))
        self.assertTrue(probe.scheduler_id_matches_api_id(api+'-ab6613ba',api))
        for wrong in [api+'-ab6613b',api+'-ab6613baa',api+'-ab6613bg',
                      api+'-AB6613BA',api+'-ab6613ba-extra','prefix-'+api+'-ab6613ba',
                      api.removeprefix('chatcmpl-')+'-ab6613ba',
                      'chatcmpl-other-request-ab6613ba']:
            self.assertFalse(probe.scheduler_id_matches_api_id(wrong,api),wrong)
        before,after,responses=self.fixture()
        after['records'][0]['num_scheduled_tokens']={
            'chatcmpl-one-ab6613ba':1,'chatcmpl-two-0123abcd':93}
        proof=probe.scheduler_evidence(before,after,responses)
        self.assertTrue(proof['proven'])
        self.assertEqual(proof['scheduler_request_ids'],
            ['chatcmpl-one-ab6613ba','chatcmpl-two-0123abcd'])

    def test_ambiguous_or_duplicate_internal_mapping_cannot_claim_two_requests(self):
        before,after,responses=self.fixture()
        responses[1]['response_request_id']='chatcmpl-one-ab6613ba'
        after['records'][0]['num_scheduled_tokens']={
            'chatcmpl-one-ab6613ba':1,'chatcmpl-one-ab6613ba-0123abcd':2}
        proof=probe.scheduler_evidence(before,after,responses)
        self.assertFalse(proof['proven'])
        self.assertEqual(proof['ambiguous_scheduler_steps'],1)
        before,after,responses=self.fixture()
        after['records'][0]['num_scheduled_tokens']={
            'chatcmpl-one-ab6613ba':1,'chatcmpl-one-0123abcd':2,'chatcmpl-two':3}
        self.assertFalse(probe.scheduler_evidence(before,after,responses)['proven'])

    def test_default_plan_does_not_issue_http_or_create_output(self):
        with patch.object(probe.follow,'http',side_effect=AssertionError('HTTP')):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                probe.main(['--gate-dir','unused','--decoder','http://unused',
                    '--service-http','http://unused','--trace-url','http://unused',
                    '--session-id','cpu','--output','unused'])
        self.assertEqual(json.loads(output.getvalue())['status'],'PLAN_ONLY')


if __name__=='__main__':unittest.main(verbosity=2)
