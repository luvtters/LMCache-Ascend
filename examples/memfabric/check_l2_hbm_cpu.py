"""CPU contract checks only; stub SDK calls do not establish H2G/NPU correctness."""
import importlib.util
import json
import logging
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]


def load(name, relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


class Checks(unittest.TestCase):
    def test_bm_selected_pointer_capacity_and_cpu_directions(self):
        module=load('hbm_bm_check','lmcache_ascend/v1/distributed/memfabric_bm.py')
        for media in ['host','device']:
            with self.subTest(media=media):
                calls=[]
                class Handle:
                    def local_mem_size(self, kind):
                        calls.append(('size',kind)); return 1<<30
                    def set_group_event_handler(self, fn): pass
                    def join(self): return 0
                    def peer_rank_ptr(self, rank, kind):
                        calls.append(('pointer',rank,kind)); return 8192+kind*4096
                    def gva_to_va(self, address, kind): return address
                    def copy_data(self,*args): calls.append(('copy',*args)); return 0
                    def wait(self): return 0
                    def leave(self): return 0
                    def destroy(self): calls.append(('destroy',))
                def create(**options): calls.append(('create',options)); return Handle()
                bm=SimpleNamespace(BmConfig=SimpleNamespace,BmDataOpType=SimpleNamespace(SDMA=1),
                    BmMemType=SimpleNamespace(HOST=3,DEVICE=2,LOCAL_DEVICE=0),
                    BmCopyType=SimpleNamespace(H2G=3,G2H=2),initialize=lambda *a:0,
                    create2=create,uninitialize=lambda:None)
                mf=SimpleNamespace(bm=bm,initialize=lambda:0,uninitialize=lambda:None)
                sdk={'torch':SimpleNamespace(npu=SimpleNamespace(set_device=lambda d:None)),
                     'torch_npu':SimpleNamespace(),'memfabric_hybrid':mf}
                with patch.dict(sys.modules,sdk),patch.object(module.importlib.metadata,'version',return_value='1.2.1'):
                    session=module.BMSession(device_id=0,rank_id=0,world_size=2,pool_gb=1,
                                            store_url='tcp://127.0.0.1:1',pool_memory_type=media)
                    try:
                        expected_type=2 if media=='device' else 3
                        self.assertEqual(session.peer_address(1),8192+expected_type*4096)
                        self.assertEqual(session.local_pool_bytes,1<<30)
                        session.copy(1,2,4096,store=True); session.copy(2,3,4096,store=False)
                        options=next(row[1] for row in calls if row[0]=='create')
                        self.assertEqual(options['local_dram_size'],1<<30)
                        self.assertEqual(options.get('local_hbm_size',0),(1<<30) if media=='device' else 0)
                        self.assertEqual([row[4] for row in calls if row[0]=='copy'],[3,2])
                        self.assertIn(('size',expected_type),calls)
                    finally: session.close()
                self.assertFalse(module.BMSession._active)
        with self.assertRaises(ValueError):
            module.BMSession(device_id=0,rank_id=0,world_size=1,pool_gb=1,store_url='unused',pool_memory_type='hbm')

    def test_adapter_propagation_status_and_memory_tags(self):
        made=[]
        def session(**kwargs):
            made.append(kwargs)
            return SimpleNamespace(local_pool_bytes=1<<30,peer_address=lambda rank:12345,
                peer_departed=threading.Event(),wait=lambda:None,close=lambda:None)
        class Base:
            def __init__(self,**kwargs): pass
        class Event:
            def close(self): pass
        def stub(name,**fields):
            result=ModuleType(name); result.__dict__.update(fields); return result
        modules={
            'lmcache.lmcache_native':stub('native',Bitmap=object),
            'lmcache.logging':stub('log',init_logger=logging.getLogger),
            'lmcache.v1.distributed.api':stub('api',MemoryResidency=SimpleNamespace(LOCAL='local'),
                MemorySegment=SimpleNamespace(DRAM='dram',VRAM='vram'),MemoryTags=lambda *a:a),
            'lmcache.v1.distributed.internal_api':stub('internal',L2StoreResult=object),
            'lmcache.v1.distributed.l2_adapters.base':stub('base',L2AdapterInterface=Base),
            'lmcache.v1.platform':stub('platform',create_event_notifier=Event),
            'lmcache_ascend.v1.distributed.memfabric_bm':stub('bm',BMSession=session)}
        with patch.dict(sys.modules,modules):
            module=load('hbm_adapter_check','lmcache_ascend/v1/distributed/memfabric_l2_adapter.py')
            for params,expected in [({},'host'),({'pool_memory_type':'device'},'device')]:
                adapter=module.MemFabricL2Adapter(params)
                try:
                    self.assertEqual(made[-1]['pool_memory_type'],expected)
                    status=adapter.report_status()
                    self.assertEqual(status['pool_native_mem_type'],'DEVICE' if expected=='device' else 'HOST')
                    self.assertEqual(status['local_selected_pool_bytes'],1<<30)
                    self.assertEqual(status['storage_peer_gva'],12345)
                    self.assertEqual(adapter.get_memory_tags()[0],'vram' if expected=='device' else 'dram')
                finally: adapter.close()
            with self.assertRaises(ValueError): module.MemFabricL2Adapter({'pool_memory_type':'bad'})

    def test_config_default_override_and_invalid(self):
        module=load('hbm_config_check','examples/memfabric/cross_host_20261030/l2_config.py')
        raw=dict(run_root='/private/runs',run_id='fresh',core_path='/private/core',
                 ascend_path='/private/ascend',communication_ip='192.168.0.223')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.json'
            for media in [None,'device','invalid']:
                cfg=dict(raw)
                if media is not None: cfg['pool_memory_type']=media
                path.write_text(json.dumps(cfg),encoding='utf-8')
                if media=='invalid':
                    with self.assertRaises(ValueError): module.read_config(path)
                else:
                    cfg=module.read_config(path)
                    for driver in [False,True]:
                        self.assertEqual(module.adapter_config(cfg,driver)['adapter_params']['pool_memory_type'],media or 'host')


if __name__=='__main__': unittest.main(verbosity=2)
