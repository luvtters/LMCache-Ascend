# SPDX-License-Identifier: Apache-2.0
"""L2-only CPU fixtures extracted from the verified revision's test module."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

def load_bm():
    path = Path(__file__).parents[3] / "lmcache_ascend/v1/distributed/memfabric_bm.py"
    spec = importlib.util.spec_from_file_location("bm_review_cpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class FakeTensor(SimpleNamespace):
    pass

def tensor(pointer=1024, size=32):
    return FakeTensor(device=SimpleNamespace(type="npu", index=0),
        data_ptr=lambda: pointer, numel=lambda: size, element_size=lambda: 1,
        is_contiguous=lambda: True)

class RdmaContracts(unittest.TestCase):
    def make_session(self):
        module = load_bm()
        handle = MagicMock()
        handle.join.return_value = handle.leave.return_value = handle.wait.return_value = 0
        handle.register.return_value = handle.unregister.return_value = handle.copy_data.return_value = 0
        handle.peer_rank_ptr.return_value = 4096
        bm = SimpleNamespace(BmConfig=lambda: SimpleNamespace(set_nic=MagicMock()),
            initialize=MagicMock(return_value=0), create2=MagicMock(return_value=handle),
            uninitialize=MagicMock(), BmDataOpType=SimpleNamespace(SDMA=1, DEVICE_RDMA=2),
            BmCopyType=SimpleNamespace(L2G=3, G2L=4),
            BmMemType=SimpleNamespace(HOST=10, DEVICE=11, LOCAL_DEVICE=12))
        mf = SimpleNamespace(bm=bm, initialize=MagicMock(return_value=0), uninitialize=MagicMock())
        torch = SimpleNamespace(npu=SimpleNamespace(set_device=MagicMock()))
        with patch.dict("sys.modules", {"torch": torch, "torch_npu": SimpleNamespace(),
                "memfabric_hybrid": mf}), patch.object(
                    module.importlib.metadata, "version", return_value="1.2.1"):
            session = module.BMSession(device_id=0, rank_id=0, world_size=2,
                pool_gb=1, store_url="cpu-only", data_op_type="DEVICE_RDMA",
                device_nic="tcp://127.0.0.1:10005", enable_56bits_gva=True)
        return module, session, handle, bm


    def test_invalid_options_fail_before_sdk_import(self):
        module = load_bm()
        for supplied in ({"data_op_type": "rdma"}, {"data_op_type": "DEVICE_RDMA"},
                {"device_nic": "invalid"}, {"device_nic": "tcp://host:70000"},
                {"device_nic": 1}, {"enable_56bits_gva": "true"}):
            with self.subTest(supplied=supplied), self.assertRaises(ValueError):
                module.BMSession(device_id=0, rank_id=0, world_size=2,
                    pool_gb=1, store_url="cpu-only", **supplied)
