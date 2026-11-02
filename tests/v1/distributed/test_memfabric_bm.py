# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch checks for the BM wrapper; no MemFabric SDK is initialized."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import threading
import unittest


class BMCopyTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[3] / "lmcache_ascend/v1/distributed/memfabric_bm.py"
        spec = importlib.util.spec_from_file_location("memfabric_bm_cpu_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module = module
        self.session = module.BMSession.__new__(module.BMSession)
        self.session.thread_id = threading.get_ident()
        self.session.closed = False
        self.session.peer_departed = threading.Event()
        self.session.departed_rank = None
        self.session.bm = SimpleNamespace(BmCopyType=SimpleNamespace(
            H2G="host-to-global", G2H="global-to-host",
            L2G="local-device-to-global", G2L="global-to-local-device"))
        self.session.handle = SimpleNamespace(copy_data=MagicMock(return_value=0))

    def test_compatibility_and_direct_directions_remain_distinct_and_sync(self):
        session = self.session
        calls = [
            (lambda: session.copy(1, 2, 32, store=True), "host-to-global"),
            (lambda: session.copy(1, 2, 32, store=False), "global-to-host"),
            (lambda: session.copy_npu_to_global(1, 2, 32), "local-device-to-global"),
            (lambda: session.copy_global_to_npu(1, 2, 32), "global-to-local-device"),
        ]
        for copy, expected in calls:
            with self.subTest(direction=expected):
                session.handle.copy_data.reset_mock()
                copy()
                session.handle.copy_data.assert_called_once_with(1, 2, 32, expected, 0)

    def test_copy_keeps_owner_and_native_error_checks(self):
        self.session.thread_id = -1
        with self.assertRaisesRegex(RuntimeError, "owner thread"):
            self.session.copy_npu_to_global(1, 2, 32)
        self.session.handle.copy_data.assert_not_called()
        self.session.thread_id = threading.get_ident()
        self.session.handle.copy_data.return_value = 7
        with self.assertRaisesRegex(RuntimeError, "BM copy failed: 7"):
            self.session.copy_global_to_npu(1, 2, 32)

    def test_startup_timeout_defaults_to_90_and_can_be_extended(self):
        for supplied, expected in (({}, 90), ({"init_timeout": 300}, 300)):
            with self.subTest(config=supplied):
                handle = MagicMock()
                handle.join.return_value = handle.wait.return_value = handle.leave.return_value = 0
                handle.peer_rank_ptr.return_value = 4096
                handle.gva_to_va.return_value = 8192
                bm = SimpleNamespace(BmConfig=SimpleNamespace,
                    initialize=MagicMock(return_value=0),
                    create2=MagicMock(return_value=handle), uninitialize=MagicMock(),
                    BmDataOpType=SimpleNamespace(SDMA=1),
                    BmMemType=SimpleNamespace(HOST=1, LOCAL_DEVICE=2))
                mf = SimpleNamespace(bm=bm, initialize=MagicMock(return_value=0),
                                     uninitialize=MagicMock())
                torch = SimpleNamespace(npu=SimpleNamespace(set_device=MagicMock()))
                with patch.dict("sys.modules", {"torch": torch,
                        "torch_npu": SimpleNamespace(), "memfabric_hybrid": mf}), \
                        patch.object(self.module.importlib.metadata, "version", return_value="1.2.1"):
                    session = self.module.BMSession(device_id=0, rank_id=0,
                        world_size=2, pool_gb=1, store_url="test-only", **supplied)
                    native_config = bm.initialize.call_args.args[3]
                    self.assertEqual(native_config.init_timeout, expected)
                    self.assertEqual(native_config.create_timeout, expected)
                    self.assertEqual(native_config.operation_timeout, 90)
                    session.close()

    def test_startup_timeout_rejects_invalid_values_before_native_import(self):
        for value in (0, -1, 3601, True, 1.5, "300"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "init_timeout"):
                self.module.BMSession(device_id=0, rank_id=0, world_size=2,
                    pool_gb=1, store_url="test-only", init_timeout=value)


if __name__ == "__main__":
    unittest.main()
