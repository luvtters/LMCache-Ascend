# SPDX-License-Identifier: Apache-2.0
"""Execute real PP policy and existing lookup methods with CPU-only fixtures.

No torch, vLLM, plugin initializer, network or device runtime is imported.
"""

import array
import ast
from dataclasses import dataclass
import importlib.util
import math
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import time
import unittest
from unittest.mock import patch


PLUGIN = Path(__file__).resolve().parents[2]
CORE = Path(os.environ.get("KS_LMCACHE_SOURCE", PLUGIN.parent / "lmcache-2p1d-luv"))
ADAPTER = CORE / "lmcache/integration/vllm/vllm_multi_process_adapter.py"
POLICY = PLUGIN / "lmcache_ascend/integration/vllm/pp_stage_local.py"
PIN = PLUGIN / "lmcache_ascend/v1/pp_stage_pinning.py"


def real_class(path: Path, name: str, namespace: dict, names: set | None = None) -> type:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    if names is not None:
        cls.decorator_list = []
        cls.bases = [ast.Name(id="_FixtureBase", ctx=ast.Load())] if "_FixtureBase" in namespace else []
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
        for node in cls.body:
            node.decorator_list = [d for d in node.decorator_list if isinstance(d, ast.Name) and d.id == "property"]
    module = ast.Module(body=[cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


ParallelStrategy = real_class(ADAPTER, "ParallelStrategy", {"dataclass": dataclass, "math": math})


def base_strategy(config: SimpleNamespace, n_servers: int) -> object:
    pc = config.parallel_config
    return ParallelStrategy(config.model_config.mla_only, pc.world_size, pc.rank,
                            pc.tensor_parallel_size, pc.pipeline_parallel_size, n_servers,
                            pc.decode_context_parallel_size)


def load_policy(base_connector: type | None = None) -> ModuleType:
    if not POLICY.exists():
        # The upstream strategy is the red-stage reference behavior.
        return SimpleNamespace(StageLocalParallelStrategy=ParallelStrategy)
    module = ModuleType("lmcache.integration.vllm.lmcache_mp_connector")
    module.LMCacheMPConnector = base_connector or type("LMCacheMPConnector", (), {})
    module.build_parallel_strategy_from_vllm_config = base_strategy
    adapter = ModuleType("lmcache.integration.vllm.vllm_multi_process_adapter")
    adapter.ParallelStrategy = ParallelStrategy
    vllm = ModuleType("vllm.config")
    vllm.VllmConfig = SimpleNamespace
    modules = {module.__name__: module, adapter.__name__: adapter, "vllm.config": vllm}
    spec = importlib.util.spec_from_file_location("ks_pp_policy_under_test", POLICY)
    loaded = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(loaded)
    return loaded


def config(rank: int = 0, mla: bool = True) -> SimpleNamespace:
    extra = {"lmcache.mp.server_urls": ["tcp://192.0.2.10:5565", "tcp://192.0.2.20:5565"]}
    return SimpleNamespace(
        parallel_config=SimpleNamespace(world_size=32, rank=rank, tensor_parallel_size=16,
                                        pipeline_parallel_size=2, data_parallel_size=1,
                                        decode_context_parallel_size=1, prefill_context_parallel_size=1),
        device_config=SimpleNamespace(device_type="npu"),
        model_config=SimpleNamespace(mla_only=mla),
        kv_transfer_config=SimpleNamespace(get_from_extra_config=lambda name, default: extra.get(name, default)),
    )


class GeometryTests(unittest.TestCase):
    def test_local_cache_handshake_preserves_mooncake_metadata(self) -> None:
        cls = load_policy().AscendPPStageLocalConnector
        connector = cls.__new__(cls)
        metadata = {(0, 0): "front-peer", (1, 0): "rear-peer"}
        self.assertIsNone(connector.set_xfer_handshake_metadata_pp_aware(metadata))
        self.assertEqual(metadata, {(0, 0): "front-peer", (1, 0): "rear-peer"})

    def strategy(self, rank: int, mla: bool = True) -> object:
        cls = load_policy().StageLocalParallelStrategy
        return cls(mla, 32, rank, 16, 2, 2, 1)

    def test_each_server_holds_one_mla_piece_and_has_16_readers(self) -> None:
        for rank in range(32):
            strategy = self.strategy(rank)
            self.assertEqual(strategy.kv_world_size, 1)
            self.assertEqual(strategy.kv_worker_id, 0)
            self.assertEqual(strategy.num_kv_readers, 16)

    def test_one_mla_writer_in_each_stage(self) -> None:
        writers = [r for r in range(32) if self.strategy(r).is_kv_writer]
        self.assertEqual(writers, [0, 16])

    def test_sharded_kv_keeps_all_16_local_pieces(self) -> None:
        for stage in range(2):
            workers = [self.strategy(stage * 16 + tp, mla=False) for tp in range(16)]
            self.assertEqual([s.kv_worker_id for s in workers], list(range(16)))
            self.assertTrue(all(s.kv_world_size == 16 and s.num_kv_readers == 1 for s in workers))

    def test_actual_constructor_routes_all_32_workers_and_both_scheduler_clients(self) -> None:
        class Base:
            def __init__(self, cfg: object, role: object, cache: object) -> None:
                self._vllm_config, self._role, self._kv_cache_config = cfg, role, cache

        roles = SimpleNamespace(SCHEDULER="scheduler", WORKER="worker")
        namespace = {
            "_FixtureBase": Base, "Any": object, "ParallelStrategy": ParallelStrategy,
            "KVConnectorRole": roles, "math": math, "sys": sys,
            "KVConnectorTransferResults": None,
            "LMCacheMPConnectorStats": lambda: object(),
            "validate_mamba_step_alignment": lambda *_: None,
            "validate_kv_cache_groups": lambda *_: None,
            "validate_dcp_support": lambda *_: None,
            "get_group_tokens_per_block": lambda *_: [128],
            "get_vllm_scheduler_block_size": lambda *_: 128,
            "get_dcp_decorated_model_name": lambda *_: "glm-5",
            "is_mp_server_autostart_enabled": lambda *_: False,
            "_ensure_transport_scheme": lambda url: url,
            "build_parallel_strategy_from_vllm_config": base_strategy,
            "zmq": SimpleNamespace(Context=SimpleNamespace(instance=lambda: object())),
            "logger": SimpleNamespace(info=lambda *_: None),
            "print_banner_once": lambda *_: None,
            "LMCacheMPSchedulerAdapter": lambda **kw: SimpleNamespace(**kw, lmcache_tokens_per_chunk=512),
            "LMCacheMPWorkerAdapter": lambda **kw: SimpleNamespace(**kw),
        }
        core_connector = real_class(
            CORE / "lmcache/integration/vllm/lmcache_mp_connector.py", "LMCacheMPConnector", namespace,
            {"__init__", "build_parallel_strategy", "role", "transfer_intermediate_tensors"},
        )
        cls = load_policy(core_connector).AscendPPStageLocalConnector
        with patch.dict(os.environ, {"LMCACHE_ASCEND_PP_STAGE_LOCAL": "1"}):
            for rank in range(32):
                cfg = config(rank)
                cfg.cache_config = SimpleNamespace(mamba_cache_mode="none")
                cfg.kv_transfer_config.is_kv_producer = True
                cfg.kv_transfer_config.kv_connector_extra_config = {}
                obj = cls(cfg, roles.WORKER)
                expected = "tcp://192.0.2.10:5565" if rank < 16 else "tcp://192.0.2.20:5565"
                self.assertEqual(obj.worker_adapter.server_url, expected)
                self.assertEqual(obj.worker_adapter.parallel_strategy.num_kv_readers, 16)
            obj = cls(cfg, roles.SCHEDULER)
            self.assertEqual(len(obj.scheduler_adapter.server_urls), 2)
            self.assertEqual(obj.scheduler_adapter.parallel_strategy.kv_world_size, 1)

    def test_customer_policy_rejects_other_devices_and_geometries(self) -> None:
        policy = load_policy()
        self.assertTrue(hasattr(policy, "AscendPPStageLocalConnector"), "No customer connector exists")
        connector = policy.AscendPPStageLocalConnector.__new__(policy.AscendPPStageLocalConnector)
        with patch.dict(os.environ, {"LMCACHE_ASCEND_PP_STAGE_LOCAL": "1"}):
            self.assertEqual(connector.build_parallel_strategy(config(31), 2).num_kv_readers, 16)
            bad = config()
            bad.device_config.device_type = "cuda"
            with self.assertRaisesRegex(ValueError, "NPU"):
                connector.build_parallel_strategy(bad, 2)
            for field, value in (("tensor_parallel_size", 8), ("pipeline_parallel_size", 1),
                                 ("decode_context_parallel_size", 2), ("data_parallel_size", 2),
                                 ("world_size", 16)):
                bad = config()
                setattr(bad.parallel_config, field, value)
                with self.assertRaises(ValueError, msg=field):
                    connector.build_parallel_strategy(bad, 2)

    def test_policy_requires_explicit_activation_and_two_reachable_endpoints(self) -> None:
        policy = load_policy()
        self.assertTrue(hasattr(policy, "AscendPPStageLocalConnector"), "No customer connector exists")
        connector = policy.AscendPPStageLocalConnector.__new__(policy.AscendPPStageLocalConnector)
        with patch.dict(os.environ, {"LMCACHE_ASCEND_PP_STAGE_LOCAL": "0"}):
            with self.assertRaisesRegex(ValueError, "LMCACHE_ASCEND_PP_STAGE_LOCAL"):
                connector.build_parallel_strategy(config(), 2)
        with patch.dict(os.environ, {"LMCACHE_ASCEND_PP_STAGE_LOCAL": "1"}):
            bad = config()
            bad.kv_transfer_config.get_from_extra_config = lambda *_: ["tcp://127.0.0.1:5565"] * 2
            with self.assertRaises(ValueError):
                connector.build_parallel_strategy(bad, 2)


class PinTests(unittest.TestCase):
    def test_context_factory_is_lazy_idempotent_and_npu_only(self) -> None:
        path = PLUGIN / "lmcache_ascend/v1/pp_stage_local_activation.py"
        self.assertTrue(path.exists())
        npu = ModuleType("lmcache.v1.platform.devices.npu")
        npu.NpuDeviceSpec = type("NpuDeviceSpec", (), {})
        cuda_factory = lambda *_: "cuda-original"
        cuda = type("CudaSpec", (), {"create_cache_context": cuda_factory})
        pin = ModuleType("lmcache_ascend.v1.pp_stage_pinning")
        pin.PinnedNpuCacheContext = lambda *args, **kwargs: (args, kwargs)
        spec = importlib.util.spec_from_file_location("ks_activation_under_test", path)
        loaded = importlib.util.module_from_spec(spec)
        # The context module is absent during activation: imports must be lazy.
        with patch.dict(sys.modules, {npu.__name__: npu}):
            spec.loader.exec_module(loaded)
            loaded.install_npu_stage_pinning()
            factory = npu.NpuDeviceSpec.create_cache_context
            loaded.install_npu_stage_pinning()
            self.assertIs(npu.NpuDeviceSpec.create_cache_context, factory)
            with patch.dict(sys.modules, {pin.__name__: pin}):
                self.assertEqual(npu.NpuDeviceSpec().create_cache_context("front", layers=41),
                                 (("front",), {"layers": 41}))
        self.assertIs(cuda.create_cache_context, cuda_factory)

    def test_npu_staging_preserves_groups_and_uses_npu_pin_allocator(self) -> None:
        self.assertTrue(PIN.exists(), "No NPU-only staging implementation exists")
        calls = []

        class Tensor:
            def __init__(self, values: object) -> None:
                self.values = list(values)

            def pin_memory(self, device: str) -> "Tensor":
                calls.append(device)
                return self

        class Buffer:
            shape = (8,)

            def __getitem__(self, key: slice) -> object:
                return SimpleNamespace(copy_=lambda source, non_blocking: calls.append((source.values, non_blocking)))

        torch = SimpleNamespace(Tensor=Tensor, long="int64", frombuffer=lambda values, dtype: Tensor(values))
        cls = real_class(PIN, "PinnedNpuCacheContext", {"array": array, "torch": torch}, {"stage_block_ids"})
        obj = cls()
        obj.block_ids_buffer_ = Buffer()
        self.assertEqual(len(obj.stage_block_ids([[12, 35], [], [47]])), 3)
        self.assertEqual(calls, ["npu", ([12, 35, 47], True)])
        with self.assertRaises(ValueError):
            obj.stage_block_ids([list(range(9))])


class LookupTests(unittest.TestCase):
    def make_scheduler(self, first: object, second: object) -> object:
        class Key(SimpleNamespace):
            def no_worker_id_version(self) -> "Key":
                return self

        class Client:
            def __init__(self, value: object) -> None:
                self.value = value
                self.polls = 0
                self.released = []
                self.lookups = []
                self.ended = []

            def lookup(self, key: Key, tp_size: int) -> object:
                self.lookups.append((key, tp_size))
                return SimpleNamespace(query=lambda: True, result=lambda **_: None)

            def query_prefetch_status(self, request_id: str) -> object:
                self.polls += 1
                return SimpleNamespace(query=lambda: True, result=lambda **_: self.value)

            def free_lookup_locks(self, key: Key, tp_size: int) -> None:
                self.released.append((key, tp_size))

            def end_session(self, request_id: str) -> None:
                self.ended.append(request_id)

        names = {"maybe_submit_lookup_request", "check_lookup_result", "free_lookup_locks",
                 "end_session", "_free_inconsistent_lookup_locks", "_create_key",
                 "is_healthy", "world_size", "tp_size"}
        cls = real_class(ADAPTER, "LMCacheMPSchedulerAdapter",
                         {"time": time, "IPCCacheServerKey": Key, "Any": object, "_LookupAck": lambda **kw: SimpleNamespace(**kw),
                          "logger": SimpleNamespace(warning=lambda *_: None)}, names)
        obj = cls.__new__(cls)
        obj.parallel_strategy = GeometryTests().strategy(0)
        obj.model_name = "glm-5"
        obj.lmcache_tokens_per_chunk = 512
        obj.req_clients = {"front": Client(first), "rear": Client(second)}
        obj._server_urls = ["front", "rear"]
        obj._pending_lookups = {"r"}
        obj._unacked_lookups = {}
        obj._finished_lookup_results = {}
        obj._per_server_hits = {}
        obj._lookup_status = {}
        obj._lookup_params = {"r": (list(range(8192)), "", None)}
        obj._nonblocking_lookup_status = True
        obj._mq_timeout = 5
        obj._health_events = {u: SimpleNamespace(is_set=lambda: True) for u in obj._server_urls}
        obj._ensure_heartbeat_started = lambda: None
        return obj

    def test_lookup_and_cancel_send_matching_local_keys_to_both_servers(self) -> None:
        obj = self.make_scheduler(8, 8)
        obj._pending_lookups.clear()
        tokens = list(range(4097))
        obj.maybe_submit_lookup_request("r", tokens, reserve_last_token=True)
        self.assertEqual(len(obj._unacked_lookups["r"].futures), 2)
        for client in obj.req_clients.values():
            key, readers = client.lookups[0]
            self.assertEqual((key.end, key.world_size, key.num_kv_readers, readers), (4096, 1, 16, 16))
        self.assertEqual(obj.check_lookup_result("r"), 4096)
        obj.free_lookup_locks(tokens, 0, 4096, "r")
        obj.end_session("r")
        for client in obj.req_clients.values():
            key, readers = client.released[0]
            self.assertEqual((key.world_size, key.num_kv_readers, readers), (1, 16, 16))
            self.assertEqual(client.ended, ["r"])

    def test_incomplete_rear_status_does_not_publish_front_only_hit(self) -> None:
        obj = self.make_scheduler(8, None)
        self.assertIsNone(obj.check_lookup_result("r"))
        obj.req_clients["rear"].value = 8
        self.assertEqual(obj.check_lookup_result("r"), 4096)
        self.assertEqual(obj.req_clients["front"].polls, 1)

    def test_unhealthy_stage_prevents_external_hit(self) -> None:
        obj = self.make_scheduler(8, 8)
        obj._health_events["rear"] = SimpleNamespace(is_set=lambda: False)
        self.assertEqual(obj.check_lookup_result("r"), 0)

    def test_both_stages_agree_and_result_is_stable(self) -> None:
        obj = self.make_scheduler(8, 8)
        self.assertEqual(obj.check_lookup_result("r"), 4096)
        self.assertEqual(obj.check_lookup_result("r"), 4096)
        self.assertEqual([c.polls for c in obj.req_clients.values()], [1, 1])

    def test_partial_hit_releases_only_excess_tail_with_16_readers(self) -> None:
        obj = self.make_scheduler(16, 8)
        self.assertEqual(obj.check_lookup_result("r"), 4096)
        key, readers = obj.req_clients["front"].released[0]
        self.assertEqual((key.start, key.end, key.world_size, key.num_kv_readers, readers), (4096, 8192, 1, 16, 16))
        self.assertEqual(obj.req_clients["rear"].released, [])

    def test_miss_in_one_stage_cannot_skip_prefill(self) -> None:
        self.assertEqual(self.make_scheduler(8, 0).check_lookup_result("r"), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
