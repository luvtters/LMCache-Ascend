# SPDX-License-Identifier: Apache-2.0
"""CPU wiring checks with stand-ins for vLLM and native-dependent MP adapters.

These checks cover connector routing and lifetime ordering, not real vLLM,
RPC, BM, or NPU execution. The False-result check executes the actual core
``get_finished`` method in isolation from its native imports.
"""

# Standard
import ast
import importlib.util
import logging
import os
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from typing import Any
import sys
import threading
import unittest
from unittest.mock import Mock, patch


def load_connector() -> ModuleType:
    """Import only the connector, replacing its external integration edges."""
    connector_name = "lmcache.integration.vllm.lmcache_mp_connector"
    adapter_name = "lmcache.integration.vllm.vllm_multi_process_adapter"
    direct_name = "lmcache_ascend.v1.distributed.memfabric_direct"
    connector = ModuleType(connector_name)
    adapter = ModuleType(adapter_name)
    direct = ModuleType(direct_name)

    class BaseConnector:
        def __init__(
            self, config: Any, role: Any, kv_cache_config: Any = None
        ) -> None:
            self.base_initialized = True

        def create_lmcache_mp_scheduler_adapter(self, **kwargs: Any) -> Any:
            raise AssertionError("stock scheduler factory must not be selected")

    class BaseScheduler:
        def __init__(
            self, urls: list[str], context: Any, model: str,
            block_size: int, strategy: Any, **kwargs: Any,
        ) -> None:
            self.req_clients = {url: Mock() for url in urls}
            self.model_name = model
            self.parallel_strategy = strategy
            self.lmcache_tokens_per_chunk = 128
            self.tp_size = 1
            self._mq_timeout = kwargs["mq_timeout"]
            self._nonblocking_lookup_status = True
            self._heartbeat_lock = threading.Lock()
            self._heartbeats = None
            self.lookups = {}
            self.lifecycle = []

        def _create_key(
            self, tokens: bytes, start: int, end: int, request_id: str,
            salt: str, configs: Any,
        ) -> SimpleNamespace:
            key = SimpleNamespace(
                token_bytes=tokens, start=start, end=end, request_id=request_id,
                cache_salt=salt, request_configs=configs,
            )
            key.no_worker_id_version = lambda: key
            return key

        def cleanup_lookup_result(self, request_id: str) -> None:
            lookup = self.lookups.pop(request_id, None)
            if lookup is not None:
                lookup.close()

        def _create_lookup_state(self, key: Any) -> Any:
            raise AssertionError("stock lookup must not be selected")

        def shutdown(self) -> None:
            self.lifecycle.append("clients-close")
            for client in self.req_clients.values():
                client.close()

    class BaseWorker:
        server_features = set()

        def __init__(
            self, url: str, context: Any, model: str,
            block_size: int, strategy: Any, **kwargs: Any,
        ) -> None:
            self.req_client = Mock()
            self.model_name = model
            self.parallel_strategy = strategy
            self.instance_id = 101
            self.lmcache_tokens_per_chunk = 128
            self.blocks_in_chunk = 8
            self._mq_timeout = kwargs["mq_timeout"]
            self.experimental = self.server_features
            self.transfer_ctx = None

        def register_kv_caches(
            self, caches: dict[str, Any], infos: Any = (), hints: Any = None
        ) -> None:
            self.kv_caches = caches
            self.transfer_ctx = self._create_transfer_context(caches)
            self.transfer_ctx.register(caches, engine_group_infos=infos)

        def shutdown(self) -> None:
            self.req_client.close()

    connector.LMCacheMPConnector = BaseConnector
    adapter.LMCacheMPSchedulerAdapter = BaseScheduler
    adapter.LMCacheMPWorkerAdapter = BaseWorker
    adapter.DEFAULT_MQ_TIMEOUT = 5.0
    adapter.DEFAULT_HEARTBEAT_INTERVAL = 10.0
    adapter.ParallelStrategy = SimpleNamespace
    direct.create_scheduler_backend = Mock(return_value=Mock())
    direct.create_worker_transfer_context = Mock(return_value=Mock())
    path = (
        Path(__file__).parents[3]
        / "lmcache_ascend/integration/vllm/memfabric_direct_connector.py"
    )
    spec = importlib.util.spec_from_file_location("memfabric_direct_connector", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        connector_name: connector, adapter_name: adapter, direct_name: direct,
    }):
        spec.loader.exec_module(module)
    return module


class ConnectorTests(unittest.TestCase):
    """Exercise explicit direct routing and rejection before device access."""

    def setUp(self) -> None:
        self.environment = patch.dict(os.environ, {"LMCACHE_MP_TRANSFER_MODE": "auto"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.module = load_connector()
        self.strategy = SimpleNamespace(
            tp_size=1, pp_size=1, dcp_size=1, vllm_world_size=1,
            vllm_worker_id=0, n_servers=1, mla_only=False,
        )
        self.config = {"lmcache.mp.direct": {
            "device_id": 0, "rank_id": 1, "world_size": 3,
            "pool_gb": 2, "store_url": "tcp://127.0.0.1:28680",
        }}

    def worker(self, **kwargs: Any) -> Any:
        """Build the worker through its public constructor with CPU stand-ins."""
        options = dict(extra_config=self.config)
        options.update(kwargs)
        return self.module.MemFabricDirectWorkerAdapter(
            "tcp://localhost:5555", None, "qwen3", 16, self.strategy, **options
        )

    def scheduler(self) -> Any:
        """Build the scheduler through its public constructor."""
        return self.module.MemFabricDirectSchedulerAdapter(
            ["tcp://localhost:5555"], None, "qwen3", 16, self.strategy,
            extra_config=self.config,
        )

    def test_connector_selects_only_direct_adapters(self) -> None:
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_connector_extra_config=self.config),
            kv_events_config=None,
        )
        connector = self.module.MemFabricDirectConnector(config, "scheduler")
        scheduler = connector.create_lmcache_mp_scheduler_adapter(
            server_urls=["tcp://localhost:5555"], context=None, model_name="qwen3",
            vllm_block_size=16, parallel_strategy=self.strategy,
            extra_config=self.config,
        )
        worker = connector.create_lmcache_mp_worker_adapter(
            server_url="tcp://localhost:5555", context=None, model_name="qwen3",
            vllm_block_size=16, parallel_strategy=self.strategy,
            extra_config=self.config,
        )
        self.assertIsInstance(scheduler, self.module.MemFabricDirectSchedulerAdapter)
        self.assertIsInstance(worker, self.module.MemFabricDirectWorkerAdapter)

    def test_worker_forwards_direct_config_and_never_returns_none(self) -> None:
        worker = self.worker()
        caches = {"layer": object()}
        worker.register_kv_caches(caches)
        factory = self.module.create_worker_transfer_context
        self.assertEqual(factory.call_args.kwargs["extra_config"], self.config)
        self.assertIs(factory.call_args.kwargs["req_client"], worker.req_client)
        self.assertEqual(factory.call_args.kwargs["tokens_per_chunk"], 128)
        worker.transfer_ctx.register.assert_called_once()
        factory.return_value = None
        with self.assertRaisesRegex(TypeError, "direct context"):
            self.worker().register_kv_caches(caches)
        with self.assertRaisesRegex(RuntimeError, "already registered"):
            worker.register_kv_caches(caches)

    def test_partial_core_patch_is_rejected_before_client_creation(self) -> None:
        del self.module.LMCacheMPSchedulerAdapter._create_lookup_state
        with self.assertRaisesRegex(RuntimeError, "lookup-state hook"):
            self.scheduler()
        self.module.create_scheduler_backend.assert_not_called()

    def test_scheduler_releases_after_local_result_cleanup(self) -> None:
        scheduler = self.scheduler()
        key = SimpleNamespace(request_id="request")
        lookup = scheduler._create_lookup_state(key)
        backend = scheduler.direct_backend
        backend.create_lookup.assert_called_once_with(
            key, tp_size=1, tokens_per_chunk=128, timeout=5.0,
            blocking_lookup=False,
        )
        scheduler.lookups["request"] = lookup
        scheduler.cleanup_lookup_result("request")
        self.assertNotIn("request", scheduler.lookups)
        scheduler.end_session("request")
        backend.end_session.assert_called_once_with("request")
        scheduler.free_lookup_locks(
            b"tokens", 0, 128, "request", extra_ranges=((128, 256),)
        )
        self.assertEqual(
            backend.free_lookup_locks.call_args.args[1:], (1, ((128, 256),))
        )
        scheduler.reset_cache()
        backend.reset_cache.assert_called_once()

    def test_shutdown_retains_clients_until_successful_drain(self) -> None:
        scheduler = self.scheduler()
        scheduler.direct_backend.close.side_effect = (
            lambda: scheduler.lifecycle.append("pool-drain")
        )
        scheduler.shutdown()
        self.assertEqual(scheduler.lifecycle, ["pool-drain", "clients-close"])
        scheduler = self.scheduler()
        scheduler.direct_backend.close.side_effect = TimeoutError("lookup pending")
        with self.assertRaises(TimeoutError):
            scheduler.shutdown()
        next(iter(scheduler.req_clients.values())).close.assert_not_called()

    def test_fixed_group_recovery_never_rejoins_or_reregisters(self) -> None:
        worker = self.worker()
        scheduler = self.scheduler()
        self.assertFalse(worker._reregister_kv_caches_callback())
        self.assertFalse(scheduler._recover_server_chunk_size("url", Mock()))
        self.module.create_worker_transfer_context.assert_not_called()

    def test_rejects_unsupported_options_before_transfer(self) -> None:
        with self.assertRaisesRegex(ValueError, "KV events"):
            self.worker(enable_kv_events=True)
        for name in ("tp_size", "pp_size", "dcp_size", "n_servers", "vllm_world_size"):
            with self.subTest(topology=name):
                setattr(self.strategy, name, 2)
                with self.assertRaisesRegex(ValueError, "TP1"):
                    self.worker()
                setattr(self.strategy, name, 1)
        with self.assertRaisesRegex(ValueError, "mapping"):
            self.worker(extra_config={})
        with self.assertRaisesRegex(ValueError, "transfer mode"):
            self.worker(extra_config={
                **self.config, "lmcache.mp.mp_transfer_mode": "engine_driven",
            })
        with patch.dict(os.environ, {"LMCACHE_MP_TRANSFER_MODE": "lmcache_driven"}):
            with self.assertRaisesRegex(ValueError, "transfer mode"):
                self.worker()
        self.module.create_worker_transfer_context.assert_not_called()

    def test_rejects_non_full_attention_groups_and_server_experiments(self) -> None:
        worker = self.worker()
        full = dict(engine_group_id=0, sw_size_tokens=-1, recurrent_state=False,
                    extra_object_group_tag=0)
        for group in (
            {**full, "engine_group_id": 1},
            {**full, "sw_size_tokens": 128},
            {**full, "recurrent_state": True},
            {**full, "extra_object_group_tag": 1},
        ):
            with self.subTest(group=group):
                with self.assertRaisesRegex(ValueError, "full-attention"):
                    worker.register_kv_caches({}, [SimpleNamespace(**group)])
        with self.assertRaisesRegex(ValueError, "full-attention"):
            worker.register_kv_caches({}, [SimpleNamespace(**full)] * 2)
        self.module.LMCacheMPWorkerAdapter.server_features = {"blend"}
        with self.assertRaisesRegex(ValueError, "experimental"):
            self.worker()
        self.module.create_worker_transfer_context.assert_not_called()

    def test_core_false_retrieve_marks_blocks_failed(self) -> None:
        core_root = Path(os.environ.get(
            "LMCACHE_CORE_SOURCE", str(Path(__file__).parents[4] / "LMCache")
        ))
        path = core_root / "lmcache/integration/vllm/vllm_multi_process_adapter.py"
        if not path.is_file():
            self.skipTest("Set LMCACHE_CORE_SOURCE to the matching core checkout")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        worker = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name == "LMCacheMPWorkerAdapter")
        method = next(node for node in worker.body if isinstance(node, ast.FunctionDef)
                      and node.name == "get_finished")
        method.decorator_list = []
        extracted = ast.Module(body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0,
            ),
            method,
        ], type_ignores=[])
        namespace = {"logger": logging.getLogger(__name__)}
        exec(
            compile(ast.fix_missing_locations(extracted), str(path), "exec"), namespace
        )
        state = SimpleNamespace(
            dispatcher=None, is_healthy=True, store_futures={},
            retrieve_futures={"r": (
                SimpleNamespace(query=lambda: True, result=lambda timeout: False),
                [7, 8],
            )},
            store_events={}, retrieve_events={}, error_block_ids=set(),
            _failed_request_ids=set(), _dropped_retrieves=set(),
            _process_finished_stores=lambda finished, engine: set(),
        )
        get_finished = MethodType(namespace["get_finished"], state)
        self.assertEqual(get_finished(set()), (set(), {"r"}))
        self.assertEqual(state.error_block_ids, {7, 8})
        self.assertEqual(state._failed_request_ids, {"r"})
        self.assertEqual(get_finished(set()), (set(), set()))


if __name__ == "__main__":
    unittest.main()
