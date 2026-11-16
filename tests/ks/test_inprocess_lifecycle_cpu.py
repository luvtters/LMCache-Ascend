# SPDX-License-Identifier: Apache-2.0
"""Run production lifecycle methods with isolated LMCache/vLLM base fixtures."""

import ast
import __future__
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
from itertools import count
import os
from pathlib import Path
import queue
import sys
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch


ROOT = Path(__file__).resolve().parents[2]
CORE = Path(os.environ.get("KS_LMCACHE_SOURCE", ROOT.parent / "lmcache-inprocess-ascend"))
INTEGRATION = ROOT / "lmcache_ascend/integration/vllm"


def classes(path, dependencies):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    # Use actual method bodies. Replace heavyweight imports and annotations only.
    exec(compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag),
         dependencies)
    return NS(**dependencies)


def method(path, cls, name, **dependencies):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = tree if cls is None else next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls
    )
    node = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    tree.body = [node]
    exec(compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag), dependencies)
    return dependencies[name]


class BaseEngine:
    def __init__(self, config, metadata, *_):
        self.config, self.metadata = config, metadata
        self.kv_events_enabled = False
        self.storage_manager = Mock()
        self.gpu_connector = Mock()
        self.closed = 0

    def close(self):
        self.closed += 1

    def is_healthy(self):
        return True

    def _is_passive(self):
        return False

    def _get_req_id(self, kwargs):
        return kwargs["req_id"]


engine_cls = classes(ROOT / "lmcache_ascend/v1/cache_engine.py", dict(
    LMCacheEngine=BaseEngine, torch=torch, queue=queue, threading=threading,
    time=time, logger=logging.getLogger(__name__),
)).AscendLMCacheEngine


def new_engine(store_async):
    cfg = NS(store_async=store_async, store_async_max_queue_size=2,
             get_extra_config_value=lambda _, default: default)
    return engine_cls(cfg, NS(worker_id=16), None, None, None, None)


class EngineTests(unittest.TestCase):
    def test_broadcast_failure_releases_owner_read_reference(self):
        engine = new_engine(False)
        engine.metadata.first_rank = 0
        engine.metadata.is_first_rank = lambda: True
        engine.broadcast_stream = Mock()
        engine._broadcast_metadata_table = Mock(return_value={"total": 1})
        engine._pipeline_broadcast_and_load = Mock(side_effect=RuntimeError("allocation failed"))
        chunk = Mock()
        with self.assertRaisesRegex(RuntimeError, "allocation failed"):
            engine._pipelined_sharded_broadcast_and_load([("key", chunk, 0, 4)], None)
        engine.broadcast_stream.synchronize.assert_called_once()
        chunk.ref_count_down.assert_called_once()

    def test_sync_close_and_repeat(self):
        engine = new_engine(False)
        engine.close()
        engine.close()
        self.assertEqual(engine.closed, 1)

    def test_finish_and_shutdown_wait_for_last_store(self):
        engine = new_engine(True)
        start, release = threading.Event(), threading.Event()
        def blocked_copy(*_):
            start.set()
            if not release.wait(3):
                raise TimeoutError("test did not release copy")
        engine._run_store_pipeline = blocked_copy
        engine.store([10], req_id="r")
        self.assertTrue(start.wait(2))
        self.assertEqual(engine.get_finished_stores({"r"}), set())
        waiting = threading.Event()
        wait_for = engine._store_cv.wait_for
        def observed_wait(predicate):
            waiting.set()
            return wait_for(predicate)
        engine._store_cv.wait_for = observed_wait
        with ThreadPoolExecutor(2) as pool:
            drain = pool.submit(engine.wait_for_pending_stores, {"r"})
            self.assertTrue(waiting.wait(2))
            close = pool.submit(engine.close)
            self.assertFalse(drain.done())
            self.assertFalse(close.done())
            self.assertEqual(engine.closed, 0)
            release.set()
            self.assertEqual(drain.result(timeout=3), {"r"})
            close.result(timeout=3)
        self.assertEqual(engine.closed, 1)
        self.assertFalse(engine._store_worker_thread.is_alive())
        self.assertEqual(engine.get_finished_stores(set()), {"r"})
        self.assertEqual(engine.get_finished_stores({"r"}), set())
        with self.assertRaisesRegex(RuntimeError, "closing"):
            engine.store([11], req_id="late")


@dataclass
class Meta:
    requests: list = field(default_factory=list)


class BaseConnector:
    def __init__(self, vllm_config, role, kv_cache_config):
        self.config = vllm_config

    def _get_connector_metadata(self):
        return self.step


class BaseAdapter:
    def __init__(self, cfg, role, parent):
        self._manager = NS(is_healthy=lambda: True)
        factory = self.service_factory_cls()
        factory.role = role.name.lower()
        factory.vllm_config = cfg
        self.lookup_client = factory.maybe_create_lookup_client()
        self.lmcache_engine = new_engine(False) if role.name == "WORKER" else None
        self.handle_preemptions = Mock()

    def start_load_kv(self, *_):
        pass

    def build_connector_meta(self, output):
        return Meta(output.requests)

    def request_finished(self, request, blocks):
        return True, None

    def update_state_after_alloc(self, request, tokens):
        self.lookup_client.clear_lookup_status(request.request_id)


wrapper = classes(CORE / "lmcache/integration/vllm/lmcache_connector_v1.py",
                  dict(KVConnectorBase_V1=BaseConnector, LMCacheConnectorV1Impl=BaseAdapter))
lookup = classes(INTEGRATION / "inprocess_lookup.py", dict(dataclass=dataclass, count=count))


class BaseFactory:
    def get_or_create_metadata(self):
        return self.metadata


connector = classes(INTEGRATION / "lmcache_ascend_connector_v1.py", dict(
    dataclass=dataclass, LMCacheConnectorMetadata=Meta, KVConnectorWorkerMetadata=object,
    LMCacheConnectorV1Dynamic=wrapper.LMCacheConnectorV1Dynamic, VllmServiceFactory=BaseFactory,
    LMCacheAscendConnectorV1Impl=BaseAdapter, SupportsHMA=type("SupportsHMA", (), {}),
    AscendLMCacheEngine=engine_cls, StepLookupClient=lookup.StepLookupClient,
    LookupRequest=lookup.LookupRequest,
))


def create(role):
    groups = [NS(kv_cache_spec=NS(block_size=b)) for b in [2, 4]]
    return connector.LMCacheAscendConnectorV1Dynamic(
        NS(parallel_config=NS(world_size=2)), NS(name=role), NS(kv_cache_groups=groups))


def step(*, new=(), reqs=(), finished=(), preempted=(), cached=None):
    return NS(scheduled_new_reqs=new, requests=reqs, finished_req_ids=set(finished),
              preempted_req_ids=set(preempted), scheduled_cached_reqs=cached or
              NS(req_ids=[], new_block_ids=[], resumed_req_ids=set()))


class ConnectorTests(unittest.TestCase):
    def test_mtp_finish_before_save_is_reported_once(self):
        path = INTEGRATION / "vllm_v1_adapter.py"
        cls = "LMCacheAscendConnectorV1Impl"
        engine = new_engine(True)
        req = NS(req_id="r", save_spec=NS(can_save=True, skip_leading_tokens=0),
                 token_ids=[1, 2, 3, 4])
        impl = NS(kv_role="kv_both", lmcache_engine=engine, _wait_for_save_done=False,
                  _parent=NS(_get_connector_metadata=lambda: Meta([req])),
                  _finished_req_ids_waiting_for_save=set(), _late_finished_sending=set())
        may_save = method(path, cls, "_may_register_store_after_wait_for_save")
        impl._may_register_store_after_wait_for_save = lambda req: may_save(impl, req)
        finished = method(path, cls, "get_finished", LMCacheConnectorMetadata=Meta)
        self.assertEqual(finished(impl, {"r"}), (None, None))
        self.assertEqual(impl._finished_req_ids_waiting_for_save, {"r"})
        engine._pending_store_reqs["r"] = 1
        impl._wait_for_save_done = True
        method(path, cls, "_replay_finished_stores_after_save")(impl)
        self.assertEqual(finished(impl, set()), (None, None))
        engine._pending_store_reqs.clear()
        self.assertEqual(finished(impl, set()), ({"r"}, None))
        self.assertEqual(finished(impl, set()), (None, None))
        engine.close()

    def test_register_updates_storage_geometry_before_post_init(self):
        class Transfer:
            def register(self, config, caches, sizes, chunk):
                self.row_bytes = 48

        path = ROOT / "lmcache_ascend/v1/npu_connector/__init__.py"
        make = method(path, None, "CreateNPUConnector", EngineType=NS(VLLM="vllm"))
        worker = create("WORKER")
        impl = worker._lmcache_engine
        engine = impl.lmcache_engine
        module = "lmcache_ascend.v1.npu_connector.inprocess_blocks"
        with patch.dict(sys.modules, {module: NS(RegisteredBlockTransfer=Transfer)}):
            engine.gpu_connector = make(None, NS(storage_owner_rank=16), "vllm")
        impl.kv_caches, impl._lmcache_chunk_size = {}, 8
        def post_init(caches):
            self.assertEqual(engine.metadata.kv_shape, (1, 1, 8, 1, 48))
            self.assertEqual(engine.metadata.kv_dtype, torch.uint8)
            impl.kv_caches = caches
        impl.register_kv_caches = post_init
        register = worker.register_kv_caches
        with patch.dict(register.__func__.__globals__, {"RegisteredBlockTransfer": Transfer, "torch": torch}):
            register({"L": torch.empty(1)})
            with self.assertRaisesRegex(RuntimeError, "already registered"):
                register({"L": torch.empty(1)})

    def test_production_store_receives_table_and_compute_event(self):
        req = NS(req_id="r", token_ids=[1, 2, 3, 4], slot_mapping=torch.arange(4),
                 block_tables=((2, 3), (1,)), save_spec=NS(can_save=True, skip_leading_tokens=0),
                 disagg_spec=None, request_configs=None, is_last_prefill=True)
        engine = Mock()
        engine._is_passive.return_value = False
        impl = NS(_parent=NS(_get_connector_metadata=lambda: Meta([req])), kv_role="kv_both",
                  lmcache_engine=engine, use_layerwise=False, kv_caches={"L": torch.zeros(4)},
                  _local_persist_skip=lambda *_: None, _lmcache_chunk_size=4,
                  _replay_finished_stores_after_save=Mock())
        save = method(INTEGRATION / "vllm_v1_adapter.py", "LMCacheAscendConnectorV1Impl",
                      "wait_for_save", LMCacheConnectorMetadata=Meta, torch=torch, logger=Mock(),
                      get_pp_group=lambda: NS(is_last_rank=True))
        event = Mock()
        with patch.object(torch, "npu", NS(Event=lambda: event), create=True):
            save(impl)
        event.record.assert_called_once()
        engine.store.assert_called_once()
        self.assertEqual(engine.store.call_args.kwargs["block_tables"], req.block_tables)
        self.assertIs(engine.store.call_args.kwargs["ordering_event"], event)
        self.assertEqual(req.save_spec.skip_leading_tokens, 4)

    def test_tp_owner_in_second_stage_uses_group_local_broadcast_source(self):
        source = CORE / "lmcache/v1/metadata.py"
        is_owner = method(source, "LMCacheMetadata", "is_first_rank")
        factory = connector.InProcessServiceFactory()
        factory.vllm_config = NS(parallel_config=NS(tensor_parallel_size=16))
        for rank in (0, 1, 16, 17):
            factory.metadata = NS(worker_id=rank, first_rank=0)
            meta = factory.get_or_create_metadata()
            self.assertEqual(is_owner(meta), rank in (0, 16))
            self.assertEqual(meta.first_rank, 0)
        self.assertFalse(is_owner(NS(worker_id=16, first_rank=0, storage_owner_rank=None)))

    def test_apc_prefix_does_not_hide_a_failed_external_block(self):
        source = CORE / "lmcache/integration/vllm/vllm_v1_adapter.py"
        load = method(source, "LMCacheConnectorV1Impl", "start_load_kv", torch=torch,
                      LMCacheConnectorMetadata=Meta, logger=Mock())
        request = NS(req_id="r", load_spec=NS(can_load=True, vllm_cached_tokens=2,
                     lmcache_cached_tokens=4), token_ids=[1, 2, 3, 4],
                     slot_mapping=torch.arange(4), request_configs=None, block_tables=((1, 2),))
        engine = Mock()
        engine.retrieve.return_value = torch.tensor([True, True, True, False])
        adapter = NS(kv_caches={"L": torch.zeros(4)}, lmcache_engine=engine,
                     _parent=NS(_get_connector_metadata=lambda: Meta([request])),
                     _stats_monitor=Mock(), device="cpu", _lmcache_chunk_size=4,
                     use_layerwise=False, async_loading=False, _invalid_block_ids=set(),
                     record_failed_blocks=Mock(return_value={2}))
        load(adapter, NS(attn_metadata=True))
        self.assertEqual(adapter._invalid_block_ids, {2})
        self.assertEqual(engine.retrieve.call_args.kwargs["block_tables"], ((1, 2),))
        engine.lookup_unpin.assert_called_once_with("r")

    def test_builder_retains_an_engine_that_failed_to_drain(self):
        builder = NS(_instances={}, _stat_loggers={"w": Mock()})
        engine = Mock()
        engine.metadata.storage_owner_rank = 16
        engine.close.side_effect = TimeoutError("copy in flight")
        builder._instances["w"] = engine
        destroy = method(CORE / "lmcache/v1/cache_engine.py", "LMCacheEngineBuilder",
                         "destroy", logger=Mock())
        with self.assertRaises(TimeoutError):
            destroy(builder, "w")
        self.assertIs(builder._instances["w"], engine)

    def test_query_only_step_and_pin_release(self):
        scheduler = create("SCHEDULER")
        client = scheduler._lmcache_engine.lookup_client
        self.assertIsNone(client.lookup([7, 8, 9], "r"))
        metadata = scheduler.build_connector_meta(step())
        replies = []
        workers = [create("WORKER"), create("WORKER")]
        for rank, worker in enumerate(workers):
            worker.step = metadata
            engine = worker._lmcache_engine.lmcache_engine
            engine.save_only_first_rank = False
            engine.metadata.worker_id = rank
            engine.lookup, engine.lookup_unpin = Mock(return_value=3 - rank), Mock()
            worker.start_load_kv(None)
            self.assertTrue(engine.lookup.call_args.kwargs["pin"])
            replies.append(worker.build_connector_worker_meta())
            self.assertIsNone(worker.build_connector_worker_meta())
        scheduler.update_connector_output(NS(kv_connector_worker_meta=replies[0].aggregate(replies[1])))
        self.assertEqual(client.lookup_cache("r"), 2)
        request = NS(request_id="r")
        self.assertEqual(scheduler.request_finished_all_groups(request, ([], [])), (True, None))
        for worker in workers:
            worker.step = scheduler.build_connector_meta(step(finished=["r"], preempted=["p"]))
            worker.handle_preemptions(worker.step)
            worker._lmcache_engine.handle_preemptions.assert_called_once_with({"p"})
            worker.start_load_kv(None)
            worker._lmcache_engine.lmcache_engine.lookup_unpin.assert_called_once_with("r")

    def test_alloc_delta_and_resume_replace_all_groups(self):
        scheduler = create("SCHEDULER")
        first = scheduler.build_connector_meta(step(new=[NS(req_id="r", block_ids=([1], [3]))],
                                                    reqs=[NS(req_id="r")]))
        def cached(ids, resumed):
            return step(reqs=[NS(req_id="r")], cached=NS(
                req_ids=["r"], new_block_ids=[ids], resumed_req_ids=resumed))
        second = scheduler.build_connector_meta(cached(([2], [4]), set()))
        scheduler.build_connector_meta(step(preempted=["r"]))
        fresh = scheduler.build_connector_meta(cached(([5], [6]), {"r"}))
        self.assertEqual(first.requests[0].block_tables, ((1,), (3,)))
        self.assertEqual(second.requests[0].block_tables, ((1, 2), (3, 4)))
        self.assertEqual(fresh.requests[0].block_tables, ((5,), (6,)))
        scheduler.build_connector_meta(step(finished=["r"]))
        self.assertFalse(scheduler._tables)

    def test_services_and_consumer(self):
        factory = connector.InProcessServiceFactory()
        self.assertIsNone(factory.maybe_create_lookup_server())
        self.assertIsNone(factory.maybe_create_offload_server())
        self.assertIsNone(factory.maybe_create_internal_api_server(None))
        impl = create("WORKER")._lmcache_engine
        impl.store_async, impl.kv_role = True, "kv_consumer"
        self.assertEqual(impl.get_finished({"r"}), (None, None))
        peer_metadata = {(1, 0): "mooncake-peer"}
        self.assertIsNone(create("SCHEDULER").set_xfer_handshake_metadata_pp_aware(peer_metadata))
        self.assertEqual(peer_metadata, {(1, 0): "mooncake-peer"})


if __name__ == "__main__":
    unittest.main()
