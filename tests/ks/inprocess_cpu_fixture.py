# SPDX-License-Identifier: Apache-2.0
"""Tiny CPU storage and copy backend for the real in-process request methods."""
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
import logging
import queue
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import Mock

import torch

import test_inprocess_lifecycle_cpu as src

LOG = logging.getLogger(__name__)
LOCATION = "LocalCPUBackend"
ABORT = "FINISHED_ABORTED"
STATUS = NS(FINISHED_ABORTED=ABORT)
CORE_ENGINE = src.CORE / "lmcache/v1/cache_engine.py"
CORE_ADAPTER = src.CORE / "lmcache/integration/vllm/vllm_v1_adapter.py"


class ChunkDB:
    def process_tokens(self, tokens, hashes=None, offsets=None, mask=None, request_configs=None):
        for end in range(2, len(tokens) + 1, 2):
            if mask is None or mask[end-1]:
                yield end-2, end, tuple(tokens[:end])


class Object:
    def __init__(self, shape):
        self.tensor = torch.empty(shape, dtype=torch.int32, device="cpu")
        self.refs = 1

    def get_size(self):
        return self.tensor.numel() * self.tensor.element_size()

    def ref_count_down(self):
        if self.refs <= 0:
            raise RuntimeError("CPU object released twice")
        self.refs -= 1


class Memory:
    def __init__(self):
        self.data = {}
        self.pins = Counter()
        self.objects = []

    def allocate(self, shapes, dtypes, **kwargs):
        obj = Object(shapes[0])
        self.objects.append(obj)
        return obj

    def batched_put(self, keys, objects, **kwargs):
        for key, obj in zip(keys, objects, strict=True):
            if key in self.data:
                self.data[key].ref_count_down()
            self.data[key] = obj  # transfer the allocation reference to storage

    def batched_contains(self, keys, locations, pin):
        found = []
        for key in keys:
            if key not in self.data:
                break
            found.append(key)
        if pin:
            self.pins.update(found)
        return len(found), {LOCATION: found} if found else {}

    def batched_unpin(self, keys, locations):
        for key in keys:
            if self.pins[key] <= 0:
                raise RuntimeError("CPU object unpinned twice")
            self.pins[key] -= 1

    def touch_cache(self):
        pass

    def get_block_mapping(self, chunks):
        return {LOCATION: [item for item in chunks if item[0] in self.data]}

    def batched_get(self, keys, location):
        objects = [self.data.get(k) for k in keys]
        for obj in objects:
            if obj is not None:
                obj.refs += 1
        return objects


class Copies:
    def __init__(self, gate, entered):
        self.gate = gate
        self.entered = entered
        self.fail = False

    def batched_from_gpu(self, objects, starts, ends, *, source, **kwargs):
        self.entered.set()
        if not self.gate.wait(30):
            raise TimeoutError("CPU test copy gate expired")
        if self.fail:
            raise ValueError("injected copy failure")
        for obj, start, end in zip(objects, starts, ends, strict=True):
            obj.tensor.copy_(source[start:end])

    def batched_to_gpu(self, objects, starts, ends, *, destination, **kwargs):
        for obj, start, end in zip(objects, starts, ends, strict=True):
            destination[start:end].copy_(obj.tensor)


class CPUEngineBase(src.BaseEngine):
    lookup = src.method(CORE_ENGINE, "LMCacheEngine", "lookup", CacheEngineKey=tuple)
    lookup_unpin = src.method(CORE_ENGINE, "LMCacheEngine", "lookup_unpin",
                              EventType=NS(LOADING=1), EventStatus=NS(NOT_FOUND=0))
    _process_tokens_internal = src.method(CORE_ENGINE, "LMCacheEngine", "_process_tokens_internal",
                                          CacheEngineKey=tuple, logger=LOG)

    def close(self):
        for obj in self.storage_manager.data.values():
            obj.ref_count_down()
        self.storage_manager.data.clear()
        super().close()


Engine = src.classes(src.ROOT / "lmcache_ascend/v1/cache_engine.py", dict(
    LMCacheEngine=CPUEngineBase, torch=torch, queue=queue, threading=threading,
    time=time, logger=LOG, CacheEngineKey=tuple)).AscendLMCacheEngine


def engine_for(rank=0, asynchronous=True, gate=None, entered=None):
    if gate is None:
        gate = threading.Event()
        gate.set()
    if entered is None:
        entered = threading.Event()
    config = NS(store_async=asynchronous, store_async_max_queue_size=4,
                get_extra_config_value=lambda key, default: default)
    metadata = NS(worker_id=rank, get_shapes=lambda n: [(n, rank+1)],
                  get_dtypes=lambda: [torch.int32])
    engine = Engine(config, metadata, None, None, None, None)
    engine.token_database, engine.storage_manager = ChunkDB(), Memory()
    engine.gpu_connector = Copies(gate, entered)
    engine.lookup_pins = {}
    engine.event_manager = NS(get_event_status=lambda *_: 0)
    engine.use_layerwise = engine.save_only_first_rank = False
    engine.async_loading = engine.remove_after_retrieve = False
    engine.fmt = None
    engine.store_location, engine.retrieve_locations = LOCATION, [LOCATION]
    engine._log_kvcache_for_check = lambda **_: None
    engine.is_frozen = lambda: False
    stats = NS(profile_process_tokens=nullcontext, profile_put=nullcontext,
               profile_to_gpu=nullcontext, time_to_store=lambda: 1., time_to_retrieve=lambda: 1.,
               process_tokens_time=0., put_time=0.)
    engine.stats_monitor = Mock()
    engine.stats_monitor.on_store_request.return_value = stats
    engine.stats_monitor.on_retrieve_request.return_value = stats
    return engine


class AdapterBase:
    request_finished = src.method(CORE_ADAPTER, "LMCacheConnectorV1Impl", "request_finished",
                                  RequestStatus=STATUS, logger=LOG)
    start_load_kv = src.method(CORE_ADAPTER, "LMCacheConnectorV1Impl", "start_load_kv",
                               torch=torch, LMCacheConnectorMetadata=src.Meta, logger=LOG)

    def __init__(self, cfg, role, parent):
        self._parent = parent
        self.lmcache_engine = cfg.engine
        self.config = NS(store_async=True, get_extra_config_value=lambda key, default: default)
        self.async_loading = self.use_layerwise = False
        self.kv_role = "kv_both"
        self.kv_caches = {"cpu": torch.empty(0)}
        factory = self.service_factory_cls()
        factory.role, factory.vllm_config = role.name.lower(), cfg
        self.lookup_client = factory.maybe_create_lookup_client()
        self._manager = NS(is_healthy=lambda: True)

    def build_connector_meta(self, output):
        return src.Meta(output.requests)


AscendAdapter = src.classes(src.INTEGRATION / "vllm_v1_adapter.py", dict(
    LMCacheConnectorV1Impl=AdapterBase, torch=torch, logger=LOG, RequestStatus=STATUS,
    LMCacheConnectorMetadata=src.Meta, _lmcache_nvtx_annotate=lambda f: f,
)).LMCacheAscendConnectorV1Impl

api = src.classes(src.INTEGRATION / "lmcache_ascend_connector_v1.py", dict(
    dataclass=dataclass, LMCacheConnectorMetadata=src.Meta, KVConnectorWorkerMetadata=object,
    LMCacheConnectorV1Dynamic=src.wrapper.LMCacheConnectorV1Dynamic,
    VllmServiceFactory=src.BaseFactory, LMCacheAscendConnectorV1Impl=AscendAdapter,
    SupportsHMA=type("SupportsHMA", (), {}), AscendLMCacheEngine=Engine,
    StepLookupClient=src.lookup.StepLookupClient, LookupRequest=src.lookup.LookupRequest))


def connector(engine=None):
    role = NS(name="SCHEDULER" if engine is None else "WORKER")
    return api.LMCacheAscendConnectorV1Dynamic(
        NS(engine=engine, parallel_config=NS(world_size=2)), role,
        NS(kv_cache_groups=[NS(kv_cache_spec=NS(block_size=2))]))


def tensor(rank, value, n):
    return torch.arange(n*(rank+1), dtype=torch.int32).reshape(n, rank+1) + value


def snapshot(engine):
    memory = engine.storage_manager
    with engine._store_lock:
        pending = dict(engine._pending_store_reqs)
    return dict(keys=len(memory.data), pins=sum(memory.pins.values()),
                refs=[obj.refs for obj in memory.objects], pending=pending,
                thread_alive=bool(engine._store_worker_thread and engine._store_worker_thread.is_alive()))
