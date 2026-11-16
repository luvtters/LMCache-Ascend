# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass

# Third Party
from lmcache.integration.vllm.lmcache_connector_v1 import LMCacheConnectorV1Dynamic
from lmcache.integration.vllm.vllm_service_factory import VllmServiceFactory
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorMetadata
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorWorkerMetadata,
    SupportsHMA,
)

# First Party
from lmcache_ascend.integration.vllm.inprocess_lookup import LookupRequest, StepLookupClient
from lmcache_ascend.integration.vllm.vllm_v1_adapter import LMCacheAscendConnectorV1Impl
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine
from lmcache_ascend.v1.npu_connector.inprocess_blocks import RegisteredBlockTransfer


@dataclass(kw_only=True)
class AscendStepMetadata(LMCacheConnectorMetadata):
    queries: list[LookupRequest]
    preempted: set[str]
    release_pins: set[str]


@dataclass
class WorkerLookups(KVConnectorWorkerMetadata):
    replies: dict[int, dict[int, int]]

    def aggregate(self, other):
        for seq, ranks in other.replies.items():
            collected = self.replies.setdefault(seq, {})
            for rank, count in ranks.items():
                if collected.setdefault(rank, count) != count:
                    raise ValueError("Inconsistent lookup result for the same worker")
        return self


class InProcessServiceFactory(VllmServiceFactory):
    def get_or_create_metadata(self):
        metadata = super().get_or_create_metadata()
        tp = self.vllm_config.parallel_config.tensor_parallel_size
        metadata.storage_owner_rank = metadata.worker_id // tp * tp
        return metadata

    def maybe_create_lookup_client(self):
        if self.role == "scheduler":
            return StepLookupClient(self.vllm_config.parallel_config.world_size)
        return None

    def maybe_create_lookup_server(self):
        return None

    def maybe_create_offload_server(self):
        return None

    def maybe_create_internal_api_server(self, lmcache_manager):
        # Its lookup management endpoints can recreate standalone ZMQ services.
        return None


class InProcessAdapter(LMCacheAscendConnectorV1Impl):
    service_factory_cls = InProcessServiceFactory

    def _apply_extra_config(self, config, vllm_config):
        super()._apply_extra_config(config, vllm_config)
        if (
            config.enable_scheduler_bypass_lookup or config.enable_async_loading
            or config.external_lookup_client or config.enable_controller
            or config.enable_p2p or config.enable_pd
        ):
            raise ValueError("In-process lookup requires vLLM step metadata")
        if (
            not config.local_cpu or config.remote_url or config.local_disk
            or config.storage_plugins or config.remote_storage_plugins
        ):
            # Packed byte chunks have a process-local registration and layout.
            raise ValueError("Registered block transfer requires the worker-local CPU cache")
        if config.use_layerwise or config.enable_blending:
            raise ValueError("Registered block transfer requires whole-chunk KV")
        if getattr(config, "enable_chunk_hashes_return", False):
            raise ValueError("Step lookup does not export chunk hashes")

    def get_finished(self, finished_req_ids):
        if not self.store_async or self.kv_role == "kv_consumer":
            return None, None
        return super().get_finished(finished_req_ids)

    def record_failed_blocks(self, request_id, expected_mask, ret_mask, slot_mapping):
        if len(self._parent.block_sizes) > 1:
            # 0.26's scheduler unpacks exactly one block table on a load error.
            raise RuntimeError(f"Incomplete pinned multi-group restore for {request_id}")
        return super().record_failed_blocks(request_id, expected_mask, ret_mask, slot_mapping)


class LMCacheAscendConnectorV1Dynamic(LMCacheConnectorV1Dynamic, SupportsHMA):
    """Ascend in-process connector using vLLM step metadata for pinned lookups."""

    adapter_cls = InProcessAdapter

    def __init__(self, vllm_config, role, kv_cache_config):
        parallel = vllm_config.parallel_config
        if (getattr(parallel, "decode_context_parallel_size", 1) > 1
                or getattr(parallel, "prefill_context_parallel_size", 1) > 1):
            raise ValueError("Registered block transfer requires unsharded token positions")
        self.block_sizes = []
        for group in kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            leaves = getattr(spec, "kv_cache_specs", None) or {"": spec}
            for leaf in leaves.values():
                if (getattr(leaf, "compress_ratio", 1) not in (None, 1)
                        or getattr(leaf, "sliding_window", None)
                        or getattr(leaf, "tokens_per_state", 1) not in (None, 1)
                        or leaf.block_size != spec.block_size):
                    raise ValueError("Expected dense KV with a common block size in each group")
            self.block_sizes.append(spec.block_size)
        if not self.block_sizes:
            raise ValueError("Missing KV allocation groups")
        self._group_config = kv_cache_config
        self._tables = {}
        super().__init__(vllm_config, role, kv_cache_config)
        self._lmcache_engine._block_size = self.block_sizes[0]
        self._lmcache_engine._discard_partial_chunks = True
        self._release_pins = set()
        self._worker_replies = {}
        if role.name == "WORKER":
            ready = isinstance(self._lmcache_engine.lmcache_engine, AscendLMCacheEngine)
        else:
            ready = isinstance(self._lmcache_engine.lookup_client, StepLookupClient)
        if not ready or not self._lmcache_engine._manager.is_healthy():
            self.shutdown()
            raise RuntimeError("In-process connector initialization failed for " + role.name)

    def register_kv_caches(self, kv_caches):
        impl = self._lmcache_engine
        if impl.kv_caches:
            raise RuntimeError("KV caches are already registered")
        engine = impl.lmcache_engine
        chunk = impl._lmcache_chunk_size
        transfer = engine.gpu_connector
        if not isinstance(transfer, RegisteredBlockTransfer):
            raise RuntimeError("Unexpected GPU connector for in-process Ascend")
        transfer.register(self._group_config, kv_caches, self.block_sizes, chunk)
        engine.metadata.kv_shape = (1, 1, chunk, 1, transfer.row_bytes)
        engine.metadata.kv_dtype = torch.uint8
        super().register_kv_caches(kv_caches)
        if not impl._manager.is_healthy():
            self.shutdown()
            raise RuntimeError("Ascend in-process Engine post_init failed")

    def build_connector_meta(self, scheduler_output):
        base = super().build_connector_meta(scheduler_output)
        client = self._lmcache_engine.lookup_client
        preempted = set(scheduler_output.preempted_req_ids or ())
        for req_id in scheduler_output.finished_req_ids | preempted:
            client.clear_lookup_status(req_id)
            self._tables.pop(req_id, None)
        for req in scheduler_output.scheduled_new_reqs:
            self._tables[req.req_id] = tuple(tuple(ids) for ids in req.block_ids)
        cached = scheduler_output.scheduled_cached_reqs
        for req_id, tables in zip(cached.req_ids, cached.new_block_ids, strict=True):
            if tables is None:
                continue
            new = tuple(tuple(ids) for ids in tables)
            if req_id in cached.resumed_req_ids:
                self._tables[req_id] = new
            else:
                self._tables[req_id] = tuple(
                    old + more for old, more in zip(self._tables[req_id], new, strict=True)
                )
        for req in base.requests:
            req.block_tables = self._tables[req.req_id]
            if len(req.block_tables) != len(self.block_sizes):
                raise ValueError(f"Incomplete allocation groups for {req.req_id}")
        release = self._release_pins | scheduler_output.finished_req_ids
        self._release_pins.clear()
        return AscendStepMetadata(
            requests=base.requests,
            queries=client.take_requests(),
            preempted=preempted,
            release_pins=release,
        )

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if not num_external_tokens:
            self._release_pins.add(request.request_id)
        super().update_state_after_alloc(request, blocks, num_external_tokens)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        matched, async_load = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if matched and any(num_computed_tokens % bs for bs in self.block_sizes):
            # A destination block shared with APC cannot be partially overwritten.
            self._lmcache_engine.load_specs[request.request_id].lmcache_cached_tokens = 0
            return 0, False
        return matched, async_load

    def set_xfer_handshake_metadata_pp_aware(self, metadata):
        # MultiConnector fans out its P/D sibling's handshake to every child.
        # Local Engine lookup needs no transfer endpoints.
        pass

    def handle_preemptions(self, kv_connector_metadata):
        if kv_connector_metadata.preempted:
            self._lmcache_engine.handle_preemptions(kv_connector_metadata.preempted)

    def start_load_kv(self, forward_context, **kwargs):
        meta = self._get_connector_metadata()
        engine = self._lmcache_engine.lmcache_engine
        for req_id in meta.release_pins:
            engine.lookup_unpin(req_id)
        for query in meta.queries:
            if engine.save_only_first_rank and not engine.metadata.is_first_rank():
                hit = len(query.tokens)  # The TP owner supplies this rank's KV.
            else:
                hit = engine.lookup(
                    tokens=query.tokens,
                    lookup_id=query.request_id,
                    request_configs=query.configs,
                    pin=True,
                )
            self._worker_replies[query.sequence] = {engine.metadata.worker_id: hit}
        try:
            super().start_load_kv(forward_context, **kwargs)
        except Exception:
            for request in meta.requests:
                engine.lookup_unpin(request.req_id)
            raise

    def build_connector_worker_meta(self):
        replies, self._worker_replies = self._worker_replies, {}
        return WorkerLookups(replies) if replies else None

    def update_connector_output(self, connector_output):
        meta = connector_output.kv_connector_worker_meta
        if meta is not None:
            self._lmcache_engine.lookup_client.receive(meta.replies)

    def request_finished(self, request, block_ids):
        self._lmcache_engine.lookup_client.clear_lookup_status(request.request_id)
        self._release_pins.add(request.request_id)
        return super().request_finished(request, block_ids)

    def request_finished_all_groups(self, request, block_ids):
        # The inherited finish callback owns a request-wide STORE fence. Physical
        # IDs are already captured per group in each submitted STORE snapshot.
        return self.request_finished(request, block_ids[0])
