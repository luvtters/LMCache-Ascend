# SPDX-License-Identifier: Apache-2.0
"""Opt-in MP connector for synchronous homogeneous TP NPU-to-pool transfers.

Select ``MemFabricDirectConnector`` through vLLM's
``kv_connector_module_path``. This module requires the matching LMCache MP
scheduler hooks; it never falls back to the CPU L1 compatibility adapter.
"""

from __future__ import annotations

# Standard
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
import os

# Third Party
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_MQ_TIMEOUT,
    LMCacheMPSchedulerAdapter,
    LMCacheMPWorkerAdapter,
    ParallelStrategy,
)

# First Party
from lmcache_ascend.v1.distributed.memfabric_direct import (
    create_scheduler_backend,
    create_worker_transfer_context,
)

if TYPE_CHECKING:
    # Third Party
    import torch
    from lmcache.v1.gpu_connector.utils import LayoutHints
    from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo
    from lmcache.v1.multiprocess.transfer_context import TransferContext
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.kv_cache_interface import KVCacheConfig


def _direct_config(extra_config: dict[str, Any] | None) -> dict[str, Any]:
    """Validate connector options before opening a client or joining BM."""
    config = dict(extra_config or {})
    if not isinstance(config.get("lmcache.mp.direct"), dict):
        raise ValueError("MemFabric direct requires a lmcache.mp.direct mapping")
    mode = config.get(
        "lmcache.mp.mp_transfer_mode", os.getenv("LMCACHE_MP_TRANSFER_MODE", "auto")
    )
    if mode != "auto":
        raise ValueError(
            "MemFabricDirectConnector selects its own transfer context; remove "
            "the explicit MP transfer mode or set it to auto"
        )
    if config.get("lmcache.mp.lazy_offload") or config.get(
        "lmcache.mp.transfer_intermediate_tensors"
    ):
        raise ValueError(
            "MemFabric direct does not support lazy offload or experimental transfers"
        )
    return config


def _validate_strategy(strategy: ParallelStrategy) -> None:
    """Reject layouts outside homogeneous TP and one metadata service."""
    if (
        strategy.tp_size not in (1, 2, 4, 8)
        or strategy.pp_size != 1
        or strategy.dcp_size != 1
        or strategy.vllm_world_size != strategy.tp_size
        or strategy.n_servers != 1
        or strategy.mla_only
    ):
        raise ValueError(
            "MemFabric direct supports homogeneous TP1/2/4/8, PP1/DCP1 GQA and one service only"
        )


class MemFabricDirectSchedulerAdapter(LMCacheMPSchedulerAdapter):
    """Keep MP scheduling while routing lookup protection to the BM pool."""

    def __init__(
        self,
        server_urls: list[str],
        context: Any,
        model_name: str,
        vllm_block_size: int,
        parallel_strategy: ParallelStrategy,
        *,
        mq_timeout: float = DEFAULT_MQ_TIMEOUT,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        extra_config: dict[str, Any] | None = None,
        required_chunk_alignment: int | None = None,
    ) -> None:
        """Connect one scheduler to the pool's metadata service."""
        _validate_strategy(parallel_strategy)
        if not callable(
            getattr(LMCacheMPSchedulerAdapter, "_create_lookup_state", None)
        ):
            raise RuntimeError("Install the matching LMCache lookup-state hook patch")
        if len(server_urls) != 1:
            raise ValueError("MemFabric direct requires exactly one service")
        self.direct_config = _direct_config(extra_config)
        if self.direct_config["lmcache.mp.direct"].get("tp_size", 1) != parallel_strategy.tp_size:
            raise ValueError("Homogeneous TP requires explicit matching direct.tp_size")
        super().__init__(
            server_urls,
            context,
            model_name,
            vllm_block_size,
            parallel_strategy,
            mq_timeout=mq_timeout,
            heartbeat_interval=heartbeat_interval,
            extra_config=self.direct_config,
            required_chunk_alignment=required_chunk_alignment,
        )
        self.direct_backend = create_scheduler_backend(
            req_clients=self.req_clients,
            parallel_strategy=self.parallel_strategy,
            tokens_per_chunk=self.lmcache_tokens_per_chunk,
            mq_timeout=self._mq_timeout,
        )
        if self.direct_backend is None:
            super().shutdown()
            raise TypeError("create_scheduler_backend must return a direct backend")

    def free_lookup_locks(
        self,
        packed_token_ids: bytes,
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
        request_configs: dict[str, Any] | None = None,
        extra_ranges: tuple[tuple[int, int], ...] = (),
    ) -> None:
        """Release pool lookup pins for skipped, overlapping, or cancelled KV."""
        key = self._create_key(
            packed_token_ids,
            start,
            end,
            request_id,
            cache_salt,
            request_configs,
        ).no_worker_id_version()
        self.direct_backend.free_lookup_locks(key, self.tp_size, extra_ranges)

    def end_session(self, request_id: str) -> None:
        """Drain and release even after local lookup-result cleanup.

        The direct backend retains outstanding RPCs independently of the
        adapter's result map, so a late lookup cannot recreate an orphaned pin.
        """
        self.direct_backend.end_session(request_id)

    def reset_cache(self) -> bool:
        """Return False: invalidating fixed membership is not a cache flush."""
        return self.direct_backend.reset_cache()

    def shutdown(self) -> None:
        """Drain pool protections before closing the control clients.

        A failed drain raises and retains the clients for a later cleanup
        attempt; RPC timeout alone does not release a pool protection.
        """
        with self._heartbeat_lock:
            if self._heartbeats is not None:
                for heartbeat in self._heartbeats.values():
                    heartbeat.stop()
        self.direct_backend.close()
        super().shutdown()

    def _create_lookup_state(self, key: IPCCacheServerKey) -> Any:
        return self.direct_backend.create_lookup(
            key,
            tp_size=self.tp_size,
            tokens_per_chunk=self.lmcache_tokens_per_chunk,
            timeout=self._mq_timeout,
            blocking_lookup=not self._nonblocking_lookup_status,
        )

    def _recover_server_chunk_size(self, server_url: str, req_client: Any) -> bool:
        # Fixed BM members and epochs require a coordinated restart.
        return False


class MemFabricDirectWorkerAdapter(LMCacheMPWorkerAdapter):
    """Use the direct transfer context and retain stock completion/error APIs."""

    def __init__(
        self,
        server_url: str,
        context: Any,
        model_name: str,
        vllm_block_size: int,
        parallel_strategy: ParallelStrategy,
        *,
        mq_timeout: float = DEFAULT_MQ_TIMEOUT,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        extra_config: dict[str, Any] | None = None,
        enable_kv_events: bool = False,
        required_chunk_alignment: int | None = None,
    ) -> None:
        """Construct the explicit direct adapter without selecting a stock mode."""
        _validate_strategy(parallel_strategy)
        self.direct_config = _direct_config(extra_config)
        if self.direct_config["lmcache.mp.direct"].get("tp_size", 1) != parallel_strategy.tp_size:
            raise ValueError("Homogeneous TP requires explicit matching direct.tp_size")
        if enable_kv_events:
            raise ValueError("MemFabric direct does not support KV events")
        super().__init__(
            server_url,
            context,
            model_name,
            vllm_block_size,
            parallel_strategy,
            mq_timeout=mq_timeout,
            heartbeat_interval=heartbeat_interval,
            extra_config=self.direct_config,
            enable_kv_events=False,
            required_chunk_alignment=required_chunk_alignment,
        )
        if self.experimental:
            super().shutdown()
            raise ValueError(
                "MemFabric direct does not support experimental server features"
            )

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        layout_hints: LayoutHints | None = None,
    ) -> None:
        """Register a single full-attention group; reject SW/recurrent layouts."""
        if self.transfer_ctx is not None:
            raise RuntimeError(
                "Fixed BM group is already registered; restart to replace it"
            )
        if len(engine_group_infos) > 1 or any(
            info.engine_group_id != 0
            or info.sw_size_tokens != -1
            or info.recurrent_state
            or info.extra_object_group_tag != 0
            for info in engine_group_infos
        ):
            raise ValueError("MemFabric direct requires one full-attention KV group")
        super().register_kv_caches(kv_caches, engine_group_infos, layout_hints)

    def _create_transfer_context(self, kv_caches: dict) -> TransferContext:
        transfer = create_worker_transfer_context(
            parallel_strategy=self.parallel_strategy,
            instance_id=self.instance_id,
            req_client=self.req_client,
            tokens_per_chunk=self.lmcache_tokens_per_chunk,
            mq_timeout=self._mq_timeout,
            extra_config=self.direct_config,
        )
        if transfer is None:
            raise TypeError(
                "create_worker_transfer_context must return a direct context"
            )
        return transfer

    def _reregister_kv_caches_callback(self) -> bool:
        # The heartbeat thread must never recreate a fixed BM group or NPU KV.
        return False


class MemFabricDirectConnector(LMCacheMPConnector):
    """External vLLM connector selecting the MemFabric-specific MP adapters."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        """Validate opt-in options, then retain the normal MP connector flow."""
        transfer_config = vllm_config.kv_transfer_config
        if transfer_config is None:
            raise ValueError("MemFabric direct requires vLLM KV transfer configuration")
        _direct_config(transfer_config.kv_connector_extra_config)
        direct = transfer_config.kv_connector_extra_config["lmcache.mp.direct"]
        if direct.get("tp_size", 1) > 1:
            # Source the total from the loaded model config, never a guessed
            # TP1 chunk byte count or a replicated-head assumption.
            direct["total_kv_heads"] = vllm_config.model_config.get_total_num_kv_heads()
        event_config = vllm_config.kv_events_config
        if event_config is not None and event_config.enable_kv_cache_events:
            raise ValueError("MemFabric direct does not support KV events")
        if not callable(
            getattr(LMCacheMPConnector, "create_lmcache_mp_scheduler_adapter", None)
        ):
            raise RuntimeError("Install the matching LMCache scheduler-factory patch")
        super().__init__(vllm_config, role, kv_cache_config)

    def create_lmcache_mp_scheduler_adapter(
        self, **adapter_kwargs: Any
    ) -> MemFabricDirectSchedulerAdapter:
        """Build the direct scheduler instead of the CPU-L1 lookup path."""
        return MemFabricDirectSchedulerAdapter(**adapter_kwargs)

    def create_lmcache_mp_worker_adapter(
        self, **adapter_kwargs: Any
    ) -> MemFabricDirectWorkerAdapter:
        """Build the direct worker instead of a stock transfer context."""
        return MemFabricDirectWorkerAdapter(**adapter_kwargs)
