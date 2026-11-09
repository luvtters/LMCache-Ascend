# SPDX-License-Identifier: Apache-2.0
"""Customer TP16/PP2: one complete pipeline stage per local MP server.

Each server owns independent L1 storage and sees only its stage's KV pieces.
The existing scheduler queries both servers and uses their common hit prefix.
"""

from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit
import ipaddress
import os

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (
    LMCacheMPConnector,
    build_parallel_strategy_from_vllm_config,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import ParallelStrategy

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class StageLocalParallelStrategy(ParallelStrategy):
    """KV identities and reader counts within one complete TP16 stage.

    Used only after AscendPPStageLocalConnector validates PP2, two servers,
    DP1 and no context-parallel token sharding. Server routing continues to
    use the original global worker rank, independently of these local IDs.
    """

    @property
    def kv_world_size(self) -> int:
        """Return one replicated MLA piece, or all local TP shards."""
        return 1 if self.mla_only else self.tp_size

    @property
    def kv_worker_id(self) -> int:
        """Return the piece ID in this server's independent L1 namespace."""
        return 0 if self.mla_only else self.vllm_worker_id % self.tp_size

    @property
    def kv_tp_size(self) -> int:
        """Return the complete TP group served by this stage's server."""
        return self.tp_size

    @property
    def is_kv_writer(self) -> bool:
        """Elect one MLA writer per stage, retaining all sharded writers."""
        return not self.mla_only or self.vllm_worker_id % self.tp_size == 0


class AscendPPStageLocalConnector(LMCacheMPConnector):
    """Opt-in local MP caches for the customer's two-host Prefill engine."""

    def set_xfer_handshake_metadata_pp_aware(
        self, metadata: dict[tuple[int, int], Any]
    ) -> None:
        """Leave the peer PD handshake to the sibling Mooncake connector.

        This connector registers its local KV through MP IPC. MultiConnector
        still forwards the unchanged metadata to Mooncake for P-to-D transfer.
        """
        return None

    def build_parallel_strategy(
        self, vllm_config: VllmConfig, n_servers: int
    ) -> ParallelStrategy:
        """Validate the customer layout and use server-local KV geometry.

        Args:
            vllm_config: Native vLLM configuration; no framework patch needed.
            n_servers: Endpoints ordered by PP stage (front, rear).

        Returns:
            Strategy with matching local registration, lookup and read locks.

        Raises:
            ValueError: Activation, device, topology or endpoints do not match.
        """
        if os.environ.get("LMCACHE_ASCEND_PP_STAGE_LOCAL") != "1":
            raise ValueError(
                "Set LMCACHE_ASCEND_PP_STAGE_LOCAL=1 in P and cache terminals"
            )
        if vllm_config.device_config.device_type != "npu":
            raise ValueError("AscendPPStageLocalConnector requires an NPU device")
        pc = vllm_config.parallel_config
        if (
            n_servers != 2
            or pc.pipeline_parallel_size != 2
            or pc.tensor_parallel_size != 16
            or pc.world_size != 32
            or pc.data_parallel_size != 1
            or getattr(pc, "decode_context_parallel_size", 1) != 1
            or getattr(pc, "prefill_context_parallel_size", 1) != 1
        ):
            raise ValueError(
                "Requires TP16/PP2, DP1, two stage-local servers and DCP/PCP=1"
            )
        self._validate_stage_endpoints(vllm_config)
        original = build_parallel_strategy_from_vllm_config(vllm_config, n_servers)
        return StageLocalParallelStrategy(**vars(original))

    @staticmethod
    def _validate_stage_endpoints(vllm_config: VllmConfig) -> None:
        """Require distinct non-loopback TCP addresses in stage order.

        The native adapters perform the subsequent connection checks.
        """
        value = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache.mp.server_urls", None
        )
        if not isinstance(value, (list, str)):
            raise ValueError("Provide both stage addresses in lmcache.mp.server_urls")
        urls = value if isinstance(value, list) else value.split(",")
        if len(urls) != 2 or not all(isinstance(url, str) for url in urls):
            raise ValueError("Provide exactly two stage-local server URLs")
        hosts: list[str] = []
        for url in urls:
            parsed = urlsplit(url.strip() if "://" in url else "tcp://" + url.strip())
            host = parsed.hostname
            if parsed.scheme != "tcp" or not host or not parsed.port:
                raise ValueError("Stage servers need tcp://host:port addresses")
            try:
                loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                loopback = host.lower() == "localhost"
            if loopback:
                raise ValueError("Use reachable host addresses, not per-node localhost")
            hosts.append(host.lower())
        if hosts[0] == hosts[1]:
            raise ValueError("The two PP stages require different cache hosts")
