# SPDX-License-Identifier: Apache-2.0
"""vLLM connector entry point for native MemFabric L1 worker transfer.

Set ``kv_connector`` to ``MemFabricL1MPConnector`` and
``kv_connector_module_path`` to
``lmcache_ascend.integration.vllm.memfabric_l1_connector``.
The core connector continues to own scheduler behavior; this subclass changes
only the worker adapter so BM joins before its first control RPC.
"""

from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector

from lmcache_ascend.integration.vllm.memfabric_l1_worker import (
    MemFabricL1MPWorkerAdapter,
)


class MemFabricL1MPConnector(LMCacheMPConnector):
    """LMCache MP connector with MemFabric BM slots on worker ranks."""

    def create_lmcache_mp_worker_adapter(self, **adapter_kwargs):
        return MemFabricL1MPWorkerAdapter(**adapter_kwargs)


__all__ = ["MemFabricL1MPConnector"]
