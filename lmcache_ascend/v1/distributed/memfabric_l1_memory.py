# SPDX-License-Identifier: Apache-2.0
"""Native LMCache L1 allocation over a BM-owned HOST arena.

This component supplies memory only. Native L1Manager/StorageManager own keys,
staging, read/write protection and eviction. It does not use PoolIndex.
The caller owns the BM session and must close it AFTER the storage manager.
"""

import ctypes
import threading

import torch

from lmcache.v1.distributed.api import L1BackendType
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator


class MemFabricL1MemoryManager:
    """Use the local BM HOST pool directly as TensorMemoryObj backing data."""

    def __init__(self, session, *, size_bytes=None, align_bytes=4096):
        session._check_thread()
        if session.pool_memory_type != "host":
            raise ValueError("MemFabric L1 currently requires a HOST pool")
        self.session = session
        self.size_bytes = size_bytes or session.local_dram_bytes
        if not 0 < self.size_bytes <= session.local_dram_bytes:
            raise ValueError("L1 arena must fit the local BM HOST contribution")
        self.align_bytes = align_bytes
        self.global_base = session.peer_address(session.rank_id)
        self.local_base = session.handle.gva_to_va(
            self.global_base, session.bm.BmMemType.LOCAL_HOST
        )
        if not self.local_base:
            raise RuntimeError("BM did not expose a local HOST mapping")
        self._raw = (ctypes.c_ubyte * self.size_bytes).from_address(self.local_base)
        self.buffer = torch.frombuffer(self._raw, dtype=torch.uint8)
        self._allocator = TensorMemoryAllocator(self.buffer, align_bytes=align_bytes)
        self._lock = threading.Lock()
        self._closed = False

    def allocate(self, layout_desc, count):
        with self._lock:
            if self._closed:
                raise RuntimeError("L1 arena is closed")
            objects = self._allocator.batched_allocate(
                layout_desc.shapes, layout_desc.dtypes, count
            )
            return (L1Error.OUT_OF_MEMORY, []) if objects is None else (L1Error.SUCCESS, objects)

    def free(self, objects):
        with self._lock:
            self._allocator.batched_free(objects)
        return L1Error.SUCCESS

    def get_memory_usage(self):
        with self._lock:
            return self._allocator.total_allocated_size, self.size_bytes

    def get_backend_type(self, memory_obj):
        return L1BackendType.DRAM

    def get_l1_memory_desc(self):
        return L1MemoryDesc(self.local_base, self.size_bytes, self.align_bytes)

    def global_address(self, memory_obj):
        """Translate an actual native object's range into the same BM arena."""
        if self._closed or not memory_obj.is_valid():
            raise ValueError("L1 object is not live")
        offset = memory_obj.tensor.data_ptr() - self.local_base
        size = memory_obj.get_size()
        if offset < 0 or size <= 0 or offset + size > self.size_bytes:
            raise ValueError("Object does not belong to this BM L1 arena")
        return self.global_base + offset

    def memcheck(self):
        with self._lock:
            return self._allocator.memcheck()

    def close(self):
        with self._lock:
            if self._allocator.total_allocated_size:
                raise RuntimeError("Native L1 objects must be freed before arena close")
            self._closed = True
        # Tensor/ctypes views do not own the BM allocation. No CPU free/unpin.
