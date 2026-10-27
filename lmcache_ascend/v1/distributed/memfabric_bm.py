# SPDX-License-Identifier: Apache-2.0
"""One fixed-membership, SDMA-only BM session, owned by its calling thread."""

import importlib.metadata
import threading
import time

from packaging.version import Version


class BMSession:
    _guard = threading.Lock()
    _active = False

    def __init__(self, *, device_id, rank_id, world_size, pool_gb, store_url,
                 init_timeout=90):
        if type(init_timeout) is not int or not 1 <= init_timeout <= 3600:
            raise ValueError("BM init_timeout must be an integer in [1, 3600] seconds")
        if not (0 <= rank_id < world_size and device_id >= 0 and pool_gb >= 1):
            raise ValueError("Invalid BM device, rank, world size, or pool capacity")
        if not isinstance(pool_gb, int):
            raise ValueError("pool_gb must be an integer (BM uses 1 GiB alignment)")
        if Version(importlib.metadata.version("memfabric-hybrid")) < Version("1.2.1"):
            raise RuntimeError("Use MemFabric >=1.2.1; the tested 1.2.0 wheel double-frees on exit")
        with self._guard:
            if BMSession._active:
                raise RuntimeError("Only one MemFabric BM session per process is supported")
            BMSession._active = True
        self.thread_id = threading.get_ident()
        self.rank_id = rank_id
        self.departed_rank = None
        self.peer_departed = threading.Event()
        self.handle = None
        self.joined = False
        self.initialized = False
        self.mf_initialized = False
        self.closed = False
        try:
            import torch
            import torch_npu  # noqa: F401
            import memfabric_hybrid as mf
            from memfabric_hybrid import bm

            self.mf, self.bm = mf, bm
            torch.npu.set_device(device_id)
            self.check(mf.initialize(), "mf.initialize")
            self.mf_initialized = True
            config = bm.BmConfig()
            config.auto_ranking = False
            config.rank_id = rank_id
            config.start_store = rank_id == 0
            config.init_timeout = init_timeout
            config.create_timeout = init_timeout
            config.operation_timeout = 90
            self.check(bm.initialize(store_url, world_size, device_id, config),
                       "bm.initialize")
            self.initialized = True
            self.handle = bm.create2(id=0, local_dram_size=pool_gb << 30,
                                     max_dram_size=pool_gb << 30,
                                     data_op_type=bm.BmDataOpType.SDMA)
            if self.handle is None:
                raise RuntimeError("BM returned an empty handle")
            self.handle.set_group_event_handler(self._on_group_event)
            self.check(self.handle.join(), "BM join")
            self.joined = True
            # join() completes local membership; peer_rank_ptr() may already
            # return a nonzero GVA before the peer's mapping has arrived.
            # JOIN callbacks do not replay peers present before this rank joined.
            # SDMA requires an actual device VA mapping for every peer segment.
            deadline = time.monotonic() + config.init_timeout
            while not self.peer_departed.is_set():
                if all(self.handle.gva_to_va(
                        self.peer_address(peer), bm.BmMemType.LOCAL_DEVICE)
                       for peer in range(world_size) if peer != rank_id):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("BM peer DRAM mappings did not become ready")
                self.peer_departed.wait(0.1)
            if self.peer_departed.is_set():
                raise RuntimeError("BM peer left while the fixed group was joining")
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(result, operation):
        if result != 0:
            raise RuntimeError(f"{operation} failed: {result}")

    def _check_thread(self):
        if threading.get_ident() != self.thread_id:
            raise RuntimeError("BM operations must run on the session owner thread")

    def _on_group_event(self, rank, event):
        # Native callback thread: only publish state, never issue BM/NPU work.
        # Fixed membership cannot recover an old index after a peer restarts.
        if rank != self.rank_id and event == self.bm.BmGroupEvent.LEAVE_EVENT:
            self.departed_rank = rank
            self.peer_departed.set()

    def peer_address(self, rank):
        self._check_thread()
        address = self.handle.peer_rank_ptr(rank, self.bm.BmMemType.HOST)
        if not address:
            raise RuntimeError(f"No DRAM mapping for BM rank {rank}")
        return address

    def copy(self, src, dst, size, *, store):
        self._copy(src, dst, size, self.bm.BmCopyType.H2G if store
                   else self.bm.BmCopyType.G2H)

    def copy_npu_to_global(self, src, dst, size):
        """Synchronously copy an actual local NPU pointer into a BM GVA."""
        self._copy(src, dst, size, self.bm.BmCopyType.L2G)

    def copy_global_to_npu(self, src, dst, size):
        """Synchronously copy a BM GVA into an actual local NPU pointer."""
        self._copy(src, dst, size, self.bm.BmCopyType.G2L)

    def _copy(self, src, dst, size, kind):
        self._check_thread()
        if self.closed:
            raise RuntimeError("BM session is closed")
        if self.peer_departed.is_set():
            raise RuntimeError(f"BM peer rank {self.departed_rank} left the fixed group")
        # flags=0 selects synchronous copies; no asynchronous buffer recycling.
        self.check(self.handle.copy_data(src, dst, size, kind, 0), "BM copy")

    def wait(self):
        self._check_thread()
        if self.handle is not None:
            self.check(self.handle.wait(), "BM wait")

    def close(self):
        self._check_thread()
        if self.closed:
            return
        if self.handle is not None:
            self.wait()
            if self.joined:
                self.check(self.handle.leave(), "BM leave")
            self.handle.destroy()
            self.handle = None
        if self.initialized:
            self.bm.uninitialize()
        if self.mf_initialized:
            self.mf.uninitialize()
        self.closed = True
        with self._guard:
            BMSession._active = False
