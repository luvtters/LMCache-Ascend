# SPDX-License-Identifier: Apache-2.0
"""One fixed-membership, SDMA-only BM session, owned by its calling thread."""

import importlib.metadata
import threading

from packaging.version import Version


class BMSession:
    _guard = threading.Lock()
    _active = False

    def __init__(self, *, device_id, rank_id, world_size, pool_gb, store_url):
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
            config.init_timeout = 90
            config.create_timeout = 90
            config.operation_timeout = 90
            self.check(bm.initialize(store_url, world_size, device_id, config),
                       "bm.initialize")
            self.initialized = True
            self.handle = bm.create2(id=0, local_dram_size=pool_gb << 30,
                                     max_dram_size=pool_gb << 30,
                                     data_op_type=bm.BmDataOpType.SDMA)
            if self.handle is None:
                raise RuntimeError("BM returned an empty handle")
            self.check(self.handle.join(), "BM join")
            self.joined = True
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

    def peer_address(self, rank):
        self._check_thread()
        address = self.handle.peer_rank_ptr(rank, self.bm.BmMemType.HOST)
        if not address:
            raise RuntimeError(f"No DRAM mapping for BM rank {rank}")
        return address

    def copy(self, src, dst, size, *, store):
        self._check_thread()
        if self.closed:
            raise RuntimeError("BM session is closed")
        kind = self.bm.BmCopyType.H2G if store else self.bm.BmCopyType.G2H
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
