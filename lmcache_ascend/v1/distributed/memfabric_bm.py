# SPDX-License-Identifier: Apache-2.0
"""One fixed-membership BM session, owned by its calling thread."""

import importlib.metadata
import threading
import time
from urllib.parse import urlparse

from packaging.version import Version


class BMSession:
    _guard = threading.Lock()
    _active = False
    # A caller may drop a poisoned context after close fails. Keep the native
    # session and registered tensors reachable until process exit in that case.
    _uncertain_sessions = []

    def __init__(self, *, device_id, rank_id, world_size, pool_gb, store_url,
                 init_timeout=90, pool_memory_type="host", data_op_type="SDMA",
                 device_nic=None, enable_56bits_gva=False):
        if data_op_type not in ("SDMA", "DEVICE_RDMA"):
            raise ValueError("data_op_type must be SDMA or DEVICE_RDMA")
        if type(enable_56bits_gva) is not bool:
            raise ValueError("enable_56bits_gva must be a bool")
        if device_nic is not None:
            try:
                parsed = urlparse(device_nic)
                valid_nic = (isinstance(device_nic, str) and parsed.scheme == "tcp"
                             and parsed.hostname and parsed.port)
            except (TypeError, ValueError, AttributeError):
                valid_nic = False
            if not valid_nic:
                raise ValueError("device_nic requires tcp://host:port")
        if data_op_type == "DEVICE_RDMA" and device_nic is None:
            raise ValueError("DEVICE_RDMA requires explicit device_nic")
        if pool_memory_type not in ("host", "device"):
            raise ValueError("pool_memory_type must be host or device")
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
        self.device_id = device_id
        self.data_op_type = data_op_type
        self._registered_tensors = {}
        self._unregistered_pointers = set()
        self._completion_unknown = False
        self.pool_memory_type = pool_memory_type
        # The first device-pool candidate deliberately preserves the 1 GiB
        # DRAM contribution used by the verified media probe on every rank.
        self.local_dram_bytes = (pool_gb if pool_memory_type == "host" else 1) << 30
        self.local_hbm_bytes = (pool_gb << 30) if pool_memory_type == "device" else 0
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
            if device_nic is not None:
                config.set_nic(device_nic)
            self.check(bm.initialize(store_url, world_size, device_id, config),
                       "bm.initialize")
            self.initialized = True
            memory = (dict(local_dram_size=self.local_dram_bytes, max_dram_size=self.local_dram_bytes)
                      if pool_memory_type == "host" else
                      dict(local_dram_size=self.local_dram_bytes, max_dram_size=self.local_dram_bytes,
                           local_hbm_size=self.local_hbm_bytes, max_hbm_size=self.local_hbm_bytes))
            options = {}
            if data_op_type == "DEVICE_RDMA" or enable_56bits_gva:
                options.update(enable_56bits_gva=enable_56bits_gva, flags=0)
            self.handle = bm.create2(id=0, data_op_type=getattr(bm.BmDataOpType, data_op_type),
                                     **memory, **options)
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
            while data_op_type == "SDMA" and not self.peer_departed.is_set():
                if all(self.handle.gva_to_va(
                        self.peer_address(peer), bm.BmMemType.LOCAL_DEVICE)
                       for peer in range(world_size) if peer != rank_id):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"BM peer {self.pool_memory_type} pool mappings did not become ready")
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
        memory_type = (self.bm.BmMemType.HOST if self.pool_memory_type == "host"
                       else self.bm.BmMemType.DEVICE)
        address = self.handle.peer_rank_ptr(rank, memory_type)
        if not address:
            raise RuntimeError(f"No {self.pool_memory_type} pool mapping for BM rank {rank}")
        return address

    def copy(self, src, dst, size, *, store):
        if getattr(self, "data_op_type", "SDMA") == "DEVICE_RDMA":
            raise RuntimeError("External CPU MemoryObj RDMA path is not validated")
        self._copy(src, dst, size, self.bm.BmCopyType.H2G if store
                   else self.bm.BmCopyType.G2H)

    def copy_npu_to_global(self, src, dst, size):
        """Synchronously copy an actual local NPU pointer into a BM GVA."""
        self._check_registered_npu(src, size)
        self._copy(src, dst, size, self.bm.BmCopyType.L2G)

    def copy_global_to_npu(self, src, dst, size):
        """Synchronously copy a BM GVA into an actual local NPU pointer."""
        self._check_registered_npu(dst, size)
        self._copy(src, dst, size, self.bm.BmCopyType.G2L)

    def register_npu_tensor(self, tensor):
        """Retain the actual allocation through native teardown; never cache bare VAs."""
        self._check_thread()
        if self.closed or self._completion_unknown:
            raise RuntimeError("BM session is closed or completion is unknown")
        if self.data_op_type != "DEVICE_RDMA":
            return
        if (tensor.device.type != "npu" or tensor.device.index != self.device_id
                or not tensor.is_contiguous()):
            raise ValueError("Registration requires a contiguous tensor on the BM NPU")
        pointer, size = tensor.data_ptr(), tensor.numel() * tensor.element_size()
        if pointer <= 0 or size <= 0:
            raise ValueError("Registration requires a nonempty NPU allocation")
        if pointer in self._registered_tensors:
            old, old_size = self._registered_tensors[pointer]
            if old is tensor and old_size == size:
                return
            raise ValueError("NPU pointer already belongs to another retained tensor")
        if any(pointer < ptr + length and ptr < pointer + size
               for ptr, (_, length) in self._registered_tensors.items()):
            raise ValueError("NPU registration overlaps another retained allocation")
        # Even a failed native registration may have partially installed state.
        self._registered_tensors[pointer] = (tensor, size)
        try:
            self.check(self.handle.register(pointer, size), "BM register NPU tensor")
        except BaseException:
            self._retain_unknown_completion()
            raise

    def _retain_unknown_completion(self):
        if not self._completion_unknown:
            self._completion_unknown = True
            self._uncertain_sessions.append(self)

    def _check_registered_npu(self, pointer, size):
        if getattr(self, "data_op_type", "SDMA") != "DEVICE_RDMA":
            return
        if size <= 0 or not any(ptr <= pointer and pointer + size <= ptr + length
                for ptr, (_, length) in self._registered_tensors.items()):
            raise ValueError("RDMA NPU copy requires a retained registered tensor range")

    def _copy(self, src, dst, size, kind):
        self._check_thread()
        if self.closed:
            raise RuntimeError("BM session is closed")
        if getattr(self, "_completion_unknown", False):
            raise RuntimeError("BM copy completion is unknown; retain registered allocations")
        if self.peer_departed.is_set():
            raise RuntimeError(f"BM peer rank {self.departed_rank} left the fixed group")
        # flags=0 selects synchronous copies; no asynchronous buffer recycling.
        try:
            self.check(self.handle.copy_data(src, dst, size, kind, 0), "BM copy")
        except BaseException:
            if getattr(self, "data_op_type", "SDMA") == "DEVICE_RDMA":
                self._retain_unknown_completion()
            raise

    def wait(self):
        self._check_thread()
        if getattr(self, "_completion_unknown", False):
            # DEVICE_RDMA handle.wait() is a no-op, not recovery evidence.
            raise RuntimeError("BM RDMA completion unknown; resources retained until process exit")
        if self.handle is not None:
            self.check(self.handle.wait(), "BM wait")

    def close(self):
        self._check_thread()
        if self.closed:
            return
        if self.handle is not None:
            self.wait()
            for pointer in list(self._registered_tensors):
                if pointer not in self._unregistered_pointers:
                    self.check(self.handle.unregister(pointer), "BM unregister NPU tensor")
                    self._unregistered_pointers.add(pointer)
                # Retain the tensor itself through leave/destroy/uninitialize.
            if self.joined:
                self.check(self.handle.leave(), "BM leave")
            self.handle.destroy()
            self.handle = None
        if self.initialized:
            self.bm.uninitialize()
        if self.mf_initialized:
            self.mf.uninitialize()
        self._registered_tensors.clear()
        self.closed = True
        with self._guard:
            BMSession._active = False
