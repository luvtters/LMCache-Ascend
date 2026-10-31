# SPDX-License-Identifier: Apache-2.0
"""One fixed-membership BM session, owned by its calling thread."""

import importlib.metadata
import json
import threading
import time
from urllib.parse import quote, urlparse
import urllib.request
import uuid

from packaging.version import Version


class BMSession:
    _guard = threading.Lock()
    _active = False
    # A caller may drop a poisoned context after close fails. Keep the native
    # session and registered tensors reachable until process exit in that case.
    _uncertain_sessions = []

    def __init__(self, *, device_id, rank_id, world_size, pool_gb, store_url,
                 init_timeout=90, pool_memory_type="host", data_op_type="SDMA",
                 device_nic=None, enable_56bits_gva=False,
                 join_coord_url=None, join_group_id=None):
        if bool(join_coord_url) != bool(join_group_id):
            raise ValueError("join_coord_url and join_group_id must be configured together")
        if join_coord_url and (urlparse(join_coord_url).scheme not in ("http", "https")
                or not urlparse(join_coord_url).netloc):
            raise ValueError("join_coord_url requires an HTTP(S) endpoint")
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
        self.world_size = world_size
        self._local_join_complete = threading.Event()
        self._join_barrier_complete = False
        self._join_nonce = uuid.uuid4().hex
        self._join_contract = dict(world_size=world_size, store_url=store_url,
            pool_gb=pool_gb, pool_memory_type=pool_memory_type,
            data_op_type=data_op_type, enable_56bits_gva=enable_56bits_gva)
        self.device_id = device_id
        self.data_op_type = data_op_type
        self._registered_tensors = {}
        self._registered_cpu_objects = {}
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
            if join_coord_url:
                self.wait_fixed_group_ready(join_coord_url, join_group_id, init_timeout)
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
        elif rank == self.rank_id and event == self.bm.BmGroupEvent.JOIN_EVENT:
            # SDK JoinHandle emits this only after import, mmap and its final
            # group-result barrier. Never count missing callbacks for old peers.
            self._local_join_complete.set()

    def wait_fixed_group_ready(self, coord_url, group_id, timeout):
        """All ranks acknowledge their own after-mmap JOIN before any payload I/O."""
        self._check_thread()
        deadline = time.monotonic() + timeout
        if not self._local_join_complete.wait(max(0, deadline - time.monotonic())):
            raise TimeoutError("BM local after-mmap JOIN callback did not complete")
        base = coord_url.rstrip("/") + "/runs/" + quote(group_id, safe="") + "/events"
        name = "bm-joined-" + str(self.rank_id)
        own = dict(name=name, rank=self.rank_id, nonce=self._join_nonce,
                   contract=self._join_contract, protocol="bm-after-mmap-join-v1")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def request(url, data=None):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("BM fixed-group JOIN metadata barrier timed out")
            req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                headers={"Content-Type": "application/json"})
            with opener.open(req, timeout=min(5, remaining)) as response:
                return json.load(response)

        request(base + "/" + name, own)
        expected = {"bm-joined-" + str(rank) for rank in range(self.world_size)}
        while True:
            if self.peer_departed.is_set():
                raise RuntimeError("BM peer left during JOIN metadata barrier")
            events = request(base)["events"]
            for key, value in events.items():
                if not key.startswith("bm-joined-"):
                    continue
                if (key not in expected or value.get("name") != key
                        or value.get("rank") != int(key.removeprefix("bm-joined-"))
                        or value.get("protocol") != "bm-after-mmap-join-v1"
                        or value.get("contract") != self._join_contract):
                    raise ValueError("BM fixed-group JOIN metadata contract mismatch")
            if expected <= events.keys():
                if events[name] != own:
                    raise ValueError("BM JOIN event belongs to another session")
                self._join_barrier_complete = True
                print(f"MEMFABRIC_JOIN_BARRIER_READY rank={self.rank_id} world_size={self.world_size} "
                      f"group_id={group_id} contract=bm-after-mmap-join-v1", flush=True)
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("BM fixed-group JOIN metadata barrier timed out")
            self.peer_departed.wait(min(0.02, remaining))

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

    def copy_cpu_object(self, obj, address, size, *, store):
        """Register exactly one CPU MemoryObj while holding its allocator lease."""
        self._check_thread()
        if self.data_op_type != "DEVICE_RDMA":
            raise ValueError("Registered CPU MemoryObj copies require DEVICE_RDMA")
        if self.closed or self._completion_unknown:
            raise RuntimeError("BM session is closed or completion is unknown")
        if not self._join_barrier_complete:
            raise RuntimeError("RDMA CPU MemoryObj requires the fixed-group JOIN metadata barrier")
        raw = obj.raw_tensor
        pointer = obj.data_ptr
        if (raw is None or raw.device.type != "cpu" or not raw.is_contiguous()
                or pointer != raw.data_ptr() or size != obj.get_size() or size <= 0
                or size > raw.numel() * raw.element_size()):
            raise ValueError("RDMA requires the exact contiguous CPU MemoryObj range")
        if pointer <= 0 or pointer % 4096:
            raise ValueError(f"RDMA CPU MemoryObj requires 4096-byte address alignment: "
                             f"pointer={pointer:#x} modulo4096={pointer % 4096} bytes={size}")
        if pointer in self._registered_cpu_objects:
            raise RuntimeError("CPU MemoryObj is already registered")
        obj.ref_count_up()
        # A Tensor reference alone does not stop LMCache recycling this slice.
        self._registered_cpu_objects[pointer] = (obj, raw, size)
        try:
            self.check(self.handle.register(pointer, size),
                       f"BM register CPU MemoryObj bytes={size} modulo4096={pointer % 4096}")
            src, dst = (pointer, address) if store else (address, pointer)
            self._copy(src, dst, size, self.bm.BmCopyType.H2G if store else self.bm.BmCopyType.G2H)
            self.check(self.handle.unregister(pointer), "BM unregister CPU MemoryObj")
        except BaseException:
            self._retain_unknown_completion()
            raise
        # Only successful synchronous copy AND unregister permit allocator reuse.
        obj.ref_count_down()
        del self._registered_cpu_objects[pointer]

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
        try:
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
        except BaseException:
            if self.data_op_type == "DEVICE_RDMA":
                self._retain_unknown_completion()
            raise
        self._registered_tensors.clear()
        self._registered_cpu_objects.clear()
        self.closed = True
        with self._guard:
            BMSession._active = False
