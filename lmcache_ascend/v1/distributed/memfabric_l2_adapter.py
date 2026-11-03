# SPDX-License-Identifier: Apache-2.0
"""LMCache L2 backed by a fixed MemFabric DRAM rank.

Keys are immutable and the index is volatile. Fixed-size slots keep allocation
simple. One I/O thread owns BM; lookup locks protect slots from eviction until
unlock. Two independent services can opt into a common volatile directory.
"""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import threading
import time
from types import MappingProxyType

from lmcache.lmcache_native import Bitmap
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryResidency, MemorySegment, MemoryTags
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.platform import create_event_notifier

from lmcache_ascend.v1.distributed.memfabric_bm import BMSession

logger = init_logger(__name__)


@dataclass
class _Entry:
    slot: int
    size: int
    layout: tuple
    pins: int = 0
    delete_pending: bool = False


class MemFabricL2Adapter(L2AdapterInterface):
    def __init__(self, params, l1_memory_desc=None):
        allowed = {"device_id", "world_size", "pool_gb", "storage_rank",
                   "rank_id", "shared_directory",
                   "slot_size_mb", "store_url", "data_op_type", "device_nic", "enable_56bits_gva",
                   "join_coord_url", "join_group_id", "register_l1_arena", "l1_registration_chunk_mb"}
        if set(params) - allowed:
            raise ValueError(f"Unknown MemFabric options: {set(params) - allowed}")
        device = params.get("device_id", 0)
        self._data_op_type = params.get("data_op_type", "SDMA")
        self._register_l1_arena = params.get("register_l1_arena", False)
        if type(self._register_l1_arena) is not bool:
            raise ValueError("register_l1_arena must be a bool")
        chunk_mb = params.get("l1_registration_chunk_mb", 0)
        if type(chunk_mb) is not int or chunk_mb < 0 or (chunk_mb and not self._register_l1_arena):
            raise ValueError("l1_registration_chunk_mb requires a nonnegative integer and register_l1_arena")
        self._registration_chunk_bytes = chunk_mb << 20
        self._registration_arena_bytes = 0
        self._registration_fallback_reason = ""
        if self._register_l1_arena and self._data_op_type != "DEVICE_RDMA":
            raise ValueError("register_l1_arena requires DEVICE_RDMA")
        if self._register_l1_arena:
            # Official core describes final capacity separately from the stable
            # prefix. In particular POSIX SHM exposes None for this public field.
            # Never turn size, an allocator class or private state into a lease.
            stable_size = getattr(l1_memory_desc, "stable_registration_size", None)
            pointer = getattr(l1_memory_desc, "ptr", None)
            capacity = getattr(l1_memory_desc, "size", None)
            if stable_size is None:
                self._registration_fallback_reason = "stable_prefix_not_exposed"
            elif (type(pointer) is not int or type(capacity) is not int
                    or type(stable_size) is not int or pointer <= 0 or pointer % 4096
                    or stable_size <= 0 or stable_size % 4096 or stable_size > capacity):
                raise ValueError("Invalid public stable L1 prefix for BM registration")
            else:
                self._registration_arena_bytes = stable_size
        self._enable_56bits_gva = params.get("enable_56bits_gva", False)
        if self._data_op_type == "DEVICE_RDMA" and not (
                params.get("join_coord_url") and params.get("join_group_id")):
            raise ValueError("L2 DEVICE_RDMA requires explicit join_coord_url and fresh join_group_id")
        world = params.get("world_size", 1)
        pool_gb = params.get("pool_gb", 1)
        self._rank = params.get("rank_id", 0)
        shared = params.get("shared_directory", False)
        if type(shared) is not bool or type(self._rank) is not int:
            raise ValueError("shared_directory must be a bool and rank_id an integer")
        if shared and (world != 2 or self._rank not in (0, 1) or "rank_id" not in params
                       or self._data_op_type != "DEVICE_RDMA"):
            raise ValueError("shared_directory requires two DEVICE_RDMA adapters with explicit rank_ids 0 and 1")
        if shared and ("storage_rank" not in params or not params.get("store_url")):
            raise ValueError("shared_directory requires an explicit storage_rank and BM store_url")
        if not shared and self._rank != 0:
            raise ValueError("Nonzero adapter rank_id requires shared_directory")
        self.storage_rank = params.get("storage_rank", world - 1)
        slot_mb = params.get("slot_size_mb", 32)
        values = (device, world, pool_gb, self.storage_rank, slot_mb)
        if any(type(value) is not int for value in values):
            raise ValueError("BM device, ranks, and capacities must be integers")
        if not (0 <= self.storage_rank < world and slot_mb > 0 and pool_gb > 0):
            raise ValueError("Invalid MemFabric pool geometry")
        self._slot_size = slot_mb << 20
        capacity = pool_gb << 30
        if capacity % self._slot_size:
            raise ValueError("slot_size_mb must divide pool_gb * 1024")
        super().__init__(max_capacity_bytes=capacity)
        self._slot_count = capacity // self._slot_size
        self._free = [] if shared else list(reversed(range(self._slot_count)))
        self._directory = None
        self._entries = OrderedDict()
        self._quarantine = []
        self._state_lock = threading.RLock()
        self._task_lock = threading.Lock()
        self._closed = False
        self._poisoned = False
        self._last_error = ""
        self._written_bytes = 0
        self._read_bytes = 0
        self._evicted_objects = 0
        self._next_id = 0
        self._tasks = {kind: {} for kind in ("store", "lookup", "load")}
        self._events = {kind: create_event_notifier() for kind in self._tasks}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mf-bm")
        try:
            optional = {name: params[name] for name in (
                "data_op_type", "device_nic", "enable_56bits_gva", "join_coord_url", "join_group_id") if name in params}
            store_url = params.get("store_url", "tcp://127.0.0.1:28580")
            if shared:
                from lmcache_ascend.v1.distributed.memfabric_directory_rpc import DirectoryClient

                contract = dict(world_size=world, storage_rank=self.storage_rank,
                                pool_gb=pool_gb, slot_size_mb=slot_mb, store_url=store_url)
                self._directory = DirectoryClient(params["join_coord_url"], params["join_group_id"],
                                                  self._rank, contract)
                self._directory.on_retired = self._notify_retired
                optional["directory_epoch"] = self._directory.epoch
            self._session = self._executor.submit(
                BMSession, device_id=device, rank_id=self._rank, world_size=world,
                pool_gb=pool_gb, store_url=store_url, **optional
            ).result()
            self._base = self._executor.submit(
                self._session.peer_address, self.storage_rank
            ).result()
            if self._registration_arena_bytes:
                self._executor.submit(self._session.set_cpu_registration_arena,
                                      pointer, stable_size, self._registration_chunk_bytes).result()
        except BaseException:
            if hasattr(self, "_session"):
                self._executor.submit(self._session.close).result()
            self._executor.shutdown(wait=True)
            for event in self._events.values():
                event.close()
            raise
        logger.info("MEMFABRIC_READY protocol=%s storage_rank=%d capacity=%d slot=%d",
                    self._data_op_type, self.storage_rank, capacity, self._slot_size)
        if shared:
            logger.info("MEMFABRIC_SHARED_DIRECTORY rank=%d owner=%d epoch=%s",
                        self._rank, self.storage_rank, self._directory.epoch)
        if self._register_l1_arena:
            logger.info("MEMFABRIC_ARENA_SOURCE source=%s bytes=%d fallback_reason=%s",
                        "stable_registration_size" if self._registration_arena_bytes else "per_object",
                        self._registration_arena_bytes, self._registration_fallback_reason or "none")

    def get_store_event_fd(self):
        return self._events["store"].fileno()

    def get_lookup_and_lock_event_fd(self):
        return self._events["lookup"].fileno()

    def get_load_event_fd(self):
        return self._events["load"].fileno()

    def _submit(self, kind, operation, *args):
        with self._task_lock:
            if self._closed:
                raise RuntimeError("MemFabric adapter is closed")
            if self._poisoned:
                raise RuntimeError(self._last_error or "BM is unavailable")
            tid = self._next_id
            self._next_id += 1
            future = self._executor.submit(operation, *args)
            self._tasks[kind][tid] = future
            future.add_done_callback(lambda _: self._events[kind].notify())
            return tid

    def submit_store_task(self, keys, objects):
        if len(keys) != len(objects):
            raise ValueError("keys and objects must have equal lengths")
        return self._submit("store", self._store, list(keys), list(objects))

    def pop_completed_store_tasks(self):
        with self._task_lock:
            done = {}
            for tid, future in list(self._tasks["store"].items()):
                if not future.done():
                    continue
                try:
                    done[tid] = future.result()
                except BaseException:
                    # Consume this failed completion before propagating it, or
                    # every later pop will replay the same exception forever.
                    # Keep successful futures queued until the full scan returns
                    # so a later failure cannot discard results the caller missed.
                    del self._tasks["store"][tid]
                    raise
            for tid in done:
                del self._tasks["store"][tid]
            return done

    def submit_lookup_and_lock_task(self, keys, group_layout_descs):
        return self._submit("lookup", self._lookup, list(keys), dict(group_layout_descs))

    def _query(self, kind, tid):
        with self._task_lock:
            future = self._tasks[kind].get(tid)
            if future is None or not future.done():
                return None
            del self._tasks[kind][tid]
            return future.result()

    def query_lookup_and_lock_result(self, task_id):
        return self._query("lookup", task_id)

    def submit_load_task(self, keys, objects):
        if len(keys) != len(objects):
            raise ValueError("keys and objects must have equal lengths")
        return self._submit("load", self._load, list(keys), list(objects))

    def query_load_result(self, task_id):
        return self._query("load", task_id)

    def _enqueue(self, operation, keys):
        with self._task_lock:
            if self._closed:
                raise RuntimeError("MemFabric adapter is closed")
            if self._poisoned:
                raise RuntimeError(self._last_error or "BM is unavailable")
            future = self._executor.submit(operation, list(keys))
            if self._directory:
                future.add_done_callback(self._observe_background_error)

    def _observe_background_error(self, future):
        # Core provides no result handle for delete/unlock. Surface a rejected
        # control operation through health and the next submitted task.
        exc = future.exception()
        if exc is not None:
            with self._state_lock:
                self._poisoned = True
                self._last_error = str(exc)
            logger.error("MemFabric shared-directory operation failed: %s", exc)

    def submit_unlock(self, keys):
        self._enqueue(self._unlock, keys)

    def delete(self, keys):
        self._enqueue(self._delete, keys)

    @staticmethod
    def _layout(obj):
        return (tuple(tuple(shape) for shape in obj.get_shapes()),
                tuple(obj.get_dtypes()), obj.get_memory_format())

    def _buffer(self, obj):
        raw = obj.raw_tensor
        size = obj.get_size()
        if (raw is None or raw.device.type != "cpu" or not raw.is_contiguous()
                or not 0 < size <= self._slot_size
                or size > raw.numel() * raw.element_size()):
            raise ValueError("Expected a contiguous CPU MemoryObj fitting one BM slot")
        return obj.data_ptr, size, self._layout(obj)

    def _remove(self, key):
        entry = self._entries.pop(key)
        self._free.append(entry.slot)
        self._notify_keys_deleted([key], [entry.size])

    def _reserve(self):
        if not self._free:
            victim = next((k for k, e in self._entries.items() if not e.pins), None)
            if victim is None:
                raise RuntimeError("BM pool is full and all slots are locked")
            self._remove(victim)
            self._evicted_objects += 1
        return self._free.pop()

    @staticmethod
    def _wire_key(key):
        # Official core includes KV shard, object group and cache salt.
        return json.dumps(asdict(key.to_encoded_object_key()), sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _wire_layout(shapes, dtypes, memory_format=None):
        result = [[list(shape) for shape in shapes], [str(dtype) for dtype in dtypes]]
        if memory_format is not None:
            result.append(str(memory_format))
        return result

    def _notify_retired(self, retired):
        # The directory delivers a retired key once to the rank that published
        # it, even when the peer triggered deletion/eviction. This balances the
        # original core notification and its per-publisher byte accounting.
        from lmcache.v1.distributed.api import EncodedObjectKey

        keys = [EncodedObjectKey(**json.loads(key)).to_object_key() for key, _ in retired]
        self._notify_keys_deleted(keys, [size for _, size in retired])

    def _directory_call(self, op, **args):
        try:
            return self._directory.call(op, **args)
        except ValueError:
            raise  # Definitive rejection, e.g. a full pool.
        except Exception as exc:
            # The reply can be lost *after* a mutation. Never retry or free an
            # uncertain reservation/pin. Rebuild both ranks with a fresh group.
            self._poisoned = True
            self._last_error = str(exc)
            raise

    def _slot_address(self, slot):
        if type(slot) is not int or not 0 <= slot < self._slot_count:
            self._poisoned = True
            self._last_error = f"Invalid BM slot returned: {slot!r}"
            raise RuntimeError(self._last_error)
        return self._base + slot * self._slot_size

    def _copy(self, obj, address, size, *, store):
        if self._data_op_type == "DEVICE_RDMA":
            try:
                logger.info("MEMFABRIC_CPU_REGISTER pointer=%#x modulo4096=%d bytes=%d direction=%s",
                            obj.data_ptr, obj.data_ptr % 4096, size, "H2G" if store else "G2H")
                started = time.perf_counter_ns()
                registration = self._session.copy_cpu_object(obj, address, size, store=store)
                elapsed = time.perf_counter_ns() - started
                logger.info("MEMFABRIC_CPU_COPY registration=%s bytes=%d direction=%s elapsed_ns=%d",
                            registration, size, "H2G" if store else "G2H", elapsed)
            except BaseException as exc:
                self._poisoned = True
                self._last_error = str(exc)
                raise
            return
        # Keep an additional allocator reference on error: a timeout alone does
        # not establish that device access has stopped. close() drains BM first.
        obj.ref_count_up()
        try:
            src, dst = (obj.data_ptr, address) if store else (address, obj.data_ptr)
            self._session.copy(src, dst, size, store=store)
        except Exception:
            self._quarantine.append(obj)
            self._poisoned = True
            raise
        else:
            obj.ref_count_down()

    def _store(self, keys, objects):
        total = 0
        with self._state_lock:
            try:
                self._observe_peer_departure()
                if self._poisoned:
                    raise RuntimeError(self._last_error or "BM is unavailable")
                for key, obj in zip(keys, objects, strict=True):
                    _, size, layout = self._buffer(obj)
                    if self._directory:
                        wire_key = self._wire_key(key)
                        slot = self._directory_call("reserve", key=wire_key, size=size,
                                                    layout=self._wire_layout(*layout), cache_salt=key.cache_salt)
                        if slot is None:
                            continue
                    elif key in self._entries:
                        old = self._entries[key]
                        if (old.size, old.layout) != (size, layout):
                            raise ValueError("An existing key has a different layout")
                        continue
                    else:
                        slot = self._reserve()
                    self._copy(obj, self._slot_address(slot), size, store=True)
                    if self._directory:
                        self._directory_call("commit", key=wire_key, slot=slot)
                    else:
                        self._entries[key] = _Entry(slot, size, layout)
                    self._notify_keys_stored([key], [size])
                    total += size
                    self._written_bytes += size
                    logger.debug("MEMFABRIC_L2_OBJECT_COMPLETE direction=STORE key=%r bytes=%d", key, size)
                logger.info("MEMFABRIC_STORE keys=%d bytes=%d rank=%d",
                            len(keys), total, self.storage_rank)
                return L2StoreResult(True, total)
            except Exception as exc:
                self._last_error = str(exc)
                logger.exception("MemFabric store failed")
                if self._data_op_type == "DEVICE_RDMA":
                    raise
                return L2StoreResult(False, total)

    def _lookup(self, keys, layouts):
        found = Bitmap(len(keys))
        with self._state_lock:
            self._observe_peer_departure()
            if self._poisoned:
                if self._data_op_type == "DEVICE_RDMA":
                    raise RuntimeError(self._last_error or "BM RDMA is unavailable")
                return found
            for i, key in enumerate(keys):
                hint = layouts.get(key.object_group_id)
                if self._directory:
                    match = None if hint is None else self._wire_layout(hint.shapes, hint.dtypes)
                    entry = self._directory_call("pin", key=self._wire_key(key), hint=match)
                    if entry is not None:
                        found.set(i)
                    continue
                entry = self._entries.get(key)
                if entry is None or entry.delete_pending:
                    continue
                if hint is not None:
                    expected = (tuple(tuple(s) for s in hint.shapes), tuple(hint.dtypes))
                    if entry.layout[:2] != expected:
                        continue
                entry.pins += 1
                self._entries.move_to_end(key)
                found.set(i)
            logger.info("MEMFABRIC_LOOKUP keys=%d hits=%d", len(keys), found.popcount())
        return found

    def _load(self, keys, objects):
        found = Bitmap(len(keys))
        total = 0
        with self._state_lock:
            self._observe_peer_departure()
            if self._poisoned and self._data_op_type == "DEVICE_RDMA":
                raise RuntimeError(self._last_error or "BM RDMA is unavailable")
            for i, (key, obj) in enumerate(zip(keys, objects, strict=True)):
                if self._poisoned:
                    break
                try:
                    if self._directory:
                        pinned = self._directory_call("load", key=self._wire_key(key))
                        if pinned is None:
                            continue
                        slot = pinned["slot"]
                    else:
                        entry = self._entries.get(key)
                        if entry is None:
                            continue
                        slot = entry.slot
                    _, size, layout = self._buffer(obj)
                    if self._directory:
                        if pinned["size"] != size or pinned["layout"] != self._wire_layout(*layout):
                            continue
                    elif (entry.size, entry.layout) != (size, layout):
                        continue
                    self._copy(obj, self._slot_address(slot), size, store=False)
                    if not self._directory:
                        self._entries.move_to_end(key)
                    found.set(i)
                    total += size
                    self._read_bytes += size
                    logger.debug("MEMFABRIC_L2_OBJECT_COMPLETE direction=LOAD key=%r bytes=%d", key, size)
                except Exception as exc:
                    self._last_error = str(exc)
                    logger.exception("MemFabric load failed")
                    if self._data_op_type == "DEVICE_RDMA":
                        raise
            logger.info("MEMFABRIC_LOAD keys=%d bytes=%d rank=%d",
                        len(keys), total, self.storage_rank)
        return found

    def _unlock(self, keys):
        with self._state_lock:
            if self._poisoned:
                raise RuntimeError(self._last_error or "BM is unavailable")
            for key in keys:
                if self._directory:
                    self._directory_call("unpin", key=self._wire_key(key))
                    continue
                entry = self._entries.get(key)
                if entry is not None and entry.pins:
                    entry.pins -= 1
                    if not entry.pins and entry.delete_pending:
                        self._remove(key)

    def _delete(self, keys):
        with self._state_lock:
            if self._poisoned:
                raise RuntimeError(self._last_error or "BM is unavailable")
            for key in keys:
                if self._directory:
                    self._directory_call("delete", key=self._wire_key(key))
                    continue
                entry = self._entries.get(key)
                if entry is not None:
                    if entry.pins:
                        entry.delete_pending = True
                    else:
                        self._remove(key)

    def get_memory_tags(self):
        return MemoryTags(MemorySegment.DRAM, MemoryResidency.REMOTE if self._directory else MemoryResidency.LOCAL)

    def get_usage(self):
        if not self._directory:
            return super().get_usage()
        from lmcache.v1.distributed.l2_adapters.base import AdapterUsage

        with self._state_lock:
            stats = self._directory_call("status")
        return AdapterUsage(stats["total_bytes_used"], self._slot_count * self._slot_size,
                            MappingProxyType(stats["bytes_by_cache_salt"]))

    def _observe_peer_departure(self):
        if not self._poisoned and self._session.peer_departed.is_set():
            self._poisoned = True
            self._last_error = (
                f"BM peer rank {self._session.departed_rank} left the fixed group; "
                "restart the service and pool together")
            logger.error(self._last_error)

    def report_status(self):
        with self._state_lock:
            self._observe_peer_departure()
            shared_stats = {}
            if self._directory:
                try:
                    shared_stats = self._directory_call("status")
                except Exception:
                    # Preserve the diagnostic path even when the directory is
                    # offline. A missing response cannot establish an empty pool.
                    shared_stats = dict.fromkeys(("objects", "free_slots", "locked_objects",
                                                  "total_pins", "evicted_objects", "pending_writes"))
            return {"is_healthy": not self._closed and not self._poisoned,
                    "protocol": self._data_op_type, "enable_56bits_gva": self._enable_56bits_gva,
                    "register_l1_arena": self._register_l1_arena,
                    "registration_arena_bytes": self._registration_arena_bytes,
                    "registration_source": "stable_registration_size" if self._registration_arena_bytes else "per_object",
                    "registration_fallback_reason": self._registration_fallback_reason,
                    "registration_chunk_bytes": self._registration_chunk_bytes,
                    "registered_l1_bytes": sum(getattr(self._session, "_cpu_arena_ranges", {}).values()),
                    "registered_l1_regions": len(getattr(self._session, "_cpu_arena_ranges", {})),
                    "pool_memory_type": "host", "storage_rank": self.storage_rank,
                    "objects": len(self._entries), "free_slots": len(self._free),
                    "locked_objects": sum(e.pins > 0 for e in self._entries.values()),
                    "total_pins": sum(e.pins for e in self._entries.values()),
                    "evicted_objects": self._evicted_objects,
                    "store_bytes": self._written_bytes, "load_bytes": self._read_bytes,
                    "rank_id": self._rank, "shared_directory": self._directory is not None,
                    "last_error": self._last_error, **shared_stats}

    def _close_bm(self):
        self._session.wait()
        for obj in self._quarantine:
            obj.ref_count_down()
        self._quarantine.clear()
        self._session.close()

    def close(self):
        with self._task_lock:
            if self._closed:
                return
            self._closed = True
            drained = self._executor.submit(self._close_bm)
        # No timeout/cancel: all queued transfers must finish before unmapping.
        try:
            drained.result()
        except BaseException as exc:
            if self._data_op_type == "DEVICE_RDMA":
                self._closed = False
                self._poisoned = True
                self._last_error = str(exc)
            raise
        self._executor.shutdown(wait=True)
        for event in self._events.values():
            event.close()
