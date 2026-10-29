# SPDX-License-Identifier: Apache-2.0
"""Experimental single-service LMCache L2 backed by a selected MemFabric pool.

Keys are immutable and the index is volatile. Fixed-size slots keep allocation
simple. One I/O thread owns BM; lookup locks protect slots from eviction until
unlock. This first version targets a fixed group on one A3 host.
"""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading

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
                   "slot_size_mb", "store_url", "pool_memory_type"}
        if set(params) - allowed:
            raise ValueError(f"Unknown MemFabric options: {set(params) - allowed}")
        device = params.get("device_id", 0)
        world = params.get("world_size", 1)
        pool_gb = params.get("pool_gb", 1)
        self.pool_memory_type = params.get("pool_memory_type", "host")
        if self.pool_memory_type not in ("host", "device"):
            raise ValueError("pool_memory_type must be host or device")
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
        self._free = list(reversed(range(capacity // self._slot_size)))
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
            self._session = self._executor.submit(
                BMSession, device_id=device, rank_id=0, world_size=world,
                pool_gb=pool_gb, store_url=params.get("store_url", "tcp://127.0.0.1:28580"),
                pool_memory_type=self.pool_memory_type
            ).result()
            self._base = self._executor.submit(
                self._session.peer_address, self.storage_rank
            ).result()
        except BaseException:
            if hasattr(self, "_session"):
                self._executor.submit(self._session.close).result()
            self._executor.shutdown(wait=True)
            for event in self._events.values():
                event.close()
            raise
        logger.info("MEMFABRIC_READY protocol=SDMA storage_rank=%d capacity=%d slot=%d pool_memory_type=%s",
                    self.storage_rank, capacity, self._slot_size, self.pool_memory_type)

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
            done = {tid: future.result() for tid, future in self._tasks["store"].items()
                    if future.done()}
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
            self._executor.submit(operation, list(keys))

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

    def _copy(self, obj, address, size, *, store):
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
                    if key in self._entries:
                        old = self._entries[key]
                        if (old.size, old.layout) != (size, layout):
                            raise ValueError("An existing key has a different layout")
                        continue
                    slot = self._reserve()
                    self._copy(obj, self._base + slot * self._slot_size, size, store=True)
                    self._entries[key] = _Entry(slot, size, layout)
                    self._notify_keys_stored([key], [size])
                    total += size
                    self._written_bytes += size
                logger.info("MEMFABRIC_STORE keys=%d bytes=%d rank=%d",
                            len(keys), total, self.storage_rank)
                return L2StoreResult(True, total)
            except Exception as exc:
                self._last_error = str(exc)
                logger.exception("MemFabric store failed")
                return L2StoreResult(False, total)

    def _lookup(self, keys, layouts):
        found = Bitmap(len(keys))
        with self._state_lock:
            self._observe_peer_departure()
            if self._poisoned:
                return found
            for i, key in enumerate(keys):
                entry = self._entries.get(key)
                if entry is None or entry.delete_pending:
                    continue
                hint = layouts.get(key.object_group_id)
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
            for i, (key, obj) in enumerate(zip(keys, objects, strict=True)):
                if self._poisoned:
                    break
                entry = self._entries.get(key)
                if entry is None:
                    continue
                try:
                    _, size, layout = self._buffer(obj)
                    if (entry.size, entry.layout) != (size, layout):
                        continue
                    self._copy(obj, self._base + entry.slot * self._slot_size,
                               size, store=False)
                    self._entries.move_to_end(key)
                    found.set(i)
                    total += size
                    self._read_bytes += size
                except Exception as exc:
                    self._last_error = str(exc)
                    logger.exception("MemFabric load failed")
            logger.info("MEMFABRIC_LOAD keys=%d bytes=%d rank=%d",
                        len(keys), total, self.storage_rank)
        return found

    def _unlock(self, keys):
        with self._state_lock:
            for key in keys:
                entry = self._entries.get(key)
                if entry is not None and entry.pins:
                    entry.pins -= 1
                    if not entry.pins and entry.delete_pending:
                        self._remove(key)

    def _delete(self, keys):
        with self._state_lock:
            for key in keys:
                entry = self._entries.get(key)
                if entry is not None:
                    if entry.pins:
                        entry.delete_pending = True
                    else:
                        self._remove(key)

    def get_memory_tags(self):
        segment = MemorySegment.VRAM if self.pool_memory_type == "device" else MemorySegment.DRAM
        return MemoryTags(segment, MemoryResidency.LOCAL)

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
            return {"is_healthy": not self._closed and not self._poisoned,
                    "protocol": "SDMA", "storage_rank": self.storage_rank,
                    "pool_memory_type": self.pool_memory_type,
                    "pool_memory_medium": "HBM" if self.pool_memory_type == "device" else "DRAM",
                    "pool_native_mem_type": "DEVICE" if self.pool_memory_type == "device" else "HOST",
                    "local_selected_pool_bytes": self._session.local_pool_bytes,
                    "storage_peer_gva": self._base,
                    "actual_export_seg_type": "NOT_VERIFIED_READ_NATIVE_LOG",
                    "objects": len(self._entries), "free_slots": len(self._free),
                    "locked_objects": sum(e.pins > 0 for e in self._entries.values()),
                    "total_pins": sum(e.pins for e in self._entries.values()),
                    "evicted_objects": self._evicted_objects,
                    "store_bytes": self._written_bytes, "load_bytes": self._read_bytes,
                    "last_error": self._last_error}

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
        drained.result()
        self._executor.shutdown(wait=True)
        for event in self._events.values():
            event.close()
