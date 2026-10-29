# SPDX-License-Identifier: Apache-2.0
"""Volatile metadata for a fixed BM DRAM pool; never owns KV payloads.

No expiry reclaims reservations or pins: only an explicit DMA completion may
make their slots reusable. A fixed-group failure requires a coordinated restart.
"""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import threading
import uuid


@dataclass
class _Entry:
    descriptor: dict
    layout: object
    pins: int = 0


class PoolIndex:
    def __init__(self, *, pool_bytes, slot_size, storage_rank, world_size,
                 ready=True, tp_size=1):
        values = (pool_bytes, slot_size, storage_rank, world_size)
        if (any(type(v) is not int for v in values) or pool_bytes <= 0
                or slot_size <= 0 or pool_bytes % slot_size
                or world_size < 2 or not 0 <= storage_rank < world_size):
            raise ValueError("Invalid fixed pool geometry")
        self.epoch = uuid.uuid4().hex
        self.pool_bytes, self.slot_size = pool_bytes, slot_size
        self.storage_rank, self.world_size = storage_rank, world_size
        if type(tp_size) is not int or tp_size not in (1, 2, 4, 8):
            raise ValueError("Only homogeneous TP1/2/4/8 is supported")
        if tp_size > 1 and (storage_rank != 0 or world_size != 1 + 2 * tp_size):
            raise ValueError("TP pool requires owner=0 and world_size=1+2*tp_size")
        self.tp_size = tp_size
        self._member_shards = {}
        self._layout = None
        self._chunk_size = None
        self._ready, self._closed, self._error = ready, False, ""
        self._lock = threading.RLock()
        self._members = {}
        self._free = list(reversed(range(pool_bytes // slot_size)))
        self._generation = [0] * len(self._free)
        self._entries = OrderedDict()
        self._writes = {}
        self._leases = {}
        self._lookups = {}
        self._evictions = 0

    def _shard(self, shard=0, tp_size=1):
        if (type(shard) is not int or type(tp_size) is not int
                or tp_size != self.tp_size or not 0 <= shard < tp_size):
            raise ValueError("Shard does not match the fixed homogeneous TP pool")

    def _key(self, key, shard):
        return key if self.tp_size == 1 else (key, shard)

    def _group(self, key):
        return [self._entries.get(self._key(key, s)) for s in range(self.tp_size)]

    def _check_member_shard(self, rank, shard):
        if self.tp_size > 1 and self._member_shards.get(rank) != shard:
            raise ValueError("Worker cannot access another TP shard")

    def _available(self):
        if self._closed:
            raise RuntimeError("Pool is closed")
        if self._error:
            raise RuntimeError(self._error)
        if not self._ready:
            raise RuntimeError("BM transfer is not ready")

    def _owner(self, epoch, rank, member_id):
        if epoch != self.epoch:
            raise RuntimeError("Stale pool epoch")
        if self._members.get(rank) != member_id:
            raise RuntimeError("Unknown fixed-group member")

    def mark_ready(self):
        with self._lock:
            if not self._closed and not self._error:
                self._ready = True

    def invalidate(self, reason="fixed group invalidated"):
        with self._lock:
            self._error = "BM fixed group unavailable: " + str(reason)
            self._ready = False
            return self.status()

    def register(self, *, rank, member_id, shard=0, tp_size=1, layout=None, size=None):
        self._shard(shard, tp_size)
        with self._lock:
            if self._closed or self._error:
                self._available()
            if (type(rank) is not int or not 0 <= rank < self.world_size
                    or rank == self.storage_rank or not isinstance(member_id, str)
                    or not member_id):
                raise ValueError("Invalid worker member identity")
            old = self._members.get(rank)
            if old is not None and old != member_id:
                self.invalidate("member replacement at rank %s" % rank)
                raise RuntimeError("Fixed-group member replacement requires restart")
            if tp_size > 1:
                if (rank - 1) % tp_size != shard:
                    raise ValueError("BM rank must identify the real TP shard")
                if not isinstance(layout, dict) or type(size) is not int or not 0 < size <= self.slot_size:
                    raise ValueError("TP registration requires actual layout and chunk size")
                if self._layout is not None and (self._layout != layout or self._chunk_size != size):
                    raise ValueError("All homogeneous TP members must have identical actual KV layouts")
                self._layout, self._chunk_size = deepcopy(layout), size
            self._members[rank] = member_id
            self._member_shards[rank] = shard
            return self.status()

    def _descriptor(self, key, slot, size, layout, shard=0, tp_size=1):
        return dict(epoch=self.epoch, slot=slot, generation=self._generation[slot],
                    key=key, shard=shard, tp_size=tp_size, size=size,
                    offset=slot * self.slot_size, storage_rank=self.storage_rank,
                    layout=deepcopy(layout))

    def prepare_store(self, *, epoch, rank, member_id, key, size, layout=None,
                      shard=0, tp_size=1):
        self._shard(shard, tp_size)
        if not isinstance(key, str) or not key or type(size) is not int or not 0 < size <= self.slot_size:
            raise ValueError("Expected a stable key and size fitting one slot")
        with self._lock:
            self._available()
            self._owner(epoch, rank, member_id)
            self._check_member_shard(rank, shard)
            if tp_size > 1 and (layout != self._layout or size != self._chunk_size):
                raise ValueError("Write differs from registered TP layout")
            entry_key = self._key(key, shard)
            if entry_key in self._entries:
                entry = self._entries[entry_key]
                if entry.descriptor["size"] != size or entry.layout != layout:
                    raise ValueError("Immutable key has a different size or layout")
                self._entries.move_to_end(entry_key)
                return dict(existing=True, descriptor=deepcopy(entry.descriptor), write_id=None)
            if any(self._key(w["entry"].descriptor["key"], w["entry"].descriptor["shard"]) == entry_key for w in self._writes.values()):
                raise RuntimeError("Key already has an active or quarantined write")
            if not self._free:
                victim = next((k for k, e in self._entries.items()
                               if not any(x is not None and x.pins for x in self._group(e.descriptor["key"]))), None)
                if victim is None:
                    raise RuntimeError("BM pool full: slots are writing, quarantined, or pinned")
                victim_key = self._entries[victim].descriptor["key"]
                for s in range(self.tp_size):
                    removed = self._entries.pop(self._key(victim_key, s), None)
                    if removed is not None:
                        self._free.append(removed.descriptor["slot"])
                        self._evictions += 1
            slot = self._free.pop()
            self._generation[slot] += 1
            entry = _Entry(self._descriptor(key, slot, size, layout, shard, tp_size), deepcopy(layout))
            write_id = uuid.uuid4().hex
            self._writes[write_id] = dict(entry=entry, owner=(rank, member_id), aborted=False)
            return dict(existing=False, descriptor=deepcopy(entry.descriptor), write_id=write_id)

    def _write(self, write_id, rank, member_id):
        write = self._writes.get(write_id)
        if write is None or write["owner"] != (rank, member_id):
            raise RuntimeError("Unknown or stale write reservation")
        return write

    def commit_store(self, *, epoch, rank, member_id, write_id):
        with self._lock:
            self._available()
            self._owner(epoch, rank, member_id)
            write = self._write(write_id, rank, member_id)
            if write["aborted"]:
                raise RuntimeError("Cannot commit an aborted write")
            entry = write["entry"]
            self._entries[self._key(entry.descriptor["key"], entry.descriptor["shard"])] = entry
            del self._writes[write_id]
            return dict(descriptor=deepcopy(entry.descriptor))

    def abort_store(self, *, epoch, rank, member_id, write_id, dma_complete=False):
        if type(dma_complete) is not bool:
            raise ValueError("dma_complete must be a boolean")
        with self._lock:
            self._owner(epoch, rank, member_id)
            write = self._write(write_id, rank, member_id)
            write["aborted"] = True
            if dma_complete:
                self._free.append(write["entry"].descriptor["slot"])
                del self._writes[write_id]
            return dict(quarantined=not dma_complete)

    def lookup(self, *, keys, shard=0, tp_size=1, request_id=None):
        self._shard(shard, tp_size)
        if not isinstance(keys, list) or any(not isinstance(k, str) or not k for k in keys):
            raise ValueError("keys must contain stable key strings")
        if request_id is not None and (not isinstance(request_id, str) or not request_id):
            raise ValueError("Invalid lookup request_id")
        with self._lock:
            self._available()
            if request_id in self._lookups:
                old = self._lookups[request_id]
                if old["keys"] != keys or old["shard"] != shard:
                    raise ValueError("Lookup request_id reused with different keys")
                return dict(epoch=self.epoch, descriptors=[deepcopy(e.descriptor) for e in old["entries"] if e.descriptor["shard"] == shard],
                            lease_id=old["lease_id"])
            entries = []
            for key in keys:
                group = self._group(key)
                if any(entry is None for entry in group):
                    break
                entries.extend(group)
                for s in range(self.tp_size):
                    self._entries.move_to_end(self._key(key, s))
            lease_id = None
            if request_id is not None:
                lease_id = uuid.uuid4().hex
                for entry in entries:
                    entry.pins += 1
                self._lookups[request_id] = dict(keys=list(keys), entries=entries, lease_id=lease_id, shard=shard)
            return dict(epoch=self.epoch, descriptors=[deepcopy(e.descriptor) for e in entries if e.descriptor["shard"] == shard],
                        lease_id=lease_id)

    def acquire(self, *, epoch, rank, member_id, descriptor):
        with self._lock:
            self._available()
            self._owner(epoch, rank, member_id)
            if not isinstance(descriptor, dict):
                raise ValueError("Expected object descriptor")
            self._check_member_shard(rank, descriptor.get("shard"))
            entry = self._entries.get(self._key(descriptor.get("key"), descriptor.get("shard")))
            if entry is None or descriptor != entry.descriptor:
                raise RuntimeError("Object descriptor is stale")
            lease_id = uuid.uuid4().hex
            entry.pins += 1
            self._entries.move_to_end(self._key(descriptor["key"], descriptor["shard"]))
            self._leases[lease_id] = (entry, rank, member_id)
            return dict(lease_id=lease_id, descriptor=deepcopy(entry.descriptor))

    def release(self, *, epoch, dma_complete=False, lease_id=None, request_id=None,
                rank=None, member_id=None, keys=None):
        if type(dma_complete) is not bool:
            raise ValueError("dma_complete must be a boolean")
        if keys is not None and (not isinstance(keys, list)
                                 or any(not isinstance(k, str) for k in keys)):
            raise ValueError("keys must be a list of stable key strings")
        with self._lock:
            if epoch != self.epoch:
                raise RuntimeError("Stale pool epoch")
            if request_id is not None:
                lookup = self._lookups.get(request_id)
                if lookup is None:
                    return dict(released=True)
                entries = [e for e in lookup["entries"]
                           if keys is None or e.descriptor["key"] in keys]
            else:
                self._owner(epoch, rank, member_id)
                lease = self._leases.get(lease_id)
                if lease is None or lease[1:] != (rank, member_id):
                    raise RuntimeError("Unknown or stale read lease")
                entries = [lease[0]]
            if dma_complete:
                for entry in entries:
                    entry.pins -= 1
                if request_id is not None:
                    lookup["entries"] = [e for e in lookup["entries"]
                                         if all(e is not selected for selected in entries)]
                    if not lookup["entries"]:
                        del self._lookups[request_id]
                else:
                    del self._leases[lease_id]
            return dict(released=dma_complete)

    def status(self):
        with self._lock:
            return dict(epoch=self.epoch, ready=self._ready and not self._closed and not self._error,
                        registration_ready=len(self._members) == self.world_size - 1,
                        transfer_ready=self._ready and not self._closed and not self._error,
                        closed=self._closed, last_error=self._error, protocol="SDMA",
                        storage_rank=self.storage_rank, world_size=self.world_size,
                        tp_size=self.tp_size, layout=deepcopy(self._layout), chunk_size=self._chunk_size,
                        pool_bytes=self.pool_bytes, slot_size=self.slot_size,
                        objects=len(self._entries), free_slots=len(self._free),
                        pending_writes=len(self._writes),
                        quarantined_slots=sum(w["aborted"] for w in self._writes.values()),
                        total_pins=sum(e.pins for e in self._entries.values()),
                        evicted_objects=self._evictions)

    def close(self):
        with self._lock:
            self._closed = True
            self._ready = False
