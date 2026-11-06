# SPDX-License-Identifier: Apache-2.0
"""Volatile metadata for a fixed group of adapters sharing BM owner segments."""
from collections import OrderedDict, deque
from copy import deepcopy
from dataclasses import dataclass, field
import json
from threading import Lock


def _validate_layout(value, *, hint=False):
    expected = 2 if hint else 3
    if not isinstance(value, list) or len(value) != expected:
        raise ValueError("Invalid layout field count")
    shapes, dtypes = value[:2]
    if (not isinstance(shapes, list) or not isinstance(dtypes, list)
            or len(shapes) != len(dtypes)
            or any(not isinstance(shape, list) or
                   any(type(dim) is not int or dim < 0 for dim in shape) for shape in shapes)
            or any(not isinstance(dtype, str) for dtype in dtypes)):
        raise ValueError("Invalid layout shapes or dtypes")
    if not hint and type(value[2]) not in (str, int):
        raise ValueError("Invalid memory format")


@dataclass
class _Entry:
    slot: int
    size: int
    layout: list
    cache_salt: str
    publisher: int
    published: bool = False
    pins: dict[int, int] = field(default_factory=dict)
    delete_pending: bool = False

    def descriptor(self):
        return {"slot": self.slot, "size": self.size, "layout": deepcopy(self.layout)}


class PoolDirectory:
    """Linearize slot reservations, publication and reader pins for one group.

    A reserved slot remains unavailable until its writer commits after a
    successful BM copy. An uncertain copy or a lost participant requires a
    coordinated group restart. There is no timeout or per-client reclamation.
    """

    def __init__(self, slot_count, slot_size, *, owner_ranks=()):
        if any(type(n) is not int or n <= 0 for n in (slot_count, slot_size)):
            raise ValueError("slot_count and slot_size must be positive integers")
        self._slot_size = slot_size
        self._owner_ranks = tuple(owner_ranks)
        self._slots_per_owner = slot_count
        # Global slot IDs interleave owners. Existing reserve/commit and pin
        # semantics still protect exactly one unique slot across the group.
        self._free = list(reversed(range(slot_count * (len(self._owner_ranks) or 1))))
        self._entries = OrderedDict()
        self._evicted = 0
        self._retired = {}
        self._lock = Lock()

    def apply(self, rank, op, args):
        """Apply one request. Keys are opaque canonical strings from the adapter."""
        if type(rank) is not int or rank < 0 or not isinstance(args, dict):
            raise ValueError("Expected a nonnegative rank and an argument object")
        if op not in ("reserve", "commit", "pin", "load", "unpin", "delete", "status"):
            raise ValueError(f"Unknown directory operation: {op!r}")
        if op != "status" and (not isinstance(args.get("key"), str) or not args["key"]):
            raise ValueError("Expected a nonempty key string")
        with self._lock:
            try:
                return getattr(self, "_" + op)(rank, **args)
            except TypeError as exc:
                raise ValueError(f"Invalid arguments for {op}: {exc}") from exc

    def _reserve(self, rank, key, size, layout, cache_salt=""):
        if type(size) is not int or not 0 < size <= self._slot_size:
            raise ValueError("Object must fit in one slot")
        # The whole retirement record must always fit in a reply, including
        # JSON quoting/escaping of the already-encoded key.
        if len(json.dumps([key, size])) > 4096:
            raise ValueError("Encoded [key, size] exceeds 4 KiB")
        if not isinstance(cache_salt, str):
            raise ValueError("cache_salt must be a string")
        _validate_layout(layout)
        old = self._entries.get(key)
        if old is not None:
            if not old.published or old.delete_pending:
                raise ValueError("Key has a pending write or delete")
            if (old.size, old.layout, old.cache_salt) != (size, layout, cache_salt):
                raise ValueError("Existing key has a different size, layout or salt")
            return None

        if self._free:
            slot = self._free.pop()
        else:
            victim = next((k for k, e in self._entries.items()
                           if e.published and not e.pins), None)
            if victim is None:
                raise ValueError("All slots are reserved or pinned")
            slot = self._retire(victim)
            self._evicted += 1
        self._entries[key] = _Entry(slot, size, deepcopy(layout), cache_salt, rank)
        # Drain an older generation of this key in the reserve response,
        # before the publisher notifies its local core about the new store.
        pending = self._retired.get(rank)
        if pending and key in pending:
            pending.move_to_end(key, last=False)
        return slot

    def _commit(self, rank, key, slot):
        entry = self._entries.get(key)
        if (entry is None or entry.published or entry.publisher != rank
                or type(slot) is not int or entry.slot != slot):
            raise ValueError("Commit does not match the writer and reserved slot")
        entry.published = True
        self._entries.move_to_end(key)

    def _pin(self, rank, key, hint=None):
        if hint is not None:
            _validate_layout(hint, hint=True)
        entry = self._entries.get(key)
        if entry is None or not entry.published or entry.delete_pending:
            return None
        if hint is not None and entry.layout[:2] != hint:
            return None
        entry.pins[rank] = entry.pins.get(rank, 0) + 1
        self._entries.move_to_end(key)
        return entry.descriptor()

    def _load(self, rank, key):
        entry = self._entries.get(key)
        if entry is None or rank not in entry.pins:
            return None
        self._entries.move_to_end(key)
        return entry.descriptor()

    def _unpin(self, rank, key):
        entry = self._entries.get(key)
        if entry is None or rank not in entry.pins:
            return
        entry.pins[rank] -= 1
        if entry.pins[rank] == 0:
            del entry.pins[rank]
        if entry.delete_pending and not entry.pins:
            self._free.append(self._retire(key))

    def _delete(self, rank, key):
        entry = self._entries.get(key)
        if entry is None:
            return
        if not entry.published:
            raise ValueError("Cannot delete an in-flight write")
        if entry.pins:
            entry.delete_pending = True
        else:
            self._free.append(self._retire(key))

    def _retire(self, key):
        entry = self._entries.pop(key)
        pending = self._retired.setdefault(entry.publisher, OrderedDict())
        pending.setdefault(key, deque()).append(entry.size)
        return entry.slot

    def take_retired(self, rank):
        """Drain this publisher's notices, up to 32 KiB with default json.dumps."""
        with self._lock:
            pending = self._retired.get(rank)
            batch = []
            wire_bytes = 2  # surrounding []
            while pending:
                key = next(iter(pending))
                sizes = pending[key]
                notice = [key, sizes[0]]
                added = len(json.dumps(notice)) + (2 if batch else 0)
                if wire_bytes + added > 32 * 1024:
                    break
                wire_bytes += added
                batch.append(notice)
                sizes.popleft()
                if not sizes:
                    del pending[key]
            if not pending:
                self._retired.pop(rank, None)
            return batch

    def _status(self, rank):
        pending = locked = pins = 0
        usage = {}
        owners = {str(owner): dict(objects=0, pending_writes=0, used_bytes=0,
                                   free_slots=self._slots_per_owner,
                                   capacity_bytes=self._slots_per_owner * self._slot_size)
                  for owner in self._owner_ranks}
        for entry in self._entries.values():
            if owners:
                owner = str(self._owner_ranks[entry.slot % len(self._owner_ranks)])
                stats = owners[owner]
                stats['free_slots'] -= 1
                if entry.published:
                    stats['objects'] += 1
                    stats['used_bytes'] += entry.size
                else:
                    stats['pending_writes'] += 1
            if not entry.published:
                pending += 1
                continue
            locked += bool(entry.pins)
            pins += sum(entry.pins.values())
            usage[entry.cache_salt] = usage.get(entry.cache_salt, 0) + entry.size
        status = {
            "objects": len(self._entries) - pending,
            "free_slots": len(self._free),
            "locked_objects": locked,
            "total_pins": pins,
            "evicted_objects": self._evicted,
            "pending_writes": pending,
            "total_bytes_used": sum(usage.values()),
            "bytes_by_cache_salt": usage,
        }
        if owners:
            status['owners'] = owners
        return status
