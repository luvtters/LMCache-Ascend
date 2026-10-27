# SPDX-License-Identifier: Apache-2.0
"""Opt-in TP1 worker data plane and scheduler control plane for MemFabric.

The server-module RPC carries JSON metadata only. No KV tensor is serialized,
registered as CPU L1, or passed through shared host staging.
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import uuid

from lmcache.utils import EngineType
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.protocols.server_module import ServerModuleCallRequest
from lmcache.v1.multiprocess.transfer_context.worker_transfer import TransferContext

def chunk_keys(key, tokens_per_chunk):
    """Identify chunks by full token prefix, model and salt, never request ID.

    ``request_configs`` is request metadata, not cache identity, matching
    IPCCacheServerKey. Layout compatibility is checked before any pool copy.
    """
    if (key.world_size != 1 or key.worker_id not in (None, 0)
            or tokens_per_chunk <= 0 or key.start < 0
            or key.start % tokens_per_chunk or key.end < key.start
            or key.end % tokens_per_chunk
            or key.end * 4 > len(key.token_bytes)):
        raise ValueError("Direct keys require TP1 and an aligned valid token range")
    namespace = json.dumps([key.model_name, key.world_size, key.cache_salt,
                            tokens_per_chunk], ensure_ascii=True,
                           separators=(",", ":")).encode()
    result = []
    for end in range(key.start + tokens_per_chunk,
                     key.end + 1, tokens_per_chunk):
        digest = hashlib.blake2b(digest_size=32)
        digest.update(len(namespace).to_bytes(8, "big"))
        digest.update(namespace)
        digest.update(key.token_bytes[:end * 4])
        result.append(digest.hexdigest())
    return result


def _validate_strategy(strategy):
    if (strategy.tp_size != 1 or strategy.pp_size != 1
            or strategy.dcp_size != 1 or strategy.n_servers != 1
            or strategy.vllm_world_size != 1 or strategy.mla_only):
        raise ValueError("MemFabric direct supports TP1/PP1/DCP1 GQA and one service only")


class _ControlRejectedError(RuntimeError):
    """The pool explicitly rejected a metadata operation without applying it."""


class _Control:
    def __init__(self, client, timeout):
        self.client, self.timeout = client, timeout

    def submit(self, method, payload):
        return self.client.server_module_call(ServerModuleCallRequest(
            method="memfabric_direct." + method,
            payload=json.dumps(payload, separators=(",", ":")).encode()))

    @staticmethod
    def decode(response):
        if not response.success:
            raise RuntimeError(response.error or "MemFabric control RPC failed")
        payload = json.loads(response.payload)
        if isinstance(payload, dict) and payload.get("ok") is False:
            raise _ControlRejectedError(payload.get("error", "MemFabric operation rejected"))
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise RuntimeError(payload.get("error", "Invalid MemFabric control reply")
                               if isinstance(payload, dict) else "Invalid control reply")
        return payload

    def call(self, method, payload):
        return self.decode(self.submit(method, payload).result(timeout=self.timeout))


def _done(value):
    future = MessagingFuture()
    future.set_result(value)
    return future


def _failed(error):
    future = MessagingFuture()
    future.set_exception(error)
    return future


class MemFabricDirectTransferContext(TransferContext):
    def __init__(self, *, instance_id, req_client, tokens_per_chunk,
                 mq_timeout, extra_config, session_factory=None,
                 transfer_factory=None, **_kwargs):
        super().__init__(instance_id, req_client)
        self.control = _Control(req_client, mq_timeout)
        self.tokens_per_chunk = tokens_per_chunk
        self.config = dict(extra_config.get("lmcache.mp.direct", {}))
        self.member_id = self.config.pop("member_id", f"worker-{instance_id}")
        self.session = None
        self.transfer = None
        self.registration = None
        self.poisoned = False
        self._session_factory = session_factory
        self._transfer_factory = transfer_factory
        self._owner = ThreadPoolExecutor(max_workers=1,
                                         thread_name_prefix="memfabric-worker-owner")
        self._dispatch_lock = threading.RLock()
        self._writes = {}
        self._read_leases = {}
        self._prepares = []
        self._acquires = []
        self._commits = {}
        self._aborts = {}
        self._releases = []

    def _run(self, operation, *args, **kwargs):
        with self._dispatch_lock:
            if self._is_closed():
                raise RuntimeError("Direct context is closed")
            return self._owner.submit(operation, *args, **kwargs).result()

    def register(self, kv_caches, model_name, world_size, blocks_in_chunk,
                 mq_timeout, layout_hints=None, engine_group_infos=(),
                 engine_type=EngineType.VLLM, lmcache_chunk_tokens=None):
        """Initialize BM, NPU scratch and registration on one owner thread."""
        return self._run(self._register, kv_caches, model_name, world_size,
                         blocks_in_chunk, mq_timeout, layout_hints,
                         engine_group_infos, engine_type, lmcache_chunk_tokens)

    def _register(self, kv_caches, model_name, world_size, blocks_in_chunk,
                  mq_timeout, layout_hints, engine_group_infos, engine_type,
                  lmcache_chunk_tokens):
        if self._is_closed() or self.session is not None:
            raise RuntimeError("Direct context is closed or already registered")
        if world_size != 1 or len(engine_group_infos) > 1:
            raise ValueError("Direct transfer supports one full-attention TP1 group")
        if lmcache_chunk_tokens not in (None, self.tokens_per_chunk):
            raise ValueError("Direct chunk token count differs from adapter")
        required = {"device_id", "rank_id", "world_size", "pool_gb", "store_url"}
        if set(self.config) != required:
            raise ValueError(f"Direct BM config requires exactly {sorted(required)}")
        if self.config["rank_id"] == 0:
            raise ValueError("Direct worker must use a nonzero BM client rank")
        if self._transfer_factory is None:
            from lmcache_ascend.v1.distributed.memfabric_direct_transfer import NpuDirectTransfer

            self._transfer_factory = NpuDirectTransfer
        if self._session_factory is None:
            from lmcache_ascend.v1.distributed.memfabric_bm import BMSession

            self._session_factory = BMSession
        self.transfer = self._transfer_factory(kv_caches, self.tokens_per_chunk,
            blocks_in_chunk, layout_hints=layout_hints, engine_type=engine_type)
        if self.transfer.device.index != self.config["device_id"]:
            raise ValueError("BM device_id must match the KV tensor logical NPU device")
        self.model_name = model_name
        self.session = self._session_factory(**self.config)
        try:
            self.registration = self.control.call("register", {
                "rank": self.config["rank_id"], "member_id": self.member_id,
                "shard": 0, "tp_size": 1})
            deadline = time.monotonic() + mq_timeout
            while not (self.registration["registration_ready"]
                       and self.registration["transfer_ready"]):
                if self.registration["last_error"] or self.registration["closed"]:
                    raise RuntimeError(self.registration["last_error"]
                                       or "MemFabric pool closed during registration")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("MemFabric fixed pool group did not become ready")
                time.sleep(min(0.01, remaining))
                self.registration = self.control.decode(self.control.submit(
                    "status", {}).result(timeout=remaining))
            if self.transfer.size > self.registration["slot_size"]:
                raise ValueError("NPU KV chunk does not fit one MemFabric slot")
            self.base = self.session.peer_address(self.registration["storage_rank"])
        except Exception:
            self.session.close()
            self.session = None
            self.registration = None
            raise

    def _identity(self):
        if self._is_closed() or self.registration is None or self.poisoned:
            raise RuntimeError("Direct context is unregistered, closed or poisoned")
        return {"epoch": self.registration["epoch"],
                "rank": self.config["rank_id"], "member_id": self.member_id}

    def _address(self, descriptor, key):
        expected = self.registration
        if (descriptor["epoch"] != expected["epoch"]
                or descriptor["storage_rank"] != expected["storage_rank"]
                or descriptor["key"] != key or descriptor["shard"] != 0
                or descriptor["tp_size"] != 1 or descriptor["size"] != self.transfer.size
                or descriptor.get("layout") != self.transfer.layout
                or type(descriptor["offset"]) is not int
                or not 0 <= descriptor["offset"] <= expected["pool_bytes"] - descriptor["size"]):
            raise ValueError("MemFabric descriptor does not match this pool and KV chunk")
        return self.base + descriptor["offset"]

    def _blocks(self, block_ids, blocks_in_chunk, key):
        self._identity()
        if (len(block_ids) != 1 or blocks_in_chunk != self.transfer.blocks_per_chunk
                or key.model_name != self.model_name):
            raise ValueError("Direct transfer requires the registered single group layout")
        keys = chunk_keys(key, self.tokens_per_chunk)
        if len(block_ids[0]) != len(keys) * blocks_in_chunk:
            raise ValueError("Direct block IDs and token key must cover the same full chunks")
        return keys, [block_ids[0][i * blocks_in_chunk:(i + 1) * blocks_in_chunk]
                      for i in range(len(keys))]

    def create_recorded_event(self):
        """Use device synchronization rather than exporting an IPC event."""
        self._run(self._identity)
        return None

    def submit_store(self, request_id, key, kv_caches, block_ids, event,
                     blocks_in_chunk):
        """Finish gather and synchronous L2G before reporting store completion."""
        try:
            return self._run(self._store, key, block_ids, event, blocks_in_chunk)
        except Exception as error:
            self.poisoned = True
            return _failed(error)

    def _store(self, key, block_ids, event, blocks_in_chunk):
        keys, chunks = self._blocks(block_ids, blocks_in_chunk, key)
        if event is not None:
            event.wait()
        self.transfer.synchronize()  # order after engine KV production
        for object_key, ids in zip(keys, chunks):
            identity = self._identity()
            future = self.control.submit("prepare_store", dict(identity,
                key=object_key, shard=0, tp_size=1, size=self.transfer.size,
                layout=self.transfer.layout))
            pending = (identity, future)
            self._prepares.append(pending)
            response = future.result(timeout=self.control.timeout)
            self._prepares.remove(pending)
            reservation = self.control.decode(response)
            write_id = reservation.get("write_id")
            descriptor = reservation["descriptor"]
            if write_id is None:
                self._address(descriptor, object_key)
                continue
            self._writes[write_id] = identity
            dma_complete = True
            try:
                address = self._address(descriptor, object_key)
                self.transfer.gather(ids)
                dma_complete = False
                self.session.copy_npu_to_global(self.transfer.buffer.data_ptr(),
                                                address, self.transfer.size)
                dma_complete = True
                future = self.control.submit("commit_store", dict(identity, write_id=write_id))
                self._commits[write_id] = future
                response = future.result(timeout=self.control.timeout)
                try:
                    self.control.decode(response)
                except Exception:
                    del self._commits[write_id]
                    raise
                del self._commits[write_id]
                del self._writes[write_id]
            except Exception:
                self.poisoned = True  # retain NPU scratch until all work drains
                if write_id not in self._commits:
                    self._abort_write(write_id, identity, dma_complete)
                raise
        return _done(True)

    def submit_retrieve(self, request_id, key, kv_caches, block_ids, event,
                        blocks_in_chunk, skip_first_n_tokens=0):
        """Finish G2L and paged-KV scatter before reporting retrieval success."""
        try:
            return self._run(self._retrieve, key, block_ids, event,
                             blocks_in_chunk, skip_first_n_tokens)
        except Exception as error:
            self.poisoned = True
            return _failed(error)

    def _retrieve(self, key, block_ids, event, blocks_in_chunk,
                  skip_first_n_tokens):
        keys, chunks = self._blocks(block_ids, blocks_in_chunk, key)
        if (skip_first_n_tokens < 0
                or skip_first_n_tokens % self.transfer.block_size):
            raise ValueError("Direct retrieve prefix must be block aligned")
        if event is not None:
            event.wait()
        self.transfer.synchronize()
        descriptors = self.control.call("lookup", {
            "keys": keys, "shard": 0, "tp_size": 1})["descriptors"]
        if len(descriptors) != len(keys):
            return _done(False)
        for index, (object_key, ids, descriptor) in enumerate(zip(keys, chunks, descriptors)):
            identity = self._identity()
            future = self.control.submit("acquire", dict(identity, descriptor=descriptor))
            pending = (identity, future)
            self._acquires.append(pending)
            response = future.result(timeout=self.control.timeout)
            self._acquires.remove(pending)
            acquired = self.control.decode(response)
            lease_id = acquired["lease_id"]
            self._read_leases[lease_id] = identity
            dma_complete = True
            try:
                address = self._address(acquired["descriptor"], object_key)
                dma_complete = False
                self.session.copy_global_to_npu(address,
                    self.transfer.buffer.data_ptr(), self.transfer.size)
                dma_complete = True
                skip = min(self.tokens_per_chunk, max(0,
                    skip_first_n_tokens - index * self.tokens_per_chunk))
                self.transfer.scatter(ids, skip_tokens=skip)
            except Exception:
                self.poisoned = True
                raise
            finally:
                if dma_complete:
                    self._release_read(lease_id, identity)
        return _done(True)

    def unregister(self):
        """Keep BM membership until close has drained device work."""
        return None  # BM membership leaves only after close has drained DMA.

    def flush_inflight_stores(self):
        """Synchronize scratch on its owner thread before worker cleanup."""
        self._run(self._synchronize)

    def _synchronize(self):
        if self.transfer is not None:
            self.transfer.synchronize()

    def _release_read(self, lease_id, identity):
        future = self.control.submit("release", dict(identity, lease_id=lease_id,
                                                    dma_complete=True))
        pending = (lease_id, future)
        self._releases.append(pending)
        response = future.result(timeout=self.control.timeout)
        try:
            self.control.decode(response)
        except _ControlRejectedError:
            self._releases.remove(pending)
            raise
        self._releases.remove(pending)
        del self._read_leases[lease_id]

    def _abort_write(self, write_id, identity, dma_complete):
        future = self.control.submit("abort_store", dict(identity,
            write_id=write_id, dma_complete=dma_complete))
        self._aborts[write_id] = (future, dma_complete)
        response = future.result(timeout=self.control.timeout)
        try:
            self.control.decode(response)
        except _ControlRejectedError:
            del self._aborts[write_id]
            raise
        del self._aborts[write_id]
        if dma_complete:
            del self._writes[write_id]

    def _close_owned(self):
        self.poisoned = True
        self._synchronize()
        if self.session is not None:
            self.session.wait()
        # Timed-out control operations retain their futures. Resolve their
        # actual result before releasing reservations or read protection.
        for identity, future in list(self._prepares):
            reply = future.result(timeout=self.control.timeout)
            try:
                response = self.control.decode(reply)
            except _ControlRejectedError:
                self._prepares.remove((identity, future))
                continue
            if response["write_id"] is not None:
                self._writes[response["write_id"]] = identity
            self._prepares.remove((identity, future))
        for identity, future in list(self._acquires):
            reply = future.result(timeout=self.control.timeout)
            try:
                response = self.control.decode(reply)
            except _ControlRejectedError:
                self._acquires.remove((identity, future))
                continue
            self._read_leases[response["lease_id"]] = identity
            self._acquires.remove((identity, future))
        for write_id, future in list(self._commits.items()):
            reply = future.result(timeout=self.control.timeout)
            try:
                self.control.decode(reply)
            except _ControlRejectedError:
                # Commit rejection leaves the write reservation in the pool.
                # The owner has now synchronized NPU work and waited for BM,
                # so the normal abort below can safely reclaim that slot.
                del self._commits[write_id]
                continue
            del self._commits[write_id]
            del self._writes[write_id]
        for write_id, (future, dma_complete) in list(self._aborts.items()):
            reply = future.result(timeout=self.control.timeout)
            try:
                self.control.decode(reply)
            except _ControlRejectedError:
                # A rejected abort did not release the reservation. Keep the
                # write tracked and retry a real abort after draining replies.
                del self._aborts[write_id]
                continue
            del self._aborts[write_id]
            if dma_complete:
                del self._writes[write_id]
        for lease_id, future in list(self._releases):
            reply = future.result(timeout=self.control.timeout)
            try:
                self.control.decode(reply)
            except _ControlRejectedError:
                # Rejected release is not evidence that the lease is gone.
                self._releases.remove((lease_id, future))
                continue
            self._releases.remove((lease_id, future))
            del self._read_leases[lease_id]
        for write_id, identity in list(self._writes.items()):
            self._abort_write(write_id, identity, True)
        for lease_id, identity in list(self._read_leases.items()):
            self._release_read(lease_id, identity)
        if self.session is not None:
            self.session.close()
            self.session = None
        self.transfer = None

    def close(self):
        """Drain and release on the BM owner; a failed drain retains resources."""
        with self._dispatch_lock:
            if self._is_closed():
                return
            self._owner.submit(self._close_owned).result()
            self._mark_closed()
            self._owner.shutdown(wait=True)


class _Lookup:
    def __init__(self, backend, key, timeout, blocking_lookup):
        self.backend, self.key = backend, key
        self.timeout, self.blocking = timeout, blocking_lookup
        self.request_id = uuid.uuid4().hex
        self.keys = chunk_keys(key, backend.tokens_per_chunk)
        self.future = None
        self.result = None
        self.deadline = 0
        self.lock = threading.Lock()

    def submit(self):
        with self.lock:
            if self.future is not None:
                raise RuntimeError("Direct lookup was already submitted")
            self.deadline = time.monotonic() + self.timeout
            self.future = self.backend.control.submit("lookup", {
                "keys": self.keys, "shard": 0, "tp_size": 1,
                "request_id": self.request_id})
            self.backend.lookups.setdefault(self.key.request_id, []).append(self)

    def on_scheduler_check(self):
        with self.lock:
            if self.result is not None or self.future is None:
                return []
            if self.blocking:
                self.future.wait(timeout=max(0, self.deadline - time.monotonic()))
            if self.future.query():
                payload = self.backend.control.decode(self.future.result(timeout=0))
                self.epoch = payload["epoch"]
                self.result = self.key.start + len(payload["descriptors"]) * self.backend.tokens_per_chunk
                return []
            if time.monotonic() >= self.deadline:
                self.result = 0
                return [self.backend.url]
            return []

    def get_result(self):
        return self.result

    def close(self):
        with self.lock:
            if self.result is None:
                self.result = 0

    def drain(self):
        self.close()
        if self.future is not None and not self.future.wait(
                timeout=self.timeout):
            return self.backend.url
        return None


class MemFabricDirectSchedulerBackend:
    def __init__(self, *, req_clients, parallel_strategy, tokens_per_chunk,
                 mq_timeout, **_kwargs):
        _validate_strategy(parallel_strategy)
        if len(req_clients) != 1:
            raise ValueError("MemFabric direct requires exactly one service")
        self.url, client = next(iter(req_clients.items()))
        self.control = _Control(client, mq_timeout)
        self.tokens_per_chunk = tokens_per_chunk
        self.lookups = {}

    def create_lookup(self, key, *, tp_size, tokens_per_chunk, timeout,
                      blocking_lookup):
        if tp_size != 1 or tokens_per_chunk != self.tokens_per_chunk:
            raise ValueError("Direct lookup topology or chunk size changed")
        return _Lookup(self, key, timeout, blocking_lookup)

    def free_lookup_locks(self, key, tp_size, extra_ranges=()):
        if tp_size != 1:
            raise ValueError("Direct lookup locks require TP1")
        ranges = ((key.start, key.end),) + tuple(extra_ranges)
        for lookup in self.lookups.get(key.request_id, ()):
            # Drain before releasing: a late lookup reply must not recreate
            # pins after a request was cancelled or skipped by engine APC.
            if lookup.drain() is not None:
                raise TimeoutError("Direct lookup did not drain before lock release")
            response = self.control.decode(lookup.future.result(timeout=0))
            selected = []
            for index, object_key in enumerate(lookup.keys):
                start = lookup.key.start + index * self.tokens_per_chunk
                if any(start >= lo // self.tokens_per_chunk * self.tokens_per_chunk
                       and start + self.tokens_per_chunk <= hi for lo, hi in ranges):
                    selected.append(object_key)
            if selected:
                self.control.call("release", {"epoch": response["epoch"],
                    "request_id": lookup.request_id, "dma_complete": True,
                    "keys": selected})

    def end_session(self, request_id):
        for lookup in self.lookups.get(request_id, ()):
            if lookup.drain() is not None:
                raise TimeoutError("Direct lookup did not drain before session release")
            response = self.control.decode(lookup.future.result(timeout=0))
            self.control.call("release", {"epoch": response["epoch"],
                "request_id": lookup.request_id, "dma_complete": True})
        self.lookups.pop(request_id, None)

    def reset_cache(self):
        # invalidate poisons fixed membership; it is not a cache flush.
        return False

    def close(self):
        for request_id in list(self.lookups):
            self.end_session(request_id)


def create_worker_transfer_context(*, parallel_strategy, **kwargs):
    _validate_strategy(parallel_strategy)
    return MemFabricDirectTransferContext(**kwargs)


def create_scheduler_backend(**kwargs):
    return MemFabricDirectSchedulerBackend(**kwargs)
