# SPDX-License-Identifier: Apache-2.0
"""MemFabric HOST backing for native L1, with metadata-only engine RPCs.

The service owns the BM arena and native L1 object lifecycle. Workers gather KV
on NPU and copy directly to these allocations. No second CPU pool or separate
cache index is introduced. This first integration deliberately supports one
owner and one full-attention KV group; it fails closed on unsupported layouts.
"""
from collections import OrderedDict
from dataclasses import dataclass
import json
import threading
import uuid

from lmcache.logging import init_logger
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.multiprocess.custom_types import PrepareRetrieveResponse, PrepareStoreResponse
from lmcache.v1.multiprocess.modules.server_transfer import TransferStrategy

from .memfabric_bm import BMSession
from .memfabric_l1_memory import MemFabricL1MemoryManager

logger = init_logger(__name__)


def create_runtime(storage_config, config):
    return MemFabricL1Runtime(storage_config, config)


class MemFabricL1Runtime:
    def __init__(self, storage_config, config):
        if len(storage_config.l1_manager_configs) != 1:
            raise ValueError("MemFabric L1 currently requires exactly one native L1 owner")
        if storage_config.l2_adapter_config.adapters:
            raise ValueError("MemFabric L1 initial integration requires L2 disabled")
        self.session_id = config.get("session_id")
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("MemFabric L1 needs an explicit session_id")
        bm = dict(config["bm"])
        if bm.get("rank_id") != 0 or bm.get("pool_memory_type", "host") != "host":
            raise ValueError("MemFabric L1 service must own BM rank 0 HOST memory")
        self.session = BMSession(**bm)
        self.memory = None
        self.strategies = []
        self.registration_metadata = {
            "transport": "memfabric_l1", "version": 1,
            "session_id": self.session_id, "owner_rank": 0,
            "pool_base": self.session.peer_address(0),
            "pool_size": self.session.local_dram_bytes,
        }
        logger.info("MEMFABRIC_L1_RUNTIME session=%s rank=0 world=%s transport=%s",
                    self.session_id, bm["world_size"], bm.get("data_op_type", "SDMA"))

    def create_memory_manager(self, config):
        if self.memory is not None:
            raise RuntimeError("MemFabric arena may only be bound to one L1 manager")
        self.memory = MemFabricL1MemoryManager(
            self.session, size_bytes=config.memory_config.size_in_bytes
        )
        return self.memory

    def create_transfer_strategy(self, storage_manager, payload, metadata, **kwargs):
        if payload.world_size != 1:
            raise ValueError("MemFabric L1 model integration currently admits TP1 only")
        if metadata.use_mla or (metadata.group_layouts and len(metadata.group_layouts) > 1):
            raise ValueError("MemFabric L1 requires one full-attention K/V group")
        if metadata.group_windows and any(window != -1 for window in metadata.group_windows):
            raise ValueError("MemFabric L1 windowed groups are not implemented")
        strategy = MemFabricL1Strategy(
            storage_manager._l1_manager, self.memory, payload.instance_id, self.session_id
        )
        self.strategies.append(strategy)
        return strategy

    def close(self):
        if any(strategy.has_pending_io for strategy in self.strategies):
            raise RuntimeError("Unconfirmed worker I/O retains the MemFabric L1 session")
        if self.memory is not None and self.memory.get_memory_usage()[0]:
            raise RuntimeError("Native L1 objects must close before BM session")
        self.session.close()
        logger.info("MEMFABRIC_L1_RUNTIME_CLOSED session=%s", self.session_id)


@dataclass
class _Operation:
    operation_id: str
    keys: list
    slots: list


class MemFabricL1Strategy(TransferStrategy):
    """Native reservations with identity-bound pins, never KV bytes over RPC."""
    def __init__(self, manager, memory, instance_id, session_id):
        self.manager, self.memory = manager, memory
        self.instance_id, self.session_id = instance_id, session_id
        self._lock = threading.RLock()
        self._pending = {}
        self._completed = OrderedDict()
        self._released = False

    @property
    def has_pending_io(self):
        with self._lock:
            return bool(self._pending)

    def _identity(self, kind, key, instance_id):
        if instance_id != self.instance_id or not key.request_id:
            raise ValueError("MemFabric transfer requires its registered worker and request ID")
        # Cache-key equality deliberately excludes request_id. Transfer identity
        # must include it so equal prompts cannot steal each other's operation.
        return kind, instance_id, key.request_id, key

    def _slot(self, obj, chunk_index):
        tensor = obj.tensor
        if tensor is None or not tensor.is_contiguous():
            raise ValueError("MemFabric L1 requires a contiguous native object")
        return {"address": self.memory.global_address(obj), "size": tensor.numel() * tensor.element_size(),
                "chunk_index": chunk_index, "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).split(".")[-1]}

    @staticmethod
    def _context(operation):
        return {"transport": "memfabric_l1", "version": 1,
                "operation_id": operation.operation_id, "slots": operation.slots}

    def _remember(self, identity, token):
        self._completed[identity] = token
        self._completed.move_to_end(identity)
        while len(self._completed) > 128:
            self._completed.popitem(last=False)

    def _check_open(self):
        if self._released:
            raise RuntimeError("Worker context has been released; pending I/O is quarantined")

    def prepare_store(self, key, instance_id, context, resolve_obj_keys,
                      group_keys=None, skipped_chunks=()):
        if group_keys is not None or any(skipped_chunks):
            raise ValueError("MemFabric L1 grouped store is not implemented")
        identity = self._identity("write", key, instance_id)
        with self._lock:
            self._check_open()
            if identity in self._pending:
                return PrepareStoreResponse(context=self._context(self._pending[identity]))
            keys = resolve_obj_keys(key)
            token = uuid.uuid4().hex
            results = self.manager.reserve_write_io(
                keys, [False] * len(keys), context.layout_desc, token,
                tag="memfabric_l1",
            )
            # Once reserved, record ownership before translating descriptors:
            # any subsequent exception must retain/quarantine these allocations.
            selected = [k for k in keys if results[k][1] is not None]
            operation = _Operation(token, selected, [])
            if selected:
                self._pending[identity] = operation
            for index, k in enumerate(keys):
                error, obj = results[k]
                if obj is not None:
                    operation.slots.append(self._slot(obj, index))
                elif error == L1Error.OUT_OF_MEMORY:
                    # Native store is best effort; reserve misses remain cache
                    # misses. A partial store is not reported as full coverage.
                    logger.warning("MEMFABRIC_L1_STORE_CAPACITY request=%s chunk=%d", key.request_id, index)
            logger.info("MEMFABRIC_L1_PREPARE_STORE request=%s instance=%s operation=%s objects=%d bytes=%d",
                        key.request_id, instance_id, token, len(selected), sum(s["size"] for s in operation.slots))
            return PrepareStoreResponse(context=self._context(operation))

    def commit_store(self, key, instance_id, cpu_data, context, resolve_obj_keys, group_keys=None):
        identity = self._identity("write", key, instance_id)
        if not cpu_data or len(cpu_data) > 1024:
            raise ValueError("MemFabric commit carries bounded completion metadata, never KV payload")
        completion = json.loads(cpu_data)
        token = completion.get("operation_id")
        with self._lock:
            self._check_open()
            operation = self._pending.get(identity)
            if operation is None:
                return self._completed.get(identity) == token and token is not None
            expected_bytes = sum(slot["size"] for slot in operation.slots)
            if token != operation.operation_id or completion.get("bytes") != expected_bytes:
                raise ValueError("MemFabric write completion does not match prepared operation")
            result = self.manager.finish_write_io(operation.keys, token, tag="memfabric_l1")
            if any(error != L1Error.SUCCESS for error in result.values()):
                raise RuntimeError("Native L1 rejected MemFabric write completion")
            del self._pending[identity]
            self._remember(identity, token)
            logger.info("MEMFABRIC_L1_COMMIT_STORE request=%s instance=%s operation=%s objects=%d completed_bytes=%d",
                        key.request_id, instance_id, token, len(operation.keys), expected_bytes)
            return True

    def prepare_retrieve(self, key, instance_id, resolve_obj_keys, group_keys=None, group_first_chunk=()):
        if group_keys is not None or any(group_first_chunk):
            raise ValueError("MemFabric L1 grouped retrieve is not implemented")
        identity = self._identity("read", key, instance_id)
        with self._lock:
            self._check_open()
            if identity in self._pending:
                return PrepareRetrieveResponse(success=True, context=self._context(self._pending[identity]))
            keys = resolve_obj_keys(key)
            token = uuid.uuid4().hex
            results = self.manager.read_prefetched_io(keys, token)
            selected = [k for k in keys if results[k][1] is not None]
            if len(selected) != len(keys):
                self.manager.finish_read_io(selected, token)
                logger.info("MEMFABRIC_L1_RETRIEVE_MISS request=%s objects=%d/%d", key.request_id, len(selected), len(keys))
                return PrepareRetrieveResponse(success=False)
            operation = _Operation(token, selected, [])
            self._pending[identity] = operation
            operation.slots = [self._slot(results[k][1], i) for i, k in enumerate(keys)]
            logger.info("MEMFABRIC_L1_PREPARE_RETRIEVE request=%s instance=%s operation=%s source=L1 objects=%d bytes=%d",
                        key.request_id, instance_id, token, len(selected), sum(s["size"] for s in operation.slots))
            return PrepareRetrieveResponse(success=True, context=self._context(operation))

    def commit_retrieve(self, key, instance_id):
        identity = self._identity("read", key, instance_id)
        token = (key.request_configs or {}).get("_memfabric_l1_operation_id")
        with self._lock:
            self._check_open()
            operation = self._pending.get(identity)
            if operation is None:
                return self._completed.get(identity) == token and token is not None
            if token != operation.operation_id:
                raise ValueError("MemFabric read completion does not match prepared operation")
            result = self.manager.finish_read_io(operation.keys, token)
            if any(error != L1Error.SUCCESS for error in result.values()):
                raise RuntimeError("Native L1 rejected MemFabric read completion")
            del self._pending[identity]
            self._remember(identity, token)
            logger.info("MEMFABRIC_L1_COMMIT_RETRIEVE request=%s instance=%s operation=%s objects=%d completed_bytes=%d",
                        key.request_id, instance_id, token, len(operation.keys), sum(s["size"] for s in operation.slots))
            return True

    def release(self):
        with self._lock:
            self._released = True
            if self._pending:
                logger.error("MEMFABRIC_L1_QUARANTINE instance=%s operations=%d; completion unknown, retaining objects",
                             self.instance_id, len(self._pending))
