# SPDX-License-Identifier: Apache-2.0
"""Worker-side native L1 transfer over MemFabric BM slots."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import threading
from typing import Any

import torch

from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPWorkerAdapter,
)
from lmcache.v1.multiprocess.custom_types import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
)
from lmcache.v1.multiprocess.transfer_context.base import (
    EngineDrivenContext,
    EngineDrivenContextMetadata,
)
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
    _get_kv_device,
    _single_group_block_ids,
    TransferContext,
)
from lmcache.v1.multiprocess.transfer_context.base import (
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.platform import synchronize_device
from lmcache.utils import init_logger

logger = init_logger(__name__)


_TRANSPORT = "memfabric_l1"
_VERSION = 1
_RETRIEVE_TOKEN = "_memfabric_l1_operation_id"


class BMAffineWorker:
    """Own a BMSession and every SDK call on one dedicated thread."""

    _uncertain_workers: list["BMAffineWorker"] = []

    def __init__(self, session_config: dict[str, Any]):
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="memfabric-bm-worker"
        )
        self._closed = False
        self.unregister_acknowledged = False
        self._poisoned = False
        self._scratch_cache = {}
        self._scratch_lock = threading.Lock()
        self.session_id = session_config["session_id"]
        self.device_id = session_config["device_id"]
        try:
            self._session = self._submit(self._open, dict(session_config))
        except BaseException:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=False)
            raise

    @staticmethod
    def _open(config: dict[str, Any]):
        from lmcache_ascend.v1.distributed.memfabric_bm import BMSession

        config.pop("session_id", None)
        return BMSession(**config)

    def _submit(self, fn, *args):
        if self._closed:
            raise RuntimeError("MemFabric BM worker is closed")
        return self._executor.submit(fn, *args).result()

    def call(self, fn, *args):
        """Run a callable with the session on its owning thread."""
        return self._submit(fn, self._session, *args)

    @staticmethod
    def _register(session, tensor):
        session.register_npu_tensor(tensor)

    @staticmethod
    def _copy_to_pool(session, tensor, address: int, size: int):
        session.copy_npu_to_global(tensor.data_ptr(), address, size)

    @staticmethod
    def _copy_from_pool(session, address: int, tensor, size: int):
        session.copy_global_to_npu(address, tensor.data_ptr(), size)

    def register(self, tensor: torch.Tensor) -> None:
        self.call(self._register, tensor)

    def scratch(self, shape, dtype, position: int) -> torch.Tensor:
        """Return a retained per-slot NPU scratch tensor, allocating once."""
        if self._poisoned:
            raise RuntimeError("MemFabric worker is poisoned by uncertain completion")
        key = (tuple(shape), dtype, position)
        with self._scratch_lock:
            tensor = self._scratch_cache.get(key)
            if tensor is None:
                tensor = torch.empty(
                    shape, dtype=dtype, device=f"npu:{self.device_id}"
                )
                self.register(tensor)
                self._scratch_cache[key] = tensor
            return tensor

    def mark_uncertain(self) -> None:
        self._poisoned = True

    def copy_to_pool(self, tensor: torch.Tensor, address: int, size: int) -> None:
        self.call(self._copy_to_pool, tensor, address, size)

    def copy_from_pool(self, address: int, tensor: torch.Tensor, size: int) -> None:
        self.call(self._copy_from_pool, address, tensor, size)

    @staticmethod
    def _close(session):
        session.close()

    def close(self) -> None:
        if self._closed:
            return
        if self._poisoned:
            BMAffineWorker._uncertain_workers.append(self)
            logger.warning(
                "Keeping MemFabric BM worker alive after uncertain device completion"
            )
            return
        try:
            self._submit(self._close, self._session)
        except BaseException:
            self._poisoned = True
            BMAffineWorker._uncertain_workers.append(self)
            raise
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)


def _key_identity(instance_id: int, key: Any) -> tuple[int, str, Any]:
    return instance_id, key.request_id, key


def _operation_metadata(operation_id: str, byte_count: int) -> bytes:
    return json.dumps(
        {"operation_id": operation_id, "bytes": byte_count},
        separators=(",", ":"),
    ).encode("utf-8")


class MemFabricL1EngineDrivenContext(EngineDrivenContext):
    """Use service-negotiated BM GVAs as native L1 transfer slots."""

    def __init__(self, metadata, req_client, mq_timeout, bm_worker, pool_base, pool_size):
        super().__init__(metadata, req_client, mq_timeout)
        self._bm_worker = bm_worker
        self._expected_layout = metadata.layout_desc
        self._pool_base = pool_base
        self._pool_size = pool_size
        self._store_ops: dict[tuple[int, str, Any], tuple[str, list[torch.Tensor], list[dict]]]
        self._store_ops = {}
        self._retrieve_ops: dict[tuple[int, str, Any], tuple[str, list[torch.Tensor], list[dict]]]
        self._retrieve_ops = {}
        self._lock = threading.Lock()
        self._closed = False

    def _validate_slots(self, context: Any, expected_count: int | None = None):
        if not isinstance(context, dict):
            raise RuntimeError("MemFabric L1 response is missing slot metadata")
        if context.get("transport") != _TRANSPORT or context.get("version") != _VERSION:
            raise RuntimeError("MemFabric L1 slot transport/version mismatch")
        operation_id = context.get("operation_id")
        slots = context.get("slots")
        if not isinstance(operation_id, str) or not operation_id:
            raise RuntimeError("MemFabric L1 response is missing operation_id")
        if not isinstance(slots, list):
            raise RuntimeError("MemFabric L1 response slots must be a list")
        if expected_count is not None and len(slots) != expected_count:
            raise RuntimeError(
                f"MemFabric L1 returned {len(slots)} slots; expected {expected_count}"
            )
        previous = -1
        for slot in slots:
            if not isinstance(slot, dict):
                raise RuntimeError("MemFabric L1 slot descriptor must be a mapping")
            address, size, chunk_index = (
                slot.get("address"), slot.get("size"), slot.get("chunk_index")
            )
            if (type(address) is not int or address <= 0 or type(size) is not int
                    or size <= 0 or type(chunk_index) is not int
                    or chunk_index <= previous
                    or address < self._pool_base
                    or address + size > self._pool_base + self._pool_size):
                raise RuntimeError("MemFabric L1 returned an invalid slot descriptor")
            previous = chunk_index
        return operation_id, slots

    def _allocate(self, slot: dict, position: int) -> torch.Tensor:
        shape = slot.get("shape")
        dtype_name = slot.get("dtype")
        if not isinstance(shape, list) or not shape or any(
            type(dim) is not int or dim <= 0 for dim in shape
        ):
            raise RuntimeError("MemFabric L1 slot has an invalid tensor shape")
        dtype = getattr(torch, dtype_name, None) if isinstance(dtype_name, str) else None
        if not isinstance(dtype, torch.dtype):
            raise RuntimeError("MemFabric L1 slot has an unsupported dtype")
        if (torch.Size(shape) != self._expected_layout.shapes[0]
                or dtype != self._expected_layout.dtypes[0]):
            raise RuntimeError("MemFabric L1 slot shape/dtype differs from registration")
        tensor = self._bm_worker.scratch(shape, dtype, position)
        size = tensor.numel() * tensor.element_size()
        if size != slot["size"]:
            raise RuntimeError("MemFabric L1 slot byte size does not match its tensor")
        return tensor

    def prepare_store(self, key, instance_id):
        identity = _key_identity(instance_id, key)
        with self._lock:
            if identity in self._store_ops:
                raise RuntimeError("Duplicate unresolved MemFabric L1 store prepare")
        try:
            response = self.req_client.prepare_store(key, instance_id).result(
                timeout=self.mq_timeout
            )
        except TimeoutError as exc:
            self._bm_worker.mark_uncertain()
            raise RuntimeError("MemFabric L1 store prepare timed out with unknown completion") from exc
        if not isinstance(response, PrepareStoreResponse):
            raise RuntimeError("Unexpected MemFabric L1 prepare_store response")
        operation_id, slots = self._validate_slots(response.context)
        if not slots:
            return [], []
        tensors = [self._allocate(slot, index) for index, slot in enumerate(slots)]
        with self._lock:
            self._store_ops[identity] = (operation_id, tensors, slots)
        return tensors, [slot["chunk_index"] for slot in slots]

    def commit_store(self, key, instance_id, chunks):
        identity = _key_identity(instance_id, key)
        with self._lock:
            operation = self._store_ops.get(identity)
            if operation is None:
                return False
            operation_id, tensors, slots = operation
        try:
            if len(chunks) != 0:
                raise RuntimeError("MemFabric L1 commit_store accepts no CPU payload")
            for tensor, slot in zip(tensors, slots, strict=True):
                try:
                    self._bm_worker.copy_to_pool(
                        tensor, slot["address"], slot["size"]
                    )
                    logger.info(
                        "MEMFABRIC_L1_BM_COMPLETE direction=npu_to_global "
                        "request_id=%s operation_id=%s bytes=%d",
                        key.request_id, operation_id, slot["size"],
                    )
                except Exception:
                    self._bm_worker.mark_uncertain()
                    raise
            payload = _operation_metadata(
                operation_id, sum(slot["size"] for slot in slots)
            )
            ok = bool(
                self.req_client.commit_store(key, instance_id, payload).result(
                    timeout=self.mq_timeout
                )
            )
            if not ok:
                raise RuntimeError("MemFabric L1 COMMIT_STORE was rejected")
            with self._lock:
                self._store_ops.pop(identity, None)
            return ok
        except Exception:
            # Keep allocations alive if native copy completion is uncertain.
            raise

    def discard_store_prepare(self, key, instance_id):
        """Forget an all-cached prepare that needs no commit RPC."""
        with self._lock:
            self._store_ops.pop(_key_identity(instance_id, key), None)

    def prepare_retrieve(self, key, instance_id, expected_count):
        identity = _key_identity(instance_id, key)
        with self._lock:
            if identity in self._retrieve_ops:
                raise RuntimeError("Duplicate unresolved MemFabric L1 retrieve prepare")
        try:
            response = self.req_client.prepare_retrieve(key, instance_id).result(
                timeout=self.mq_timeout
            )
        except TimeoutError as exc:
            self._bm_worker.mark_uncertain()
            raise RuntimeError("MemFabric L1 retrieve prepare timed out with unknown completion") from exc
        if not isinstance(response, PrepareRetrieveResponse):
            raise RuntimeError("Unexpected MemFabric L1 prepare_retrieve response")
        if not response.success:
            return None
        operation_id, slots = self._validate_slots(response.context, expected_count)
        if [slot["chunk_index"] for slot in slots] != list(range(expected_count)):
            raise RuntimeError("MemFabric L1 retrieve slots are not full and ordered")
        tensors = [self._allocate(slot, index) for index, slot in enumerate(slots)]
        with self._lock:
            self._retrieve_ops[identity] = (operation_id, tensors, slots)
        try:
            for tensor, slot in zip(tensors, slots, strict=True):
                self._bm_worker.copy_from_pool(
                    slot["address"], tensor, slot["size"]
                )
                logger.info(
                    "MEMFABRIC_L1_BM_COMPLETE direction=global_to_npu "
                    "request_id=%s operation_id=%s bytes=%d",
                    key.request_id, operation_id, slot["size"],
                )
        except Exception:
            self._bm_worker.mark_uncertain()
            raise
        return tensors

    def commit_retrieve(self, key, instance_id):
        identity = _key_identity(instance_id, key)
        with self._lock:
            operation = self._retrieve_ops.get(identity)
            if operation is None:
                return False
            operation_id, _tensors, _slots = operation
        configs = dict(key.request_configs or {})
        configs[_RETRIEVE_TOKEN] = operation_id
        commit_key = replace(key, request_configs=configs)
        try:
            result = bool(
                self.req_client.commit_retrieve(commit_key, instance_id).result(
                    timeout=self.mq_timeout
                )
            )
        except TimeoutError as exc:
            raise RuntimeError(
                "MemFabric L1 COMMIT_RETRIEVE timed out with unknown completion"
            ) from exc
        if not result:
            raise RuntimeError("MemFabric L1 COMMIT_RETRIEVE was rejected")
        with self._lock:
            self._retrieve_ops.pop(identity, None)
        return result

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # Uncertain operations retain their tensors in this context until
            # BM close has attempted its own conservative completion checks.
            pending = list(self._store_ops.values()) + list(self._retrieve_ops.values())
            self._store_ops.clear()
            self._retrieve_ops.clear()
        if pending or self._bm_worker._poisoned or not self._bm_worker.unregister_acknowledged:
            self._uncertain_operations = pending
            BMAffineWorker._uncertain_workers.append(self._bm_worker)
            logger.warning(
                "Keeping MemFabric BM worker alive because a transfer completion "
                "or engine-driven unregistration is uncertain"
            )
            return
        try:
            self._bm_worker.close()
        except Exception:
            # Do not release uncertain tensor references after SDK failure.
            self._uncertain_operations = pending
            raise


class _UnregisterAckFuture:
    """Record whether the server confirmed unregistration before BM close."""

    def __init__(self, transfer_context, future):
        self._transfer_context = transfer_context
        self._future = future

    def result(self, timeout=None):
        try:
            result = self._future.result(timeout=timeout)
        except BaseException:
            self._transfer_context._bm_worker.unregister_acknowledged = False
            raise
        self._transfer_context._bm_worker.unregister_acknowledged = True
        return result


class MemFabricL1TransferContext(EngineDrivenTransferContext):
    """Select the BM slot transport only after an explicit server handshake."""

    def __init__(self, instance_id, req_client, bm_worker):
        super().__init__(instance_id, req_client)
        self._bm_worker = bm_worker
        self._io_lock = threading.Lock()

    def unregister(self):
        future = super().unregister()
        if future is None:
            return None
        return _UnregisterAckFuture(self, future)

    def close(self):
        if self._engine_driven_context is None:
            if self._bm_worker.unregister_acknowledged or not self._registration_request_sent:
                self._bm_worker.close()
            else:
                BMAffineWorker._uncertain_workers.append(self._bm_worker)
                logger.warning(
                    "Keeping MemFabric BM worker alive because registration "
                    "was not confirmed unregistered"
                )
        super().close()

    def register(
        self, kv_caches, model_name, world_size, blocks_in_chunk, mq_timeout,
        layout_hints=None, engine_group_infos=(), engine_type=None,
        lmcache_chunk_tokens=None,
    ):
        if world_size != 1:
            raise NotImplementedError("MemFabric L1 first supports TP1 only")
        if len(engine_group_infos) > 1:
            raise NotImplementedError("MemFabric L1 first supports one KV group")
        kwargs = {
            "layout_hints": layout_hints,
            "engine_group_infos": engine_group_infos,
            "lmcache_chunk_tokens": lmcache_chunk_tokens,
        }
        if engine_type is not None:
            kwargs["engine_type"] = engine_type
        super().register(
            kv_caches, model_name, world_size, blocks_in_chunk, mq_timeout, **kwargs
        )

    def _create_engine_driven_context(self, metadata, response, mq_timeout):
        transport = getattr(response, "transport_metadata", {})
        if not isinstance(transport, dict) or transport.get("transport") != _TRANSPORT:
            raise RuntimeError("Server did not negotiate MemFabric L1 transport")
        if (transport.get("version") != _VERSION
                or transport.get("session_id") != self._bm_worker.session_id
                or transport.get("owner_rank") != 0):
            raise RuntimeError("Server returned incompatible MemFabric L1 metadata")
        pool_base, pool_size = transport.get("pool_base"), transport.get("pool_size")
        if (type(pool_base) is not int or pool_base <= 0
                or type(pool_size) is not int or pool_size <= 0):
            raise RuntimeError("Server returned invalid MemFabric L1 pool bounds")
        return MemFabricL1EngineDrivenContext(
            metadata, self._req_client, mq_timeout, self._bm_worker,
            pool_base, pool_size,
        )

    def _gather_to_npu_scratch(
        self, kv_caches, block_ids, blocks_per_chunk, outputs, chunk_indices
    ):
        gather_paged_kv_to_cpu(
            kv_caches,
            block_ids,
            blocks_per_chunk,
            layout_hints=self._layout_hints,
            engine_kv_format=self._engine_kv_format,
            out=outputs,
            chunk_indices=chunk_indices,
            engine_type=self._engine_type,
        )

    def submit_store(
        self, request_id, key, kv_caches, block_ids, event, blocks_in_chunk
    ):
        with self._io_lock:
            return self._submit_store_sync(
                request_id, key, kv_caches, block_ids, event, blocks_in_chunk
            )

    def _submit_store_sync(
        self, request_id, key, kv_caches, block_ids, event, blocks_in_chunk
    ):
        future = MessagingFuture()
        try:
            if len(self._group_states) > 1 or len(block_ids) != 1:
                raise NotImplementedError("MemFabric L1 currently supports one KV group")
            synchronize_device(_get_kv_device(kv_caches))
            result = self._engine_driven_context.prepare_store(key, self._instance_id)
            if result is None:
                future.set_result(False)
                return future
            outputs, chunk_indices = result
            if not outputs:
                self._engine_driven_context.discard_store_prepare(
                    key, self._instance_id
                )
                future.set_result(True)
                return future
            self._gather_to_npu_scratch(
                kv_caches, _single_group_block_ids(block_ids), blocks_in_chunk,
                outputs, chunk_indices,
            )
            synchronize_device(_get_kv_device(kv_caches))
            future.set_result(
                self._engine_driven_context.commit_store(key, self._instance_id, [])
            )
        except Exception as exc:
            self._bm_worker.mark_uncertain()
            logger.exception("MemFabric L1 store failed")
            future.set_exception(exc)
        return future

    def submit_retrieve(
        self, request_id, key, kv_caches, block_ids, event, blocks_in_chunk,
        skip_first_n_tokens=0,
    ):
        with self._io_lock:
            return self._submit_retrieve_sync(
                request_id, key, kv_caches, block_ids, event, blocks_in_chunk,
                skip_first_n_tokens,
            )

    def _submit_retrieve_sync(
        self, request_id, key, kv_caches, block_ids, event, blocks_in_chunk,
        skip_first_n_tokens=0,
    ):
        # Slot retrieval and completion remain fail-closed. The native H2D
        # scatter uses the same page-format descriptor as the store gather.
        future = MessagingFuture()
        try:
            if len(self._group_states) > 1 or len(block_ids) != 1:
                raise NotImplementedError("MemFabric L1 currently supports one KV group")
            full_ids = _single_group_block_ids(block_ids)
            if len(full_ids) % blocks_in_chunk:
                raise ValueError("Engine block count must be divisible by blocks_per_chunk")
            count = len(full_ids) // blocks_in_chunk
            chunks = self._engine_driven_context.prepare_retrieve(
                key, self._instance_id, count
            )
            if chunks is None:
                future.set_result(False)
                return future
            if event is not None:
                event.wait()
            scatter_cpu_to_paged_kv(
                kv_caches,
                full_ids,
                chunks,
                blocks_in_chunk,
                skip_first_n_tokens=skip_first_n_tokens,
                layout_hints=self._layout_hints,
                engine_kv_format=self._engine_kv_format,
                engine_type=self._engine_type,
            )
            synchronize_device(_get_kv_device(kv_caches))
            future.set_result(
                self._engine_driven_context.commit_retrieve(key, self._instance_id)
            )
        except Exception as exc:
            self._bm_worker.mark_uncertain()
            logger.exception("MemFabric L1 retrieve failed")
            future.set_exception(exc)
        return future


class MemFabricL1MPWorkerAdapter(LMCacheMPWorkerAdapter):
    """Join BM membership before the core adapter's first server RPC."""

    def __init__(self, *, extra_config=None, **kwargs):
        config = (extra_config or {}).get("memfabric_l1")
        if not isinstance(config, dict):
            raise ValueError("memfabric_l1 worker configuration is required")
        self._memfabric_session_id = config.get("session_id")
        bm_config = config.get("bm")
        if not isinstance(self._memfabric_session_id, str) or not self._memfabric_session_id:
            raise ValueError("memfabric_l1.session_id is required")
        if not isinstance(bm_config, dict):
            raise ValueError("memfabric_l1.bm must contain BMSession fields")
        self._memfabric_session_config = dict(bm_config)
        self._memfabric_session_config["session_id"] = self._memfabric_session_id
        self._bm_worker = BMAffineWorker(self._memfabric_session_config)
        try:
            super().__init__(extra_config=extra_config, **kwargs)
        except BaseException:
            self._bm_worker.close()
            raise

    def _create_transfer_context(self, kv_caches: dict) -> TransferContext:
        return MemFabricL1TransferContext(
            self.instance_id, self.req_client, self._bm_worker
        )


__all__ = ["MemFabricL1MPWorkerAdapter"]
