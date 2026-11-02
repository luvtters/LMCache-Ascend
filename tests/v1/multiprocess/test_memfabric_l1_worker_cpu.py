# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for native L1 worker transport selection and BM affinity."""

import sys
import threading
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from lmcache.v1.multiprocess.custom_types import PrepareStoreResponse

from lmcache_ascend.integration.vllm.memfabric_l1_worker import (
    BMAffineWorker,
    MemFabricL1EngineDrivenContext,
    MemFabricL1TransferContext,
)


def test_bm_calls_and_close_stay_on_the_session_owner_thread():
    calls = []

    class FakeSession:
        def __init__(self):
            self.owner = threading.get_ident()

        def register_npu_tensor(self, tensor):
            calls.append(("register", threading.get_ident(), tensor))

        def copy_npu_to_global(self, src, dst, size):
            calls.append(("copy", threading.get_ident(), (src, dst, size)))

        def close(self):
            calls.append(("close", threading.get_ident(), None))

    fake_module = ModuleType("lmcache_ascend.v1.distributed.memfabric_bm")
    fake_module.BMSession = lambda **_kwargs: FakeSession()
    with patch.dict(sys.modules, {fake_module.__name__: fake_module}):
        worker = BMAffineWorker({"session_id": "test", "device_id": 0})
        owner = worker._session.owner
        worker.register(object())
        worker.copy_to_pool(SimpleNamespace(data_ptr=lambda: 12), 34, 56)
        worker.close()

    assert owner != threading.get_ident()
    assert [entry[0] for entry in calls] == ["register", "copy", "close"]
    assert {entry[1] for entry in calls} == {owner}


def test_slot_scratch_is_allocated_and_registered_once(monkeypatch):
    worker = BMAffineWorker.__new__(BMAffineWorker)
    worker.device_id = 0
    worker._poisoned = False
    worker._scratch_cache = {}
    worker._scratch_lock = threading.Lock()
    worker.register = MagicMock()
    allocated = MagicMock()
    allocate = MagicMock(return_value=allocated)
    monkeypatch.setattr("torch.empty", allocate)

    first = worker.scratch((2, 4), torch.bfloat16, 0)
    second = worker.scratch((2, 4), torch.bfloat16, 0)

    assert first is second is allocated
    allocate.assert_called_once_with(
        (2, 4), dtype=torch.bfloat16, device="npu:0"
    )
    worker.register.assert_called_once_with(allocated)


def test_memfabric_context_requires_matching_service_transport_metadata():
    bm_worker = SimpleNamespace(session_id="session-a")
    context = MemFabricL1TransferContext(1, MagicMock(), bm_worker)
    metadata = SimpleNamespace(layout_desc=object())
    bad_response = SimpleNamespace(
        transport_metadata={"transport": "inline", "version": 1}
    )

    with pytest.raises(RuntimeError, match="did not negotiate"):
        context._create_engine_driven_context(metadata, bad_response, 1.0)

    good_response = SimpleNamespace(
        transport_metadata={
            "transport": "memfabric_l1",
            "version": 1,
            "session_id": "session-a",
            "owner_rank": 0,
            "pool_base": 4096,
            "pool_size": 4096,
        }
    )
    engine_context = context._create_engine_driven_context(metadata, good_response, 1.0)
    assert isinstance(engine_context, MemFabricL1EngineDrivenContext)
    assert engine_context._bm_worker is bm_worker


def test_slot_validation_rejects_duplicate_or_out_of_order_chunk_indices():
    context = MemFabricL1EngineDrivenContext.__new__(MemFabricL1EngineDrivenContext)
    context._pool_base = 100
    context._pool_size = 256
    slots = [
        {"address": 100, "size": 64, "chunk_index": 0},
        {"address": 200, "size": 64, "chunk_index": 0},
    ]

    with pytest.raises(RuntimeError, match="invalid slot descriptor"):
        context._validate_slots(
            {"transport": "memfabric_l1", "version": 1,
             "operation_id": "op", "slots": slots}
        )


def test_all_cached_store_does_not_leave_a_pending_operation():
    context = MemFabricL1EngineDrivenContext.__new__(MemFabricL1EngineDrivenContext)
    context.req_client = SimpleNamespace(
        prepare_store=MagicMock(
            return_value=SimpleNamespace(
                result=MagicMock(return_value=PrepareStoreResponse(
                    context={"transport": "memfabric_l1", "version": 1,
                             "operation_id": "op", "slots": []}
                ))
            )
        )
    )
    context.mq_timeout = 0.1
    context._store_ops = {}
    context._lock = threading.Lock()

    assert context.prepare_store(_Key("req"), 1) == ([], [])
    assert context._store_ops == {}


def test_one_group_state_is_supported_for_store_prepare():
    transfer = MemFabricL1TransferContext.__new__(MemFabricL1TransferContext)
    transfer._io_lock = threading.Lock()
    transfer._group_states = [object()]
    transfer._instance_id = 7
    context = SimpleNamespace(
        prepare_store=MagicMock(return_value=([], [])),
        discard_store_prepare=MagicMock(),
    )
    transfer._engine_driven_context = context

    future = transfer.submit_store(
        "req", object(), {"layer": torch.empty(1)}, [[1]], None, 1
    )

    assert future.result() is True
    context.discard_store_prepare.assert_called_once()


def test_store_prepare_timeout_is_uncertain_not_a_cache_miss():
    worker = SimpleNamespace(mark_uncertain=MagicMock())
    context = MemFabricL1EngineDrivenContext.__new__(MemFabricL1EngineDrivenContext)
    context.req_client = SimpleNamespace(
        prepare_store=MagicMock(
            return_value=SimpleNamespace(result=MagicMock(side_effect=TimeoutError))
        )
    )
    context.mq_timeout = 0.1
    context._bm_worker = worker
    context._store_ops = {}
    context._lock = threading.Lock()

    with pytest.raises(RuntimeError, match="prepare timed out"):
        context.prepare_store(_Key("req"), 1)
    worker.mark_uncertain.assert_called_once()


def test_malformed_retrieve_response_is_not_reported_as_a_miss():
    context = MemFabricL1EngineDrivenContext.__new__(MemFabricL1EngineDrivenContext)
    context.req_client = SimpleNamespace(
        prepare_retrieve=MagicMock(
            return_value=SimpleNamespace(result=MagicMock(return_value=object()))
        )
    )
    context.mq_timeout = 0.1
    context._retrieve_ops = {}
    context._lock = threading.Lock()

    with pytest.raises(RuntimeError, match="Unexpected.*prepare_retrieve"):
        context.prepare_retrieve(_Key("req"), 1, 1)


@dataclass(unsafe_hash=True)
class _Key:
    request_id: str
    request_configs: dict | None = field(default=None, hash=False)


def test_retrieve_commit_timeout_keeps_operation_for_quarantine():
    key = _Key("req")
    operation = ("op", [object()], [{"address": 1, "size": 4}])
    context = MemFabricL1EngineDrivenContext.__new__(MemFabricL1EngineDrivenContext)
    context._lock = threading.Lock()
    context._retrieve_ops = {(1, "req", key): operation}
    context.req_client = SimpleNamespace(
        commit_retrieve=MagicMock(
            return_value=SimpleNamespace(result=MagicMock(side_effect=TimeoutError))
        )
    )
    context.mq_timeout = 0.1

    with pytest.raises(RuntimeError, match="COMMIT_RETRIEVE timed out"):
        context.commit_retrieve(key, 1)
    assert context._retrieve_ops[(1, "req", key)] is operation
