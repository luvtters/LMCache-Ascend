# SPDX-License-Identifier: Apache-2.0
"""CPU lifecycle checks for the native-L1 MemFabric strategy."""

from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.l1_runtime import create_l1_runtime
from lmcache_ascend.v1.distributed import memfabric_l1_runtime as runtime_module
from lmcache_ascend.v1.distributed.memfabric_l1_runtime import (
    MemFabricL1Runtime,
    MemFabricL1Strategy,
)
from lmcache_ascend.integration.vllm.memfabric_l1_worker import (
    MemFabricL1TransferContext,
)


class _MemoryObj:
    def __init__(self, shape, dtype):
        self.tensor = torch.empty(shape, dtype=dtype)
        self._valid = True

    def get_size(self):
        return self.tensor.numel() * self.tensor.element_size()

    def is_valid(self):
        return self._valid

    def set_l1_manager(self, _manager_id):
        pass


class _FakeMemoryManager:
    """Allocator fake; L1Manager remains the real native object manager."""

    def __init__(self):
        self.objects = []

    def allocate(self, layout, count):
        objects = [
            _MemoryObj(shape, dtype)
            for _ in range(count)
            for shape, dtype in zip(layout.shapes, layout.dtypes, strict=True)
        ]
        self.objects.extend(objects)
        return L1Error.SUCCESS, objects

    def free(self, objects):
        for obj in objects:
            obj._valid = False
        return L1Error.SUCCESS

    def get_memory_usage(self):
        used = sum(obj.get_size() for obj in self.objects if obj.is_valid())
        return used, 1 << 20

    def get_backend_type(self, _obj):
        return L1BackendType.DRAM

    def get_l1_memory_desc(self):
        return None

    def memcheck(self):
        return True

    def close(self):
        pass


@pytest.fixture
def native_l1():
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=1 << 20,
            use_lazy=True,
            init_size_in_bytes=1 << 20,
            shm_name="",
        ),
        write_ttl_seconds=1,
        read_ttl_seconds=1,
    )
    manager = L1Manager(config, memory_manager=_FakeMemoryManager())
    yield manager
    try:
        manager.close()
    except RuntimeError:
        # The release/quarantine test intentionally leaves an actual-I/O pin.
        pass


@pytest.fixture
def layout():
    return MemoryLayoutDesc(
        shapes=[torch.Size([2, 4, 8])], dtypes=[torch.float32]
    )


def _ipc_key(request_id):
    return IPCCacheServerKey.from_token_ids(
        model_name="runtime-test",
        world_size=1,
        worker_id=17,
        token_ids=[1, 2, 3, 4],
        start=0,
        end=4,
        request_id=request_id,
    )


def _object_key():
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(17),
        model_name="runtime-test",
        kv_rank=0,
    )


class _FakeBMStorage:
    def global_address(self, obj):
        return obj.tensor.data_ptr()


def _strategy(manager):
    return MemFabricL1Strategy(manager, _FakeBMStorage(), 17, "session-a")


def test_store_is_staging_until_matching_completion_then_native_prefetch_read(
    native_l1, layout
):
    strategy = _strategy(native_l1)
    obj_key = _object_key()
    key = _ipc_key("request-a")
    response = strategy.prepare_store(
        key, 17, SimpleNamespace(layout_desc=layout), lambda _key: [obj_key]
    )
    operation_id = response.context["operation_id"]
    slot = response.context["slots"][0]

    assert native_l1.reserve_read([obj_key])[obj_key] == (L1Error.KEY_NOT_EXIST, None)
    wrong_completion = json.dumps(
        {"operation_id": "wrong-token", "bytes": slot["size"]}
    ).encode()
    with pytest.raises(ValueError, match="does not match"):
        strategy.commit_store(key, 17, wrong_completion, None, lambda _key: [obj_key])
    assert native_l1.finish_write([obj_key], tag="memfabric_l1")[obj_key] == (
        L1Error.KEY_IN_WRONG_STATE
    )

    completion = json.dumps(
        {"operation_id": operation_id, "bytes": slot["size"]}
    ).encode()
    assert strategy.commit_store(key, 17, completion, None, lambda _key: [obj_key])
    assert native_l1.get_object_state(obj_key) is not None

    # Prefetch has already installed one native read reservation before the
    # runtime atomically converts it to an actual-I/O pin.
    assert native_l1.reserve_read([obj_key])[obj_key][0] == L1Error.SUCCESS
    retrieve = strategy.prepare_retrieve(key, 17, lambda _key: [obj_key])
    assert retrieve.success
    retrieve_id = retrieve.context["operation_id"]
    wrong_read_key = replace(
        key, request_configs={"_memfabric_l1_operation_id": "wrong-token"}
    )
    with pytest.raises(ValueError, match="does not match"):
        strategy.commit_retrieve(wrong_read_key, 17)
    assert native_l1.delete([obj_key], force=True)[obj_key] == L1Error.KEY_IS_LOCKED

    read_key = replace(
        key, request_configs={"_memfabric_l1_operation_id": retrieve_id}
    )
    assert strategy.commit_retrieve(read_key, 17)
    assert native_l1.delete([obj_key])[obj_key] == L1Error.SUCCESS


def test_equal_cache_keys_with_distinct_request_ids_keep_separate_read_tokens(
    native_l1, layout
):
    strategy = _strategy(native_l1)
    obj_key = _object_key()
    store_key = _ipc_key("request-store")
    response = strategy.prepare_store(
        store_key, 17, SimpleNamespace(layout_desc=layout), lambda _key: [obj_key]
    )
    slot = response.context["slots"][0]
    assert strategy.commit_store(
        store_key,
        17,
        json.dumps(
            {"operation_id": response.context["operation_id"], "bytes": slot["size"]}
        ).encode(),
        None,
        lambda _key: [obj_key],
    )

    first = _ipc_key("request-read-1")
    second = _ipc_key("request-read-2")
    assert first == second  # request_id is excluded from cache-key equality.
    native_l1.reserve_read([obj_key], read_locks=2)
    first_response = strategy.prepare_retrieve(first, 17, lambda _key: [obj_key])
    second_response = strategy.prepare_retrieve(second, 17, lambda _key: [obj_key])
    first_id = first_response.context["operation_id"]
    second_id = second_response.context["operation_id"]
    assert first_id != second_id

    cross_token_key = replace(
        second, request_configs={"_memfabric_l1_operation_id": first_id}
    )
    with pytest.raises(ValueError, match="does not match"):
        strategy.commit_retrieve(cross_token_key, 17)
    assert native_l1.delete([obj_key], force=True)[obj_key] == L1Error.KEY_IS_LOCKED

    for request_key, operation_id in ((first, first_id), (second, second_id)):
        completion_key = replace(
            request_key,
            request_configs={"_memfabric_l1_operation_id": operation_id},
        )
        assert strategy.commit_retrieve(completion_key, 17)
    assert native_l1.delete([obj_key])[obj_key] == L1Error.SUCCESS


def test_release_quarantines_unconfirmed_write_without_publishing(native_l1, layout):
    strategy = _strategy(native_l1)
    obj_key = _object_key()
    response = strategy.prepare_store(
        _ipc_key("request-release"),
        17,
        SimpleNamespace(layout_desc=layout),
        lambda _key: [obj_key],
    )
    operation_id = response.context["operation_id"]
    slots = response.context["slots"]
    assert slots

    strategy.release()
    with pytest.raises(RuntimeError, match="released"):
        strategy.commit_store(
            _ipc_key("request-release"),
            17,
            json.dumps(
                {"operation_id": operation_id, "bytes": slots[0]["size"]}
            ).encode(),
            None,
            lambda _key: [obj_key],
        )
    assert native_l1.reserve_read([obj_key])[obj_key] == (L1Error.KEY_NOT_EXIST, None)
    assert native_l1.finish_write([obj_key], tag="memfabric_l1")[obj_key] == (
        L1Error.KEY_IN_WRONG_STATE
    )
    assert native_l1.delete([obj_key], force=True)[obj_key] == L1Error.KEY_IS_LOCKED
    assert strategy.has_pending_io
    with pytest.raises(RuntimeError, match="active I/O pins"):
        native_l1.close()


def test_runtime_init_and_factory_metadata_fail_closed():
    storage = SimpleNamespace(
        l1_manager_configs=[object()],
        l2_adapter_config=SimpleNamespace(adapters=[]),
    )
    with patch.object(runtime_module, "BMSession") as bm_session:
        with pytest.raises(ValueError, match="session_id"):
            MemFabricL1Runtime(storage, {"bm": {"rank_id": 0}})
        with pytest.raises(ValueError, match="rank 0"):
            MemFabricL1Runtime(
                storage,
                {"session_id": "session-a", "bm": {"rank_id": 1}},
            )
        bm_session.assert_not_called()

    with pytest.raises(ValueError, match="config must be a mapping"):
        create_l1_runtime(
            {"l1_runtime": {"factory": "not.imported:create", "config": []}},
            storage,
        )


def test_worker_rejects_unsupported_transport_version():
    transfer = MemFabricL1TransferContext.__new__(MemFabricL1TransferContext)
    transfer._bm_worker = SimpleNamespace(session_id="session-a")
    transfer._req_client = object()
    response = SimpleNamespace(
        transport_metadata={
            "transport": "memfabric_l1",
            "version": 2,
            "session_id": "session-a",
            "owner_rank": 0,
        }
    )
    with pytest.raises(RuntimeError, match="incompatible"):
        transfer._create_engine_driven_context(SimpleNamespace(), response, 1.0)
