# SPDX-License-Identifier: Apache-2.0
"""CPU lifecycle tests; BM, NPU and messaging transport are fake boundaries."""

import importlib.util
import json
from pathlib import Path
import threading
import time
import types
import unittest
from unittest.mock import patch


class Future:
    def __init__(self):
        self.done = threading.Event()
        self.value = None
        self.error = None

    def query(self):
        return self.done.is_set()

    def wait(self, timeout=None):
        return self.done.wait(timeout)

    def result(self, timeout=None):
        if not self.wait(timeout):
            raise TimeoutError("test RPC pending")
        if self.error is not None:
            raise self.error
        return self.value

    def set_result(self, value):
        self.value = value
        self.done.set()

    def set_exception(self, error):
        self.error = error
        self.done.set()


class ContextBase:
    def __init__(self, instance_id, req_client):
        self.closed = False

    def _is_closed(self):
        return self.closed

    def _mark_closed(self):
        self.closed = True


def load_modules():
    directory = Path(__file__).parents[3] / "lmcache_ascend/v1/distributed"
    modules = {}
    for name in (
        "lmcache.utils",
        "lmcache.v1.multiprocess.futures",
        "lmcache.v1.multiprocess.protocols.server_module",
        "lmcache.v1.multiprocess.transfer_context.worker_transfer",
    ):
        modules[name] = types.ModuleType(name)
    modules["lmcache.utils"].EngineType = types.SimpleNamespace(VLLM="vllm")
    modules["lmcache.v1.multiprocess.futures"].MessagingFuture = Future
    modules["lmcache.v1.multiprocess.protocols.server_module"].ServerModuleCallRequest = types.SimpleNamespace
    modules["lmcache.v1.multiprocess.transfer_context.worker_transfer"].TransferContext = ContextBase
    direct_spec = importlib.util.spec_from_file_location("memfabric_direct_test_module", directory / "memfabric_direct.py")
    direct = importlib.util.module_from_spec(direct_spec)
    index_spec = importlib.util.spec_from_file_location("memfabric_index_test_module", directory / "memfabric_pool_index.py")
    index = importlib.util.module_from_spec(index_spec)
    modules[index_spec.name] = index
    with patch.dict("sys.modules", modules):
        direct_spec.loader.exec_module(direct)
        index_spec.loader.exec_module(index)
    return direct, index.PoolIndex


DIRECT, PoolIndex = load_modules()


def key(start=0, end=128, *, request="request", worker=0, salt="", configs=None):
    return types.SimpleNamespace(
        model_name="qwen3", world_size=1, worker_id=worker,
        token_bytes=b"".join(value.to_bytes(4, "big") for value in range(end)),
        start=start, end=end, request_id=request, cache_salt=salt,
        request_configs=configs,
    )


STRATEGY = types.SimpleNamespace(tp_size=1, pp_size=1, dcp_size=1,
    n_servers=1, vllm_world_size=1, mla_only=False)


class Client:
    def __init__(self):
        self.index = PoolIndex(pool_bytes=1 << 30, slot_size=32 << 20,
                               storage_rank=0, world_size=2)
        self.delay = None
        self.pending = []
        self.registration_not_ready_once = False
        self.calls = []

    def server_module_call(self, request):
        method = request.method.split(".")[-1]
        payload = json.loads(request.payload)
        self.calls.append((method, payload))
        future = Future()
        if method == self.delay:
            self.pending.append((method, payload, future))
        else:
            self.complete(method, payload, future)
        return future

    def complete(self, method, payload, future):
        try:
            result = getattr(self.index, method)(**payload)
            if method == "register" and self.registration_not_ready_once:
                result["ready"] = result["transfer_ready"] = False
                self.registration_not_ready_once = False
            result = dict(ok=True, **result)
        except (ValueError, RuntimeError) as error:
            result = dict(ok=False, error=str(error))
        future.set_result(types.SimpleNamespace(success=True, error="",
            payload=json.dumps(result).encode()))

    def flush(self):
        for item in self.pending:
            self.complete(*item)
        self.pending.clear()


class Session:
    def __init__(self, **kwargs):
        self.owner = threading.get_ident()
        self.operations = []
        self.fail_copy = False
        self.fail_wait = False
        self.closed = False

    def record(self, operation):
        if self.owner != threading.get_ident():
            raise RuntimeError("wrong BM owner")
        self.operations.append(operation)

    def peer_address(self, rank):
        self.record("peer_address")
        return 4096

    def copy_npu_to_global(self, src, dst, size):
        self.record("L2G")
        if self.fail_copy:
            raise RuntimeError("DMA completion unknown")

    def copy_global_to_npu(self, src, dst, size):
        self.record("G2L")

    def wait(self):
        self.record("wait")
        if self.fail_wait:
            raise RuntimeError("DMA still pending")

    def close(self):
        self.record("close")
        self.closed = True


class Transfer:
    def __init__(self, kv, tokens, blocks, **kwargs):
        self.owner = threading.get_ident()
        self.device = types.SimpleNamespace(index=0)
        self.size = 32
        self.block_size = 128
        self.blocks_per_chunk = 1
        self.buffer = types.SimpleNamespace(data_ptr=lambda: 1024)
        self.layout = {"format": "test-NPU-layout"}
        self.operations = []
        self.fail_gather = False
        self.fail_scatter = False

    def record(self, operation):
        if threading.get_ident() != self.owner:
            raise RuntimeError("wrong NPU owner")
        self.operations.append(operation)

    def synchronize(self):
        self.record("synchronize")

    def gather(self, ids):
        self.record("gather")
        if self.fail_gather:
            raise RuntimeError("gather failed before DMA")

    def scatter(self, ids, skip_tokens=0):
        self.record("scatter")
        if self.fail_scatter:
            raise RuntimeError("partial scatter failed")


def context(client, *, init_timeout=None, session_factory=Session):
    config = dict(device_id=0, rank_id=1, world_size=2,
                  pool_gb=1, store_url="test-only")
    if init_timeout is not None:
        config["init_timeout"] = init_timeout
    result = DIRECT.MemFabricDirectTransferContext(instance_id=1,
        req_client=client, tokens_per_chunk=128, mq_timeout=1,
        extra_config={"lmcache.mp.direct": config},
        session_factory=session_factory, transfer_factory=Transfer)
    result.register({"layer": object()}, "qwen3", 1, 1, 1)
    return result


def backend(client, timeout=1):
    return DIRECT.MemFabricDirectSchedulerBackend(req_clients={"test": client},
        parallel_strategy=STRATEGY, tokens_per_chunk=128, mq_timeout=timeout)


def add_object(client, object_key):
    client.index.register(rank=1, member_id="fixture")
    identity = dict(epoch=client.index.epoch, rank=1, member_id="fixture")
    reservation = client.index.prepare_store(**identity, key=object_key, size=32)
    client.index.commit_store(**identity, write_id=reservation["write_id"])


class DirectTests(unittest.TestCase):
    def test_optional_startup_timeout_is_forwarded_without_changing_default(self):
        for value in (None, 300):
            with self.subTest(value=value):
                received = []

                def factory(**kwargs):
                    received.append(kwargs)
                    return Session(**kwargs)

                item = context(Client(), init_timeout=value, session_factory=factory)
                if value is None:
                    self.assertNotIn("init_timeout", received[0])
                else:
                    self.assertEqual(received[0]["init_timeout"], value)
                item.close()

    def test_optional_startup_timeout_is_validated_before_device_creation(self):
        for value in (0, -1, 3601, True, 1.5, "300"):
            with self.subTest(value=value):
                item = DIRECT.MemFabricDirectTransferContext(instance_id=1,
                    req_client=Client(), tokens_per_chunk=128, mq_timeout=1,
                    extra_config={"lmcache.mp.direct": dict(device_id=0,
                        rank_id=1, world_size=2, pool_gb=1, store_url="test-only",
                        init_timeout=value)}, session_factory=Session,
                    transfer_factory=Transfer)
                try:
                    with self.assertRaisesRegex(ValueError, "init_timeout"):
                        item.register({}, "qwen3", 1, 1, 1)
                finally:
                    item.close()

    def test_keys_share_identity_across_workers_requests_and_metadata(self):
        full = DIRECT.chunk_keys(key(end=256), 128)
        self.assertEqual(full, DIRECT.chunk_keys(key(end=256, request="D",
            worker=None, configs={"priority": 4}), 128))
        self.assertEqual(full[1:], DIRECT.chunk_keys(key(start=128, end=256), 128))
        self.assertNotEqual(full, DIRECT.chunk_keys(key(end=256, salt="other"), 128))
        with self.assertRaises(ValueError):
            DIRECT.chunk_keys(key(end=129), 128)

    def test_owner_thread_covers_register_store_retrieve_and_foreign_close(self):
        client = Client()
        item = context(client)
        session, transfer = item.session, item.transfer
        self.assertNotEqual(session.owner, threading.get_ident())
        saved = []
        thread = threading.Thread(target=lambda: saved.append(item.submit_store(
            "store", key(), {}, [[1]], None, 1).result()))
        thread.start()
        thread.join(2)
        self.assertEqual(saved, [True])
        self.assertTrue(item.submit_retrieve("load", key(), {}, [[2]], None, 1).result())
        thread = threading.Thread(target=item.close)
        thread.start()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(session.closed)
        self.assertIn("L2G", session.operations)
        self.assertIn("G2L", session.operations)
        self.assertIn("gather", transfer.operations)
        self.assertIn("scatter", transfer.operations)
        self.assertEqual(client.index.status()["total_pins"], 0)

    def test_registration_polls_service_background_join_readiness(self):
        client = Client()
        client.registration_not_ready_once = True
        item = context(client)
        self.assertTrue(any(method == "status" for method, _ in client.calls))
        item.close()

    def test_unknown_dma_keeps_slot_and_scratch_until_successful_drain(self):
        client = Client()
        item = context(client)
        session, transfer = item.session, item.transfer
        session.fail_copy = True
        session.fail_wait = True
        with self.assertRaisesRegex(RuntimeError, "DMA completion unknown"):
            item.submit_store("save", key(), {}, [[1]], None, 1).result()
        self.assertEqual(client.index.status()["quarantined_slots"], 1)
        with self.assertRaisesRegex(RuntimeError, "DMA still pending"):
            item.close()
        self.assertIs(item.transfer, transfer)
        self.assertFalse(session.closed)
        self.assertEqual(client.index.status()["pending_writes"], 1)
        session.fail_wait = False
        item.close()
        self.assertEqual(client.index.status()["pending_writes"], 0)
        self.assertTrue(session.closed)

    def test_partial_scatter_is_a_failed_future_not_direct_success(self):
        client = Client()
        item = context(client)
        item.submit_store("save", key(), {}, [[1]], None, 1).result()
        item.transfer.fail_scatter = True
        with self.assertRaisesRegex(RuntimeError, "partial scatter"):
            item.submit_retrieve("load", key(), {}, [[2]], None, 1).result()
        self.assertTrue(item.poisoned)
        self.assertEqual(client.index.status()["total_pins"], 0)
        item.close()

    def test_lookup_reports_absolute_prefix_end_for_nonzero_start(self):
        client = Client()
        lookup_key = key(start=128, end=256, worker=None)
        add_object(client, DIRECT.chunk_keys(lookup_key, 128)[0])
        owner = backend(client)
        lookup = owner.create_lookup(lookup_key, tp_size=1,
            tokens_per_chunk=128, timeout=1, blocking_lookup=False)
        lookup.submit()
        self.assertEqual(lookup.on_scheduler_check(), [])
        self.assertEqual(lookup.get_result(), 256)
        owner.end_session(lookup_key.request_id)
        missing = owner.create_lookup(key(start=128, end=256, salt="miss"),
            tp_size=1, tokens_per_chunk=128, timeout=1, blocking_lookup=False)
        missing.submit()
        missing.on_scheduler_check()
        self.assertEqual(missing.get_result(), 128)
        owner.close()

    def test_cleanup_before_late_lookup_still_drains_and_releases_pins(self):
        client = Client()
        lookup_key = key(worker=None)
        add_object(client, DIRECT.chunk_keys(lookup_key, 128)[0])
        client.delay = "lookup"
        owner = backend(client)
        lookup = owner.create_lookup(lookup_key, tp_size=1,
            tokens_per_chunk=128, timeout=1, blocking_lookup=False)
        lookup.submit()
        lookup.close()
        errors = []

        def end():
            try:
                owner.end_session(lookup_key.request_id)
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=end)
        thread.start()
        time.sleep(0.01)
        client.flush()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(client.index.status()["total_pins"], 0)
        self.assertEqual(owner.lookups, {})

    def test_timeout_retains_lookup_until_late_reply_can_be_released(self):
        client = Client()
        lookup_key = key(worker=None)
        add_object(client, DIRECT.chunk_keys(lookup_key, 128)[0])
        client.delay = "lookup"
        owner = backend(client)
        lookup = owner.create_lookup(lookup_key, tp_size=1,
            tokens_per_chunk=128, timeout=0.01, blocking_lookup=False)
        lookup.submit()
        with self.assertRaises(TimeoutError):
            owner.end_session(lookup_key.request_id)
        self.assertIn(lookup_key.request_id, owner.lookups)
        client.flush()
        owner.end_session(lookup_key.request_id)
        self.assertEqual(client.index.status()["total_pins"], 0)

    def test_partial_lock_release_preserves_other_lookup_chunks(self):
        client = Client()
        lookup_key = key(end=256, worker=None)
        for object_key in DIRECT.chunk_keys(lookup_key, 128):
            add_object(client, object_key)
        owner = backend(client)
        lookup = owner.create_lookup(lookup_key, tp_size=1,
            tokens_per_chunk=128, timeout=1, blocking_lookup=False)
        lookup.submit()
        owner.free_lookup_locks(key(start=128, end=256), 1)
        self.assertEqual(client.index.status()["total_pins"], 1)
        owner.end_session(lookup_key.request_id)
        self.assertEqual(client.index.status()["total_pins"], 0)

    def test_timed_out_prepare_keeps_future_for_cleanup_without_dma(self):
        client = Client()
        item = context(client)
        item.control.timeout = 0.01
        client.delay = "prepare_store"
        with self.assertRaises(TimeoutError):
            item.submit_store("save", key(), {}, [[1]], None, 1).result()
        client.flush()
        self.assertEqual(client.index.status()["pending_writes"], 1)
        self.assertNotIn("L2G", item.session.operations)
        item.close()
        self.assertEqual(client.index.status()["pending_writes"], 0)

    def test_timed_out_acquire_keeps_future_for_cleanup_without_read(self):
        client = Client()
        item = context(client)
        item.submit_store("save", key(), {}, [[1]], None, 1).result()
        item.control.timeout = 0.01
        client.delay = "acquire"
        with self.assertRaises(TimeoutError):
            item.submit_retrieve("load", key(), {}, [[2]], None, 1).result()
        client.flush()
        self.assertEqual(client.index.status()["total_pins"], 1)
        self.assertNotIn("G2L", item.session.operations)
        item.close()
        self.assertEqual(client.index.status()["total_pins"], 0)

    def test_timed_out_commit_is_resolved_without_aborting_committed_object(self):
        client = Client()
        item = context(client)
        item.control.timeout = 0.01
        client.delay = "commit_store"
        with self.assertRaises(TimeoutError):
            item.submit_store("save", key(), {}, [[1]], None, 1).result()
        client.flush()
        self.assertEqual(client.index.status()["objects"], 1)
        item.close()
        self.assertEqual(client.index.status()["objects"], 1)
        self.assertFalse(any(method == "abort_store" for method, _ in client.calls))

    def test_timed_out_abort_is_not_reissued_after_late_acknowledgement(self):
        client = Client()
        item = context(client)
        item.control.timeout = 0.01
        item.transfer.fail_gather = True
        client.delay = "abort_store"
        with self.assertRaises(TimeoutError):
            item.submit_store("save", key(), {}, [[1]], None, 1).result()
        client.flush()
        self.assertEqual(client.index.status()["pending_writes"], 0)
        item.close()
        self.assertEqual(sum(method == "abort_store" for method, _ in client.calls), 1)

    def test_timed_out_release_is_not_reissued_after_late_acknowledgement(self):
        client = Client()
        item = context(client)
        item.submit_store("save", key(), {}, [[1]], None, 1).result()
        item.control.timeout = 0.01
        client.delay = "release"
        with self.assertRaises(TimeoutError):
            item.submit_retrieve("load", key(), {}, [[2]], None, 1).result()
        client.flush()
        self.assertEqual(client.index.status()["total_pins"], 0)
        item.close()
        self.assertEqual(sum(method == "release" for method, _ in client.calls), 1)

    def test_late_prepare_rejection_does_not_prevent_owner_close(self):
        client = Client()
        item = context(client)
        session = item.session
        item.control.timeout = 0.01
        client.delay = "prepare_store"
        with self.assertRaises(TimeoutError):
            item.submit_store("save", key(), {}, [[1]], None, 1).result()
        client.index.invalidate("pool unavailable before prepare")
        client.flush()
        self.assertEqual(client.index.status()["pending_writes"], 0)
        self.assertNotIn("L2G", session.operations)
        item.close()
        self.assertTrue(session.closed)

    def test_late_acquire_rejection_does_not_prevent_owner_close(self):
        client = Client()
        item = context(client)
        session = item.session
        item.submit_store("save", key(), {}, [[1]], None, 1).result()
        item.control.timeout = 0.01
        client.delay = "acquire"
        with self.assertRaises(TimeoutError):
            item.submit_retrieve("load", key(), {}, [[2]], None, 1).result()
        client.index.invalidate("pool unavailable before acquire")
        client.flush()
        self.assertEqual(client.index.status()["total_pins"], 0)
        self.assertNotIn("G2L", session.operations)
        item.close()
        self.assertTrue(session.closed)

    def test_late_commit_rejection_waits_for_dma_then_aborts_reservation(self):
        client = Client()
        item = context(client)
        session = item.session
        item.control.timeout = 0.01
        client.delay = "commit_store"
        with self.assertRaises(TimeoutError):
            item.submit_store("save", key(), {}, [[1]], None, 1).result()
        client.index.invalidate("pool unavailable before commit")
        client.flush()
        self.assertEqual(client.index.status()["pending_writes"], 1)
        session.fail_wait = True
        with self.assertRaisesRegex(RuntimeError, "DMA still pending"):
            item.close()
        self.assertEqual(client.index.status()["pending_writes"], 1)
        self.assertFalse(session.closed)
        session.fail_wait = False
        item.close()
        self.assertEqual(client.index.status()["pending_writes"], 0)
        self.assertEqual(client.index.status()["objects"], 0)
        self.assertTrue(session.closed)


if __name__ == "__main__":
    unittest.main()
