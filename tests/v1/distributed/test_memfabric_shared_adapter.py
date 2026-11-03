# SPDX-License-Identifier: Apache-2.0
"""Two real adapter processes and HTTP directory; only core/SDK buffers are fakes."""
from dataclasses import dataclass, replace
import importlib.util
import multiprocessing
from pathlib import Path
import sys
import tempfile
import threading
import time
import traceback
import unittest

from test_memfabric_l2_rdma import MemoryObj, Session, load_adapter

ROOT = Path(__file__).parents[3]


def source(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class WireKey:
    chunk_hash_hex: str = "ab12"
    model_name: str = "test-model"
    kv_rank: int = 1
    object_group_id: int = 0
    cache_salt: str = "tenant"

    def to_encoded_object_key(self):
        return self

    def to_object_key(self):
        return self


def adapter_process(rank, url, backing, pipe):
    # A single slot is enough to expose premature reuse. Each process has its
    # own GVA for the same owner. The backing file only simulates BM payload I/O.
    base = (rank + 1) << 32
    class FileSession(Session):
        def peer_address(self, owner):
            assert owner == 0
            return base

        def copy_cpu_object(self, obj, address, size, *, store):
            assert self.owner == threading.get_ident() and address == base
            if self.failure:
                raise RuntimeError("injected unknown copy completion")
            if store:
                backing.write_bytes(obj.payload)
            else:
                obj.payload = backing.read_bytes()
            assert len(obj.payload) == size
            return "cpu-test-file"

    def wait_result(query):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            value = query()
            if value is not None:
                return value
            time.sleep(.001)
        raise TimeoutError("adapter task did not finish")

    name = "lmcache_ascend.v1.distributed.memfabric_directory_rpc"
    rpc = source(name, "lmcache_ascend/v1/distributed/memfabric_directory_rpc.py")
    adapter_type = load_adapter(FileSession)
    signed_events = []
    adapter_type._notify_keys_stored = lambda self, keys, sizes: signed_events.extend(
        (k, size) for k, size in zip(keys, sizes, strict=True))
    adapter_type._notify_keys_deleted = lambda self, keys, sizes: signed_events.extend(
        (k, -size) for k, size in zip(keys, sizes, strict=True))
    modules = adapter_type._cpu_import_modules
    modules["lmcache.v1.distributed.api"].EncodedObjectKey = WireKey
    modules["lmcache.v1.distributed.l2_adapters.base"].AdapterUsage = lambda used, capacity, salts: (
        used, capacity, dict(salts))
    sys.modules.update({**modules, name: rpc})  # Confined to this spawned process.
    try:
        adapter = adapter_type(dict(world_size=2, rank_id=rank, storage_rank=0, shared_directory=True,
                                    pool_gb=1, slot_size_mb=1024, data_op_type="DEVICE_RDMA",
                                    join_coord_url=url, join_group_id="attempt", device_nic="tcp://fake:1",
                                    store_url="tcp://fake:2"))
        pipe.send(("ok", (adapter._session.config["rank_id"], adapter._session.config["directory_epoch"])))
        while True:
            op, key, payload = pipe.recv()
            if op == "close":
                adapter.close()
                pipe.send(("ok", None))
                break
            try:
                if op in ("store", "load", "fail_store"):
                    obj = MemoryObj(size=len(payload))
                    obj.payload = payload
                    if op == "load":
                        tid = adapter.submit_load_task([key], [obj])
                        bitmap = wait_result(lambda: adapter.query_load_result(tid))
                        value = bitmap.popcount(), obj.payload
                    else:
                        adapter._session.failure = op == "fail_store"
                        tid = adapter.submit_store_task([key], [obj])
                        value = wait_result(lambda: adapter.pop_completed_store_tasks().get(tid)).size
                elif op == "pin":
                    tid = adapter.submit_lookup_and_lock_task([key], {})
                    value = wait_result(lambda: adapter.query_lookup_and_lock_result(tid)).popcount()
                elif op in ("delete", "unpin"):
                    (adapter.delete if op == "delete" else adapter.submit_unlock)([key])
                    # Fence this adapter's fire-and-forget operation via its public FIFO.
                    tid = adapter.submit_lookup_and_lock_task([], {})
                    value = wait_result(lambda: adapter.query_lookup_and_lock_result(tid)).popcount()
                elif op == "status":
                    value = adapter.report_status()
                elif op == "usage":
                    value = adapter.get_usage()
                elif op == "events":
                    value = list(signed_events)
                else:
                    raise AssertionError(op)
                pipe.send(("ok", value))
            except (ValueError, RuntimeError) as exc:
                pipe.send(("rejected", str(exc)))
    except BaseException:
        pipe.send(("error", traceback.format_exc()))
        raise
    finally:
        pipe.close()


class SharedAdapterTest(unittest.TestCase):
    def test_remote_lookup_locked_delete_and_unknown_copy(self):
        key = WireKey()
        other = replace(key, cache_salt="other-tenant")
        payload = bytes(range(64))
        ctx = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as tmp:
            coord = source("pool_coord", "examples/memfabric/pool_coordination.py")
            server = coord.make_server("127.0.0.1", 0, tmp, "attempt",
                                       shared_directory=dict(storage_rank=0, pool_gb=1, slot_size_mb=1024))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            processes, pipes = [], []
            def reply(rank, expected="ok"):
                self.assertTrue(pipes[rank].poll(10), f"rank {rank} response timed out")
                result, value = pipes[rank].recv()
                self.assertEqual(result, expected, value)
                return value

            def call(rank, op, key=key, data=payload, expected="ok"):
                pipes[rank].send((op, key, data))
                return reply(rank, expected)

            try:
                for rank in (0, 1):
                    parent, child = ctx.Pipe()
                    process = ctx.Process(target=adapter_process, args=(
                        rank, f"http://127.0.0.1:{server.server_port}", Path(tmp) / "payload", child))
                    process.start()
                    child.close()
                    pipes.append(parent)
                    processes.append(process)
                a, b = reply(0), reply(1)
                self.assertEqual((a[0], b[0]), (0, 1))
                self.assertTrue(a[1] and a[1] == b[1])
                self.assertEqual(call(1, "pin"), 0)
                self.assertEqual(call(0, "store"), len(payload))
                self.assertEqual(call(1, "usage"), (64, 1 << 30, {"tenant": 64}))
                self.assertEqual(call(1, "store"), 0)
                for isolated in (other, replace(key, model_name="other-model"),
                                 replace(key, kv_rank=2), replace(key, object_group_id=1)):
                    self.assertEqual(call(1, "pin", isolated), 0)
                self.assertEqual(call(1, "pin"), 1)
                call(0, "delete")
                self.assertEqual(call(0, "pin"), 0)
                self.assertIn("pinned", call(0, "store", other, expected="rejected"))
                self.assertEqual(call(1, "load", data=b"?" * 64), (1, payload))
                call(1, "unpin")
                self.assertEqual(call(1, "load", data=b"?" * 64), (0, b"?" * 64))
                self.assertEqual(call(0, "status")["free_slots"], 1)
                self.assertEqual(call(0, "events"), [(key, 64), (key, -64)])
                self.assertEqual(call(1, "events"), [])
                self.assertEqual(call(1, "store", other), len(payload))
                self.assertEqual(call(0, "pin", other), 1)
                self.assertEqual(call(0, "load", other, b"?" * 64), (1, payload))
                call(0, "unpin", other)
                self.assertIn("unknown copy", call(0, "fail_store", expected="rejected"))
                self.assertFalse(call(0, "status")["is_healthy"])
                self.assertEqual(call(1, "pin"), 0)
                self.assertEqual(call(1, "status")["pending_writes"], 1)
                self.assertIn("reserved", call(1, "store", other, expected="rejected"))
                for rank, proc in enumerate(processes):
                    call(rank, "close")
                    proc.join(5)
                    self.assertEqual(proc.exitcode, 0)
            finally:
                for proc in processes:
                    if proc.is_alive():
                        proc.terminate()
                    proc.join(5)
                for pipe in pipes:
                    pipe.close()
                server.shutdown()
                server.server_close()
                thread.join()
