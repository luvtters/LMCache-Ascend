# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression and cost probes for the in-process connector."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
from pathlib import Path
import pickle
import platform
from statistics import median
import subprocess
import threading
from time import perf_counter_ns, sleep
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

import inprocess_cpu_fixture as cpu
import test_inprocess_lifecycle_cpu as src


def step_lookup(mode):
    engine = cpu.engine_for(asynchronous=False)
    scheduler, worker = cpu.connector(), cpu.connector(engine)
    client = scheduler._lmcache_engine.lookup_client
    context = NS(attn_metadata=None)
    try:
        engine.store([1, 2], req_id="seed", source=cpu.tensor(0, 3, 2))
        client.lookup([1, 2], "a")
        if mode != "replay":
            client.lookup([1, 2], "b")
        worker.step = scheduler.build_connector_meta(src.step())
        original = engine.lookup
        def lookup(**kwargs):
            hit = original(**kwargs)
            if mode == "lookup_error" and kwargs["lookup_id"] == "b":
                raise ValueError("injected after pin")
            return hit
        with patch.object(engine, "lookup", side_effect=lookup) as calls:
            if mode == "replay":
                worker.start_load_kv(context)
                first = worker.build_connector_worker_meta().replies
                worker.start_load_kv(context)
                repeat = worker.build_connector_worker_meta().replies
                worker.step = scheduler.build_connector_meta(src.step(finished=["a"]))
                worker.start_load_kv(context)
                return dict(pins=sum(engine.storage_manager.pins.values()), calls=calls.call_count,
                            same_hit=first == repeat == {0: {0: 2}})
            with patch.object(cpu.AdapterBase, "start_load_kv", side_effect=(
                ValueError("injected load failure") if mode == "load_error" else None
            )):
                try:
                    worker.start_load_kv(context)
                except ValueError as exc:
                    if "injected" not in str(exc):
                        raise
                else:
                    raise AssertionError("fault injection did not run")
            pins = sum(engine.storage_manager.pins.values())
            stale_reply = worker.build_connector_worker_meta() is not None
        worker.start_load_kv(context)
        retry = worker.build_connector_worker_meta()
        worker.step = scheduler.build_connector_meta(src.step(finished=["a", "b"]))
        worker.start_load_kv(context)
        return dict(pins=pins, stale_reply=stale_reply, retry_ok=retry.replies ==
                    {0: {0: 2}, 1: {0: 2}}, final_pins=sum(engine.storage_manager.pins.values()))
    finally:
        engine.close()


def allocation_fault():
    engine = cpu.engine_for(asynchronous=False)
    mem = engine.storage_manager
    allocate = mem.allocate
    def fail_second(*args, **kwargs):
        if mem.objects:
            raise ValueError("second chunk allocation failed")
        return allocate(*args, **kwargs)
    try:
        with patch.object(mem, "allocate", side_effect=fail_second):
            try:
                engine.store([1, 2, 3, 4], req_id="fail", source=cpu.tensor(0, 1, 4))
            except ValueError as exc:
                if str(exc) != "second chunk allocation failed":
                    raise
            else:
                raise AssertionError("allocator did not raise")
        return dict(refs=[o.refs for o in mem.objects], published=len(mem.data))
    finally:
        engine.close()


def stale_step():
    engine = cpu.engine_for()
    sched, worker = cpu.connector(), cpu.connector(engine)
    context = NS(attn_metadata=None)
    try:
        engine.store([1, 2], req_id="seed", source=cpu.tensor(0, 1, 2))
        engine.wait_for_pending_stores({"seed"})
        old = sched.build_connector_meta(src.step(preempted=["reused"], finished=["reused"]))
        worker.step = old
        worker.handle_preemptions(old)
        worker.start_load_kv(context)
        sched._lmcache_engine.lookup_client.lookup([1, 2], "reused")
        worker.step = sched.build_connector_meta(src.step())
        worker.start_load_kv(context)
        worker.build_connector_worker_meta()
        worker.step = old
        worker.handle_preemptions(old)
        worker.start_load_kv(context)
        held = sum(engine.storage_manager.pins.values())
        worker.step = sched.build_connector_meta(src.step(finished=["reused"]))
        worker.start_load_kv(context)
        return dict(while_active=held, after_release=sum(engine.storage_manager.pins.values()))
    finally:
        engine.close()


def store_events():
    engine = cpu.engine_for(asynchronous=False)
    engine.kv_events_enabled = True
    engine.kv_events = []
    class Key(tuple):
        @property
        def chunk_hash(self):
            return self[-1]
    chunks = engine.token_database.process_tokens
    def keyed(*args, **kwargs):
        for start, end, key in chunks(*args, **kwargs):
            yield start, end, Key(key)
    globals_ = inspect.unwrap(engine._run_store_pipeline).__globals__
    try:
        with patch.object(engine.token_database, "process_tokens", keyed), patch.dict(
            globals_, CacheStoreEvent=NS, convert_tokens_to_list=lambda tokens, a, b: tokens[a:b]
        ):
            engine.gpu_connector.fail = True
            try:
                engine.store([1, 2], req_id="failed", source=cpu.tensor(0, 1, 2))
            except ValueError as exc:
                if str(exc) != "injected copy failure":
                    raise
            else:
                raise AssertionError("copy did not fail")
            failed = len(engine.get_kv_events())
            engine.gpu_connector.fail = False
            engine.store([1, 2, 3, 4], req_id="ok", source=cpu.tensor(0, 4, 4))
            events = list(engine.get_kv_events())
        return dict(failed_events=failed, committed=[e.block_hashes for e in events],
                    parents=[e.parent_block_hash for e in events])
    finally:
        engine.close()


def empty_save():
    save = src.method(src.INTEGRATION / "vllm_v1_adapter.py", "LMCacheAscendConnectorV1Impl",
                      "wait_for_save", torch=torch, LMCacheConnectorMetadata=src.Meta)
    impl = cpu.connector(cpu.engine_for(asynchronous=False))._lmcache_engine
    impl._parent.step = src.Meta([])
    event, replay = Mock(), Mock()
    try:
        with patch.object(torch, "npu", NS(Event=lambda: event), create=True), patch.object(
            impl, "_replay_finished_stores_after_save", replay
        ):
            save(impl)
        return dict(event_records=event.record.call_count, finish_polled=replay.call_count)
    finally:
        impl.lmcache_engine.close()


def contention():
    # Hold a copy, fill four queue slots, then independently hold publication.
    gate, copying = threading.Event(), threading.Event()
    publish_gate, publishing = threading.Event(), threading.Event()
    engine = cpu.engine_for(gate=gate, entered=copying)
    src_tensor = cpu.tensor(0, 4, 2)
    def store(i):
        engine.store([1, 2], req_id=str(i), source=src_tensor)
    put = engine.storage_manager.batched_put
    def held_publish(*args, **kwargs):
        publishing.set()
        if not publish_gate.wait(3):
            raise TimeoutError("publication gate")
        put(*args, **kwargs)
    def observed(action, entered):
        start = perf_counter_ns()
        entered.set()
        action()
        return (perf_counter_ns() - start) / 1e6
    try:
        with ThreadPoolExecutor(2) as pool, patch.object(engine.storage_manager, "batched_put", held_publish):
            try:
                store(0)
                assert copying.wait(2)
                start = perf_counter_ns()
                hit = engine.lookup([1, 2])
                free_ms = (perf_counter_ns() - start) / 1e6
                for i in range(1, 5):
                    store(i)
                enqueue_started = threading.Event()
                waiting = pool.submit(observed, lambda: store(5), enqueue_started)
                assert enqueue_started.wait(2)
                sleep(0.025)
                blocked = not waiting.done()
                gate.set()
                assert publishing.wait(2)
                lookup_started = threading.Event()
                lookup = pool.submit(observed, lambda: engine.lookup([1, 2]), lookup_started)
                assert lookup_started.wait(2)
                sleep(0.025)
                locked = not lookup.done()
            finally:
                gate.set()
                publish_gate.set()
            return dict(queue_capacity=4, injected_wait_ms_per_gate=25,
                        lookup_while_copy_hit=hit, lookup_while_copy_ms=free_ms,
                        full_queue_blocked=blocked, enqueue_ms=waiting.result(3),
                        publication_blocked_lookup=locked, lookup_ms=lookup.result(3))
    finally:
        engine.close()


def cost():
    tokens = [(i * 3571) % 100003 for i in range(8192)]
    tables = (tuple(range(1, 129)), tuple(range(129, 257)))
    rows = []
    for count in (1, 8, 32):
        samples = []
        for _ in range(10):
            client = src.lookup.StepLookupClient(32)
            start = perf_counter_ns()
            for i in range(count):
                client.lookup(tokens, str(i))
            packet = [(q.sequence, q.request_id, q.tokens, q.configs) for q in client.take_requests()]
            staged = perf_counter_ns()
            blob = pickle.dumps(packet, protocol=5)
            encoded = perf_counter_ns()
            pickle.loads(blob)
            decoded = perf_counter_ns()
            client.receive({seq: dict.fromkeys(range(32), 0) for seq, *_ in packet})
            for _, req, *_ in packet:
                assert client.lookup_cache(req) == 0
            samples.append([(end - begin) / 1e6 for begin, end in
                            ((start, staged), (staged, encoded), (encoded, decoded),
                             (decoded, perf_counter_ns()))])
        rows.append(dict(requests=count, pickle_bytes=len(blob),
                         median_ms=dict(zip(("stage", "encode", "decode", "receive_and_poll_32_ranks"),
                                            map(median, zip(*samples[1:]))))))
    # Isolate the exact slot-mapping operations from ReqMeta.from_request_tracker.
    def slot_map():
        blocks = torch.tensor(tables[0], dtype=torch.long)
        offsets = torch.arange(64, dtype=torch.long)
        return (offsets.reshape(1, 64) + blocks.reshape(-1, 1) * 64).flatten()[:8192]
    times = []
    for _ in range(101):
        start = perf_counter_ns()
        slots = slot_map()
        times.append((perf_counter_ns() - start) / 1e6)
    scheduler = src.create("SCHEDULER")
    slot_ms = median(times[1:])
    group_rows = []
    for count in (0, 1, 32):
        ids = [str(i) for i in range(count)]
        request = src.step(reqs=[NS(req_id=i) for i in ids],
                           cached=NS(req_ids=ids, new_block_ids=[((129,), (258,))] * count,
                                     resumed_req_ids=set()))
        times = []
        for _ in range(101):
            scheduler._tables.update(dict.fromkeys(ids, tables))
            start = perf_counter_ns()
            scheduler.build_connector_meta(request)
            times.append((perf_counter_ns() - start) / 1e6)
        group_rows.append(dict(requests=count, metadata_median_ms=median(times[1:])))
    return dict(lookup=rows, group_table_append=group_rows, block_ids_per_group=128, block_size=64,
                slot_tensor_bytes=slots.numel()*slots.element_size(),
                slot_build_median_ms=slot_ms, serialization="pickle protocol 5 proxy")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    path = p.parse_args().output
    torch.set_num_threads(1)
    results = {}
    for name in ("replay", "lookup_error", "load_error"):
        results[name] = step_lookup(name)
    results.update(stale_step=stale_step(), events=store_events(), allocation=allocation_fault(), empty=empty_save(),
                   contention=contention(), cost=cost())
    expected = dict(replay=dict(pins=0, calls=1, same_hit=True),
                    events=dict(failed_events=0, committed=[[2], [4]], parents=[None, 2]),
                    stale_step=dict(while_active=1, after_release=0),
                    allocation=dict(refs=[0], published=0), empty=dict(event_records=0, finish_polled=1))
    for name in ("lookup_error", "load_error"):
        expected[name] = dict(pins=0, stale_reply=False, retry_ok=True, final_pins=0)
    checks = {name: results[name] == want for name, want in expected.items()}
    checks["contention"] = all(results["contention"][k] for k in
                               ("full_queue_blocked", "publication_blocked_lookup"))
    report = dict(python=platform.python_version(), torch=torch.__version__, npu_tested=False,
                  ascend_head=subprocess.check_output(["git", "-C", str(src.ROOT), "rev-parse", "HEAD"], text=True).strip(),
                  checks=checks, results=results)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(checks)
    return int(not all(checks.values()))


if __name__ == "__main__":
    raise SystemExit(main())
