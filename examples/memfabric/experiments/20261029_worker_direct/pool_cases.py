#!/usr/bin/env python3
"""Two-slot real-NPU pool eviction/protection experiment, not inference.

Start one fresh external direct service with pool_gb=1, slot_size_mb=512,
world_size=3, storage rank 0, then start P/D in parallel. Default is a plan;
both --run and --allow-npu are required before importing any device package.
Only descriptors and rendezvous markers are written to coordination files.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


OFFSETS = {"A": 0, "B": 11, "C": 29}
MODEL = "qwen3-synthetic-pool-capacity"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("context",), default="context")
    parser.add_argument("--role", choices=("p", "d"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ascend-source", type=Path, required=True)
    parser.add_argument("--core-source", type=Path, required=True)
    parser.add_argument("--coord-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--server-url", default="tcp://127.0.0.1:25595")
    parser.add_argument("--store-url", default="tcp://127.0.0.1:28695")
    parser.add_argument("--pool-gb", type=int, default=1)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--source-blocks", default="1")
    parser.add_argument("--target-blocks", default="2")
    parser.add_argument("--layout", choices=("split-nhd",), default="split-nhd")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--init-timeout", type=int, default=300)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--allow-npu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        args.source_blocks = [int(x) for x in args.source_blocks.split(",")]
        args.target_blocks = [int(x) for x in args.target_blocks.split(",")]
    except ValueError:
        parser.error("source/target blocks must be comma-separated integers")
    if (args.pool_gb != 1 or args.device < 0 or args.timeout <= 0
            or not 1 <= args.init_timeout <= 3600
            or min(args.layers, args.kv_heads, args.head_size,
                   args.block_size, args.num_blocks) < 1):
        parser.error("this case requires a 1 GiB pool and valid geometry/timeouts")
    if (not args.source_blocks or len(args.source_blocks) != len(args.target_blocks)
            or len(set(args.target_blocks)) != len(args.target_blocks)
            or any(b < 0 or b >= args.num_blocks
                   for b in args.source_blocks + args.target_blocks)):
        parser.error("invalid source/target block lists")
    if not args.run_id or any(not (c.isalnum() or c in "-_") for c in args.run_id):
        parser.error("run-id must contain only letters, digits, '-' or '_'")
    args.skip_blocks = 0
    args.tokens = args.block_size * len(args.source_blocks)
    args.payload_bytes = 2 * args.layers * args.tokens * args.kv_heads * args.head_size * 2
    if args.payload_bytes > 512 << 20:
        parser.error("one object must fit the configured 512 MiB slot")
    args.run_dir = args.coord_dir / args.run_id
    if args.run and not args.dry_run and not args.allow_npu:
        parser.error("--run requires --allow-npu")
    return args


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def marker(args, name, **values):
    write_json(args.run_dir / (name + ".json"), dict(name=name, **values))


def wait_for(args, name):
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        failures = list(args.run_dir.glob("failed-*.json"))
        if failures:
            raise RuntimeError("peer failed: " + failures[0].read_text(encoding="utf-8"))
        target = args.run_dir / (name + ".json")
        if target.exists():
            return json.loads(target.read_text(encoding="utf-8"))
        time.sleep(.02)
    raise TimeoutError("waiting for " + name)


def helper_module(args):
    source = args.ascend_source / "examples/memfabric/validate_direct_kv.py"
    spec = importlib.util.spec_from_file_location("_pool_case_kv_helper", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load existing synthetic KV helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(args, report):
    helper = helper_module(args)
    torch, BMSession, _ = helper.real_imports(args, report)
    from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
    from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
    from lmcache_ascend.v1.distributed.memfabric_direct import (
        MemFabricDirectTransferContext, _ControlRejectedError, chunk_keys,
    )

    copies = []

    class CountedSession(BMSession):
        def copy_npu_to_global(self, src, dst, size):
            began = time.perf_counter_ns()
            result = super().copy_npu_to_global(src, dst, size)
            copies.append(dict(direction="L2G", bytes=size,
                               completed_ns=time.perf_counter_ns() - began))
            return result

        def copy_global_to_npu(self, src, dst, size):
            began = time.perf_counter_ns()
            result = super().copy_global_to_npu(src, dst, size)
            copies.append(dict(direction="G2L", bytes=size,
                               completed_ns=time.perf_counter_ns() - began))
            return result

    caches = helper.allocate_kv(torch, args, patterned=args.role == "p")
    reference = caches if args.role == "p" else helper.allocate_kv(torch, args, patterned=True)
    current_offset = 0
    client = RequestClientFactory.create(args.server_url)
    worker = MemFabricDirectTransferContext(
        instance_id=os.getpid(), req_client=client, tokens_per_chunk=args.tokens,
        mq_timeout=args.timeout, session_factory=CountedSession,
        extra_config={"lmcache.mp.direct": dict(
            device_id=args.device, rank_id=1 if args.role == "p" else 2,
            world_size=3, pool_gb=1, store_url=args.store_url,
            init_timeout=args.init_timeout, member_id=args.run_id + "-" + args.role)})
    pending = []
    lookup_ids = set()
    cleanup_reservations = []
    cleanup_leases = []
    closed = False

    def rpc(method, payload):
        future = worker.control.submit(method, payload)
        item = (method, payload, future)
        pending.append(item)
        response = future.result(timeout=args.timeout)
        pending.remove(item)
        return worker.control.decode(response)

    def variant(label):
        nonlocal current_offset
        delta = OFFSETS[label] - current_offset
        for layer in reference.values():
            for component in range(2):
                helper.plane(layer, component).add_(delta)
        current_offset = OFFSETS[label]
        torch.npu.synchronize(args.device)

    def key(label):
        return IPCCacheServerKey.from_token_ids(
            model_name=MODEL, world_size=1, worker_id=0,
            token_ids=list(range(args.tokens)), start=0, end=args.tokens,
            request_id=args.run_id + "-" + label, cache_salt=args.run_id + "-" + label)

    keys = {label: key(label) for label in OFFSETS}
    object_keys = {label: chunk_keys(value, args.tokens)[0] for label, value in keys.items()}

    def lookup_pin(label, request_id):
        lookup_ids.add(request_id)
        response = rpc("lookup", dict(keys=[object_keys[label]], request_id=request_id))
        if len(response["descriptors"]) != 1:
            raise AssertionError("Expected one committed object for " + label)
        return response["descriptors"][0]

    def store(label):
        variant(label)
        before = len(copies)
        value = worker.submit_store(keys[label].request_id, keys[label], caches,
                                    [args.source_blocks], None, len(args.source_blocks)).result(args.timeout)
        if not value or len(copies) != before + 1 or copies[-1]["direction"] != "L2G":
            raise AssertionError("Store did not complete one real L2G for " + label)

    def restore(label):
        variant(label)
        for layer in caches.values():
            for component in range(2):
                helper.plane(layer, component).fill_(-79)
        torch.npu.synchronize(args.device)
        before = len(copies)
        value = worker.submit_retrieve(keys[label].request_id, keys[label], caches,
            [args.target_blocks], None, len(args.target_blocks)).result(args.timeout)
        if not value or len(copies) != before + 1 or copies[-1]["direction"] != "G2L":
            raise AssertionError("Restore did not complete one real G2L for " + label)
        result = helper.verify_scatter(torch, caches, reference, args)
        report["restored_" + label] = result
        if not all(result.values()):
            raise AssertionError("Restored/untouched NPU content mismatch for " + label)

    def release_queries():
        for request_id in list(lookup_ids):
            response = rpc("release", dict(epoch=worker.registration["epoch"],
                           request_id=request_id, dma_complete=True))
            if not response["released"]:
                raise AssertionError("Completed metadata lookup protection did not release")
            lookup_ids.remove(request_id)

    try:
        worker.register(caches, MODEL, 1, len(args.source_blocks), args.timeout,
                        layout_hints={"kv_layout": "NHD"}, lmcache_chunk_tokens=args.tokens)
        initial = rpc("status", {})
        report.update(initial_status=initial, scratch_device=str(worker.transfer.buffer.device),
                      scratch_bytes=worker.transfer.size, native_format=str(worker.transfer.format))
        if (initial["pool_bytes"] != 1 << 30 or initial["slot_size"] != 512 << 20
                or initial["objects"] != 0 or initial["free_slots"] != 2
                or initial["pending_writes"] != 0 or initial["total_pins"] != 0):
            raise AssertionError("Use a fresh 1 GiB service with exactly two 512 MiB slots")
        marker(args, "joined-" + args.role)
        wait_for(args, "joined-p")
        wait_for(args, "joined-d")
        if args.role == "p":
            store("A")
            store("B")
            descriptors = rpc("lookup", dict(keys=[object_keys["A"], object_keys["B"]]))["descriptors"]
            if len(descriptors) != 2:
                raise AssertionError("A/B were not both committed")
            marker(args, "stored-ab", descriptors=descriptors)
            wait_for(args, "pinned-a")
            store("C")
            marker(args, "stored-c", status=rpc("status", {}))
            wait_for(args, "verified")
        else:
            ab = wait_for(args, "stored-ab")["descriptors"]
            old_b = next(value for value in ab if value["key"] == object_keys["B"])
            lookup_pin("A", args.run_id + "-pin-A")
            marker(args, "pinned-a", status=rpc("status", {}))
            wait_for(args, "stored-c")
            status = rpc("status", {})
            if status["objects"] != 2 or status["evicted_objects"] != 1 or status["total_pins"] != 1:
                raise AssertionError("Expected exactly B eviction while A remains pinned")
            if rpc("lookup", dict(keys=[object_keys["B"]]))["descriptors"]:
                raise AssertionError("Evicted B is still visible")
            identity = worker._identity()
            try:
                unexpected = rpc("acquire", dict(identity, descriptor=old_b))
            except _ControlRejectedError as error:
                if "stale" not in str(error).lower():
                    raise
                report["stale_B_rejected"] = str(error)
            else:
                cleanup_leases.append(unexpected["lease_id"])
                raise AssertionError("Old B descriptor was accepted after its slot was reused")
            restore("A")
            restore("C")
            new_c = lookup_pin("C", args.run_id + "-pin-C")
            if (new_c["slot"] != old_b["slot"]
                    or new_c["generation"] <= old_b["generation"]):
                raise AssertionError("C did not reuse B slot with a newer generation")
            report["reused_slot"] = dict(old_B=old_b, new_C=new_c)
            try:
                unexpected = rpc("acquire", dict(identity,
                    descriptor=dict(new_c, generation=old_b["generation"])))
            except _ControlRejectedError as error:
                if "stale" not in str(error).lower():
                    raise
                report["stale_generation_for_live_C_rejected"] = str(error)
            else:
                cleanup_leases.append(unexpected["lease_id"])
                raise AssertionError("Live C accepted the previous slot generation")
            before_full = rpc("status", {})
            if before_full["total_pins"] != 2:
                raise AssertionError("A/C must both be protected before the pool-full probe")
            try:
                unexpected = rpc("prepare_store", dict(identity,
                    key=args.run_id + "-full-probe", size=worker.transfer.size,
                    layout=worker.transfer.layout, shard=0, tp_size=1))
            except _ControlRejectedError as error:
                if "pool full" not in str(error).lower():
                    raise
                report["all_pinned_prepare_rejected"] = str(error)
            else:
                if unexpected.get("write_id") is not None:
                    cleanup_reservations.append(unexpected["write_id"])
                raise AssertionError("prepare_store unexpectedly succeeded while all slots were pinned")
            if worker.poisoned:
                raise AssertionError("Pure metadata rejection poisoned the NPU context")
            release_queries()
            final = rpc("status", {})
            report["final_status"] = final
            if any(final[name] != 0 for name in ("total_pins", "pending_writes", "quarantined_slots")):
                raise AssertionError("Completed work or protections did not drain")
            marker(args, "verified", status=final)
        report["status_before_close"] = rpc("status", {})
        marker(args, "drained-" + args.role)
        wait_for(args, "drained-p")
        wait_for(args, "drained-d")
        worker.close()
        client.close()
        closed = True
        report["context_and_client_closed_normally"] = True
    finally:
        report["completed_copies"] = list(copies)
        report["completed_bytes"] = {
            direction: sum(item["bytes"] for item in copies if item["direction"] == direction)
            for direction in ("L2G", "G2L")}
        if not closed:
            # First prove actual worker-owned device accesses have drained.
            # Failed drain leaves custom protections untouched, never timed out.
            try:
                worker.close()
                for method, payload, future in list(pending):
                    try:
                        response = worker.control.decode(future.result(timeout=args.timeout))
                    except _ControlRejectedError:
                        pending.remove((method, payload, future))
                        continue
                    pending.remove((method, payload, future))
                    if method == "prepare_store" and response.get("write_id"):
                        cleanup_reservations.append(response["write_id"])
                    elif method == "acquire":
                        cleanup_leases.append(response["lease_id"])
                identity = dict(epoch=worker.registration["epoch"],
                                rank=worker.config["rank_id"], member_id=worker.member_id)
                for write_id in cleanup_reservations:
                    # These probes never submitted any payload DMA.
                    rpc("abort_store", dict(identity, write_id=write_id, dma_complete=True))
                for lease_id in cleanup_leases:
                    rpc("release", dict(identity, lease_id=lease_id, dma_complete=True))
                release_queries()
                client.close()
            except BaseException as error:
                report["cleanup_error"] = repr(error)
                report["protection_completion_unknown"] = True
                report["pending_control_methods"] = [item[0] for item in pending]


def main():
    args = arguments()
    report = dict(status="NOT_EXECUTED", role=args.role, pid=os.getpid(),
                  planned_payload_bytes=args.payload_bytes, object_variants=OFFSETS,
                  pool_bytes=1 << 30, slot_size=512 << 20, slots=2,
                  kv_origin="synthetic_paged_KV_with_distinct_NPU_values",
                  model_weights_loaded=False, model_inference_verified=False,
                  actual_device_initialized=False, cpu_payload_staging_in_harness=False,
                  native_internal_staging_runtime_traced=False,
                  ascend_source=str(args.ascend_source), core_source=str(args.core_source))
    if not args.run or args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    args.run_dir.mkdir(parents=True, exist_ok=True)
    started = args.run_dir / ("started-" + args.role + ".json")
    if started.exists():
        raise RuntimeError("Role already attempted; use a fresh run-id and fixed group")
    write_json(started, dict(pid=os.getpid(), role=args.role))
    began = time.monotonic()
    try:
        execute(args, report)
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        marker(args, "failed-" + args.role, error=repr(error))
    report["elapsed_seconds"] = time.monotonic() - began
    write_json(args.run_dir / ("result-" + args.role + ".json"), report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
