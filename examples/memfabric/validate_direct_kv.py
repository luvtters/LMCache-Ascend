#!/usr/bin/env python3
"""Real-NPU synthetic paged KV validation; no model weights or inference.

primitive mode: three BMSession processes, file metadata rendezvous only.
context mode: two real MemFabricDirectTransferContext workers and an externally
started LMCache service hosting MemFabricDirectServerModule (BM storage rank 0).
The harness never sends a KV payload through its coordination files or RPCs.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("primitive", "context"), required=True)
    parser.add_argument("--role", choices=("pool", "p", "d"), required=True)
    parser.add_argument("--device", type=int, required=True,
                        help="Logical NPU id after the caller's device mask")
    parser.add_argument("--ascend-source", type=Path, required=True)
    parser.add_argument("--core-source", type=Path, required=True)
    parser.add_argument("--coord-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--store-url", default="tcp://127.0.0.1:28680")
    parser.add_argument("--server-url", default="tcp://127.0.0.1:28681")
    parser.add_argument("--pool-gb", type=int, default=1)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--source-blocks", default="1")
    parser.add_argument("--target-blocks", default="2")
    parser.add_argument("--skip-blocks", type=int, default=0)
    parser.add_argument("--layout", choices=("stacked-nhd", "split-nhd"),
                        default="split-nhd")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--allow-npu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.source_blocks = [int(value) for value in args.source_blocks.split(",")]
    args.target_blocks = [int(value) for value in args.target_blocks.split(",")]
    if args.mode == "context" and args.role == "pool":
        parser.error("context mode uses an externally started LMCache pool service")
    if (min(args.layers, args.kv_heads, args.head_size, args.block_size,
            args.num_blocks, args.pool_gb) < 1 or args.device < 0
            or args.timeout <= 0):
        parser.error("geometry, device, pool and timeout must be valid")
    if (not args.source_blocks or len(args.source_blocks) != len(args.target_blocks)
            or len(set(args.target_blocks)) != len(args.target_blocks)
            or any(block < 0 or block >= args.num_blocks
                   for block in args.source_blocks + args.target_blocks)
            or not 0 <= args.skip_blocks <= len(args.target_blocks)):
        parser.error("source/target block lists and skip range are invalid")
    if not args.run_id or any(not (c.isalnum() or c in "-_") for c in args.run_id):
        parser.error("run-id must contain only letters, digits, '-' or '_'")
    for root in (args.ascend_source, args.core_source):
        if not root.is_dir():
            parser.error("source directory does not exist: " + str(root))
    args.tokens = args.block_size * len(args.source_blocks)
    args.payload_bytes = 2 * args.layers * args.tokens * args.kv_heads * args.head_size * 2
    if args.payload_bytes > args.pool_gb << 30:
        parser.error("one KV object exceeds the BM pool")
    if not args.allow_npu and not args.dry_run:
        parser.error("real device use requires --allow-npu")
    args.run_dir = args.coord_dir / args.run_id
    return args


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def marker(args, name, **extra):
    write_json(args.run_dir / (name + ".json"), dict(name=name, **extra))


def wait_for(args, name):
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        failures = list(args.run_dir.glob("failed-*.json"))
        if failures:
            raise RuntimeError("peer failed: " + failures[0].read_text(encoding="utf-8"))
        path = args.run_dir / (name + ".json")
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        time.sleep(0.02)
    raise TimeoutError("timed out waiting for " + name)


def plane(layer, index):
    return layer[index]


def allocate_kv(torch, args, *, patterned):
    """Allocate either stacked or separate K/V planes entirely on the NPU."""
    shape = (args.num_blocks, args.block_size, args.kv_heads, args.head_size)
    elements = 1
    for size in shape:
        elements *= size
    base = torch.arange(elements, dtype=torch.int32, device=f"npu:{args.device}")
    base = base.reshape(shape)
    caches = {}
    for layer in range(args.layers):
        planes = []
        for component in range(2):
            if patterned:
                value = ((base + layer * 37 + component * 53) % 127 - 63)
                value = value.to(torch.bfloat16)
            else:
                value = torch.full(shape, -79, dtype=torch.bfloat16,
                                   device=f"npu:{args.device}")
            planes.append(value)
        caches[f"layer.{layer}"] = (torch.stack(planes, dim=0)
                                      if args.layout == "stacked-nhd" else planes)
    return caches


def expected_chunk(torch, caches, args):
    components = []
    for component in range(2):
        layers = []
        for layer in caches.values():
            selected = torch.cat([plane(layer, component)[block]
                                  for block in args.source_blocks], dim=0)
            layers.append(selected.reshape(args.tokens, args.kv_heads * args.head_size))
        components.append(torch.stack(layers, dim=0))
    return torch.stack(components, dim=0)


def clear_payload(torch, caches, scratch, args):
    for layer in caches.values():
        for component in range(2):
            plane(layer, component).zero_()
    scratch.zero_()
    torch.npu.synchronize(args.device)


def verify_scatter(torch, actual, reference, args):
    restored = True
    untouched = True
    writes = dict(zip(args.target_blocks[args.skip_blocks:],
                      args.source_blocks[args.skip_blocks:]))
    for name, layer in actual.items():
        for component in range(2):
            for block in range(args.num_blocks):
                observed = plane(layer, component)[block]
                if block in writes:
                    expected = plane(reference[name], component)[writes[block]]
                    restored = restored and torch.equal(observed, expected)
                else:
                    untouched = untouched and bool(torch.all(observed == -79).item())
    return {"restored_blocks_equal": restored,
            "unwritten_and_skipped_blocks_unchanged": untouched}


def real_imports(args, report):
    sys.path[:0] = [str(args.ascend_source.resolve()), str(args.core_source.resolve())]
    import torch
    import torch_npu  # noqa: F401
    from lmcache_ascend.v1.distributed.memfabric_bm import BMSession
    from lmcache_ascend.v1.distributed.memfabric_direct_transfer import NpuDirectTransfer
    torch.npu.set_device(args.device)
    report.update(actual_device_initialized=True, torch_version=torch.__version__,
                  loaded_bm_module=sys.modules[BMSession.__module__].__file__,
                  loaded_paged_transfer_module=sys.modules[NpuDirectTransfer.__module__].__file__)
    return torch, BMSession, NpuDirectTransfer


def primitive(args, report):
    torch, BMSession, NpuDirectTransfer = real_imports(args, report)
    session = None
    transfer = None
    try:
        caches = reference = expected = None
        if args.role != "pool":
            caches = allocate_kv(torch, args, patterned=args.role == "p")
            reference = caches if args.role == "p" else allocate_kv(torch, args, patterned=True)
            expected = expected_chunk(torch, reference, args)
            transfer = NpuDirectTransfer(caches, args.tokens, len(args.source_blocks),
                                         layout_hints={"kv_layout": "NHD"})
            report.update(scratch_device=str(transfer.buffer.device),
                          scratch_shape=list(transfer.buffer.shape),
                          native_format=str(transfer.format), bytes=transfer.size)
        session = BMSession(device_id=args.device, rank_id={"pool": 0, "p": 1, "d": 2}[args.role],
                            world_size=3, pool_gb=args.pool_gb, store_url=args.store_url)
        report["bm_join_complete"] = True
        marker(args, "joined-" + args.role)
        if args.role == "p":
            transfer.gather(args.source_blocks)
            report["gather_equal"] = torch.equal(transfer.buffer, expected)
            if not report["gather_equal"]:
                raise AssertionError("native gather does not match paged source KV")
            session.copy_npu_to_global(transfer.buffer.data_ptr(), session.peer_address(0), transfer.size)
            report["bm_l2g_bytes"] = transfer.size
            clear_payload(torch, caches, transfer.buffer, args)
            expected.zero_()
            torch.npu.synchronize(args.device)
            report["source_and_scratch_cleared_after_l2g"] = True
            marker(args, "stored")
            wait_for(args, "restored")
        elif args.role == "d":
            wait_for(args, "stored")
            session.copy_global_to_npu(session.peer_address(0), transfer.buffer.data_ptr(), transfer.size)
            report["bm_g2l_bytes"] = transfer.size
            report["pool_read_equal"] = torch.equal(transfer.buffer, expected)
            if not report["pool_read_equal"]:
                raise AssertionError("BM G2L data does not match independent NPU reference")
            transfer.scatter(args.target_blocks, skip_tokens=args.skip_blocks * args.block_size)
            report.update(verify_scatter(torch, caches, reference, args))
            if not all(report[name] for name in ("restored_blocks_equal", "unwritten_and_skipped_blocks_unchanged")):
                raise AssertionError("native scatter changed unexpected KV blocks")
            marker(args, "restored")
        else:
            wait_for(args, "restored")
        marker(args, "drained-" + args.role)
        for role in ("pool", "p", "d"):
            wait_for(args, "drained-" + role)
        if transfer is not None:
            transfer.synchronize()
        session.close()
        session = None
        report["bm_closed_normally"] = True
    finally:
        # A failure must remain visible. If the owner can prove completion it
        # may close; a failed close is reported rather than called a pass.
        if session is not None:
            if transfer is not None:
                transfer.synchronize()
            session.close()


def context_mode(args, report):
    torch, _, _ = real_imports(args, report)
    from types import SimpleNamespace
    from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
    from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
    from lmcache_ascend.v1.distributed.memfabric_direct import (
        MemFabricDirectSchedulerBackend, MemFabricDirectTransferContext,
    )
    caches = allocate_kv(torch, args, patterned=args.role == "p")
    reference = caches if args.role == "p" else allocate_kv(torch, args, patterned=True)
    client = RequestClientFactory.create(args.server_url)
    worker = MemFabricDirectTransferContext(
        instance_id=os.getpid(), req_client=client, tokens_per_chunk=args.tokens,
        mq_timeout=args.timeout, extra_config={"lmcache.mp.direct": {
            "device_id": args.device, "rank_id": 1 if args.role == "p" else 2,
            "world_size": 3, "pool_gb": args.pool_gb, "store_url": args.store_url,
            "member_id": args.run_id + "-" + args.role}})
    scheduler = None
    closed = False
    try:
        worker.register(caches, "qwen3-synthetic-paged-validation", 1,
                        len(args.source_blocks), args.timeout,
                        layout_hints={"kv_layout": "NHD"},
                        lmcache_chunk_tokens=args.tokens)
        report.update(direct_context_registered=True,
                      rpc_coverage="real_LMCache_RequestClient_and_direct_service",
                      request_client_class=type(client).__module__ + "." + type(client).__name__,
                      loaded_direct_context_module=sys.modules[MemFabricDirectTransferContext.__module__].__file__,
                      scratch_device=str(worker.transfer.buffer.device),
                      scratch_shape=list(worker.transfer.buffer.shape),
                      native_format=str(worker.transfer.format), bytes=worker.transfer.size,
                      service_status_before=worker.control.call("status", {}))
        cache_key = IPCCacheServerKey.from_token_ids(
            model_name="qwen3-synthetic-paged-validation", world_size=1, worker_id=0,
            token_ids=list(range(args.tokens)), start=0, end=args.tokens,
            request_id=args.run_id + "-d", cache_salt=args.run_id)
        marker(args, "joined-" + args.role)
        if args.role == "p":
            worker.submit_store(args.run_id + "-p", cache_key, caches,
                                [args.source_blocks], None, len(args.source_blocks)).result(args.timeout)
            report["bm_l2g_bytes_requested"] = worker.transfer.size
            clear_payload(torch, caches, worker.transfer.buffer, args)
            report["source_and_scratch_cleared_after_l2g"] = True
            marker(args, "stored")
            wait_for(args, "restored")
        else:
            wait_for(args, "stored")
            scheduler = MemFabricDirectSchedulerBackend(
                req_clients={args.server_url: client}, tokens_per_chunk=args.tokens,
                mq_timeout=args.timeout, parallel_strategy=SimpleNamespace(
                    tp_size=1, pp_size=1, dcp_size=1, n_servers=1,
                    vllm_world_size=1, mla_only=False))
            lookup = scheduler.create_lookup(cache_key.no_worker_id_version(), tp_size=1,
                tokens_per_chunk=args.tokens, timeout=args.timeout, blocking_lookup=True)
            lookup.submit()
            failed = lookup.on_scheduler_check()
            report["scheduler_hit_tokens"] = lookup.get_result()
            if failed or lookup.get_result() != args.tokens:
                raise AssertionError("direct scheduler did not find the stored KV chunk")
            restored = worker.submit_retrieve(cache_key.request_id, cache_key, caches,
                [args.target_blocks], None, len(args.target_blocks),
                skip_first_n_tokens=args.skip_blocks * args.block_size).result(args.timeout)
            if not restored:
                raise AssertionError("direct context returned a cache miss")
            report["bm_g2l_bytes_requested"] = worker.transfer.size
            report.update(verify_scatter(torch, caches, reference, args))
            if not all(report[name] for name in ("restored_blocks_equal", "unwritten_and_skipped_blocks_unchanged")):
                raise AssertionError("context restored incorrect paged KV")
            scheduler.end_session(cache_key.request_id)
            report["service_status_after"] = worker.control.call("status", {})
            if report["service_status_after"]["total_pins"] != 0:
                raise AssertionError("lookup/read protection was not released")
            marker(args, "restored")
        marker(args, "drained-" + args.role)
        for role in ("p", "d"):
            wait_for(args, "drained-" + role)
        if scheduler is not None:
            scheduler.close()
        worker.close()
        client.close()
        closed = True
        report["context_and_client_closed_normally"] = True
    finally:
        if not closed:
            if scheduler is not None:
                scheduler.close()
            worker.close()
            client.close()


def main():
    args = arguments()
    report = dict(
        status="NOT_EXECUTED", mode=args.mode, role=args.role,
        pid=os.getpid(), device=args.device, source_blocks=args.source_blocks,
        target_blocks=args.target_blocks, skip_blocks=args.skip_blocks,
        layers=args.layers, kv_heads=args.kv_heads, head_size=args.head_size,
        block_size=args.block_size, chunk_tokens=args.tokens, layout=args.layout,
        dtype="torch.bfloat16", planned_payload_bytes=args.payload_bytes,
        kv_origin="synthetic_qwen3_style_paged_tensors",
        model_weights_loaded=False, model_inference_verified=False,
        vllm_connector_or_worker_adapter_verified=False,
        actual_device_initialized=False,
        rpc_coverage=("NOT_EXECUTED" if args.mode == "context"
                      else "NOT_VERIFIED_file_metadata_rendezvous_only"),
        cpu_payload_staging_in_harness=False,
        native_internal_staging_runtime_traced=False,
        direct_parent_revision="be90c96",
        source_revision_verified_by_script=False,
        ascend_source=str(args.ascend_source.resolve()),
        core_source=str(args.core_source.resolve()),
        visible_devices={name: os.environ.get(name) for name in
            ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES")},
    )
    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    started = args.run_dir / ("started-" + args.role + ".json")
    if started.exists():
        raise RuntimeError("role already started; use a fresh run-id")
    write_json(started, dict(pid=os.getpid(), role=args.role))
    began = time.monotonic()
    try:
        (primitive if args.mode == "primitive" else context_mode)(args, report)
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=str(error), traceback=traceback.format_exc())
        marker(args, "failed-" + args.role, error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - began
        write_json(args.run_dir / ("result-" + args.role + ".json"), report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
