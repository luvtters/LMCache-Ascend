#!/usr/bin/env python3
"""Opt-in native NPU-page batch primitive, independent of LMCache integration.

Default is PLAN_ONLY. CPU coordination carries metadata only. Each process owns
one BM rank and retains its page tensors until native batch, wait and close have
returned. The outer executor must record the real process wait/exit separately.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

from batch_address_plan import plan_segments
from direct_coordination import publish, wait_event


SOURCE_BLOCKS = [3, 1]
TARGET_BLOCKS = [4, 2]
_FAILED_KEEPALIVE = []


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("pool", "p", "d"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, choices=(4, 36), default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--pool-memory-type", choices=("host", "device"), default="host")
    parser.add_argument("--store-url", default="tcp://192.168.0.50:28715")
    parser.add_argument("--coord-url", default="http://192.168.0.50:29615")
    parser.add_argument("--init-timeout", type=int, default=300)
    parser.add_argument("--timeout", type=float, default=360)
    parser.add_argument("--ascend-source", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--core-source", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--plan-only", action="store_true")
    parser.add_argument("--allow-npu", action="store_true")
    args = parser.parse_args(argv)
    if (args.device < 0 or args.timeout <= 0 or not 1 <= args.init_timeout <= 3600
            or not args.run_id or any(not (c.isalnum() or c in "-_") for c in args.run_id)):
        parser.error("invalid device, timeout, or run-id")
    if args.run and not args.allow_npu:
        parser.error("--run requires --allow-npu")
    return args


def plan(args):
    return dict(status="PLAN_ONLY", role=args.role,
        rank={"pool": 0, "p": 1, "d": 2}[args.role], world_size=3,
        run_id=args.run_id, layers=args.layers, dtype="torch.bfloat16",
        num_blocks=6, block_size=64, chunk_tokens=128, kv_heads=8, head_size=128,
        source_blocks=SOURCE_BLOCKS, target_blocks=TARGET_BLOCKS,
        payload_bytes=2 * args.layers * 128 * 8 * 128 * 2,
        expected_segments=4 * args.layers,
        device=args.device, store_url=args.store_url, coord_url=args.coord_url,
        physical_device_mask=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        pool_memory_type=args.pool_memory_type,
        declared_dram_bytes_per_rank=1 << 30,
        declared_hbm_bytes_per_rank=(1 << 30) if args.pool_memory_type == "device" else 0,
        scratch_allocated=False, production_integration=False,
        native_batch_verified=False, process_exit="NOT_OBSERVED_BY_SELF")


def session_config(args):
    config = dict(device_id=args.device, rank_id={"pool": 0, "p": 1, "d": 2}[args.role],
        world_size=3, pool_gb=1, store_url=args.store_url, init_timeout=args.init_timeout)
    if args.pool_memory_type == "device":
        config["pool_memory_type"] = "device"
    return config


def validate_local_segments(planes, segments, expected):
    if len(segments) != expected or len({s['src'] for s in segments}) != expected:
        raise AssertionError('Batch needs distinct original local page pointers')
    allocated = {(p['component'], p['layer']): p for p in planes}
    for segment in segments:
        plane = allocated[(segment['component'], segment['layer'])]
        if not (plane['base_ptr'] <= segment['src'] < segment['src'] + segment['size'] <=
                plane['base_ptr'] + plane['num_blocks'] * plane['block_stride_bytes']):
            raise AssertionError('Segment extends beyond its retained original NPU plane')


def owner_pool_addresses(session, pool_memory_type, local_device_type):
    result = dict(owner_gva=int(session.peer_address(0)))
    if pool_memory_type == 'device':
        mapped = int(session.handle.gva_to_va(result['owner_gva'], local_device_type))
        if not mapped:
            raise RuntimeError('Selected owner pool has no actual LOCAL_DEVICE mapping')
        result['owner_local_device_va'] = mapped
    return result


def batch_arrays(segments, *, retrieve=False):
    """The D planner maps its target pages to the same packed pool layout."""
    if not segments:
        raise ValueError("Cannot submit an empty batch")
    src = [s["dst"] if retrieve else s["src"] for s in segments]
    dst = [s["src"] if retrieve else s["dst"] for s in segments]
    sizes = [s["size"] for s in segments]
    return src, dst, sizes


def cpu_plane(torch, layer, component):
    """Independent complete CPU index pattern; BF16 values remain exact."""
    index = torch.arange(6 * 64 * 8 * 128, dtype=torch.int32, device="cpu")
    return ((index + layer * 37 + component * 53) % 127 - 63).to(
        torch.bfloat16).reshape(6, 64, 8, 128)


def allocate_pages(torch, args):
    pages = []
    for component in (0, 1):
        for layer in range(args.layers):
            value = (cpu_plane(torch, layer, component) if args.role == "p" else
                     torch.full((6, 64, 8, 128), -79, dtype=torch.bfloat16, device="cpu"))
            tensor = value.to(f"npu:{args.device}")
            pages.append((component, layer, tensor))
    torch.npu.synchronize(args.device)
    return pages


def describe_pages(pages):
    return [dict(component=component, layer=layer, base_ptr=int(tensor.data_ptr()),
                 block_stride_bytes=int(tensor.stride(0) * tensor.element_size()),
                 token_stride_bytes=int(tensor.stride(1) * tensor.element_size()),
                 token_bytes=int(tensor.shape[2] * tensor.shape[3] * tensor.element_size()),
                 num_blocks=int(tensor.shape[0])) for component, layer, tensor in pages]


def verify_pages(torch, pages):
    results = []
    for component, layer, tensor in pages:
        observed = tensor.cpu()
        reference = cpu_plane(torch, layer, component)
        expected = torch.full_like(reference, -79)
        for source, target in zip(SOURCE_BLOCKS, TARGET_BLOCKS):
            expected[target].copy_(reference[source])
        written = all(torch.equal(observed[target], reference[source])
                      for source, target in zip(SOURCE_BLOCKS, TARGET_BLOCKS))
        untouched = all(torch.equal(observed[block], expected[block])
                        for block in range(6) if block not in TARGET_BLOCKS)
        exact = torch.equal(observed, expected)
        results.append(dict(component=component, layer=layer,
                            written_pages_equal=written, untouched_pages_equal=untouched,
                            all_elements_equal=exact,
                            checked_elements=int(observed.numel())))
    return results


def execute(args, report, record):
    sys.path.insert(0, str(args.ascend_source.resolve()))
    if args.core_source:
        sys.path.insert(1, str(args.core_source.resolve()))
    import torch
    import torch_npu  # noqa: F401
    from lmcache_ascend.v1.distributed.memfabric_bm import BMSession
    import memfabric_hybrid as mf
    from memfabric_hybrid import bm

    report.update(torch_version=torch.__version__, memfabric_version=importlib.metadata.version(
        "memfabric-hybrid"), memfabric_module=mf.__file__, bm_wrapper_module=sys.modules[
        BMSession.__module__].__file__, host=socket.gethostname(), pid=os.getpid())
    torch.npu.set_device(args.device)
    pages = []  # Strong ownership through batch/wait/close, even on failure.
    session = None
    primary_error = None
    try:
        if args.role != "pool":
            pages = allocate_pages(torch, args)
            report["plane_metadata"] = describe_pages(pages)
        record("bm_init", "begin")
        session = BMSession(**session_config(args))
        report['session_pool_memory_type'] = getattr(session, 'pool_memory_type', 'host')
        if report['session_pool_memory_type'] != args.pool_memory_type:
            raise AssertionError('BM wrapper selected a different pool memory type')
        record("bm_init", "complete")
        publish(args.coord_url, args.run_id, "joined-" + args.role,
                dict(name="joined-" + args.role, rank=report["rank"], layers=args.layers))
        if args.role != "pool":
            report.update(owner_pool_addresses(session, args.pool_memory_type,
                                               bm.BmMemType.LOCAL_DEVICE))
            owner_gva = report['owner_gva']
            segments = plan_segments(report["plane_metadata"],
                SOURCE_BLOCKS if args.role == "p" else TARGET_BLOCKS,
                block_size=64, token_start=0, token_count=128,
                pool_base=owner_gva, pool_bytes=1 << 30)
            validate_local_segments(report['plane_metadata'], segments, report['expected_segments'])
            report["segments"] = segments
            if len(segments) != report["expected_segments"]:
                raise AssertionError("Unexpected actual plane/block segment count")
            if args.role == "d":
                stored = wait_event(args.coord_url, args.run_id, "stored", args.timeout)
                if stored["layers"] != args.layers or stored["completed_bytes"] != report["payload_bytes"]:
                    raise AssertionError("Peer geometry or successful byte count differs")
            sources, destinations, sizes = batch_arrays(segments, retrieve=args.role == "d")
            if sum(sizes) != report["payload_bytes"]:
                raise AssertionError("Batch does not cover exactly the packed chunk")
            direction = bm.BmCopyType.L2G if args.role == "p" else bm.BmCopyType.G2L
            report["batch_requested_bytes"] = sum(sizes)
            report["batch_completed_bytes"] = 0
            record("native_batch", "begin", count=len(sizes), bytes=sum(sizes),
                   direction=str(direction), flags=0, stream=0)
            result = session.handle.copy_data_batch(sources, destinations, sizes,
                                                   len(sizes), direction, 0, 0)
            report["native_batch_return_code"] = result
            record("native_batch", "returned", return_code=result)
            session.check(result, "BM native batch")
            record("batch_wait", "begin")
            session.wait()
            record("batch_wait", "complete")
            torch.npu.synchronize(args.device)
            report["batch_completed_bytes"] = sum(sizes)
            report["native_batch_verified"] = True
            record("native_batch", "complete", bytes=sum(sizes))
            if args.role == "p":
                for _, _, tensor in pages:
                    tensor.zero_()
                torch.npu.synchronize(args.device)
                report["source_pages_cleared_after_batch_wait"] = True
                publish(args.coord_url, args.run_id, "stored", dict(name="stored",
                    layers=args.layers, completed_bytes=report["batch_completed_bytes"]))
                wait_event(args.coord_url, args.run_id, "restored", args.timeout)
            else:
                report["verification"] = verify_pages(torch, pages)
                report["verification_uses_cpu_copies_after_completion"] = True
                if not all(r["all_elements_equal"] and r["written_pages_equal"]
                           and r["untouched_pages_equal"] for r in report["verification"]):
                    raise AssertionError("Full KV page verification failed")
                publish(args.coord_url, args.run_id, "restored", dict(name="restored",
                    layers=args.layers, checked_planes=len(pages), all_elements_equal=True))
        else:
            wait_event(args.coord_url, args.run_id, "restored", args.timeout)
        record("final_wait", "begin")
        session.wait()
        record("final_wait", "complete")
        publish(args.coord_url, args.run_id, "drained-" + args.role,
                dict(name="drained-" + args.role, rank=report["rank"]))
        for role in ("pool", "p", "d"):
            wait_event(args.coord_url, args.run_id, "drained-" + role, args.timeout)
        report["content_and_drain_complete"] = True
    except BaseException as exc:
        primary_error = exc
        report.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
        record("operation", "error", error=repr(exc))
        try:
            publish(args.coord_url, args.run_id, "failed-" + args.role,
                    dict(name="failed-" + args.role, error=repr(exc)))
        except Exception as coordination_error:
            report["failure_publication_error"] = repr(coordination_error)
    finally:
        if session is not None:
            try:
                record("bm_close", "begin")
                session.close()
                report["bm_closed_normally"] = True
                record("bm_close", "complete")
            except BaseException as close_error:
                report["bm_close_error"] = repr(close_error)
                record("bm_close", "error", error=repr(close_error))
                if primary_error is None:
                    primary_error = close_error
        # pages stays in this scope until close has returned or failed. If a
        # native wait fails, stop this process; never reuse its pool or pages.
    if primary_error is not None:
        # Avoid refcount/GC disposal of pages whose DMA completion may remain
        # unknown. main persists the failure then uses immediate process exit;
        # the outer executor must tear down this entire fixed group.
        _FAILED_KEEPALIVE.append((pages, session, locals().get("sources"),
                                  locals().get("destinations"), locals().get("sizes")))
        raise primary_error
    report["status"] = "NATIVE_BATCH_PRIMITIVE_PASS_PROCESS_EXIT_PENDING"


def main(argv=None):
    args = arguments(argv)
    report = plan(args)
    if not args.run:
        print(json.dumps(report, indent=2))
        return
    directory = args.output_dir / args.run_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("result-" + args.role + ".json")
    if path.exists():
        raise RuntimeError("Existing role result; use a new run-id")

    def record(operation, phase, **extra):
        event = dict(operation=operation, phase=phase, mono_ns=time.monotonic_ns(),
                     pid=os.getpid(), role=args.role, run_id=args.run_id, **extra)
        with (directory / ("events-" + args.role + ".jsonl")).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
            stream.flush()
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    report["status"] = "RUNNING"
    failed = False
    try:
        execute(args, report, record)
    except BaseException as exc:
        report.update(status="FAIL", final_error=repr(exc))
        failed = True
        try:
            publish(args.coord_url, args.run_id, "failed-" + args.role,
                    dict(name="failed-" + args.role, error=repr(exc)))
        except Exception as coordination_error:
            report["failure_publication_error"] = repr(coordination_error)
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    if failed:
        # No Python/torch/native finalizer runs on completion-unknown tensors.
        os._exit(1)


if __name__ == "__main__":
    main()
