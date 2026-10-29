#!/usr/bin/env python3
"""Metadata-only peer-left rejection probe; default PLAN_ONLY needs stdlib.

Uses the real LMCache RequestClient only with --run. Its dependencies may
passively import CPU Torch; this script never imports an accelerator backend,
registers a worker, constructs BM, or submits payload data/DMA.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid


COUNTS = ("objects", "free_slots", "total_pins", "pending_writes", "quarantined_slots")


def audit_imports():
    prefixes = ("torch_npu", "memfabric", "memfabric_hybrid", "mf_smem", "smem", "acl", "mooncake.engine")
    forbidden = sorted(name for name in sys.modules if any(
        name == prefix or name.startswith(prefix + ".") for prefix in prefixes))
    maps = Path("/proc/self/maps")
    accelerator_maps = sorted({line.rsplit(None, 1)[-1] for line in maps.read_text().splitlines()
                               if any(name in line for name in ("libtorch_npu", "libmf_smem", "libhybm", "libascendcl"))}) if maps.exists() else []
    return dict(cpu_torch_imported="torch" in sys.modules, forbidden_modules=forbidden,
                accelerator_library_mappings=accelerator_maps,
                device_backend_autoload=os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD"))


def peer_left(value):
    if not isinstance(value, dict):
        return False
    message = value.get("last_error", "")
    if not isinstance(message, str):
        return False
    return value.get("ready") is False and "fixed group unavailable" in message.lower() \
        and "peer" in message.lower() and "left" in message.lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="tcp://127.0.0.1:25595")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    plan = dict(mode="RUN" if args.run else "PLAN_ONLY", server_url=args.server_url,
                output_root=str(args.output_root), operations=["status", "lookup", "prepare_store", "status"],
                lookup_creates_pin=False, prepare_uses_unregistered_identity=True,
                registers_worker=False, initializes_device=False, submits_dma=False,
                cpu_torch_may_be_imported_indirectly=bool(args.run))
    print(json.dumps(plan, indent=2), flush=True)
    if not args.run:
        return 0
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")
    if args.output_root.exists():
        raise FileExistsError("Refuse to overwrite probe evidence")
    args.output_root.mkdir(parents=True)
    report = dict(plan=plan, started_unix=time.time(), calls=[], passed=False, client_closed=False,
                  checks={}, pending_futures=[], generation_validation_not_tested=True)
    client, pending = None, []

    def save():
        (args.output_root / "receipt.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def require_cpu_only(stage):
        record = audit_imports()
        report[stage] = record
        save()
        if record["forbidden_modules"] or record["accelerator_library_mappings"]:
            raise RuntimeError("Accelerator import/mapping observed; refusing metadata probe")

    try:
        # This is a process-local choice, never a change to the service or host.
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        require_cpu_only("before_factory_import")
        from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
        from lmcache.v1.multiprocess.protocols.server_module import ServerModuleCallRequest
        require_cpu_only("after_factory_import")
        client = RequestClientFactory.create(args.server_url)
        require_cpu_only("before_rpc")

        def call(method, payload):
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            record = dict(method="memfabric_direct." + method, request_payload_utf8=encoded.decode(),
                          submitted_unix=time.time(), outcome="PENDING")
            report["calls"].append(record)
            save()
            future = client.server_module_call(ServerModuleCallRequest(method=record["method"], payload=encoded))
            pending.append((record, future))
            try:
                response = future.result(timeout=args.timeout)
            except TimeoutError as error:
                record.update(outcome="TIMEOUT_UNKNOWN", error=repr(error))
                save()
                # Retain the original Future, do not cancel/resubmit or call a
                # timeout an operation rejection. Client close is recorded below.
                raise
            except BaseException as error:
                record.update(outcome="TRANSPORT_OR_FUTURE_ERROR", error=repr(error))
                save()
                raise
            pending.remove((record, future))
            body = response.payload.decode("utf-8", errors="replace")
            record.update(outcome="RESPONSE_RECEIVED", received_unix=time.time(),
                          outer_success=response.success, outer_error=response.error,
                          response_payload_utf8=body)
            record["response_payload"] = json.loads(body)
            save()
            if response.success is not True or not isinstance(record["response_payload"], dict):
                raise RuntimeError("RPC envelope failed; this is not an observed module rejection")
            return record["response_payload"]

        before = call("status", {})
        report["before_status"] = before
        if before.get("ok") is not True or not peer_left(before):
            raise RuntimeError("Expected an already peer-left fixed group before probing")
        epoch = before.get("epoch")
        if not isinstance(epoch, str) or not epoch:
            raise RuntimeError("Missing pool epoch")
        identity = "cpu-peerleft-no-register-" + uuid.uuid4().hex
        lookup = call("lookup", dict(keys=[], shard=0, tp_size=1))
        prepare = call("prepare_store", dict(epoch=epoch, rank=1, member_id=identity,
                                             key=identity, size=1, layout=None, shard=0, tp_size=1))
        after = call("status", {})
        report["after_status"] = after
        checks = dict(before_peer_left=peer_left(before), after_peer_left=peer_left(after),
                      status_ok=after.get("ok") is True,
                      same_epoch=after.get("epoch") == epoch,
                      all_count_fields_observed=all(type(before.get(key)) is int and type(after.get(key)) is int
                                                    for key in COUNTS),
                      metadata_counts_unchanged=all(before.get(key) == after.get(key) for key in COUNTS))
        for label, response in (("lookup", lookup), ("prepare", prepare)):
            error = response.get("error", "")
            checks[label + "_fixed_group_rejected"] = (
                response.get("ok") is False and isinstance(error, str)
                and "fixed group unavailable" in error.lower()
                and "peer" in error.lower() and "left" in error.lower())
        report["checks"] = checks
        require_cpu_only("after_rpc")
        report["passed"] = all(checks.values())
    except BaseException as error:
        report.update(passed=False, error=repr(error))
    finally:
        for record, future in pending:
            item = dict(method=record["method"], outcome=record["outcome"],
                        original_future_retained_until_close=True, cancelled_or_resubmitted=False)
            try:
                item["done_at_close"] = future.query()
                if item["done_at_close"]:
                    late = future.result(timeout=0)
                    item.update(late_outer_success=late.success,
                                late_payload_utf8=late.payload.decode("utf-8", errors="replace"))
            except BaseException as error:
                item["late_observation_error"] = repr(error)
            report["pending_futures"].append(item)
        if client is not None:
            try:
                client.close()
                report["client_closed"] = True
            except BaseException as error:
                report.update(passed=False, close_error=repr(error))
        else:
            report["client_not_created"] = True
        report["passed"] = report["passed"] and report["client_closed"] and not pending
        report["finished_unix"] = time.time()
        report["final_import_audit"] = audit_imports()
        report["passed"] = (report["passed"] and not report["final_import_audit"]["forbidden_modules"]
                            and not report["final_import_audit"]["accelerator_library_mappings"])
        save()
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
