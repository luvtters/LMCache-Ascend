# SPDX-License-Identifier: Apache-2.0
"""Run the numbered CPU acceptance cases and persist an independently readable JSON."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
from types import SimpleNamespace as NS
import unittest

import torch

import inprocess_cpu_fixture as cpu
import test_inprocess_blocks_cpu as blocks
import test_inprocess_lifecycle_cpu as src
import test_inprocess_lookup_cpu as lookup


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def unittests(*cases):
    def run():
        names = []
        for cls, name in cases:
            result = unittest.TestResult()
            cls(name).run(result)
            if not result.wasSuccessful():
                raise AssertionError((result.errors + result.failures)[0][1])
            check(not result.skipped, "unexpected skipped subcheck")
            names.append(name)
        return {"passed_methods": names}
    return run


def roundtrip():
    e = cpu.engine_for()
    tokens = [1, 2, 3, 4]
    try:
        e.store(tokens, req_id="unit", source=cpu.tensor(0, 21, 4))
        e.wait_for_pending_stores({"unit"})
        hit = e.lookup(tokens, pin=True, lookup_id="unit")
        pinned = cpu.snapshot(e)
        dst = torch.zeros(4, 1, dtype=torch.int32)
        mask = e.retrieve(tokens, req_id="unit", destination=dst)
        e.lookup_unpin("unit")
        state = cpu.snapshot(e)
        check(hit == 4 and pinned["pins"] == 2 and mask.all().item(), "incomplete hit")
        check(torch.equal(dst, cpu.tensor(0, 21, 4)), "restored bytes differ")
        check(state["pins"] == 0 and state["refs"] == [1, 1], "read reference leak")
        return {"tokens_hit": hit, "values": dst.flatten().tolist(), "after_read": state}
    finally:
        e.close()


def failed_copy():
    e = cpu.engine_for()
    try:
        e.gpu_connector.fail = True
        e.store([1, 2], req_id="fail", source=cpu.tensor(0, 7, 2))
        e.wait_for_pending_stores({"fail"})
        hit = e.lookup([1, 2])
        state = cpu.snapshot(e)
        check(hit == 0 and state["refs"] == [0] and not state["pending"], "failed STORE published or leaked")
        return dict(hit=hit, after_failure=state)
    finally:
        e.close()


def aborted():
    sched, e = cpu.connector(), cpu.engine_for()
    worker = cpu.connector(e)
    client = sched._lmcache_engine.lookup_client
    try:
        e.store([1, 2], req_id="seed", source=cpu.tensor(0, 3, 2))
        e.wait_for_pending_stores({"seed"})
        client.lookup([1, 2], "abort")
        worker.step = sched.build_connector_meta(src.step())
        worker.start_load_kv(NS(attn_metadata=None))
        late = worker.build_connector_worker_meta()
        check(cpu.snapshot(e)["pins"] == 1, "lookup did not pin")
        delayed, _ = sched.request_finished_all_groups(NS(request_id="abort", status=cpu.ABORT), ([],))
        worker.step = sched.build_connector_meta(src.step(finished=["abort"]))
        worker.start_load_kv(NS(attn_metadata=None))
        sched.update_connector_output(NS(kv_connector_worker_meta=late))
        state = cpu.snapshot(e)
        check(delayed and state["pins"] == 0 and client.lookup_cache("abort") == -1, "aborted lookup survived")
        return {"delay_free": delayed, "pins_after_abort": state["pins"], "late_lookup_status": -1}
    finally:
        e.close()


def mtp_finish():
    sched, e = cpu.connector(), cpu.engine_for()
    worker = cpu.connector(e)
    impl = worker._lmcache_engine
    req = NS(req_id="draft", save_spec=NS(can_save=True, skip_leading_tokens=0), token_ids=[1, 2])
    try:
        delay, _ = sched.request_finished_all_groups(NS(request_id="draft", status="finished"), ([1],))
        require_step = sched.build_connector_meta(src.step(reqs=[], finished=["draft"]))
        require_step.requests.append(req)
        worker.step = require_step
        worker.start_load_kv(NS(attn_metadata=None))
        early = worker.get_finished({"draft"})[0]
        check(delay and early is None and impl._finished_req_ids_waiting_for_save == {"draft"},
              "MTP finished before STORE could be submitted")
        e.store(req.token_ids, req_id="draft", source=cpu.tensor(0, 5, 2))
        e.wait_for_pending_stores({"draft"})
        impl._wait_for_save_done = True
        impl._replay_finished_stores_after_save()
        worker.step = sched.build_connector_meta(src.step())
        final = worker.get_finished(set())[0]
        repeat = worker.get_finished({"draft"})[0]
        check(final == {"draft"} and repeat is None, "MTP completion was lost or replayed")
        return {"before_save": early, "after_save": sorted(final), "replay": repeat}
    finally:
        e.close()


def unit_cases():
    c, e, l, b = src.ConnectorTests, src.EngineTests, lookup.StepLookupTests, blocks.TransferTests
    return [
        ("角色构造、服务选择与 consumer 完成语义", unittests((c, "test_services_and_consumer"), (c, "test_query_only_step_and_pin_release"))),
        ("TP 存储 owner 与 PP 广播源", unittests((c, "test_tp_owner_in_second_stage_uses_group_local_broadcast_source"))),
        ("KV 注册、重复注册与分配形状", unittests((c, "test_register_updates_storage_geometry_before_post_init"))),
        ("异构 KV 平面与 APC 前缀字节", unittests((b, "test_mixed_planes_distinct_group_tables_and_apc"))),
        ("错块号及 chunk 对齐拒绝", unittests((b, "test_bad_chunk_and_bad_table"))),
        ("LOOKUP 共同前缀与缺回复", unittests((l, "test_common_prefix_requires_both_ranks"))),
        ("同 ID 重新接纳与迟到回复", unittests((l, "test_reissued_request_ignores_late_reply"))),
        ("查询发出之前取消", unittests((l, "test_cancel_before_dispatch"))),
        ("错误 rank 和冲突回复", unittests((l, "test_invalid_rank_and_inconsistent_replay"))),
        ("APC 命中不能掩盖外部恢复失败", unittests((c, "test_apc_prefix_does_not_hide_a_failed_external_block"))),
        ("Engine 写入、命中、回读与读引用释放", roundtrip),
        ("异步拷贝失败后的可见性、引用与 pending", failed_copy),
        ("Abort 后撤销在途 LOOKUP 与 worker pin", aborted),
        ("STORE 传递分组块表与计算事件", unittests((c, "test_production_store_receives_table_and_compute_event"))),
        ("MTP 先 finish 后 STORE 的单次补报", mtp_finish),
        ("抢占恢复替换块表并保留提交时快照", unittests((c, "test_alloc_delta_and_resume_replace_all_groups"))),
        ("异步 STORE 排空、延迟通知与关闭后提交", unittests((e, "test_finish_and_shutdown_wait_for_last_store"))),
        ("同步退出幂等、关闭失败保留 Engine、异常释放读引用", unittests((e, "test_sync_close_and_repeat"), (c, "test_builder_retains_an_engine_that_failed_to_drain"), (e, "test_broadcast_failure_releases_owner_read_reference"))),
    ]


def run_cases(cases, prefix, rows):
    for i, (title, fn) in enumerate(cases, 1):
        row = dict(id=f"{prefix}{i:02}", title=title, status="FAIL")
        rows.append(row)
        start = time.monotonic()
        try:
            row["evidence"] = fn()
            row["status"] = "PASS"
        except FileNotFoundError:
            row["status"] = "BLOCKED"
            row["error"] = traceback.format_exc()
        except Exception:
            row["error"] = traceback.format_exc()
        row["elapsed_s"] = round(time.monotonic() - start, 4)
        print(row["id"], row["status"], row.get("error", ""), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["4", "5"], required=True)
    p.add_argument("--output", type=Path, required=True)
    opt = p.parse_args()
    phase = int(opt.phase)
    report = dict(phase=phase, started_utc=datetime.now(timezone.utc).isoformat(),
                  evidence_type="CPU_TENSOR_MOCK_BACKEND", npu_tested=False,
                  python=platform.python_version(), torch=torch.__version__, parent_pid=os.getpid(),
                  commits={name: subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                                         text=True).strip()
                           for name, root in [("ascend", src.ROOT), ("core", src.CORE)]}, cases=[])
    if phase == 4:
        run_cases(unit_cases(), "C", report["cases"])
    else:
        from inprocess_pp2_cpu import Workers
        vllm_root = os.environ.get("KS_VLLM_SOURCE", "")
        report["vllm_aggregator_source"] = str(
            (Path(vllm_root) / "vllm/distributed/kv_transfer/kv_connector/utils.py").resolve()
        )
        with Workers() as pair:
            run_cases(pair.cases(), "P", report["cases"])
        report["workers"] = pair.outcome()
    counts = Counter(r["status"] for r in report["cases"])
    report["counts"] = {status: counts[status] for status in ("PASS", "FAIL", "SKIP", "BLOCKED")}
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    opt.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return int(report["counts"].get("PASS", 0) != (18 if phase == 4 else 12))


if __name__ == "__main__":
    raise SystemExit(main())
