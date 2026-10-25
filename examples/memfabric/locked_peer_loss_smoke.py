# SPDX-License-Identifier: Apache-2.0
"""Real BM peer loss after lookup pins a key, before load starts.

Owns and stops its child pool_member process. Device IDs are process-local
and are mapped by ASCEND_RT_VISIBLE_DEVICES. Defaults use devices 0 and 2.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from adapter_smoke import memory, wait
from lmcache.v1.distributed.api import ObjectKey
from lmcache_ascend.v1.distributed.memfabric_l2_adapter import MemFabricL2Adapter


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    adapter = None
    with (output / "member.log").open("w") as log:
        member = subprocess.Popen([
            sys.executable, "-u", str(Path(__file__).with_name("pool_member.py")),
            "--device", str(args.member_device), "--rank", "1", "--store-url", args.store_url],
            stdout=log, stderr=subprocess.STDOUT)
    try:
        adapter = MemFabricL2Adapter({"device_id": args.owner_device, "world_size": 2,
                                     "storage_rank": 1, "pool_gb": 1,
                                     "slot_size_mb": 32,
                                     "store_url": args.store_url})
        key = ObjectKey(b"L" * 32, "locked-peer-loss", 0)
        source, target = memory(23), memory(97)
        sentinel = target.raw_tensor.clone()
        task = adapter.submit_store_task([key], [source])
        assert wait(lambda: adapter.pop_completed_store_tasks().get(task)).is_successful()
        task = adapter.submit_lookup_and_lock_task([key], {})
        assert wait(lambda: adapter.query_lookup_and_lock_result(task)).test(0)
        before = adapter.report_status()
        assert before["total_pins"] == 1
        (output / "before_exit.json").write_text(json.dumps(before, indent=2))
        member.kill()
        member.wait(timeout=20)
        deadline = time.monotonic() + 15
        while adapter.report_status()["is_healthy"]:
            assert time.monotonic() < deadline, "Departure not detected; load not attempted"
            time.sleep(0.1)
        task = adapter.submit_load_task([key], [target])
        assert not wait(lambda: adapter.query_load_result(task)).test(0)
        assert torch.equal(target.raw_tensor, sentinel)
        assert source.get_ref_count() == target.get_ref_count() == 1
        adapter.submit_unlock([key])
        deadline = time.monotonic() + 10
        while adapter.report_status()["total_pins"]:
            assert time.monotonic() < deadline, "Pin was not released"
            time.sleep(0.01)
        task = adapter.submit_lookup_and_lock_task([key], {})
        assert not wait(lambda: adapter.query_lookup_and_lock_result(task)).test(0)
        task = adapter.submit_store_task([key], [source])
        assert not wait(lambda: adapter.pop_completed_store_tasks().get(task)).is_successful()
        after = adapter.report_status()
        assert after["load_bytes"] == before["load_bytes"] == 0
        (output / "after_exit.json").write_text(json.dumps(after, indent=2))
        adapter.close()
        summary = {"lookup_pin_before_exit": 1, "pins_after_unlock": 0,
                   "load_hit_after_exit": False, "new_bm_read_bytes": 0,
                   "destination_unchanged": True, "allocator_refs": [1, 1],
                   "store_failed_after_exit": True, "close_completed": True}
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print("LOCKED_PEER_LOSS_PASS", json.dumps(summary), flush=True)
    finally:
        if adapter is not None:
            adapter.close()
        if member.poll() is None:
            member.terminate()
            try:
                member.wait(timeout=20)
            except subprocess.TimeoutExpired:
                member.kill()
                member.wait()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--owner-device", type=int, default=0,
                        help="Process-local owner ID, mapped by ASCEND_RT_VISIBLE_DEVICES")
    parser.add_argument("--member-device", type=int, default=2,
                        help="Process-local member ID, mapped by ASCEND_RT_VISIBLE_DEVICES")
    parser.add_argument("--store-url", default="tcp://127.0.0.1:28586")
    args = parser.parse_args(argv)
    if args.owner_device < 0 or args.member_device < 0:
        parser.error("owner-device and member-device must be non-negative")
    if args.owner_device == args.member_device:
        parser.error("owner-device and member-device must be different")
    return args


if __name__ == "__main__":
    main(parse_args())
