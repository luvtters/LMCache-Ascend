# SPDX-License-Identifier: Apache-2.0
"""Bounded host/pool copy workload for msprof; not an inference benchmark.

Start pool_member.py on rank 1 first. Run this rank 0 under msprof, selecting
runtime-api and task-time collection. Only MemFabric moves the payload.
"""

import argparse
import json
from pathlib import Path
import time

import torch

from lmcache_ascend.v1.distributed.memfabric_bm import BMSession


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    events = []

    def phase(name):
        events.append({"phase": name, "unix_ns": time.time_ns()})
        (output / "phases.json").write_text(json.dumps(events, indent=2))
        print(name, flush=True)

    session = BMSession(device_id=args.device, rank_id=0, world_size=2,
                        pool_gb=1, store_url=args.store_url)
    try:
        address = session.peer_address(1)
        source = torch.arange(4 << 20, dtype=torch.int32)
        expected = source.clone()
        target = torch.full_like(source, -1)
        size = source.numel() * source.element_size()
        session.copy(source.data_ptr(), address, size, store=True)
        source.fill_(-1)
        session.copy(address, target.data_ptr(), size, store=False)
        assert torch.equal(target, expected)
        source.copy_(expected)
        phase("IDLE")
        time.sleep(3)
        phase("COPY_ACTIVE")
        deadline = time.monotonic() + 5
        iterations = 0
        while time.monotonic() < deadline:
            session.copy(source.data_ptr(), address, size, store=True)
            session.copy(address, target.data_ptr(), size, store=False)
            iterations += 1
        phase("COPY_FINISHED")
        assert torch.equal(target, expected)
        summary = {"protocol": "SDMA", "storage_rank": 1,
                   "bytes_per_copy": size, "copy_pairs": iterations,
                   "payload_equal": True, "source_overwrite_check": True}
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print("PROFILE_COPY_PASS", json.dumps(summary), flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--store-url", default="tcp://127.0.0.1:28584")
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
