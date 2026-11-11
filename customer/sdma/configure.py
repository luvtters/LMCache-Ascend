#!/usr/bin/env python3
"""Write a fresh pair of SDMA L2 configs; does not initialize SDKs or devices."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-gb", type=int, default=8, help="DDR GiB per host")
    parser.add_argument("--device85", type=int, default=0)
    parser.add_argument("--device86", type=int, default=0)
    args = parser.parse_args()
    if (not args.run_id or not args.run_id.isascii()
            or any(not (c.isalnum() or c in "-_") for c in args.run_id)
            or args.pool_gb < 1 or min(args.device85, args.device86) < 0):
        parser.error("use an ASCII run ID, positive pool size and nonnegative logical devices")
    args.output.mkdir(parents=True, exist_ok=False)
    for rank, host, device in ((0, "85", args.device85), (1, "86", args.device86)):
        config = dict(type="plugin", module_path="memfabric_sdma.memfabric_l2_adapter",
                      class_name="MemFabricL2Adapter", shared=True,
                      adapter_params=dict(shared_directory=True, rank_id=rank,
                          device_id=device, world_size=2, storage_ranks=[0, 1],
                          pool_gb=args.pool_gb, slot_size_mb=32, data_op_type="SDMA",
                          enable_56bits_gva=False,
                          device_nic=f"tcp://10.48.96.{host}:28582",
                          store_url="tcp://10.48.96.85:28580",
                          join_coord_url="http://10.48.96.85:28581",
                          join_group_id=args.run_id))
        path = args.output / f"{host}.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(path.resolve())


if __name__ == "__main__":
    main()
