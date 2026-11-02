# SPDX-License-Identifier: Apache-2.0
"""Contribute one rank's DRAM; the LMCache service owns the object index."""

import argparse
import signal
import threading

from lmcache_ascend.v1.distributed.memfabric_bm import BMSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--pool-gb", type=int, default=1)
    parser.add_argument("--store-url", default="tcp://127.0.0.1:28580")
    parser.add_argument("--data-op-type", choices=("SDMA", "DEVICE_RDMA"), default="SDMA")
    parser.add_argument("--device-nic")
    parser.add_argument("--enable-56bits-gva", action="store_true")
    parser.add_argument("--join-coord-url")
    parser.add_argument("--join-group-id")
    args = parser.parse_args()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    session = BMSession(device_id=args.device, rank_id=args.rank,
                        world_size=args.world_size, pool_gb=args.pool_gb,
                        store_url=args.store_url, data_op_type=args.data_op_type,
                        device_nic=args.device_nic, enable_56bits_gva=args.enable_56bits_gva,
                        join_coord_url=args.join_coord_url, join_group_id=args.join_group_id)

    def on_member_event(rank, event):
        if rank == 0 and event == session.bm.BmGroupEvent.LEAVE_EVENT:
            stop.set()

    session.handle.set_group_event_handler(on_member_event)
    try:
        print(f"POOL_MEMBER_READY rank={args.rank} device={args.device} "
              f"protocol={args.data_op_type} pool_memory_type=host "
              f"enable_56bits_gva={args.enable_56bits_gva}", flush=True)
        stop.wait()
    finally:
        session.close()
        print("POOL_MEMBER_CLOSED", flush=True)


if __name__ == "__main__":
    main()
