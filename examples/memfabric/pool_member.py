# SPDX-License-Identifier: Apache-2.0
"""Contribute one rank's selected pool; the service owns the object index."""

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
    parser.add_argument("--pool-memory-type", choices=("host", "device"), default="host")
    parser.add_argument("--store-url", default="tcp://127.0.0.1:28580")
    args = parser.parse_args()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    session = BMSession(device_id=args.device, rank_id=args.rank,
                        world_size=args.world_size, pool_gb=args.pool_gb,
                        store_url=args.store_url, pool_memory_type=args.pool_memory_type)

    def on_member_event(rank, event):
        if rank == 0 and event == session.bm.BmGroupEvent.LEAVE_EVENT:
            stop.set()

    session.handle.set_group_event_handler(on_member_event)
    try:
        print(f"POOL_MEMBER_READY rank={args.rank} device={args.device} "
              f"pool_memory_type={session.pool_memory_type} "
              f"local_selected_pool_bytes={session.local_pool_bytes}", flush=True)
        stop.wait()
    finally:
        session.close()
        print("POOL_MEMBER_CLOSED", flush=True)


if __name__ == "__main__":
    main()
