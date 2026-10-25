# SPDX-License-Identifier: Apache-2.0
"""Real BM/MemoryObj checks. Requires the prepared A3 environment; no mocks.

Device IDs are process-local and are mapped by ASCEND_RT_VISIBLE_DEVICES.
"""

import argparse
import time

import torch

from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObjMetadata, TensorMemoryObj
from lmcache_ascend.v1.distributed.memfabric_l2_adapter import MemFabricL2Adapter


def memory(value, shape=(1024,)):
    raw = (torch.arange(1024, dtype=torch.int32) + value).remainder(251).to(torch.uint8)
    meta = MemoryObjMetadata(shape=torch.Size(shape), dtype=torch.uint8,
                             address=raw.data_ptr(), phy_size=1024, ref_count=1,
                             fmt=MemoryFormat.BINARY, shapes=[torch.Size(shape)],
                             dtypes=[torch.uint8])
    return TensorMemoryObj(raw, meta, None)


def wait(query):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        result = query()
        if result is not None:
            return result
        time.sleep(0.01)
    raise TimeoutError("Adapter task did not complete")


def main(args):
    adapter = MemFabricL2Adapter({"device_id": args.device, "world_size": 1,
                                 "storage_rank": 0, "pool_gb": 1,
                                 "slot_size_mb": 512,
                                 "store_url": args.store_url})
    keys = [ObjectKey(bytes([i]) * 32, "memfabric-smoke", 0) for i in range(5)]
    source = [memory(i * 11) for i in range(5)]
    expected = source[0].raw_tensor.clone()

    def store(ks, objs):
        tid = adapter.submit_store_task(ks, objs)
        return wait(lambda: adapter.pop_completed_store_tasks().get(tid))

    def lookup(ks, layouts=None):
        tid = adapter.submit_lookup_and_lock_task(ks, layouts or {})
        return wait(lambda: adapter.query_lookup_and_lock_result(tid))

    def load(ks, objs):
        tid = adapter.submit_load_task(ks, objs)
        return wait(lambda: adapter.query_load_result(tid))

    try:
        result = store(keys[:2], source[:2])
        assert result.is_successful() and result.bytes_transferred() == 2048
        source[0].raw_tensor.zero_()
        bits = lookup(keys[:3])
        assert bits.test(0) and bits.test(1) and not bits.test(2)
        assert not store([keys[2]], [source[2]]).is_successful()
        target = memory(99)
        assert load([keys[0]], [target]).test(0)
        assert torch.equal(target.raw_tensor, expected)
        assert source[0].get_ref_count() == 1 and target.get_ref_count() == 1
        assert not load([keys[0]], [memory(7, (32, 32))]).test(0)
        bad_layout = {0: MemoryLayoutDesc([torch.Size((32, 32))], [torch.uint8])}
        assert not lookup([keys[0]], bad_layout).test(0)
        assert store([keys[0]], [source[0]]).bytes_transferred() == 0
        adapter.submit_unlock([keys[1]])
        assert store([keys[2]], [source[2]]).is_successful()
        assert not lookup([keys[1]]).test(0)
        adapter.delete([keys[0]])
        assert not lookup([keys[0]]).test(0)
        assert load([keys[0]], [target]).test(0)
        assert torch.equal(target.raw_tensor, expected)
        adapter.submit_unlock([keys[0]])
        assert not lookup([keys[0]]).test(0)
        tid = adapter.submit_store_task([keys[3]], [source[3]])
        adapter.close()
        assert adapter.pop_completed_store_tasks()[tid].is_successful()
        print("ADAPTER_SMOKE_PASS round_trip=True miss=True layout=True "
              "locked_capacity=True duplicate=True deferred_delete=True "
              "allocator_refs=True close_drains=True", flush=True)
    finally:
        adapter.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0,
                        help="Process-local device ID, mapped by ASCEND_RT_VISIBLE_DEVICES")
    parser.add_argument("--store-url", default="tcp://127.0.0.1:28582")
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error("device must be non-negative")
    return args


if __name__ == "__main__":
    main(parse_args())
