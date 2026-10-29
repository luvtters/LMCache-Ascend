"""Actual 32/33-MiB CPU MemoryObjs/eventfds; lossless compressed fake BM only."""
import ctypes
import os
import threading
from types import SimpleNamespace
from unittest.mock import patch
import zlib

from l2_full_capacity_probe import POOL, SLOT, run_capacity_matrix
from lmcache_ascend.v1.distributed import memfabric_l2_adapter as module


def main():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    # Compress fake backing storage to avoid allocating a full 1-GiB CPU pool.
    # Every submitted byte is retained losslessly and checked by the same probe.
    blocks, events = {}, []
    def copy(src, dst, size, *, store):
        if store:
            blocks[dst] = zlib.compress(ctypes.string_at(src, size), level=1)
        else:
            raw = zlib.decompress(blocks[src])
            assert len(raw) == size
            ctypes.memmove(dst, raw, size)
    session = SimpleNamespace(peer_departed=threading.Event(), local_pool_bytes=POOL,
        peer_address=lambda rank: 4096, copy=copy, wait=lambda: None, close=lambda: None)
    with patch.object(module, "BMSession", return_value=session):
        adapter = module.MemFabricL2Adapter(dict(world_size=2, pool_gb=1, slot_size_mb=32))
    try:
        result = run_capacity_matrix(adapter, "cpu-only-full-capacity", lambda event, **kw: events.append((event, kw)))
    finally:
        adapter.close()
    assert result["initial_written_bytes"] == POOL
    assert len(result["comparisons"]) == 33
    assert all(row["compared_bytes"] == SLOT and row["mismatches"] == 0 for row in result["comparisons"])
    assert len(result["completed_bm_calls"]) == 66 and result["oversize_bm_calls"] == 0
    assert len(blocks) == 32
    assert result["final_status"]["objects"] == result["final_status"]["total_pins"] == 0
    print("PASS_CPU_ONLY: full 1GiB write/read; 33rd LRU; 33MiB reject; 33 full reads; 66 fake BM copies")


if __name__ == "__main__":
    main()
