"""Run the exact API matrix with real CPU MemoryObjs and a dictionary fake BM."""
import ctypes
import threading
from types import SimpleNamespace
from unittest.mock import patch

from l2_api_boundary_probe import MIB, run_matrix
from lmcache_ascend.v1.distributed import memfabric_l2_adapter as module


def main():
    blocks, copies, events = {}, [], []
    def copy(src, dst, size, *, store):
        copies.append(dict(store=store, bytes=size))
        if store:
            blocks[dst] = ctypes.string_at(src, size)
        else:
            ctypes.memmove(dst, blocks[src], size)
    session = SimpleNamespace(peer_departed=threading.Event(), local_pool_bytes=1 << 30,
        peer_address=lambda rank: 4096, copy=copy, wait=lambda: None, close=lambda: None)
    with patch.object(module, "BMSession", return_value=session):
        adapter = module.MemFabricL2Adapter(dict(world_size=2, pool_gb=1, slot_size_mb=256))
    try:
        result = run_matrix(adapter, "cpu-only-api-matrix", lambda event, **kw: events.append((event, kw)))
    finally:
        adapter.close()
    assert sum(row["bytes"] for row in copies if row["store"]) == 6 * MIB
    assert sum(row["bytes"] for row in copies if not row["store"]) == 5 * MIB
    assert result["final_status"]["total_pins"] == result["final_status"]["objects"] == 0
    assert len(result["comparisons"]) == 5
    assert any(name == "closed_submit_rejected" for name, _ in events)
    print("PASS_CPU_REAL_MEMORYOBJ_EVENTFD_FAKE_BM_ONLY: full API matrix, 11 copies")


if __name__ == "__main__":
    main()
