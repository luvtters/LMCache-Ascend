#!/usr/bin/env python3
"""CPU-only native allocator/lifetime check; never initializes BM or NPU.

Requires the normal LMCache native extension for TTLLock. The BM session shim
only exposes a CPU tensor as its arena, so this proves native L1 management,
not MemFabric transport or NPU compatibility.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-source", type=Path, required=True)
    parser.add_argument("--ascend-source", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.core_source))
    import torch
    from lmcache.v1.distributed.api import GroupedObjectKeys, MemoryLayoutDesc, ObjectKey, PrefetchTaskSpec
    from lmcache.v1.distributed.config import EvictionConfig, L1ManagerConfig, L1MemoryManagerConfig, StorageManagerConfig
    from lmcache.v1.distributed.error import L1Error
    from lmcache.v1.distributed.l1_manager import L1Manager
    from lmcache.v1.distributed.storage_manager import StorageManager

    # Load this leaf file directly: no Ascend plugin activation or torch_npu.
    path = args.ascend_source / "lmcache_ascend/v1/distributed/memfabric_l1_memory.py"
    spec = importlib.util.spec_from_file_location("memfabric_l1_cpu_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arena = torch.zeros(1 << 20, dtype=torch.uint8)
    session = SimpleNamespace(
        _check_thread=lambda: None, pool_memory_type="host", local_dram_bytes=arena.numel(),
        rank_id=0, peer_address=lambda rank: 1 << 40,
        bm=SimpleNamespace(BmMemType=SimpleNamespace(LOCAL_HOST="host")),
        handle=SimpleNamespace(gva_to_va=lambda gva, medium: arena.data_ptr()),
    )
    memory = module.MemFabricL1MemoryManager(session)
    config = L1ManagerConfig(L1MemoryManagerConfig(arena.numel(), False, shm_name=""))
    manager = L1Manager(config, memory_manager=memory)
    storage = StorageManager(StorageManagerConfig(config, EvictionConfig("noop")),
                             _l1_managers=(manager,))
    try:
        key = ObjectKey(b"cpu-native-l1", "synthetic-kv", 0)
        layout = MemoryLayoutDesc([torch.Size([2, 2, 8, 16])], [torch.bfloat16])
        obj = storage.reserve_write([key], layout)[key]
        assert obj.tensor.data_ptr() >= arena.data_ptr()
        assert memory.global_address(obj) == (1 << 40) + obj.meta.address
        assert manager.reserve_read([key])[key][0] != L1Error.SUCCESS
        try:
            manager.close()
        except RuntimeError:
            pass
        else:
            raise AssertionError("L1Manager close accepted a live write reservation")
        try:
            memory.close()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Allocator close accepted a live native object")
        obj.tensor.fill_(17)
        assert torch.equal(arena[:obj.get_size()].view(torch.bfloat16),
                           torch.full((obj.tensor.numel(),), 17, dtype=torch.bfloat16))
        storage.finish_write([key])
        handle = storage.submit_prefetch_task(PrefetchTaskSpec(
            [GroupedObjectKeys([key], 0, layout)], allowed_l2_memory_tags=frozenset()
        ))
        result = storage.query_prefetch_status(handle)
        assert result is not None and result.l1_hit_count == 1 and result.l2_hit_count == 0
        keys, objects = storage.unsafe_read([key])
        assert keys == [key] and objects[0] is obj
        try:
            manager.close()
        except RuntimeError:
            pass
        else:
            raise AssertionError("L1Manager close accepted a live read reservation")
        assert storage.delete_l1_keys([key]) == (0, 1)
        storage.finish_read_prefetched([key])
        assert storage.delete_l1_keys([key]) == (1, 0)
        assert storage.get_l1_usage()[0] == 0
        assert memory.memcheck()
        try:
            memory.global_address(obj)
        except ValueError:
            pass
        else:
            raise AssertionError("Freed native object retained a usable GVA")
    finally:
        storage.close()
    assert arena.numel() == (1 << 20)  # The external owner still owns its arena.
    print(json.dumps(dict(status="PASS_CPU_NATIVE_L1_ONLY", staging_hidden=True,
                          native_l1_hit=1, read_protection=True, released_bytes=0,
                          live_close_rejected=True, native_memcheck=True)))


if __name__ == "__main__":
    main()
