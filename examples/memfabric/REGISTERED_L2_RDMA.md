# Registered CPU MemoryObj L2 candidate (2026-10-31)

This independent branch starts from worker candidate `a9d30974`; it does not
replace that candidate or change LMCache core (`5b565d2` lifecycle baseline),
the compatibility archive, or SDK 1.2.1. Service rank 0 and pool-member ranks
must load this checkout with identical DEVICE_RDMA / 56-bit settings and HOST
pool geometry. The native SDK route still requires native-log verification.

## Actual operation and allocator ownership

`MemFabricL2Adapter` retains its existing single I/O owner. In explicit
DEVICE_RDMA mode it calls `BMSession.copy_cpu_object(obj, remote_gva, size,
store=...)`. That method validates the exact CPU MemoryObj range, obtains a real
`obj.ref_count_up()` allocator lease, retains both MemoryObj and raw Tensor,
registers exactly that range, completes synchronous H2G/G2H with flags=0, and
unregisters it. Only then does it return the allocator lease.

Register, copy or unregister failure retains the actual MemoryObj, raw Tensor
and allocator lease in the process-level uncertain BM session until process
exit. DEVICE_RDMA wait is not completion evidence. RDMA errors propagate from
store/load and poisoned lookup instead of becoming a miss or CPU fallback.
Normal SDMA retains the original raw-pointer copy/reference behavior.

SDK HOST MR conversion requires address modulo4096=0. Each adapter operation
logs pointer, pointer modulo4096, byte count and direction; unaligned MemoryObj
addresses fail before native registration. Neither address nor registered size
is rounded or expanded into other allocator slices. The earlier failing 64-byte
primitive also had address modulo4096=192 and failed in HAL registration; size
alone is not an established root cause. An aligned small range can still fail
native registration and is reported explicitly with its exact size and return.

## Real allocator / 18 MiB L2 gate

### Required startup synchronization after the first-device failure

Frozen `e741c05` reached CPU registration before remote HOST import. The complete
forward log's first native ERROR at 03:34:35.772916 was
`Direction mismatch: specified=0 except:1`; remote rank 1 JoinHandle did not
begin until 03:34:36.413553. In fixed SDK source,
`hybm_data_op_entry.cpp:35-50` rejects that mask before entity CopyData;
`hybm_va_manager.cpp:260-283` returns address mask 0 for reserved but unimported
GVA. Thus the source was LOCAL_HOST while the destination GVA was not imported.
This failure is earlier than a payload RDMA completion. Existing unknown-copy
and lease retention remains conservative and unchanged.

The follow-up requires `join_coord_url` and a fresh `join_group_id` for L2 RDMA.
Every rank waits for its **own** JOIN callback, then publishes an immutable
`bm-joined-<rank>` event and waits for all configured ranks. This does not count
or assume replay of JOIN events for peers that were already present. SDK
`smem_bm_entry.cpp:260-340` orders import, group-result barrier, mmap, another
group-result barrier, then JOIN callback; `GroupOpBarrier:237-255` calls
`GroupGatherResult`. The fixed-group contract requires all ranks to acknowledge
their own completion on the fresh group before payload I/O. The two-rank CPU
HTTP checks validate this control mechanism; new device acceptance is required.

Do not substitute public rank/address queries for this barrier:
`GetRankIdByGva:662-672` only computes a rank from reserved address geometry;
remote HOST RDMA mmap records `{gva,0,0,...}` in
`hybm_conn_based_segment.cpp:238`, so converting that remote address to LOCAL_HOST
or LOCAL_DEVICE is not an available readiness test. `CanReachDataOperators:1302`
reports the configured RDMA bit without checking import/connection completion.

Start the existing CPU-only coordinator first, with a fresh directory/group:

```sh
python -u examples/memfabric/direct_coordination.py --bind 192.168.0.223 \
  --port 29595 --directory /private/new-run/join --run-id l2-fresh-after-mmap
```

The new gate is PLAN_ONLY by default. With `--allow-npu`, it allocates two real
MemoryObjs through `HostMemoryAllocator(use_paging=True)`, backed by
`PagedTensorMemoryAllocator`. It records actual addresses, alignment, parent
allocator and refcounts. It stores patterns A/B through the actual L2 adapter,
destroys the source before each load, compares every restored byte, checks
refcounts return to 1, drains lookup pins and native close, then releases the
objects and checks allocator accounting. This is actual L2 adapter validation,
not worker-direct or model inference. It does not claim independent owner-side
content verification, wire transport, throughput, or server L1 allocator parity.

Use a fresh store endpoint chosen by the campaign, the authorized devices and
the verified private environment. For example, rank 0 can own a fresh store
`tcp://192.168.0.223:28731` after the caller checks that port is available.
Run both processes concurrently; both need
`MF_HYBM_RDMA_SWAP_SPACE_SIZE=0` and this checkout plus matching core on PYTHONPATH.

Pool owner on the other authorized server, with its selected physical NPU
mapped as logical device 0:

```sh
export MF_HYBM_RDMA_SWAP_SPACE_SIZE=0
python -u examples/memfabric/pool_member.py --device 0 --rank 1 --world-size 2 \
  --pool-gb 1 --store-url tcp://192.168.0.223:28731 \
  --data-op-type DEVICE_RDMA --device-nic tcp://127.0.0.1:10005 \
  --enable-56bits-gva --join-coord-url http://192.168.0.223:29595 \
  --join-group-id l2-fresh-after-mmap
```

L2 gate on rank 0, with a disjoint authorized physical NPU mapped as logical 0:

```sh
export MF_HYBM_RDMA_SWAP_SPACE_SIZE=0
python -u examples/memfabric/l2_rdma_memoryobj_gate.py --device 0 \
  --store-url tcp://192.168.0.223:28731 --storage-rank 1 --world-size 2 \
  --device-nic tcp://127.0.0.1:10005 --enable-56bits-gva \
  --join-coord-url http://192.168.0.223:29595 --join-group-id l2-fresh-after-mmap \
  --payload-bytes 18874368 --output /private/new-run/l2-memoryobj.json --allow-npu
```

Without `--allow-npu` the second command only writes a plan and imports no
accelerator runtime. The pool CLI does execute and must only be used inside
the caller's current device authorization and independent deadline guard.
The path `/private/new-run` must be replaced with the confirmed personal result
directory. Capture native logs and external process waits/exit codes for both
processes; a JSON success field is not proof of both natural process exits.

## Existing LMCache service configuration

The normal L2 plugin structure is unchanged. Its `adapter_params` may now include:

```json
{
  "device_id": 0,
  "world_size": 2,
  "storage_rank": 1,
  "pool_gb": 1,
  "slot_size_mb": 32,
  "store_url": "tcp://192.168.0.223:28731",
  "data_op_type": "DEVICE_RDMA",
  "device_nic": "tcp://127.0.0.1:10005",
  "enable_56bits_gva": true,
  "join_coord_url": "http://192.168.0.223:29595",
  "join_group_id": "l2-fresh-after-mmap"
}
```

The pool remains HOST DRAM. There is no worker connector substitution, extra
CPU staging or pointer-only registration cache. Actual service arena alignment
must be checked from its own MemoryObjs; a passing standalone allocator gate
does not establish all model/service allocation layouts.

## Local validation boundary

CPU-only `test_memfabric_l2_rdma.py` covers exact registration range and
register/copy/unregister/lease ordering, actual-object retention after caller
release on all three native failure points, alignment refusal, small-range
native failure, adapter configuration, owner-thread dispatch, explicit failure
propagation, default SDMA behavior, and PLAN_ONLY runtime isolation.

These are fake native SDK / allocator control contracts. The real allocator
gate, native registration on service arenas, NPU/RDMA transport, inference and
performance remain NOT_VERIFIED by this local development task.

Local results: 9 new L2 contracts, 5 two-rank HTTP startup contracts,
11 RDMA contracts and 4 default BM contracts pass (29 total);
changed Python files also compile and `git diff
--check` passes. No SSH, device initialization, installation or pushing was
performed by this development task.

The same BMSession constructor fields can be explicitly configured in a worker
`lmcache.mp.direct` and its direct-service module config in this follow-up branch.
That reuse does not modify the frozen `a9d30974` worker candidate or establish
that candidate's prior startup safety or new device results.

Public native `handle.wait()` also differs from the bottom-level RDMA operator:
`MemEntityDefault::Wait:889` dispatches through `HostComposeDataOp::Wait:229`,
which supports SDMA asynchronous work and returns BM_ERROR when its SDMA
operator is absent. `DataOpDeviceRDMA::Wait:522` itself is a no-op, but that is
not the public compose path. This follow-up therefore skips the unsupported
native wait only for pure DEVICE_RDMA after all wrapper copies succeeded
synchronously with flags=0. Unknown registration/copy/teardown state still
raises before that branch and retains allocations. SDMA retains native wait.
No skipped wait is labeled as independent completion or recovery evidence.
