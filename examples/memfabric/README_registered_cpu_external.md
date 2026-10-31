# External ordinary torch CPU allocation -> remote HOST RDMA probe

This independent two-rank diagnostic tests the missing prerequisite for a future
external CPU MemoryObj route. It does not change BMSession, LMCache, the actual
LMCache allocator, SDK, dependencies, world3 probe or original registered probe.
It uses fixed SDK source `106f783` and the existing private runtime environment.

## What is tested

- Rank 0 is the client and store starter; rank 1 owns remote HOST memory.
  Both use logical device 0 after independently verified physical device masks.
- Both create matching HOST 1 GiB / HBM 0 pools, DEVICE_RDMA, 56-bit GVA enabled,
  allocation flags 0 and `MF_HYBM_RDMA_SWAP_SPACE_SIZE=0`.
- Client creates two separate `torch.empty((n,), dtype=torch.uint8,
  device='cpu', pin_memory=False)` allocations. These are ordinary contiguous CPU
  Tensor allocations with their actual allocator alignment. No NPU payload,
  pinned-memory allocation, SDK-pool source, custom mmap or realignment is used.
- Actual source/restore ranges must be nonoverlapping, CPU-writable and outside
  the client's SDK HOST pool. Actual pointers, OS page size, page offsets and
  covering page intervals are reported, including overlapping covering pages if
  the ordinary allocator happens to place small allocations that way.
- Both actual CPU pointers are separately registered with `handle.register(ptr,n)`.
  Either failure aborts without changing allocation or endpoints. All public
  register/unregister/copy returns and requested addresses/types are retained.
- Client verifies source A, then calls `H2G(source_CPU_ptr, owner_HOST_GVA,n,0)`.
  Owner independently verifies every A byte at its validated local HOST HVA.
  Client fills its original source with 51 and verifies every byte; source A no
  longer remains available there. Restore is reset to 90 and fully verified.
  `G2H(owner_HOST_GVA,restore_CPU_ptr,n,0)` must restore every A byte.
- Only then does owner write and independently verify B. Client resets/verifies
  the restore sentinel again, reads B via G2H and compares all B. Both ranks
  finish the payload barrier before unregister/leave/destroy/uninitialize.
  CPU Tensor references remain alive through the entire SDK cleanup.

Coordination HTTP transports metadata only. Byte counters advance only after
zero synchronous H2G/G2H returns. Missing peers time out and published peer
failures abort waits. Payload success, public cleanup success, native warnings,
true process exits and resource release remain separate results.

## Source-confirmed opportunity and runtime uncertainty

In fixed `106f783`:

- `src/smem/csrc/smem_bm/smem_bm_entry.cpp:674-720` publicly registers and
  unregisters the supplied actual address/length.
- `src/hybm/csrc/mm/hybm_mem_segment.cpp:128-170` classifies the local pointer;
  the HOST/DEVICE_RDMA branch calls
  `HalHostRegister(addr,size,HOST_MEM_MAP_DEV,logicDeviceId,&output)` and records
  the HVA/device-VA mapping. Native failure logs include the HAL return.
- `src/hybm/csrc/entity/hybm_entity_default.cpp:240-286` follows successful local
  registration with a transport MR using `REG_MR_FLAG_DRAM` for a HOST pointer.
- `src/hybm/csrc/data_operation/host/hybm_data_op_device_rdma.cpp:323-335,459-471`
  routes cross-member HOST writes/reads to SafePut/SafeGet. At `1076-1155`, a
  registered full range goes directly to CopyRDMA; swap zero rejects an
  unregistered range instead of using a staging fallback.

These are source paths, not evidence that a particular ordinary torch malloc
address is classified correctly or accepted by HalHostRegister/transport.
Registration can fail due to alignment, range/classification or platform
constraints. Keep exact native diagnostics and do not replace the 64-byte
allocator outcome with a different aligned buffer. The public register API
does not expose a separate successful HAL return; preserve native debug/error
logs and report that distinction. Likewise public unregister zero does not
erase native `HalHostUnregisterEx` warnings.

Even a complete pass proves this particular ordinary torch CPU allocator path.
It does not validate LMCache's actual external MemoryObj allocator, ownership,
pinning, slicing or lifetime contract. BMSession's existing rejection remains
unchanged by this diagnostic.

## CPU-only checks and PLAN_ONLY

```bash
python examples/memfabric/check_registered_cpu_external_cpu.py
python examples/memfabric/registered_cpu_external_probe.py \
  --rank 0 --device 0 --payload-bytes 64 \
  --store-url tcp://115.120.85.137:19731 \
  --coord-url http://115.120.85.137:29731 --run-id externalcpu01 \
  --output /tmp/externalcpu01-plan.json
```

Six CPU checks pass: matching creation/allocator contracts, exclusive PLAN_ONLY
output with no native imports, CLI limits, real ctypes pointer-range rejection,
unaligned pointer reporting without realignment, and causal client/owner A/B
control flow plus rejection of an incomplete owner acknowledgement. These are
**CPU_CONTROL_PATH_ONLY** and never execute torch, SDK, NPU or RDMA.

Allowed payloads are exactly 64, 1048576 and 18874368 bytes. PLAN_ONLY is the
default. Every actual attempt needs new run ID, ports, coordination/result
directories, and exclusive new rank outputs.

## Native command parameters for an authorized fresh run

Run only in the existing confirmed personal containers/private environment.
Check live ownership, physical masks, time window and the independent deadline
guard first. Example: client rank 0 on 137 physical 8; owner rank 1 on 90.25
physical 8. SDK store is on client 137 because rank 0 sets `start_store=True`.
These commands assume the independently verified mask value 8 exposes the
authorized physical card as logical 0. `RUN_DIR` is an already confirmed fresh
private run directory; `python` is the already prepared private interpreter.

On 137 start metadata coordination:

```bash
python examples/memfabric/direct_coordination.py --bind 0.0.0.0 --port 29731 \
  --directory "$RUN_DIR/coord" --run-id externalcpu01 >"$RUN_DIR/coord.log" 2>&1 &
coord_pid=$!
```

Launch both commands promptly in separate terminals/processes and retain their
actual wait returns, even on failure. Under a shell using `set -e`, explicitly
guard the wait so its nonzero return can be recorded:

```bash
# Client on 137.
ASCEND_RT_VISIBLE_DEVICES=8 timeout --signal=TERM --kill-after=10s 480s \
  python -u examples/memfabric/registered_cpu_external_probe.py \
  --rank 0 --device 0 --payload-bytes 64 --init-timeout 180 --timeout 180 \
  --device-nic tcp://127.0.0.1:10005 \
  --store-url tcp://115.120.85.137:19731 \
  --coord-url http://115.120.85.137:29731 --run-id externalcpu01 \
  --output "$RUN_DIR/rank0.json" --allow-npu >"$RUN_DIR/rank0.log" 2>&1 &
client_pid=$!
client_rc=0
wait "$client_pid" || client_rc=$?
printf '%s\n' "$client_rc" >"$RUN_DIR/rank0.wait.rc"
```

```bash
# Owner on 90.25.
ASCEND_RT_VISIBLE_DEVICES=8 timeout --signal=TERM --kill-after=10s 480s \
  python -u examples/memfabric/registered_cpu_external_probe.py \
  --rank 1 --device 0 --payload-bytes 64 --init-timeout 180 --timeout 180 \
  --device-nic tcp://127.0.0.1:10005 \
  --store-url tcp://115.120.85.137:19731 \
  --coord-url http://115.120.85.137:29731 --run-id externalcpu01 \
  --output "$RUN_DIR/rank1.json" --allow-npu >"$RUN_DIR/rank1.log" 2>&1 &
owner_pid=$!
owner_rc=0
wait "$owner_pid" || owner_rc=$?
printf '%s\n' "$owner_rc" >"$RUN_DIR/rank1.wait.rc"
```

Require both true waits, both full payload/cleanup reports, native registration,
actual copy-type/transport diagnostics and independent device-release evidence.
After both ranks truly exit, stop the matching coordination process from its
original shell. Timeouts never extend a device authorization window; enforce
the earlier actual cutoff independently. Failed ordinary-allocator samples
must be retained. Do not automatically retry using aligned memory or NPU data.

At authoring time only the six CPU checks have run. Actual torch CPU allocation
classification, HAL/transport registration, H2G/G2H cross-host full contents,
native exits and release are all **NOT_VERIFIED**.
