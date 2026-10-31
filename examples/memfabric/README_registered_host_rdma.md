# Registered HOST DEVICE_RDMA primitive diagnostic

Independent diagnostic for fixed MemFabric 1.2.1. This is not a production
adapter and does not establish HCCS use, default SDMA correctness, LMCache
functionality, or a performance result. CPU verification does not validate RDMA.

Each of two ranks requests 1 GiB HOST and zero HBM, explicitly selects
`DEVICE_RDMA`, sets `MF_HYBM_RDMA_SWAP_SPACE_SIZE=0` before SDK imports, and
does not fall back to another transport. The default NIC URL is
`tcp://127.0.0.1:10005`; the SDK selects its local NPU IP. Actual transport
must be checked in preserved native logs. `--enable-56bits-gva` is explicit.

The local rank's HOST GVA is converted with `LOCAL_HOST`; source at offset 0
and restore at offset `2*n` remain within `local_mem_size(HOST)`. Before any
CPU dereference the probe rejects null/overflow/out-of-pool/overlapping
ranges and checks complete read/write coverage in `/proc/self/maps`.
It never CPU dereferences the remote GVA and never frees an SDK pool address.
The handle and SDK memory remain alive through the both-done barrier.

1. Rank 0 writes A into its registered source; rank 1 initializes its own
   target with a sentinel. Both agree on configuration and owner address.
2. Rank 0 executes synchronous H2G with flags 0 and requires raw return 0.
   Rank 1 independently reads its local HOST HVA and compares every byte to A.
3. After owner verification rank 0 destroys its original source, G2H reads
   into its separate registered restore buffer, and fully verifies A.
4. Rank 1 locally writes independently generated B, sending metadata only.
   Rank 0 re-sentinels restore, G2H reads again and fully verifies B.
5. Both-done precedes leave/destroy/uninitialization. Primary exceptions and
   cleanup errors are retained. Completion byte counters describe successful
   synchronous SDK returns; content records are separate evidence.

`handle.wait()` for DEVICE_RDMA is a no-op and is intentionally not presented
as independent completion evidence. Fixed SDK synchronous copy goes through
WriteRemote/ReadRemote, Synchronize and notify.Wait. The external launcher
must retain raw stdout/stderr, wait for both child processes and record actual
exit codes. Python PASS alone does not establish clean native process exit.

## Local checks and planning

```sh
python examples/memfabric/check_registered_host_rdma_cpu.py
python examples/memfabric/registered_host_rdma_probe.py \
  --rank 0 --device 0 --enable-56bits-gva \
  --store-url tcp://HOST0:19001 --coord-url http://HOST0:19002 \
  --run-id fresh-registered-rdma --output /private/run/rank0.json
```

The second command is PLAN_ONLY: no torch, torch_npu, MemFabric imports or
device selection. Use an output path that does not exist. Each process writes
its rank-local result and `<output>.events.jsonl`; the caller must preserve
these originals and launch a fresh coordination server/group for every retry.

## Device execution contract

Only run during a currently authorized device window, after ownership/device
mapping checks and independent deadline protection. Start `direct_coordination.py`
on the first host with a fresh run-id and an explicitly private directory.
Use the planning command for rank 0 and corresponding `--rank 1` command on
the second host, add `--allow-npu` to both, select the caller-verified logical
device, and point both at the same store and coordination endpoint. The caller
must apply physical device 8-15 masks and ensure logical mapping; this script
does not assume that a logical device number is a physical number.

Use an external group timeout (the current campaign uses 170 seconds), retain
both process waits and cleanup evidence, and never extend the authorized
deadline. Registration failures, missing mappings, configuration disagreement,
nonzero SDK returns or any content mismatch fail this attempt. Do not enable
swap or alter global hugepage settings to obtain a pass.

## Source basis and verification limits

Fixed SDK revision `106f783`:

- `hybm_conn_based_segment.cpp:386-406`: MapSlice HVA allocation,
  HalHostRegister and AddVaInfo.
- `hybm_entity_default.cpp:219-234`: HOST allocation automatically registers
  the DRAM memory region.
- `device_rdma_transport_manager.cpp:145-166`: RaRegisterMR.
- `hybm_data_op_device_rdma.cpp:1076-1125`: SafePut/Get registration lookup.

Local CPU tests use a real ctypes allocation, synthetic maps text to test
permission/range rejection, and independent A/B patterns. They do not claim
Linux live maps behavior or NPU/RDMA success. NPU initialization, HOST/HVA
mapping, actual registration and transport, full cross-host data completion,
native exit and device release remain **NOT_VERIFIED** until a separate
authorized run supplies the raw evidence.

## Explicit managed-GVA interface after D3 direction rejection

The D3 run registered HOST MR and produced valid CPU HVA mappings, but failed
before transport dispatch: `hybm_data_op_entry.cpp:49` reported
`Direction mismatch: specified=0 except:36`. Completed payload bytes were zero.
This is an interface/address-classification failure, not evidence that RDMA
payload transfer failed or succeeded.

On this A3 run the allocator's HVA `0x12c1c0000000` lies inside the SDK's
static device VA interval. `ClassifyAddressMask` falls through to
`GetLocalMemoryType` for the HVA and returns LOCAL_DEVICE (mask 4); the remote
HOST GVA returns GLOBAL_HOST (mask 2), producing `4 | (2 << 4) = 36`.
H2G requires LOCAL_HOST (mask 1), so it is rejected. Selecting L2G or AUTO
would preserve that incorrect device classification and is not the fix.

For a separate attempt add **`--copy-interface g2g` to both ranks**. The public
G2G API takes the local managed HOST GVA and remote HOST GVA for the write,
and remote HOST GVA and local restore HOST GVA for reads. `SmemBmEntry` derives
HOST-to-HOST direction 4 from both managed GVAs. `DataOpDeviceRDMA::TransformVa`
then converts the local GVA to the original registered HOST HVA and retains
the remote GVA. CPU initialization and full-byte checks continue to use only
validated HVA ranges; the RDMA backend, HOST pool and swap=0 remain fixed.

The original `h2g-g2h` interface remains the default for a direct comparison.
The selected interface is recorded and must match across ranks. Source basis:
`smem_bm_entry.cpp:590-621`, `hybm_data_op_entry.cpp:35-51`,
`hybm_va_manager.cpp:233-293`, `hybm_data_op_device_rdma.cpp:115-133,379-398`.
The CPU request checks verify exact GVA source/destination selection; they do
not prove that the device run passes direction checking or RDMA completion.

## Optional real NPU endpoint, with explicit local D2D staging

Add **`--local-endpoint device --copy-interface g2g` on both ranks** for a
separate device validation group. Defaults remain unchanged. This mode requests
1 GiB HOST plus 1 GiB HBM per rank. Rank 1 remains the remote HOST pool owner.
Rank 0 selects its own registered DEVICE GVA at source offset 0 and restore
offset `2*n`, checks `local_mem_size(DEVICE)`, nonzero/overflow-free mappings,
and contiguous `LOCAL_DEVICE` translations. Device addresses are never CPU
dereferenced. Failure to obtain the requested mapping aborts the attempt.

Rank 0 creates real contiguous uint8 `torch` NPU source and restore tensors.
The public SDK L2G interface copies source Tensor to its local HBM GVA, then
G2G copies that registered HBM GVA to the remote HOST GVA. After independent
owner verification, source Tensor and registered HBM source are overwritten.
G2G reads remote HOST into the separate registered HBM restore region; G2L
copies that region to the real restore NPU Tensor, which is then synchronized
and copied to CPU solely for full-byte verification. Owner-generated B repeats
the read with fresh sentinels. Tensor allocations stay alive through teardown.

This is a registered NPU-to-remote-HOST primitive with **local D2D staging**,
not zero-copy transmission directly from arbitrary torch scratch. No payload
is staged through a local HOST buffer during the measured cross-host copy.
Initial test-pattern upload and final CPU inspection are verification setup.
The trace records each local staging operation separately from cross-host
completion byte counters. It does not prove production integration or speedup.

Fixed SDK source: `hybm_data_op_device_rdma.cpp` local CopyLD2GD/CopyGD2LD
branches call CopyLD2LD (`ACL_MEMCPY_DEVICE_TO_DEVICE`); cross-rank
CopyGD2GH/CopyGH2GD call CopyRDMA. Public G2G selects DEVICE-to-HOST and
HOST-to-DEVICE from managed GVA regions, and TransformVa converts only local
managed pointers. `torch.npu.synchronize` brackets local staging and precedes
content inspection; synchronous SDK returns remain mandatory. CPU checks cover
the new explicit parameter contract and mapping geometry, not NPU execution.
