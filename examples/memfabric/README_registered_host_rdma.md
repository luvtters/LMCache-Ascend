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

## Optional registration of real torch NPU allocations

For a fresh independent attempt, both ranks explicitly select
**`--local-endpoint torch-registered --copy-interface l2g-g2l`**.
This requests HOST 1 GiB and HBM pool size zero. Rank 0 allocates separate real
contiguous uint8 NPU source and restore tensors, synchronizes them, rejects
null/overflow/overlapping ranges, and calls public `BigMemory.register(ptr,n)`
for each allocation. Every registration requires native return zero and is
logged separately. Any missing API or registration failure aborts; no fallback
or custom pointer wrapper is used.

The remote write is `L2G(source_tensor_ptr, rank1_HOST_GVA,n,0)` and both remote
reads are `G2L(rank1_HOST_GVA,restore_tensor_ptr,n,0)`. There is no local HAL HBM
pool, local L2G/G2L staging or intermediate HOST payload buffer. The existing
DEVICE_RDMA SafePut/SafeGet registration checks must hit the newly registered
real tensor range, then call CopyRDMA. Owner-local HVA full-byte A verification,
source overwrite, owner-written B, restore sentinel reset, independent reader
full-byte checks and both-done barrier remain required. Completion counters
advance only after a zero synchronous remote copy return.

The registered tensors remain alive through explicit `unregister` calls and
all SDK cleanup. Register and unregister raw returns are retained; native logs
must also be checked because lower SDK cleanup paths can emit warnings while
returning success. Public API basis: `docs/pythonAPI.md:332-333`,
`smem_bm.cpp:565-594`, `smem_bm_entry.cpp:674-720`,
`hybm_entity_default.cpp:240-286`, `hybm_mem_segment.cpp:128-170`.
The registered device slice produces REG_MR_FLAG_HBM on the same transport.

H2's previous staged-HBM route crashed with native SIGSEGV after entering its
first local CopyLD2LD call, before the remote write. Fixed SDK CopyLD2LD calls
DlAclApi::AclrtMemcpy directly with DEVICE_TO_DEVICE; the wrapper calls the
loaded public `aclrtMemcpy` function, not `memcpy_s`. Repeating the identical
ACL call outside the SDK has no demonstrated corrective basis. The crash does
not prove a general prohibition on HAL-mapped HBM pointers or remote RDMA
failure. The torch-registered route removes that unverified local staging
requirement and still needs separate real-device completion/cleanup evidence.

## Optional synchronous API sampling

On both ranks add `--benchmark-iterations 30 --benchmark-warmup 3` with
`--local-endpoint torch-registered --copy-interface l2g-g2l`. Default iterations
zero preserves functional-only execution. Register once; rank 0 first performs
3 warmup plus 30 measured writes of A, then owner fully checks the final A.
After source overwrite, 3 warmup plus 30 measured reads of stable owner A run,
followed by full NPU restore content verification and the original independent
owner-B write/read/full-content challenge. All original failure/cleanup gates
and external process waits remain required.

`copy_samples.write/read` retain every warmup/sample raw return and elapsed
`perf_counter_ns`, successful completed bytes and failure truncation. The loop
contains no coordination, content comparison or Python event printing. Public
SDK log level is temporarily warn during samples, then restored to debug
outside the loop; native warnings/errors remain visible. Routine debug log
printing would otherwise dominate API timings. Registration wrapper elapsed
times are separate and include event logging; they are not copy latency.

Cross-host counters include successful warmup and measured copies plus the
extra B read; at 3/30 expect write `33*n`, read `34*n`. Coordination marker
bytes still describe the final single payload verified by owner. Preserve raw
per-call samples; compare only paired same-host/cross-host groups at equal
size and conditions. These are synchronous primitive API timings, not model
TTFT, throughput, or production-adapter results.

Sampling also explicitly accepts `--local-endpoint host --copy-interface g2g`
for the verified registered HOST GVA route. It uses the identical timing loop
and unchanged owner A/B full-content verification outside that loop. Sampling
still rejects the original H2G/G2H interface and HAL DEVICE staging mode.

## Optional independent-session write/read sampling barriers

For a new explicitly paired attempt, start a separate `direct_coordination.py`
HTTP server with a fresh global run-id, separate from either BM attempt and
its normal owner/source coordination run. On each **source rank0** add:

```text
--sample-barrier-url http://137_COMMUNICATION_IP:GLOBAL_COORD_PORT
--sample-barrier-run-id fresh-global-pair01
--sample-barrier-participant A
--sample-barrier-participants A,B
```

The other source uses participant B. Both retain distinct BM store/group
processes, distinct BM run-ids, normal per-group coordination and explicitly
assigned cards. Owner rank1 need not receive barrier flags. All four barrier
options are required together; sampling must be enabled, participant IDs must
be distinct and simple ASCII identifiers, and the global ID must differ from
the local BM run-id. Default execution has no cross-session barrier.

After source allocation/registration/initialization and owner-address readiness,
source A/B publish immutable `sample-ready-write-A/B` metadata, validate both
ready records, then enter their write warmup/measured loop. After each owner's
full A verification and source overwrite, they independently rendezvous at
`sample-ready-read-A/B` before their read loop. No KV bytes travel through this
barrier. Each ready record includes exact payload/iteration/warmup/interface
contract and source process identity; duplicate BM attempts/processes,
conflicting events, participant disagreement, peer failure or timeout abort.
Source errors also publish global failed-participant metadata so a peer waiting
in the other phase can stop. Every wait shares one bounded phase deadline.
All HTTP and validation occurs outside the timed copy loop.

Each warmup and measured copy now records `start_monotonic_ns` and
`end_monotonic_ns` as outer bounds around its `perf_counter_ns` API timing.
The result carries `clock_host_id` (hostname plus boot ID where available) and
start/end monotonic-to-Unix clock anchors. Compare intervals directly only
when clock_host_id is identical. Anchors do not establish cross-host clock
synchronization. The paired source processes on137 can therefore be audited
for actual overlapping synchronous copy API intervals rather than mere process
coexistence. Report observed overlap (or lack of overlap) from raw samples;
the barrier itself does not prove overlap, physical DMA concurrency, TP2,
aggregate throughput gain or model benefit.
