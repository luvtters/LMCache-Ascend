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
