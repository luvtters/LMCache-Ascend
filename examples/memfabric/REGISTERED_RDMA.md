# Registered worker-direct RDMA candidate (2026-10-31)

Base: worker lifecycle commit `c790352`. CPU control contracts pass locally;
this production integration has not yet been exercised on NPU or with a model.
It does not change LMCache core, the compatibility baseline, or SDK 1.2.1.
Matching core lifecycle revision: `5b565d211416cd48bebf20bae12a4abc15a72ba9`.
Use this Ascend candidate on service and every worker; do not mix it with the
older SDMA-only server module for RDMA operation.

## Configuration for the actual worker/direct service

Keep all existing rank/world/device/pool/store settings. On **every** rank,
add the following fields to the direct service's `config` and the worker's
`kv_connector_extra_config["lmcache.mp.direct"]`:

```json
{
  "pool_memory_type": "host",
  "data_op_type": "DEVICE_RDMA",
  "device_nic": "tcp://127.0.0.1:10005",
  "enable_56bits_gva": true
}
```

The NIC value above is the explicit value used by the independently verified
registered-torch primitive. Select the NIC according to that runtime's verified
device/network mapping; this document does not claim that it selects a measured
wire transport. Set the same previously verified runtime environment on all
participating processes:

```sh
export MF_HYBM_RDMA_SWAP_SPACE_SIZE=0
```

Retain each role's explicit physical-device visibility and a fresh group/store
port/result directory. The existing gate and P/D/server launch commands remain
usable after injecting these fields into their JSON. `cross_host_role.py` and
`validate_direct_kv.py` have no new RDMA CLI switches in this commit: their
generated default configs continue to use SDMA. Do not assume those defaults
activate this candidate.

The service status reports configured `protocol` / `data_op_type` and
`enable_56bits_gva`. Registration uses fixed `transport_contract_version=1` and
rejects a worker whose data operation or 56-bit setting differs before adding
that member. The default SDMA/false registration remains backward compatible
with missing optional fields. The worker verifies its ready-status contract as
well. NIC endpoint strings are rank-local and therefore are not equality-checked
across hosts. Configured status is not native route proof; preserve SDK logs.

The service still owns the remote HOST pool and payloads use NPU scratch to/from
that pool via L2G/G2L; RPC remains metadata only. The service contributes no HBM
in HOST mode. No local CPU or SDK HBM staging is introduced by this integration.

## Registration and completion contract

Worker setup synchronizes its NPU scratch and calls
`BMSession.register_npu_tensor(transfer.buffer)` once. The session holds the
actual Tensor, rejects overlapping registered buffers, and rejects any RDMA
NPU copy outside retained registered ranges. Scratch is private to the worker;
callers must not resize or replace its storage while registered.

Copies remain synchronous `copy_data(..., flags=0)`. Successful return is the
API completion basis. The bottom-level RDMA Wait is a no-op, but the SDK's
public compose Wait supports SDMA only and can fail for pure DEVICE_RDMA.
This follow-up skips that unsupported wait only after synchronous success;
it never counts as independent completion evidence. Native copy or registration failure retains
the session and tensors until process exit, and close raises without unregister,
leave or destroy. It does not recycle possibly active storage or release the
associated direct write/read protection as if completion were known.

After successful copies and NPU synchronization, normal close unregisters each
scratch, leaves/destroys the handle, uninitializes SDK state, then drops retained
Tensors. Native SDK unregister warnings are not suppressed or reclassified:
the independent primitive returned public unregister=0 while emitting
HalHostUnregisterEx ret8 for DEVICE slices. Preserve full native logs and actual
process exit codes in device acceptance.

Any exception from wait, unregister, leave, destroy, or SDK uninitialization
also retains the complete RDMA session and its Tensor references globally until
process exit, even if the caller drops its context. Such a failure does not mark
the session closed or permit a replacement BM session in the same process.

## L2 CPU MemoryObj boundary

The compatibility L2 adapter currently uses external CPU `MemoryObj` addresses
with H2G/G2H, and allocator ref-counting around synchronous copies. The passing
RDMA HOST-pool G2G primitive only established SDK-owned HOST allocations; it did
not establish that `handle.register(external_cpu_ptr, n)` supports the existing
LMCache CPU arena/pinning/allocation mode.

This candidate therefore does not expose RDMA through the L2 adapter. Calling
`BMSession.copy` in RDMA mode raises explicitly. A follow-up CPU-address primitive
must use the actual MemoryObj allocation, explicit public registration, complete
remote-owner and restored-byte checks, unregister/native-warning records, and
actual process waits before changing that path. A plain Tensor reference does
not replace MemoryObj allocator ref-counting; any future integration must keep
the allocator object live for the complete registered lifetime.

## Local verification

From this checkout, CPU-only commands:

```sh
python -m unittest discover -s tests/v1/distributed -p test_memfabric_rdma.py -v
python -m unittest discover -s tests/v1/distributed -p test_memfabric_bm.py -v
python -m unittest discover -s tests/v1/distributed -p test_memfabric_direct_hbm.py -v
python -m unittest discover -s tests/v1/distributed -p test_memfabric_direct.py -v
python -m unittest discover -s tests/v1/distributed -p test_memfabric_direct_server.py -v
python -m unittest discover -s tests/v1/distributed -p test_memfabric_direct_tp.py -v
```

For the core status-aggregation check in the TP module, set
`LMCACHE_CORE_SOURCE` to the matching core lifecycle checkout above.

Result: 10 new RDMA contracts, 4 BM controls, 7 medium contracts, 22 direct
lifecycle, 8 server and 8 TP contracts pass (59 total). These fakes establish configuration,
registration ownership, direction/flags and error cleanup control only. Real
worker-direct model gate, full KV byte equality, remote HOST provenance,
multirank RDMA coexistence and performance remain NOT_VERIFIED for this commit.
The separate real-router `test_memfabric_direct_protocol.py` could not import
on this local Windows Python due to missing `msgspec`; it is not counted as a
passing check, and no dependencies were installed for this candidate.
