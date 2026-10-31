# Registered-memory cross-host DRAM primitives: selected results

Date: 2026-10-31. Two physical A3 servers in one SuperPod; unchanged
MemFabric 1.2.1 / `106f783df905fe8d8e40446375faf615cf9b74fb`.
Internal machine identifiers, raw logs, and credentials are excluded here.

## Verified HOST pool primitive

Revision `13cd85f`: managed HOST GVA-to-GVA copies with DEVICE_RDMA.
Five cases passed: forward/reverse 1 MiB, forward/reverse 18 MiB, and forward
64 bytes. All ten role processes naturally exited 0. Each case independently
checked owner A, reader A, owner B, and reader B in full.

The source was overwritten after the owner's first check; the restore buffer
was distinct and reset before reading the owner's update. Native logs confirm
HOST-to-HOST, registered MRs, CopyRDMA and one WriteRemote / two ReadRemote
operations per case. Total completed write/read: 39,845,952 / 79,691,904 bytes.

The earlier HVA H2G attempt was rejected by address-direction classification
before RDMA payload submission. It is not evidence that DEVICE_RDMA itself
failed. The managed GVA interface avoided that classification problem without
changing the SDK.

## Verified real NPU Tensor primitive

Revision `d1fd579`: `--local-endpoint torch-registered --copy-interface l2g-g2l`.
Source and restore are two real, non-overlapping torch NPU allocations,
explicitly registered through the public BM API. Synchronous L2G/G2L copies
use their actual pointers and a remote HOST pool GVA. No extra local HAL HBM
pool or D2D staging is used in this mode.

The same five cases passed. Both roles naturally exited 0 in every case;
20 independent full-content checks passed. Native DEVICE MRs, LD-to-RH writes,
RH-to-LD reads, CopyRDMA, and public register/unregister returns were audited.
Total completed write/read: 39,845,952 / 79,691,904 bytes. Test setup creates
the initial bytes on CPU and verification reads NPU results back to CPU;
those operations are outside the cross-host transfer under test.

Lifecycle qualification: public unregister returned 0 and the processes exited
normally, but each case emitted two native `HalHostUnregisterEx ... ret:8;
teardown continues` warnings. The warnings are retained and unresolved;
the result does not assert warning-free native deregistration.

An earlier registered-HAL-HBM staging variant crashed with SIGSEGV during
its first local ACL D2D operation. It did not establish cross-host NPU transfer
and was not merged with successful samples. Its peer required planned cleanup.

## Scope

These are independent synchronous functional primitives, not production
LMCache L2 or worker integration, model KV reuse, TP validation, or performance
results. The original HOST SDMA route remains failed. Native evidence identifies
DEVICE_RDMA/HCCP; no additional claim about HCCS link bytes or speedup is made.
The final narrow CPU checks passed 7 tests. SDK, drivers, production adapters,
models, and compatibility baselines were not modified.
