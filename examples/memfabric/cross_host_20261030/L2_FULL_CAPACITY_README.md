# Full selected-pool capacity and object-size boundary

Tested probe: `75345b0`; production candidates: LMCache `fa2e6ff` and Ascend
`137038a`. No production files change. Execution defaults to plan-only and
requires explicit device opt-in plus an authorized external supervisor.

On 2026-10-30, one real-NPU host-A to host-B DEVICE/HBM SDMA group passed:

- Filled the selected 1 GiB pool with 32 sequential 32 MiB objects, then fully
  read and compared all 32 objects: 1 GiB payload, zero mismatches.
- Reused one CPU source buffer only after each synchronous SDK copy returned
  and its store future was consumed. No batch reused mutable buffer contents.
- Matched actual key/slot/GVA snapshots to each BM destination; observed all
  32 slots and the final slot ending exactly at the pool boundary.
- Stored a 33rd 32 MiB object, evicted the actual LRU key, reused its observed
  slot/address, and fully compared the replacement. Total writes and compared
  reads were each 1,107,296,256 bytes (1,056 MiB), with 66 successful BM calls.
- Rejected a 33 MiB object with zero completed bytes, no additional BM call,
  no extra eviction, and the adapter still healthy.
- Finished with zero objects/pins, 32 free slots, successful SDK close, and both
  participant processes naturally exiting with real wait code zero. An
  independent guard and subsequent device/port checks confirmed release.

CPU preparation used actual 32/33 MiB MemoryObjs and eventfds with a lossless
compressed fake BM backing store to bound CPU memory use. It is separate from
the real-NPU result. No large payload files or hashes were used for acceptance.
SDK flags-zero synchronous copy returns establish each copy's completion;
close-time BM wait and supervisor process waits are recorded separately.

This is one compatible L2 CPU-MemoryObj to remote HBM correctness/boundary test.
Internal native logs and deployment details remain private. No bandwidth,
long-term reliability, or model recovery conclusion is drawn.
