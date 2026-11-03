# Compatibility L2 with an unchanged LMCache core

The runtime checks used official LMCache core
`6913259264dbebb2fa5ba1665606c40ffc5c6661`, unchanged, and Ascend
`467855d16ac5ee9a12f5647b55389d4572aeb9a8`. This delivery preserves the same
production and test contents; the validation and registration notes are
documentation updates. It does not add worker-direct hooks to the L2 route.

The adapter reads the existing public `stable_registration_size` descriptor
field. It does not require a custom core field, inspect allocator classes or
private state, or treat final capacity as a stable prefix. A missing prefix
uses per-object registration. MemoryObj leases and adapter-before-L1 shutdown
retain the existing core contract.

## Recorded single-host coverage

On an Ascend A3 host with MemFabric SDK 1.2.1 and Qwen3-8B:

| Path | Recorded result |
|---|---|
| DEVICE_RDMA TP1, TP2 and TP4 | Strict pooled-KV requests succeeded. |
| DEVICE_RDMA TP4 with BM/model device sharing | A qualified context assignment succeeded. |
| DEVICE_RDMA concurrent TP8 | P and D occupied separate sets of eight devices simultaneously. Eight shards each stored and loaded ten 2.25 MiB objects: 180 MiB per direction. D restored 1280 tokens, computed 93, and reported zero local-cache tokens; service provenance was 0 L1 / 80 L2. |
| SDMA TP2 and TP4 | Single-host strict requests succeeded; this is not cross-host SDMA evidence. |
| Default Lazy L1, full 2 GiB stable prefix | Registration reuse worked; the 180 MiB representative request touched six 32 MiB regions. |
| Fixed POSIX SHM | Missing public stable prefix correctly selected per-object registration; inference succeeded. |
| Four fixed BM members | Three independent groups selected storage owner rank 1, 2 or 3. Each passed the real MemoryObj boundary checks, including complete content and role waits. This is not automatic capacity aggregation. |
| Registration boundaries | Real L1 objects covered a 32 MiB MR boundary, a clipped tail, a wholly external range and explicit rejection of a prefix-straddling object. |
| Real Lazy L1 API growth | A 64 MiB startup prefix grew to 256 MiB. With a qualified context assignment, in-prefix and external 32 MiB objects completed round trips with full-content and reference checks. |
| Longer inputs | An 8208-token cold request computed locally; a repeat restored 8192 tokens and computed the 16-token tail. Concurrent requests retained valid answers and complete token-source accounting. |
| Scheduler concurrency | With `max_num_seqs=2` and two 128-token decode requests, raw D metrics observed two running requests. This does not establish SDK batching or kernel overlap. |
| Client disconnect | A streamed client disconnected after the first content token; observable queues/pins returned to idle and a subsequent short request succeeded. Server-side abort and SDK pending I/O were not independently established. |

The original service logs, HTTP responses, raw metrics/status queries, exact
argv, source identities and true process waits are retained in private
validation records. Script-level PASS output was not the sole evidence. The
CPU fake-SDK checks are separate from these device and model results.

## Remaining limitations

- Fixed anonymous backing failed native registration in both registration
  modes. Some device-context assignments also failed; success is limited to
  the recorded deployment profiles.
- A partially committed prefix is a startup snapshot, not a subscription to
  allocator growth. A straddling object is rejected. The standalone growth
  check does not qualify growing L1 during model inference.
- SDK `free sq_cq failed: -8`, internal worker signal escalation/zombies and
  unregister warnings remain. Successful requests and waited parent processes
  do not imply warning-free, graceful shutdown of every child.
- Exact native-extension build provenance was not established for the reused
  binary. Runtime source identity and the reused binary are recorded separately.
- The current production diff leaves BM transport, rank/address exchange,
  lookup/index and copy-completion paths unchanged. Previous physical
  cross-host DEVICE_RDMA HOST-pool results therefore provide conditional
  source-review evidence only; this candidate was not physically cross-host
  retested. Cross-host SDMA and model P/D on different physical hosts are not
  qualified by these single-host checks.
- No new performance advantage is claimed. Lower-level copies, inference
  correctness, scheduler concurrency and end-to-end performance are distinct.
