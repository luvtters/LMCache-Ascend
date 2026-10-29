# Experimental two-sequence engine scheduling evidence

Example code: `ecf9e5b`; strict installed-vLLM ID matching: `7178a1c`.
Run these examples against separately selected production candidates, tested
here with LMCache core `5b565d2` and Ascend worker candidate `1e5ef32`.

`cross_host_role.py` keeps `--max-num-seqs=1` by default and accepts an explicit
value of 2. The experiment connector records the original scheduler's positive
token assignments before delegating, preserving its input, call order and return.
Scheduler records retain their actual process identity and are distinguished
from TP worker records.

`direct_engine_mixed_probe.py` reuses an original ORCHID gate and runs two HTTP
pairs: warm/partial and warm/cold. It preserves complete HTTP inputs/responses,
tokenization, metrics, status and worker trace. A mixed claim requires two
successful HTTP requests, positive scheduled tokens in one D engine step, and
an unambiguous mapping to both complete API response IDs.

## Selected real-NPU validation, 2026-10-30

- Homogeneous TP4 P/D engines on host A, selected DEVICE/HBM pool on host B,
  engine max-num-seqs=2. Each rank also declared 1 GiB DRAM plus 2 GiB HBM.
- The original observer `ecf9e5b` retained its safe result
  `FUNCTION_PASS_SCHEDULING_NOT_PROVEN`. Its HTTP correctness passed; the
  observer did not yet account for the installed engine's internal ID suffix.
- The new independent group using `7178a1c` reported
  `ENGINE_MIXED_SCHEDULING_FUNCTION_PASS` for both pairs. Original scheduler
  records and successful HTTP responses independently confirmed the same-step
  two-request mapping. The earlier result was not overwritten.
- Actual prompt lengths were warm 1,373, partial 1,237, cold 1,113 tokens;
  common complete chunks were 10, 5 and 0. Pair metric deltas were respectively
  prompt/external/compute/prefix 2610/1920/690/0 and 2486/1280/1206/0.
- Every response was ORCHID with four output tokens. Each D shard completed
  15 or 10 G2L/native/scatter sequences using its actual 4.5 MiB NPU scratch.
  P had no new BM copy; case-end pins/pending/quarantine were zero.
- Gate/probe/executor exited naturally with code 0; P/D/proxy used controlled
  shutdown with code 0, pool service controlled 130, and trace collectors -2.
  All waits were retained; this is not an all-natural-zero exit claim.

Installed vLLM retains the complete external/API ID and optionally adds `-`
plus exactly eight lower-case hex characters for its internal scheduler ID.
Matching is fully anchored; wrong prefixes/suffixes, unrelated requests and
one-to-many/many-to-one ambiguity cannot establish a mixed claim. Eight focused
CPU checks passed. A missing mapping reports function success with scheduling
unproven rather than silently claiming batching.

One scheduling step containing two requests establishes engine scheduling
overlap. Shared-scratch transfer work remains serialized; asynchronous BM,
kernel/stream concurrency, bandwidth, speedup and long-term stability are not
established. Internal host addresses, request identities and raw logs remain
private. The first validation above covers the two mixed-request groups.

## Additional short repeated-retrieval validation

Three independent Qwen3-8B groups each completed 500 HTTP requests: TP1 with
max-num-seqs=1, TP4 with max-num-seqs=1, and TP4 with max-num-seqs=2. Each group
used a fresh pool and retained its own complete responses, counters and waits.
The first 250 requests were serial; the next 250 used two HTTP client threads.

- Every response reproduced the first-chunk context marker: 1,364 prompt tokens
  and five output tokens. Per-group metric deltas were prompt 682,000,
  external 640,000, compute 42,000, and local-prefix 0.
- TP1 completed 5,000 G2L operations of 18 MiB. Each of the four TP4 D shards
  completed 5,000 operations of 4.5 MiB. Aggregate completed bytes were
  94,371,840,000 (90,000 MiB) per group, matched independently to original
  worker identities, native returns and scatter completions. P added no copy.
- At every 50-request checkpoint, ordinary L1 held no objects and no L2 adapter was configured;
  pins, pending writes and quarantine were zero. Client pending counts covered
  this HTTP probe only; the internal engine RPC pending map was not exposed.
- In the final TP4/max-num-seqs=2 group, original scheduler records matched
  733 same-step pairs to successful HTTP requests in the two-client phase.
  The first pair scheduled 84 tokens for each request. The strict complete-ID
  contract above and both HTTP time intervals were checked independently.
- Probes/gates/executors exited naturally with code 0, P/D/proxy via controlled
  shutdown with 0, pool service via controlled SIGINT with 130, and trace
  collectors with -2. Dispatcher shutdown preceded BM teardown.

These are short functional samples. Timing was used to check the execution
budget, without a performance comparison. Same-step admission does not establish
asynchronous DMA or concurrent kernels. The production data path still uses
per-chunk NPU scratch; the separately tested native raw-page batch primitive
has not been integrated into that context.
