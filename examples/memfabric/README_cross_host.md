# Cross-host worker-direct role contract

The independent HBM candidate accepts `--pool-memory-type device` for inference
roles only. Use the same option on service/P/D and all helper role plans. Objects
select the SDK DEVICE pool; every BM rank still declares **1 GiB DRAM plus
pool_gb GiB HBM**, matching the verified media probe. This is not a pure-HBM
allocation and does not repair or validate the failed cross-host HOST DRAM route.
Default `host` retains the existing DRAM-only behavior. Status reports the
selected medium and both per-rank declarations. Physical device lists remain
explicitly overridable; no new allocation, reservation or hardware permission is
implied by a generated role plan.

These are experiment-only launchers for fixed core `8bc9edc` and Ascend baseline
`7e5d40a` plus the current cross-host changes. CPU checks do not validate RPC,
MemFabric, NPU, SDMA reachability, inference, or native shutdown.

`cross_host_role.py` prints one JSON role plan by default. It does not SSH,
reserve devices, start background work, or stop other processes. `--run` replaces
the launcher with that one foreground process; device roles also require
`--allow-npu`. The outer executor must supply the verified runtime environment,
record the actual PID/process group, wait for every child, and enforce its own
authorized deadline on BOTH hosts. It must never restart a failed BM group.
At actual execution, the runner prefixes its source paths to inherited
`PYTHONPATH` and `lmcache_ascend` to inherited `LD_LIBRARY_PATH`; verified CANN,
ACL and driver search paths remain intact. The printed plan contains prefixes
only and does not capture the planning machine's environment. Executors that
run the printed argv directly must apply the same prefix-and-preserve rule.

Default paths:

- root: `/workspace/d00896497/crosshost_20261030`
- core: `<root>/src/core_direct`; Ascend: `<root>/src/ascend_direct`
- Python: `/workspace/d00896497/venv/bin/python`
- model: `/workspace/d00896497/models/Qwen3-8B` (existing read-only original)
- local artifacts: `<root>/runs/<session-id>`; identical paths are NOT shared storage

## Topologies and addresses

| Topology | service/pool rank0 | P rank1 | D rank2 |
| --- | --- | --- | --- |
| T1 | 12 physical8 | 137 physical8 | 12 physical10 |
| T2 | 137 physical8 | 137 physical10 | 12 physical8 |
| T3 (first target) | 12 physical8 | 137 physical8 | 137 physical10 |

137 uses `192.168.0.223`; 12 uses `192.168.0.50`. Every device role masks exactly
its assigned physical device through `ASCEND_RT_VISIBLE_DEVICES`; process-local
device and BM device are 0. The outer executor must confirm the actual mapping.
BM world=3, P/D TP=1, one service, `init_timeout=300`, synchronous copy semantics.
All ranks use the pool host's BM store `:28695` and service TCP `:25595`.
Service HTTP is `:28195`. P/D APIs use `:28401/:28402`. Proxy/gate are on 137,
`:28400`, proxy telemetry `:25808`, gate telemetry `:25809`.

P/D connector config explicitly sets `lmcache.mp.autostart=false`. Start the
service explicitly. Service and rank0 owner remain the same process and host.

## Single-role invocation

From either private Ascend checkout, inspect a role without executing it:

```bash
python examples/memfabric/cross_host_role.py --topology T3 \
  --stage inference --session-id w02-t3-attempt01 --role service
```

To execute that role on the host named in the JSON plan:

```bash
python examples/memfabric/cross_host_role.py --topology T3 \
  --stage inference --session-id w02-t3-attempt01 --role service --run --allow-npu
```

The same contract applies to `p`, `d`, `proxy`, `gate`, `coordinator`, `trace137`,
and `trace12`. CPU roles omit `--allow-npu`. `--root`, `--python`, and `--model`
override paths without changing the fixed topology. Do not copy model weights.
For T1/T2 the outer executor must confirm 12's existing read-only model path.

## Launch order and evidence

Use a fresh session-id for every stage, topology, retry, or changed condition.

1. `--stage primitive`: start CPU `coordinator` on the pool host. Launch `service`
   (a bare rank0 BM process in this stage), `p`, and `d` promptly in parallel.
   They exchange only immutable completion metadata over HTTP. P clears its
   source and scratch after L2G; D compares the ENTIRE restored chunk against an
   independently generated NPU reference, then verifies scatter and untouched
   blocks. Both full (`--skip-blocks 0`) and skip (`--skip-blocks 1`) runs require
   fresh sessions. Reverse/non-contiguous block mappings are `3,1 -> 4,2`.
2. `--stage context`: start coordinator and the explicit LMCache service/rank0.
   Perform the outer executor's short real RequestClient metadata probe, then
   start P and D together. Do not wait for rank0 BM readiness before starting
   peers. Workers use the real direct context and index, acquire/commit/release,
   full content checks, and final zero pins. The service stays alive until both
   worker processes have exited and the executor records its shutdown.
3. `--stage inference`: start `trace137` and `trace12` (CPU `:29596/snapshot`),
   explicit service/rank0, then P and D concurrently. Once service and engines
   are genuinely ready, start proxy, then gate. The gate itself sends exactly
   one frozen chat request with `max_tokens=8`; it asserts 1373 prompt tokens,
   1280 external / 93 computed / 0 local-prefix tokens and 180 MiB completed in
   each direction. It never clears L1 or sends a hidden extra P request.

The CPU trace endpoints read only local raw JSONL metadata for their exact
`MF_DIRECT_SESSION_ID`. They return original records, explicit host identity,
host-local monotonic time, and counters derived from those raw completion
records. They never read KV or invoke a device operation. These counters are
not hardware link-byte counters.

Before inference the gate samples a baseline independently on each host. It
filters and orders each worker's trace only in that worker host's time domain,
including stage/native-call/BM completion pairings. It never compares host A's
monotonic time to host B's or to the gate clock. Host/session changes are errors.
The causal release proof is P store-finished callback -> same-epoch committed
pool objects with zero pending writes -> P L2G completion / no D copy -> proxy
notification allowing D. `causal_gate_events.json`, raw snapshots, pool status,
HTTP body/response, metrics, and trace pairings retain this evidence. A copied
JSONL file alone is not accepted as cross-host ordering proof.

HTTP events (`direct_coordination.py`) are immutable and persisted under their
attempt: same body is idempotent, different body gets HTTP409, any `failed-*`
event aborts waiters, and waiting has a local monotonic timeout. File rendezvous
remains available in `validate_direct_kv.py` when `--coord-url` is omitted; it is
only suitable for a genuinely shared local directory on one host.

After every stage stop new work, drain protected objects, stop and wait for P/D,
then stop and wait for service. Retain all nonzero codes/signals and native
errors separately from content/request correctness. Finally the outer executor
must collect raw resources/containers and confirm both hosts released the group.

## Local CPU check

```bash
python examples/memfabric/check_cross_host_cpu.py
```

This covers independent clock domains and identity rejection, HTTP immutable
events/duplicate/conflict/failure/timeout, three topology expansions, callback
configuration and output-token limits. It imports no accelerator module.

## Explicit homogeneous TP candidate (default remains TP1)

`--tp-size 2|4|8` is an opt-in **inference-stage candidate** requiring the separate
production TP changes (including the status-field collision fix). It must not be
deployed against the fixed TP1-only baseline. Synthetic primitive/context roles
currently remain TP1 and explicitly reject a TP candidate selection.

The first TP2 T3 role plan can be inspected with:

```bash
python examples/memfabric/cross_host_role.py --topology T3 --stage inference \
  --session-id w02-t3-tp2-attempt01 --tp-size 2 --pool-gb 2 --role p
```

Use the SAME TP/capacity/session flags for service, P, D, proxy, trace helpers and
gate. TP2 T3 defaults to P137 physical `[8,9]`, D137 `[10,11]`, pool12 `[8]`.
The P/D device mask maps to logical `[0,1]`; production context receives these
explicit `device_ids`, P base BM rank1, D base rank3, world5. Actual model config
supplies total KV heads; the launcher never fabricates that field.

Optional `--p-devices 8,9 --d-devices 10,11 --pool-device 8` overrides physical
lists. Duplicate devices, wrong list length, and overlaps between roles on the
same host are rejected. Larger default T3 candidates are P `[8..11]` / D
`[12..15]` for TP4 and P `[0..7]` / D `[8..15]` for TP8; these are plans, not
resource authorization. Service keeps 32 MiB slots; `--pool-gb 2` can hold the
single-prefix 10-chunk TP2/4 case (TP8 needs at least 80 slots, hence 3 GiB).
The caller must explicitly select capacity sufficient for its TP condition.

The gate collects one immutable P completion receipt from EVERY distinct real
`kv_rank`, refusing conflicting or unrelated request receipts. Earlier ranks
receive a local acknowledgement; they are not forwarded to the proxy, whose
current handler would release D on the first received receipt. Only the full
receipt set plus completed pool state/trace proof produces one proxy release.

Each trace record carries actual worker `kv_worker_id`, vLLM rank and TP size.
The gate requires P and D to each have all TP worker processes and the expected
individual BM ranks. For 1373 input / chunk128, each rank must complete exactly
10 transfers in its own direction, and its scratch/chunk bytes must be
`18 MiB / TP`. Pool committed objects must be `10 * TP`; total completed bytes
remain exactly 180 MiB in EACH direction. Per-rank gather/native/BM/scatter
pairings, host-local time baselines and byte checks prevent an extra transfer on
one worker from compensating for a missing or undersized shard on another.
Observer counters retain role/shard/rank/PID and remain raw-trace-derived only.

CPU checks include TP1/2/4/8 expected counts, independent time domains, missing
rank and wrong BM rank rejection, and duplicate/partial/conflicting TP receipts.
These checks do not validate the actual model, tensors, HCCS, or native runtime.
