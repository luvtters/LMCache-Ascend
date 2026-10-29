# Two-slot pool capacity experiment

This is synthetic NPU context/RPC validation, not model inference or a performance benchmark. It keeps the same P/D members alive for the whole A/B/C sequence. No payload is written to coordination files.

The external direct service must start with `pool_gb=1`, `slot_size_mb=512`, `world_size=3`, `rank_id=0`. This constrains the index to two object slots; it does **not** reduce the physical BM pool to 36 MiB. Each default object contains 18 MiB of actual BF16 NPU data.

In two prepared container terminals, set the respective physical-device mask and run P/D in parallel. Replace paths and URLs with the current case. The following omits the caller's environment setup:

```bash
python "$RUN/runtime/pool_cases.py" \
  --role p --device 0 \
  --ascend-source "$RUN/src/LMCache-Ascend" \
  --core-source "$RUN/src/LMCache" \
  --coord-dir "$RUN/capacity01" --run-id objects01 \
  --server-url tcp://127.0.0.1:25595 \
  --store-url tcp://127.0.0.1:28695 \
  --timeout 180 --run --allow-npu
```

D runs the same command with `--role d` under its own one-card mask. Without `--run`, the script prints a plan and imports no accelerator packages; `--run` without `--allow-npu` is rejected.

The sequence is:

1. P stores A/B using the real context. Their NPU values differ by 11.
2. D holds a metadata lookup pin on A. P stores C with a third distinct NPU pattern, causing B eviction.
3. D rejects the old B descriptor, restores and checks A/C content and untouched pages, and rejects C with the old slot generation.
4. With A/C both pinned, a pure `prepare_store` RPC must reject pool-full. It never enters `submit_store`, so the intentional rejection does not poison the worker context.
5. Query pins are released; pending writes, quarantines and pins must be zero. Both workers rendezvous before normal context/client close.

Each successful synchronous BM copy records direction, bytes and elapsed nanoseconds. Expected totals are P: three L2G copies / 54 MiB; D: two G2L copies / 36 MiB. These are completed API copy counters, not physical HCCS byte counters. A process supervisor must independently record exit codes; JSON PASS alone is insufficient.

On an error, worker-owned accesses must drain before the harness releases its additional metadata protections. A drain/unknown-response error is recorded as unknown completion; the harness does not infer completion from elapsed time or force slot reuse. Use a new fixed BM group after a failed case.

Local preparation checks: syntax, plan/import without device packages, and the actual pure-CPU PoolIndex A/B/C protocol sequence passed. Real NPU execution remains `NOT_VERIFIED` until the parent task runs it.
