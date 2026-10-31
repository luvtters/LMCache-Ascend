# Normal-cache compatibility-L2 comparison (2026-11-01)

This is a sanitized, small-sample diagnostic checkpoint. The comparison is
existing **Mooncake Store L2 (Ascend TE data plane)** versus the compatibility
**MemFabric L2 adapter in explicit DEVICE_RDMA mode**, not raw TE/BM copy or
worker-direct. P and D shared one compute host; the HOST DRAM pool was on a
second physical host. No MemCache, SDK changes or SDK upgrades were introduced.

## Configuration and selection

Both used Qwen3-8B BF16 TP1, eager mode, block128, max context8192,
max sequences2, each model KV budget2GiB, CPU L1 budget2GiB and LRU. Prefix
caching was enabled. Default store/prefetch policy and the original P/D proxy
were retained: no manual L1 deletion, prefix disable or extra pool-write barrier.
These are explicitly bounded capacities, not the product's factory defaults.

Each case used eight distinct synthetic sessions, followed by eight revisits;
the final128-token input chunk changed on revisit. This is not the customer's
original GLM dataset. Shared-prefix construction is not measured cache hit rate.
The table reports the revisit phase, eight requests per backend/case. Main4K
uses the completed r2 pair; short and concurrent cases use r1. Earlier main4K
r1 remains private diagnostic evidence; its official E2E metric was absent.
Planned r3 did not run. No claim of statistical significance is made.

Both remote owners allocated16GiB. Fixed32MiB MemFabric slots admit about9GiB
of18MiB KV payload, plus an unused16GiB local rank0 pool. Store has a16MiB
client segment. The approximately4.64GiB two-pass working set fits both pools,
but effective capacities and memory footprints are not identical.

## Measured client comparison

Change=(MemFabric/Store-1)*100%; lower latency/duration and higher throughput
are better. Input throughput is official input tokens divided by official
duration. The native official E2E metric was explicitly requested for these
groups. Request-success counts do not certify full content or SDK lifecycle.

| Metric | Input/output/concurrency | Store | MemFabric | Change |
|---|---|---:|---:|---:|
| Mean TTFT (ms) | 1K/128/C1 | 186.33 | 184.783 | -0.83% |
| Mean TPOT (ms/token) | 1K/128/C1 | 45.947 | 44.815 | -2.46% |
| Mean E2E (ms) | 1K/128/C1 | 6,021.609 | 5,876.273 | -2.41% |
| Input tokens/s | 1K/128/C1 | 170.04 | 174.246 | +2.47% |
| Output tokens/s | 1K/128/C1 | 21.255 | 21.781 | +2.47% |
| Requests/s | 1K/128/C1 | 0.166 | 0.17 | +2.47% |
| Round duration (s) | 1K/128/C1 | 48.177 | 47.014 | -2.41% |
| Completed requests | 1K/128/C1 | 8 | 8 | +0.00% |
| Failed requests | 1K/128/C1 | 0 | 0 | — |
| Mean TTFT (ms) | 4K/128/C1 | 309.516 | 832.809 | +169.07% |
| Mean TPOT (ms/token) | 4K/128/C1 | 44.598 | 48.382 | +8.48% |
| Mean E2E (ms) | 4K/128/C1 | 5,973.482 | 6,977.268 | +16.80% |
| Input tokens/s | 4K/128/C1 | 685.641 | 587.009 | -14.39% |
| Output tokens/s | 4K/128/C1 | 21.426 | 18.344 | -14.39% |
| Requests/s | 4K/128/C1 | 0.167 | 0.143 | -14.39% |
| Round duration (s) | 4K/128/C1 | 47.792 | 55.822 | +16.80% |
| Completed requests | 4K/128/C1 | 8 | 8 | +0.00% |
| Failed requests | 4K/128/C1 | 0 | 0 | — |
| Mean TTFT (ms) | 4K/128/C2 | 788.087 | 853.271 | +8.27% |
| Mean TPOT (ms/token) | 4K/128/C2 | 50.454 | 54.453 | +7.93% |
| Mean E2E (ms) | 4K/128/C2 | 7,195.706 | 7,768.786 | +7.96% |
| Input tokens/s | 4K/128/C2 | 1,127.713 | 1,054.404 | -6.50% |
| Output tokens/s | 4K/128/C2 | 35.241 | 32.95 | -6.50% |
| Requests/s | 4K/128/C2 | 0.275 | 0.257 | -6.50% |
| Round duration (s) | 4K/128/C2 | 29.057 | 31.077 | +6.95% |
| Completed requests | 4K/128/C2 | 8 | 8 | +0.00% |
| Failed requests | 4K/128/C2 | 0 | 0 | — |

These samples show no MemFabric benefit for the4K workload. Short revisits are
close and mostly use the local model prefix cache, so their small difference
cannot establish an L2 benefit. TPOT differences are not attributed solely to
copy. Actual restore/recompute and L1/L2 block counts differ between backends.
In main4K r2 revisit, service records include400 L2 blocks for MemFabric and346
for Store, summed over both P and D (not unique user KV blocks).

## Additional observed stage boundary

A post-hoc same-size slice of main4K r2 revisit selected service prefetch
records with requested32, retained31, L1=0 and L2=31: five records per backend.
Mean service-prefetch elapsed was355.78ms for MemFabric and47.36ms for Store.
31*18MiB=558MiB is inferred object volume, not measured wire bytes. This slice
does not replace client means or constitute independent repeated experiments.
The window includes service scheduling, indexing and backend completion;
it is not pure DMA or H2D time. It supports a slower recorded L2 recovery
window, but does not isolate BM registration, copy and unregister. Their
separate diagnostic was prepared on CPU but did not execute on devices.

Store's main4K r2 revisit native batch_get_into log had12 calls,346 memory
keys, total491.842ms and mean40.987ms. These are SDK API windows, not pure DMA.
The installed Ascend SDK route was observed; the adapter's protocol="tcp"
field is not proof of TCP payload transport. Wire HCCS/RoCE bytes were not
quantified. This is a backend comparison, not isolated transport attribution.

## Failures and incomplete work

Store4K/1000-output revisit was interrupted when the service exited with
SIGKILL(-9), with no signal sent by this supervisor. It reported2 completed
and6 failed requests, so that phase is excluded from this comparison. The
source of SIGKILL is unknown; no matching container/kernel OOM evidence was
found. The queue and following diagnostic stopped, without reclaim/restart.
Long-output fill completed on both sides and remains separate private evidence.

Other Store groups completed their measured requests but failed during
teardown: service SIGSEGV(-11), owner SIGABRT(-6). SDK-only loading also
reproduced SIGABRT without LMCache or models. A trace/free crash site is not
proof of the original corruption source. MemFabric free-sq-cq -8 warnings
remain, despite controlled SIGINT process completion in its completed groups.

Complete samples passed the requested READY-prefix check under forced output
length, but some4K full continuations differ in body text. Short outputs match.
The content-difference cause is not determined. This is not full production
correctness, lifecycle, stability or performance certification.

Both personal containers were stopped and their process IDs read back as0;
no project accelerator processes remained. Raw internal endpoints, logs,
requests, per-item outputs and release receipts remain in the private archive.

The executed wrappers are archived in [l2_rdma_runtime_20261101](l2_rdma_runtime_20261101/README.md).
Functional registration/context evidence is recorded in
[the L2 checkpoint](REGISTERED_L2_RDMA_VALIDATION_20261101.md).
