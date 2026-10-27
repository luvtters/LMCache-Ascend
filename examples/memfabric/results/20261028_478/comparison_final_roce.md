# BM / TE transfer results

Same-host CPU DRAM: BM SDMA versus TE Ascend Direct, RoCE requested. This is a path comparison; it does not isolate identical HCCS links or pure backend overhead.

Warmup=20, iterations=100, repeats=3; batch=1, outstanding=1, Python submission threads/process=1.

| MiB | Direction | Backend | n | Mean us | p50 us | p95 us | Effective GB/s | BM/TE mean time |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | write | BM SDMA | 300 | 215.503 | 213.651 | 241.357 | 4.866 | 1.674 |
| 1 | write | TE Ascend Direct, RoCE requested | 300 | 128.710 | 126.571 | 138.289 | 8.147 | 1.674 |
| 1 | read | BM SDMA | 300 | 228.842 | 226.392 | 245.828 | 4.582 | 1.823 |
| 1 | read | TE Ascend Direct, RoCE requested | 300 | 125.544 | 124.571 | 132.774 | 8.352 | 1.823 |
| 18 | write | BM SDMA | 300 | 1330.361 | 1328.265 | 1353.066 | 14.187 | 1.555 |
| 18 | write | TE Ascend Direct, RoCE requested | 300 | 855.373 | 855.106 | 861.807 | 22.066 | 1.555 |
| 18 | read | BM SDMA | 300 | 1280.134 | 1278.409 | 1300.857 | 14.744 | 1.495 |
| 18 | read | TE Ascend Direct, RoCE requested | 300 | 856.379 | 855.851 | 866.779 | 22.040 | 1.495 |
| 64 | write | BM SDMA | 300 | 3239.429 | 3158.948 | 3254.162 | 20.716 | 1.144 |
| 64 | write | TE Ascend Direct, RoCE requested | 300 | 2832.305 | 2830.910 | 2848.262 | 23.694 | 1.144 |
| 64 | read | BM SDMA | 300 | 3654.315 | 3514.959 | 3638.473 | 18.364 | 1.292 |
| 64 | read | TE Ascend Direct, RoCE requested | 300 | 2828.799 | 2828.235 | 2839.735 | 23.723 | 1.292 |

Statistics pool all individual completed-copy samples across repeats. Effective GB/s = total successful payload bytes / total synchronous API time. Per-repeat bandwidth and percentiles are not averaged. BM/TE mean time < 1 means BM has lower mean time.

- BM: registration=compat BMSession.copy, no explicit preregistration; allocator=posix_memalign(65536), ordinary pages, first-touch memset; remote allocator=BM private DRAM pool (1 GiB per rank); NUMA=worker resource snapshots show mapping location; BM pool NOT_VERIFIED; working set=fixed addresses reused; hot buffers.
- TE: registration=explicit TE register_memory; allocator=aclrtMallocHost, pinned Host, first-touch memset; remote allocator=aclrtMallocHost, pinned Host, first-touch memset; NUMA=worker resource snapshots show mapping location; BM pool NOT_VERIFIED; working set=fixed addresses reused; hot buffers.

## Residual differences

- Target memory: BM-managed pool versus the TE allocator; address ownership is different.
- Registration: the compatibility BM wrapper has no explicit preregistration; TE registers explicitly (HIXL relay skips native Host registration). Internal BM registration cost is not established.
- Physical NUMA placement of the BM pool remains NOT_VERIFIED even with matching process policies.
- Backend thread roles, hardware offload and native/Python binding layers differ; matching submission count does not make them identical.
- Fixed hot buffers and single outstanding copy do not establish streaming or inference performance.
- BM route evidence: compat wrapper requests SDMA; physical HCCS traffic not quantified.
- TE route evidence: NOT_VERIFIED; requested ascend-roce; inspect HIXL logs/counters; allocator is not memfd-backed.

Registration, allocation and NUMA differences limit attribution. These transfer measurements do not establish inference speedup.
