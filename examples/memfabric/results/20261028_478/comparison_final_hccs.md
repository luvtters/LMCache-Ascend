# BM / TE transfer results

Same-host CPU DRAM: BM SDMA versus TE Ascend Direct, HCCS relay requested. This is a path comparison; it does not isolate identical HCCS links or pure backend overhead.

Warmup=20, iterations=100, repeats=3; batch=1, outstanding=1, Python submission threads/process=1.

| MiB | Direction | Backend | n | Mean us | p50 us | p95 us | Effective GB/s | BM/TE mean time |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | write | BM SDMA | 300 | 215.503 | 213.651 | 241.357 | 4.866 | 0.805 |
| 1 | write | TE Ascend Direct, HCCS relay requested | 300 | 267.718 | 266.177 | 279.648 | 3.917 | 0.805 |
| 1 | read | BM SDMA | 300 | 228.842 | 226.392 | 245.828 | 4.582 | 0.855 |
| 1 | read | TE Ascend Direct, HCCS relay requested | 300 | 267.566 | 266.612 | 278.380 | 3.919 | 0.855 |
| 18 | write | BM SDMA | 300 | 1330.361 | 1328.265 | 1353.066 | 14.187 | 1.326 |
| 18 | write | TE Ascend Direct, HCCS relay requested | 300 | 1003.176 | 1002.342 | 1026.982 | 18.815 | 1.326 |
| 18 | read | BM SDMA | 300 | 1280.134 | 1278.409 | 1300.857 | 14.744 | 1.267 |
| 18 | read | TE Ascend Direct, HCCS relay requested | 300 | 1010.585 | 1000.423 | 1073.937 | 18.677 | 1.267 |
| 64 | write | BM SDMA | 300 | 3239.429 | 3158.948 | 3254.162 | 20.716 | 1.230 |
| 64 | write | TE Ascend Direct, HCCS relay requested | 300 | 2632.860 | 2618.148 | 2717.048 | 25.489 | 1.230 |
| 64 | read | BM SDMA | 300 | 3654.315 | 3514.959 | 3638.473 | 18.364 | 1.436 |
| 64 | read | TE Ascend Direct, HCCS relay requested | 300 | 2544.408 | 2420.297 | 2842.614 | 26.375 | 1.436 |

Statistics pool all individual completed-copy samples across repeats. Effective GB/s = total successful payload bytes / total synchronous API time. Per-repeat bandwidth and percentiles are not averaged. BM/TE mean time < 1 means BM has lower mean time.

- BM: registration=compat BMSession.copy, no explicit preregistration; allocator=posix_memalign(65536), ordinary pages, first-touch memset; remote allocator=BM private DRAM pool (1 GiB per rank); NUMA=worker resource snapshots show mapping location; BM pool NOT_VERIFIED; working set=fixed addresses reused; hot buffers.
- TE: registration=TE metadata registration; HIXL skips Host registration in relay mode; allocator=aclrtMallocHost, pinned Host, first-touch memset; remote allocator=aclrtMallocHost, pinned Host, first-touch memset; NUMA=worker resource snapshots show mapping location; BM pool NOT_VERIFIED; working set=fixed addresses reused; hot buffers.

## Residual differences

- Target memory: BM-managed pool versus the TE allocator; address ownership is different.
- Registration: the compatibility BM wrapper has no explicit preregistration; TE registers explicitly (HIXL relay skips native Host registration). Internal BM registration cost is not established.
- Physical NUMA placement of the BM pool remains NOT_VERIFIED even with matching process policies.
- Backend thread roles, hardware offload and native/Python binding layers differ; matching submission count does not make them identical.
- Fixed hot buffers and single outstanding copy do not establish streaming or inference performance.
- BM route evidence: compat wrapper requests SDMA; physical HCCS traffic not quantified.
- TE route evidence: NOT_VERIFIED; requested ascend-hccs-relay; inspect HIXL logs/counters; allocator is not memfd-backed.

Registration, allocation and NUMA differences limit attribution. These transfer measurements do not establish inference speedup.
