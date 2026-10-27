# BM / TE transfer results

Same-host CPU DRAM: BM SDMA versus TE SHM. This is a path comparison; it does not isolate identical HCCS links or pure backend overhead.

Warmup=20, iterations=100, repeats=3; batch=1, outstanding=1, Python submission threads/process=1.

| MiB | Direction | Backend | n | Mean us | p50 us | p95 us | Effective GB/s | BM/TE mean time |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | write | BM SDMA | 300 | 215.503 | 213.651 | 241.357 | 4.866 | 3.777 |
| 1 | write | TE SHM | 300 | 57.059 | 53.340 | 67.065 | 18.377 | 3.777 |
| 1 | read | BM SDMA | 300 | 228.842 | 226.392 | 245.828 | 4.582 | 4.005 |
| 1 | read | TE SHM | 300 | 57.145 | 53.761 | 71.915 | 18.349 | 4.005 |
| 18 | write | BM SDMA | 300 | 1330.361 | 1328.265 | 1353.066 | 14.187 | 1.246 |
| 18 | write | TE SHM | 300 | 1068.033 | 1020.528 | 1361.382 | 17.672 | 1.246 |
| 18 | read | BM SDMA | 300 | 1280.134 | 1278.409 | 1300.857 | 14.744 | 1.205 |
| 18 | read | TE SHM | 300 | 1062.502 | 1007.102 | 1397.803 | 17.764 | 1.205 |
| 64 | write | BM SDMA | 300 | 3239.429 | 3158.948 | 3254.162 | 20.716 | 0.879 |
| 64 | write | TE SHM | 300 | 3684.022 | 3614.110 | 4066.183 | 18.216 | 0.879 |
| 64 | read | BM SDMA | 300 | 3654.315 | 3514.959 | 3638.473 | 18.364 | 0.876 |
| 64 | read | TE SHM | 300 | 4173.681 | 3953.678 | 5113.981 | 16.079 | 0.876 |

Statistics pool all individual completed-copy samples across repeats. Effective GB/s = total successful payload bytes / total synchronous API time. Per-repeat bandwidth and percentiles are not averaged. BM/TE mean time < 1 means BM has lower mean time.

- BM: registration=compat BMSession.copy, no explicit preregistration; allocator=posix_memalign(65536), ordinary pages, first-touch memset; remote allocator=BM private DRAM pool (1 GiB per rank); NUMA=worker resource snapshots show mapping location; BM pool NOT_VERIFIED; working set=fixed addresses reused; hot buffers.
- TE: registration=explicit TE register_memory; allocator=TE allocateSharedMemory, POSIX MAP_SHARED, first-touch memset; remote allocator=TE allocateSharedMemory, POSIX MAP_SHARED, first-touch memset; NUMA=worker resource snapshots show mapping location; BM pool NOT_VERIFIED; working set=fixed addresses reused; hot buffers.

## Residual differences

- Target memory: BM-managed pool versus the TE allocator; address ownership is different.
- Registration: the compatibility BM wrapper has no explicit preregistration; TE registers explicitly (HIXL relay skips native Host registration). Internal BM registration cost is not established.
- Physical NUMA placement of the BM pool remains NOT_VERIFIED even with matching process policies.
- Backend thread roles, hardware offload and native/Python binding layers differ; matching submission count does not make them identical.
- Fixed hot buffers and single outstanding copy do not establish streaming or inference performance.
- BM route evidence: compat wrapper requests SDMA; physical HCCS traffic not quantified.
- TE route evidence: SHM-only native bridge, CPU memcpy.

Registration, allocation and NUMA differences limit attribution. These transfer measurements do not establish inference speedup.
