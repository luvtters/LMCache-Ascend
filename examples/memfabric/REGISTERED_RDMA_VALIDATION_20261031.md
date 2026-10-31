# Registered RDMA validation checkpoint (2026-10-31)

This is a sanitized result record, not a collection of internal runtime logs.
Runtime sources were Ascend `a9d30974` for the first TP1 campaign and
`25ceb46e` for the JOIN/close fixes, with core `5b565d2`. Documentation commits
after those revisions do not change the executed source identity.

## Observed functionality

- Real Qwen3-8B TP1: three two-host placements passed the strict P-to-pool-to-D
  request, including P/D on separate hosts and either host owning the DRAM pool.
- Real Qwen3-8B TP2 and TP4: split-host P/D passed with five and nine BM ranks.
  The new all-rank after-mmap JOIN acknowledgements were observed.
- Every strict request had 1373 input tokens, 1280 restored tokens, 93 locally
  computed tokens and zero local-prefix tokens; the response was ORCHID.
  Each leg completed 180 MiB in the traced synchronous call path. TP2 had
  20 objects and TP4 had 40; these are functional checks, not throughput tests.
- Actual HostMemoryAllocator MemoryObj gates passed full A/B content checks
  for forward/reverse 18 MiB and forward 32 MiB, including source destruction,
  restored content, allocator checks and zero final pins.

The SDK pool was explicitly HOST DRAM, with zero declared local HBM. The worker
route explicitly registered real NPU scratch and used DEVICE_RDMA L2G/G2L.
It did not substitute a CPU object or transport the KV payload over RPC.

## Unresolved failures and boundaries

- Real L2 inference is still blocked: after NPU IPC/model startup, registering
  an aligned 18 MiB object from the service's shared, pinned L1 arena returned
  HAL ret8 / public -1. No L2 completed payload bytes were observed in that
  attempt. Passing standalone MemoryObj gates does not resolve this case.
- Lifecycle is not clean. Model services needed a planned SIGTERM escalation;
  P/D retained internal forced-process/resource-tracker diagnostics. One L2
  pool member exited with SIGSEGV after reporting close. Preserve actual waits
  and signals separately from payload success.
- Native unregister/free-sq-cq warnings remain. Public zero returns and zero
  process exits do not establish warning-free teardown or absence of leaks.
- This does not fix the separately failing default SDMA cross-host DRAM path.
- Production native batch submission, heterogeneous TP, TP8, long-duration
  reliability, capacity eviction and complete failure-recovery coverage remain
  separate work. API concurrency is not proof of simultaneous device DMA.

Internal raw requests, source metrics, native logs, per-rank traces, actual waits
and resource-release receipts are archived privately, rather than published here.
