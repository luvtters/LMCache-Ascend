# Raw SDMA DRAM diagnostic: selected results, 2026-10-31

Scope: two physical A3 servers in the same SuperPod, MemFabric 1.2.1
(`106f783df905fe8d8e40446375faf615cf9b74fb`). This is a functional diagnostic,
not an LMCache or model benchmark. Private machine identifiers and raw internal
logs are kept outside this repository.

The 60 SDMA cases consisted of:

| Endpoint and placement | Cases | Result |
| --- | ---: | --- |
| HOST, same physical server | 8 | Full content and real process exits passed |
| DEVICE, different physical servers | 8 | Full content and real process exits passed |
| HOST, different physical servers | 44 | Failed; native write CQE9 / read CQE10 |

Controls covered reverse direction, 64-byte / 4-KiB / 1-MiB payloads,
CPU-buffer APIs including 18 MiB, 56-bit GVA, allocation flag 512,
even/odd device pairings, and explicit requested DRAM NUMA nodes 0/4.
Requested NUMA parameters were confirmed in SDK inputs; actual page placement
was not established. DEVICE cases also declare DRAM capacity and are not
described as pure-HBM allocation.

The size, CPU-buffer, and NUMA probe revisions were respectively `61e15b6`,
`682c8cd`, and `fe882113`. The final narrow CPU checks passed 15 tests.
Failure results, native returns, full-content checks and real process waits
were retained separately. An owner's successful local initialization in a
failed remote-read case is not a successful cross-host transfer.

This does not establish the unique root cause of HOST SDMA failure. A separate
registered-memory DEVICE_RDMA diagnostic subsequently established another
working cross-host HOST route; that result does not repair this SDMA path.
No driver, firmware, SDK, production adapter, model, or compatibility baseline
was changed by these diagnostic additions.
