# Four-slot L2 API boundary probe

Probe code: `ea5e9e3`; production candidates: LMCache core `fa2e6ff` and Ascend
adapter `137038a`. The CLI defaults to plan-only; actual device execution needs
both `--run` and `--allow-npu`, plus an authorized external supervisor.

## Selected validation, 2026-10-30

The exact matrix passed first with real CPU MemoryObjs/eventfds and fake BM,
then in two real-NPU groups: same-host HOST/DRAM and host-A to host-B DEVICE/HBM.
Each selected pool was 1 GiB with four 256 MiB slots; each object was 1 MiB.
HBM participants also declared 1 GiB DRAM per rank and selected DEVICE for data.

The matrix checks cold miss; fill four; pin key0 and reject duplicate overwrite;
evict unlocked key1; read key0; fail a new store with every slot pinned while
remaining healthy; hide a pinned deleted key without reclaiming its slot;
reclaim on unlock; store again; fully compare the remaining objects; release
all pins/delete all objects; close twice; explicitly reject a closed submission.

Both real groups completed 11 BM calls: 6 MiB written and 5 MiB read with five
full-content comparisons and zero mismatches. Final objects/pins were zero,
free slots four, and health true. All four participant processes exited
naturally with real wait code zero. Independent guards were cancelled after
those waits; subsequent device and port checks confirmed release.

Native logs selected SDMA and exported/imported the expected DRAM/HBM segments.
This validates the compatible L2 CPU-MemoryObj path, including possible local
SDK staging. Separate cross-host HOST SDMA failures remain unresolved. These
API passes do not establish model recovery, link bandwidth, or service speedup.
Raw native logs and internal deployment details remain private.
