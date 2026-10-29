# Experimental HBM pool for the compatibility L2 adapter

This branch keeps the existing CPU MemoryObj interface and the LMCache L1
staging path. It changes the selected BM pool, not the worker connector or the
CPU H2G/G2H copy directions. The default remains `pool_memory_type="host"`.

For the candidate, set `pool_memory_type="device"` in the adapter parameters
or the `cross_host_20261030/l2_config.py` JSON input, and pass
`--pool-memory-type device` to the matching `pool_member.py`. Both ranks must
use the same value, pool capacity and fresh BM group. There is no fallback
from DEVICE to HOST after a failure.

The device configuration keeps a DRAM segment and adds an equally sized HBM
segment on each rank, matching the isolated raw DEVICE-pool probe. Only the
DEVICE GVA is selected for L2 payloads. Thus a 1 GiB `pool_gb` reserves 1 GiB
DRAM plus 1 GiB HBM per rank; it does not mean a pure HBM-only allocation.
The adapter's usable object capacity is the selected pool's 1 GiB.

On initialization BMSession checks the selected pool's actual local capacity
with `local_mem_size(HOST|DEVICE)`, selects that type in `peer_rank_ptr`, and
waits for the peer's LOCAL_DEVICE mapping. Status exposes the selected native
memory type, the SDK-read local capacity, and the selected storage GVA.
`actual_export_seg_type` remains explicitly unverified until the native raw
ExportInner/Import records are inspected. The HBM segment tag is VRAM;
the existing residency tag and CPU-L1 target selection are unchanged.

## Evidence boundary, 2026-10-30

An isolated 1 MiB NPU source -> remote DEVICE pool -> owner NPU readback passed
full independent content verification and both real process waits. A fresh
HOST-pool group on the same endpoints failed with SDMA COMPERR and zero
completed bytes; a same-host DEVICE positive control passed. These are raw
BM medium diagnostics, not validation of this L2 candidate.

The CPU H2G/G2H path to a remote DEVICE pool, its possible SDK-local staging,
an actual L2 store/load cycle, and model P/D are all **NOT_VERIFIED** here.
The prior remote DRAM failure remains a failure. No performance conclusion or
absence of SDK-internal bounce buffers is implied.

CPU checks:

```text
python examples/memfabric/check_l2_hbm_cpu.py
python examples/memfabric/cross_host_20261030/l2_cpu_checks.py
```

The first uses stub SDK calls to check native option selection, real wrapper
control flow, HOST/DEVICE address choice, unchanged CPU directions, config
propagation and status. It does not import an accelerator SDK. The second is
the existing configuration, process-wait and L2-source-audit check.

Before any device validation, obtain a current allocation and run the 18 MiB
CPU MemoryObj L2 smoke in a fresh group. Require complete bytes, independent
readback, selected-media native logs, both actual waits and released devices
before considering model P/D. Keep a failed candidate in its own result
directory rather than reusing an old successful group's receipt.
