# HOST/DRAM cross-host diagnosis

These diagnostics keep CPU control tests separate from actual NPU/BM payload
validation. They do not change the production L2 adapter or worker transport.

## CPU-only tests

See [the CPU control probe](README_cpu_control.md). It uses the existing
standard-library `direct_coordination` module, with no accelerator imports.

In the 2026-10-30 run, two actual Linux hosts each completed 19 positive and
negative checks. Independent offline inspection matched all 54 outgoing HTTP
transactions to their incoming server records, including exact request/response
bytes, expected 201/200/409 and old-attempt rejections. Each endpoint echoed
60,873 bytes of metadata intact. Reverse callbacks, completion/failure events
and missing-event timeouts were verified against raw records.

The containers used runc, UID 65534, no devices or mounts, no added capabilities
and no accelerator-module imports. Client execs returned 0; listeners handled
planned SIGTERM and returned 0, with PID 0 and released ports afterward.
Distinct kernel boot IDs prevented ordinary same-host containers from passing
the two-host check merely because their hostnames differed. Private addresses,
raw logs, boot IDs and credentials are not included in this selected record.

This is CPU TCP/HTTP metadata evidence. It is not native LMCache RPC, BM,
SDMA, HCCS, HOST-pool payload or inference acceptance.

## Raw device probe prepared for the next authorized window

`../bm_media_probe.py` retains the independently tested push and owner-local
initialization/remote-pull directions from `510a761`. The default is a CPU
plan; importing device SDKs and doing transfers requires `--allow-npu`.

An explicit `--dram-map-host-va` option selects the fixed SDK 1.2.1 allocation
flag `SMEM_BM_FLAG_DRAM_MAP_HOST_VA` (1 << 9). It is allowed only with HOST and
SDMA. It changes `create2.flags` to 512; data-copy flags stay 0. Both ranks
must agree on the full create options before SDK initialization.

In SDK revision `106f783`, this flag affects the combination of sliced DRAM
allocation and local `MapMemBySetAccess`, rather than the default allocation
and `MapMemByImport`. It is not a documented prerequisite switch or a proven
fix. It has not been exercised on devices in this work. Production defaults,
SDK binaries and system settings are unchanged.

Eight focused CPU checks cover planning, both-rank contracts, explicit SDK
option forwarding, rejected non-HOST/non-SDMA combinations, payload pattern,
mapping lookup and preserved cleanup failures. No CPU check proves that an
imported mapping can actually be reached by remote SDMA.

Next device work should use one fixed physical endpoint pair: default HOST
push; owner-initialized HOST pull; DEVICE control; then independent HOST groups
with the alternative allocation flag. Retain complete content, native CQE/SQE,
actual process waits and cleanup evidence for every group. Failed groups are
never overwritten or relabeled when a later group passes.

## Current failure boundary

Archived HOST push and pull both reached native SDMA task completion errors
(CQE 9 and 10 respectively), after group join and nonzero mapped addresses.
An owner-local HOST initialization succeeded before remote pull failed.
Earlier DEVICE transfers succeeded. These observations locate the failure
below LMCache lookup/RPC, but do not identify a unique SDK, driver, firmware or
fabric root cause. The earlier push and pull used different physical endpoint
pairs, so they are not a strict same-pair read/write comparison.

Installed host/driver/SDK metadata alone cannot establish the required fabric
switch firmware. Querying that version is CPU-only but requires a suitable
read-only management interface. Actual remote mapping access, SDMA payload,
the alternative allocation mode and model recovery need a new device window.
