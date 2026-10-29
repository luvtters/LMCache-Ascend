# Worker direct lifecycle candidate: selected validation

This candidate keeps an explicitly rejected lookup as an error. Rechecking it
raises the original exception and never converts it to a zero-token miss.
Cleanup of an observed rejection does not decode the same reply or release
nonexistent pins again. Transport failures retain their unresolved state.

The direct service depends on the built-in LMCache-driven transfer module.
Before BM wait/leave/uninitialization, it stops and drains the built-in device
callback dispatcher. The built-in module retains its entries until its later
close; repeated quiescence does not drain a runtime already torn down.
A dispatcher that remains alive after join raises an error and retains resources.

Use the paired core candidate `8677f65` (CPU-check followup `5b565d2`), based on
`8bc9edc`. The Ascend changes are `5e8cc07` and `760d6b8`, based on the DEVICE-pool
candidate `8b0c7e1`. Original baselines remain separate.

CPU checks: 22 worker/scheduler checks, 8 service checks, and 4 focused core
callback shutdown checks passed. New failure checks reproduced the previous
behavior before the fixes. These are CPU evidence only.

Recorded real-NPU validation on 2026-10-30:

- Three fresh fixed-world idle-leave groups explicitly rejected valid store and
  lookup after an idle peer left. No BM copy was attempted; pins and writes
  remained zero. All six worker processes exited naturally with code zero.
- A normal TP1 context regression restored an 18 MiB, 36-layer, 128-token KV
  object from noncontiguous source pages into different target pages. Complete
  content and untouched-page checks passed; both workers exited zero.
- Split-host P/D TP2 tests covered a remote write/local read and a local
  write/remote read through a DEVICE-selected pool. A split-host TP4 group
  covered remote writes. All 16 workers exited zero, all shard content and
  missing-shard checks passed, and final pins/writes/quarantine were zero.
- Each service received only its planned interrupt and exited 130. Raw logs
  showed the dispatcher thread stopped before BM group leave/uninitialization;
  shutdown had no thread errors or timeouts. Coordinators were explicitly
  stopped, every real wait was recorded, and allocated devices were released.
- A separate strong-context question campaign used the same paired production
  code with Qwen3-8B at TP1 and TP4. Both runs had 1,364 actual prompt tokens.
  The cold D reference computed all 1,364 tokens, used no external KV and issued
  no BM copy. The cached D run used 1,280 external tokens plus 84 computed tokens,
  with zero local-prefix tokens, and returned the same hidden-code answer as
  the cold reference. Per-rank completion traces totaled 180 MiB in each
  direction; TP4 required all four shard receipts. P/D exited zero and the
  service exited 130 after only its planned interrupt, with the dispatcher
  stopped before BM teardown and no shutdown errors. The retrieved archive's
  selected gate summaries, cold references, waits and shutdown logs were
  checked before recording these results. Gate tooling is a separate harness.

Scope: worker-direct services had empty ordinary built-in GPU registrations.
This does not establish general shutdown safety with nonempty built-in contexts
and outstanding device streams. Earlier service SIGSEGV and lock-error samples
are retained; these results do not uniquely identify their native root cause.
The model results are functional checks; there is no throughput, hardware-link
byte, long-duration stability, or real-model TP8 claim. DEVICE selection still
contributes both DRAM and HBM;
the earlier remote HOST/DRAM failure remains unresolved.

Internal raw logs and host identifiers are intentionally kept outside this
source branch. The default probe remains plan-only; actual device use requires
explicit authorization and an external finite executor with real-wait cleanup.
