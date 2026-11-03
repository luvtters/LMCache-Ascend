# Optional L1 arena registration for DEVICE_RDMA L2

This remains a CPU L1 to remote MemFabric HOST DRAM L2 adapter. Workers and
the NPU-to-L1 connector do not change. `register_l1_arena` defaults to `false`.

The per-object register/copy/unregister path pays host registration on every
transfer. `register_l1_arena: true` uses the public optional
`L1MemoryDesc.stable_registration_size` exposed by official LMCache core
`6913259264dbebb2fa5ba1665606c40ffc5c6661`; this adapter requires no core
change. The field is the stable prefix length, while `L1MemoryDesc.size` is the
final capacity. If the descriptor or stable length is missing or the length is
`None`, the adapter keeps the per-object path. It never substitutes `size` or
reads private allocator state to infer a stable prefix.

With the default Lazy allocator, initialization and final capacity are both
2 GiB; the exposed stable prefix has been registered and used successfully.
For fixed POSIX SHM, the stable length is `None`, so registration remains
per-object; that path has passed. Fixed anonymous backing exposed a stable
length, but both per-object and arena registration failed in the tested
environment, so that profile is not qualified. These outcomes do not imply
that every backing or provider can register an arena.

The adapter configures the exposed stable prefix on its single BM owner thread.

Real Lazy growth from 64 MiB to 256 MiB has also been checked through the
official L1 manager API. The adapter kept its original 64 MiB snapshot and
copied an object wholly outside it through per-object registration. The
successful deployment used pin context logical 0 and BM logical 1; the reverse
combination failed registration on the expanded memory. This is a deployment
observation, not a provider root-cause determination or a model-service growth
qualification. Objects straddling the original prefix still fail explicitly.
Use a fully committed L1 prefix for the qualified inference arena profile.
The first covered MemoryObj registers that prefix; later covered copies hold
their ordinary allocator lease and reuse the registration. With
`l1_registration_chunk_mb: 32`, it registers touched 32 MiB ranges, clipping
the final range to the stable prefix. A copy spanning multiple MRs is split at
their boundaries with continuous pool offsets. Objects wholly outside the
stable prefix use per-object registration. An object crossing the stable
prefix boundary is currently rejected. A Lazy allocator's later growth is not
silently added to the captured prefix.

Registration is deferred until first real I/O, so its cost must be reported
as first-use setup rather than silently removed from request latency. Runtime
content, KV source, byte totals, process exits, and performance must be
reported for the specific backing and attempt; CPU contract tests do not
establish those results.

`l1_registration_chunk_mb: 32` optionally registers only touched fixed 32 MiB
regions within the stable prefix. Its default `0` keeps whole-prefix behavior.
Distinct regions remain registered until normal close; the adapter does not
assume that a provider combines them into one MR. The MemoryObj lease remains
held through its copy and is released only after successful completion. If
registration, copy, or completion is uncertain, the session is poisoned and
the object lease and established regions are retained; further mutation is
rejected. Normal close drains queued work, unregisters the arena, closes BM,
and only then allows core to close L1. No error is translated into a cache
miss.
