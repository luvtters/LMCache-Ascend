# Optional L1 arena registration for DEVICE_RDMA L2

This remains a CPU L1 to remote MemFabric HOST DRAM L2 adapter. Workers and
the NPU-to-L1 connector do not change. `register_l1_arena` defaults to `false`.

The measured per-object register/copy/unregister path pays host registration
on every transfer. Set `register_l1_arena: true` only with the accompanying
core descriptor addition `L1MemoryDesc.stable_host_registration_size`.
That field describes allocator-owned mapped bounds and lifetime, not a claim
that every provider can register them. The original io_uring-specific stable
size and its POSIX SHM restriction remain unchanged.

The adapter configures the exact stable prefix on its single BM owner thread.
The first covered MemoryObj registers that prefix once; later covered copies
hold their ordinary allocator lease and reuse the registration. Fully separate
fallback objects retain per-object registration. A partially overlapping
object fails explicitly. A lazy allocator's growing tail is not silently
included in the original snapshot.

Normal close drains queued work, unregisters the arena, closes BM, and only
then allows core to close L1. Register/copy/unregister failure poisons the
session, retains uncertain object leases, rejects further mutation, and
propagates close failure. No error is translated into a cache miss.

Registration is deferred until first real I/O, so its cost must be reported
as first-use setup rather than silently removed from request latency. Runtime
validation of the actual SHM backing, cross-host content, KV source, bytes,
process exits, and performance is required separately from CPU contract tests.
