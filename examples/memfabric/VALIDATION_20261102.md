# Single-host L2 validation, 2026-11-02

This is the compatibility L2 route: model NPU KV passes through the ordinary LMCache CPU L1 before MemFabric HOST-pool storage. The core checkout was the unchanged official `6913259`; MemFabric Hybrid was 1.2.1. Existing native extensions were reused; their exact build commit has not been established. This record is functional validation, not a performance comparison or cross-host acceptance.

## Observed coverage

- Qwen3-8B TP1, SDMA and DEVICE_RDMA: strict P/D handoff and seven supplemental cases passed. The strict request contained 1373 tokens; D restored 1280 and computed 93, with zero local prefix hits. The service recorded 0 L1 / 10 L2 retained objects and 180 MiB in each transfer direction.
- Supplemental TP1 cases covered repeat, partial prefix, short cold input, same-key concurrent clients, multi-chunk cold write, reuse of that write, and concurrent different prefixes. Same-key concurrency can fall back to recomputation when L1 reservation reports KEY_NOT_WRITABLE; responses alone do not prove every request read the pool.
- SDMA TP2 and TP4, chunk128: per-key completion logs proved STORE/LOAD for every rank and chunk. TP2 had 20 objects, 90 MiB per rank; TP4 had 40 objects, 45 MiB per rank. Aggregate payload was 180 MiB in both directions, with 1280 external / 93 computed tokens and answer ORCHID.
- The real MemoryObj boundary gate passed 14 checks per protocol: locked capacity rejection, continued work after rejection, missing keys, size mismatch, deferred deletion, rewrite, content including tail bytes, eviction after unlock, and close behavior. The accompanying store-completion fix consumes a failed future before rethrowing it, preserving successful results for the next poll.
- A separate DEVICE_RDMA test completed a 9 MiB HostMemoryAllocator object round trip and all three roles exited 0. SDMA peer departure invalidated the adapter as expected; the RDMA peer-departure attempt did not reach readiness.

## Remaining limits

TP2 model DEVICE_RDMA failed while registering CPU L1 objects, before payload copy, for both 9 MiB and 18 MiB objects. The SDK reported native ret=8 / public -1. The standalone 9 MiB success rules out treating object size alone as the explanation. Model L1 backing and BM/model device placement still need controlled isolation.

TP8 did not reach model startup: service readiness failed. No TP8 KV or performance result is claimed. Larger TP and chunk values in the gate are validation configurations, not an assertion that every combination is supported.

The original environment produced P/D teardown SIGABRT. A separate private Python dependency view excluding optional Mooncake made the tested TP1 P/D processes exit 0 for both protocols; the original installation was unchanged. Services may return 130 after planned SIGINT with complete close logs. P/D parent exit 0 does not imply every internal worker exited gracefully: TP4 shutdown included worker signal escalation. SDK `free sq_cq -8` warnings remain separately recorded.

The model gate is controlled: local prefix caching is off and native skip_l1 is used. It verifies pool provenance and full shard coverage; it does not measure natural-cache speedup. Full model capacity, long-duration stress, cancellation/timeout combinations, and production batching remain unaccepted.

## Reusable entry points

`l2_model_gate.py` accepts matching TP device lists and a `chunk_size` that is a multiple of vLLM block128. It waits for all P rank callbacks and checks original per-object completion logs before accepting multi-rank restoration.

`l2_memoryobj_boundary_gate.py` is plan-only until `--allow-npu` is supplied. It needs a matching pool member and coordination group, fresh output/group names, and explicit device/protocol parameters. It does not launch or clean up those companion processes.

Original role logs, requests, metrics queries, actual waits, failed attempts, and resource-release records are retained privately. Internal logs, models, SDK binaries, and credentials are not included in this repository.
