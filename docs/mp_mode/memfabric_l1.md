# Native MemFabric L1 MP connector

This connector keeps the MP scheduler and cache service in LMCache core. The
Ascend subclass only creates the worker adapter; workers gather KV pages into
registered NPU scratch tensors and BM copies those tensors to or from the
service-owned pool. Native `L1Manager` owns allocation and cache lifecycle. This
path does not configure an LMCache L2 adapter.

## Start one fixed BM group

Use a unique `session_id` and the same BM group settings in the service and both
model processes. For a one-node TP1 P/D pair, use three BM ranks: service 0, P
1, D 2. `device_id` is local to each process's visible NPU set.
Assign a distinct permitted physical device to each role with
`ASCEND_RT_VISIBLE_DEVICES` before starting it; the examples below use logical
`device_id: 0` inside each single-device process. The coordinator needs no NPU.

Start the small CPU JOIN coordinator first:

```bash
python examples/memfabric/bm_join_coordinator.py --host 127.0.0.1 --port 36781
```

Start the service with rank 0. The runtime factory receives the session and BM
configuration through `--runtime-plugin-config`:

```bash
lmcache server --host 127.0.0.1 --port 36555 --http-host 127.0.0.1 \
  --http-port 38081 --chunk-size 128 --l1-size-gb 1 \
  --supported-transfer-mode engine_driven \
  --runtime-plugin-config '{"l1_runtime":{"factory":"lmcache_ascend.v1.distributed.memfabric_l1_runtime:create_runtime","config":{"session_id":"demo-session","bm":{"device_id":0,"rank_id":0,"world_size":3,"pool_gb":1,"store_url":"tcp://127.0.0.1:36780","pool_memory_type":"host","data_op_type":"SDMA","join_coord_url":"http://127.0.0.1:36781","join_group_id":"demo-session"}}}}'
```

Each vLLM process uses this connector config. P sets `kv_role` to
`kv_producer` and BM `rank_id` to 1; D sets `kv_role` to `kv_consumer` and rank
2. Keep the other BM fields and `session_id` identical to the service:

```json
{
  "kv_connector": "MemFabricL1MPConnector",
  "kv_connector_module_path": "lmcache_ascend.integration.vllm.memfabric_l1_connector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "lmcache.mp.host": "127.0.0.1",
    "lmcache.mp.port": 36555,
    "lmcache.mp.mp_transfer_mode": "engine_driven",
    "memfabric_l1": {
      "session_id": "demo-session",
      "bm": {
        "device_id": 0,
        "rank_id": 1,
        "world_size": 3,
        "pool_gb": 1,
        "store_url": "tcp://127.0.0.1:36780",
        "pool_memory_type": "host",
        "data_op_type": "SDMA",
        "join_coord_url": "http://127.0.0.1:36781",
        "join_group_id": "demo-session"
      }
    }
  }
}
```

Pass this JSON as vLLM's `--kv-transfer-config`. Start the coordinator, service,
P, and D as one group without waiting for the service RPC before starting the
workers. Each rank must finish BM's after-mmap JOIN barrier before workers make
their first service RPC; waiting for the RPC before starting P/D deadlocks the
fixed group.

For a cold model start, set `init_timeout` explicitly in all three BM configs
to cover model loading (the validation used 600 seconds). Model workers join
after loading their KV layout; a short service-side join timeout can expire
before either worker arrives.

## Scope and failure handling

- Current support is TP1, one full-attention KV group. Hybrid/windowed groups
  and multi-group layouts fail closed.
- The native transfer gathers or scatters paged KV through registered NPU
  scratch tensors; this is not a zero-kernel scatter path.
- The service runtime rejects configured L2 adapters. Cache objects live in
  native L1 backed by the BM HOST pool.
- If BM completion is unknown, the worker retains registered buffers and pins
  its session. Stop and recover the whole fixed group after checking the
  original completion and shutdown results; do not reuse a partially failed
  group or session.

## Validation snapshot (2026-11-02)

The paired core implementation is `8285a3a`. On one A3 host with Qwen3-8B,
TP1 per model, bf16, 128-token chunks, a 1 GiB HOST L1 pool and SDK 1.2.1:

- The real service/worker gate restored all 72 synthetic KV tensors (36 layers)
  after an 18 MiB write and read with both SDMA and DEVICE_RDMA.
- The SDMA model gate stored and restored 180 MiB. D reported 1,280 external
  tokens, 93 computed tokens and zero local prefix-cache tokens for a
  1,373-token request. The service reported 10 L1 hits and zero L2 hits.
- With local prefix caching disabled, repeat, partial-prefix, cold and two
  concurrently submitted requests passed. The partial request restored 1,024
  of 1,382 input tokens; the cold request computed all 1,390 tokens with no BM
  read. Each concurrent request restored its own 180 MiB.
- After the model basic suite, native read/write locks, I/O pins and staging
  were zero. Service, P, D and the CPU coordinator all exited 0.

These are functional observations, not a performance comparison. SDK
`free sq_cq -8` warnings were still observed during the DEVICE_RDMA gate's
cleanup. An earlier coordinator required SIGTERM because it did not respond
to SIGINT; its explicit SIGINT handler was subsequently verified with a
CPU-only inherited-signal test and the model basic suite. Preserve failure
and teardown evidence separately from completed payload transfers.

Long-running operation-token retention, late release of expired native
prefetch reservations, multi-TP layouts and cross-host operation of this
native-L1 integration still need separate work and validation.
