# MemFabric BM L2 prototype

This example connects an independent LMCache MP service to a MemFabric DRAM
pool. The adapter lives in LMCache-Ascend and loads through LMCache's existing
Python L2 plugin interface; it requires no LMCache core changes.

```text
vLLM-Ascend + LMCacheMPConnector
    -> LMCache MP service / L1 CPU MemoryObj
    -> MemFabricL2Adapter / synchronous BM SDMA copy
    -> another BM rank's DRAM
```

The reverse path restores KV through L1 into the inference NPU. This is not
Memcache, and it is not direct NPU-to-pool zero-copy. MemFabric supplies memory
mapping and data movement; this adapter owns the volatile key index, fixed-slot
allocation, lookup pins, eviction, and LMCache task completion notifications.

## Verified configuration

On 2026-10-24, one A3 host with working internal HCCS:

- Python 3.12.13, CANN 9.1.0, driver 26.1.1.
- torch 2.10.0+cpu, torch-npu 2.10.0.post4.
- vLLM / vLLM-Ascend 0.23.0, Qwen3-8B BF16.
- LMCache commit `6913259264dbebb2fa5ba1665606c40ffc5c6661`.
- LMCache-Ascend base `c5e29f15902437883dba530c22705a48ff7ce0fb` plus these files.
- MemFabric **1.2.1**. The tested 1.2.0 ARM64 wheel aborted during BM teardown;
  the official [packaging fix](https://gitcode.com/Ascend/memfabric_hybrid/pull/1197)
  concerns duplicate native library loading.

The tested image also contained `mooncake-transfer-engine-npu==0.3.11.post1`.
Importing its native extension together with torch or gRPC reproduced a heap
error at process exit, without creating any BM pool. LMCache eagerly imports
available transfer-channel implementations. Removing this unused optional
package **inside the private experiment container** made MP import and CLI
shutdown checks exit successfully. It is not required for this BM adapter;
keep this environment-specific finding separate from MemFabric teardown.

Activate your prepared Ascend environment and Python environment in every
terminal. Install this checkout and the matching LMCache core before running.
MemFabric is an optional dependency: `pip install memfabric-hybrid==1.2.1`.
Check available disk space before downloads. Existing model files can be mounted
read-only; do not change their configuration or make unnecessary copies.

## Device and adapter checks

From this checkout:

```bash
python examples/memfabric/sdma_smoke.py --devices 0,2 --port 28572
python examples/memfabric/adapter_smoke.py
```

The first check overwrites each source after copying to peer DRAM and compares
separate destination tensors, for both CPU and NPU buffers. The second uses
real `MemoryObj` instances and checks locking, misses, layouts, duplicate stores,
deferred deletion, references, and draining on close. An expected locked-pool
store failure is part of the second check.

## Independent service and pool member

Run these in separate terminals in the **same container**. Both processes must
see the full device list. Start them within 90 seconds of each other.

```bash
# Terminal 1: contribute rank 1 DRAM using device 2.
python examples/memfabric/pool_member.py --device 2 --rank 1
```

```bash
# Terminal 2: rank 0 on device 0; store objects in rank 1 DRAM.
lmcache server --host 127.0.0.1 --port 25555 \
  --http-host 127.0.0.1 --http-port 28080 \
  --chunk-size 128 --l1-size-gb 4 --eviction-policy LRU \
  --supported-transfer-mode lmcache_driven --max-gpu-workers 2 \
  --prometheus-port 29090 \
  --l2-adapter "$(cat examples/memfabric/l2.json)"
```

Wait for `POOL_MEMBER_READY` and check `curl -f http://127.0.0.1:28080/healthcheck`.
`GET /status` includes the adapter's object count, transferred bytes, selected
storage rank, and last error. BM membership and the service index are separate:
a healthy HTTP server alone does not establish that every pool member joined.

## Two inference instances

Use the identical model path, dtype, and TP geometry for both engines. Replace
`/path/to/Qwen3-8B` with the existing read-only model path.

```bash
# Terminal 3: instance A. For instance B in terminal 4, use device 6 / port 28002.
ASCEND_RT_VISIBLE_DEVICES=4 vllm serve /path/to/Qwen3-8B \
  --served-model-name mf-test --tensor-parallel-size 1 --seed 1024 \
  --dtype bfloat16 --max-model-len 4096 --max-num-batched-tokens 1024 \
  --max-num-seqs 1 --gpu-memory-utilization 0.7 --block-size 128 \
  --enforce-eager --no-enable-prefix-caching --host 127.0.0.1 --port 28001 \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://127.0.0.1","lmcache.mp.port":25555,"lmcache.mp.mp_transfer_mode":"lmcache_driven"}}'
```

The `vllm` executable must use the Python environment containing this checkout;
activating a venv does not rewrite a console script inherited from the base
image. Check `command -v vllm` and its shebang.

With a fresh service and both engines healthy:

```bash
python examples/memfabric/inference_smoke.py --output /your/private/results
```

This explicitly clears **only that experimental service's L1**, after waiting
for the first request's L2 stores. It asserts empty L1, preserved BM objects,
positive BM load bytes, and matching generated answers on the second engine.
The measured run restored 1280 of 1373 prompt tokens from 180 MiB of BM data;
vLLM reported 1280 external-transfer tokens, 93 computed tokens, and zero local
prefix-cache tokens. This was a correctness run, not a performance benchmark.

## Boundaries

- Fixed membership on one host; one authoritative service and one selected
  storage rank. Each BM rank reserves `pool_gb` GiB, while effective object
  capacity is the selected rank's pool. Two 1 GiB ranks do not give this adapter
  2 GiB of usable object capacity.
- The metadata index is in-process and not persistent. Restarting the service
  loses the index. Restart all group members together; no membership recovery
  or cross-service shared metadata is implemented.
- Slots default to 32 MiB. One `MemoryObj` must fit a slot; locked slots cannot
  be evicted. This setting fits the tested Qwen3-8B TP1 / chunk 128 geometry.
- One I/O thread serializes BM operations. Lookups pin slots until LMCache
  unlocks them. Stores publish keys only after synchronous copies finish.
- Normal shutdown drains outstanding copies. Transfer failures poison the
  adapter and retain buffer references until BM drain; fault recovery is not
  validated. Stop inference engines, then the LMCache service, then any remaining
  pool member before stopping the container.
- SDMA is explicitly selected. No physical-link counter, cross-host HCCS,
  throughput benchmark, long-duration load, or independent multi-service
  concurrency claim is made by these checks.

For actual P/D request routing, the pinned LMCache tree contains
`examples/disagg_prefill_mp/disagg_proxy_server.py`. Its store-finished signal
does not by itself prove L2 persistence: default policy can serve D from L1.
Keep P/D routing evidence separate from the L1-clear test above.
