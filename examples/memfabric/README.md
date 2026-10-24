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
storage rank, eviction count, outstanding pins, and last error. BM membership and the service index are separate:
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
- Startup waits for every peer DRAM segment to have a device VA mapping.
  A nonzero `peer_rank_ptr` alone does not establish this. JOIN callbacks do not
  replay every earlier member, so counting callbacks is insufficient.
- Normal shutdown drains outstanding copies. Transfer failures poison the
  adapter and retain buffer references until BM drain. Peer LEAVE notifications
  also poison the adapter: new lookups miss, loads stop, and stores fail.
  Idle pool-member SIGINT and SIGKILL were tested with correct inference fallback
  and zero new BM reads. Automatic recovery and failure during DMA are untested.
  Stop inference engines, then the LMCache service, then any remaining pool
  member before stopping the container.
- SDMA is explicitly selected. No physical-link counter, cross-host HCCS,
  throughput benchmark, long-duration load, or independent multi-service
  concurrency claim is made by these checks.

For actual P/D request routing, the pinned LMCache tree contains
`examples/disagg_prefill_mp/disagg_proxy_server.py`. Its store-finished signal
does not by itself prove L2 persistence: default policy can serve D from L1.
Keep P/D routing evidence separate from the L1-clear test above.

## One P/D request through MemFabric

The additional `pd_inference_smoke.py` check sends **one inference request to
the unchanged upstream P/D proxy**. Run it against fresh P/D engines and a fresh
experimental service, with these changes to the preceding commands:

1. Add `--l2-store-policy skip_l1` to `lmcache server`. LMCache removes each L1
   object after successful L2 storage; the check never calls `/cache/clear`.
2. Start the upstream proxy with ports 28000 (client) and 25768 (telemetry),
   prefiller 28001 and decoder 28002, all on 127.0.0.1.
3. Set P's `LMCACHE_REQUEST_TELEMETRY_TYPE=fastapi` and
   `LMCACHE_REQUEST_TELEMETRY_ENDPOINT=http://127.0.0.1:25769/api/v1/telemetry`.
   The check temporarily serves this telemetry relay; leave D telemetry unset.
4. Run `python examples/memfabric/pd_inference_smoke.py --output /your/private/pd-results`
   using a new output directory.

The relay receives P's actual `request_store_finished` event, waits for BM
stores to complete and L1 object count/allocated bytes to reach zero, saves that
snapshot, then forwards the same event to the proxy. Only then can the proxy
send to D. A timeout fails the check. This relay is a single-request validation
gate, not a production router or a multi-request commit protocol.

The 2026-10-24 run passed with P on device 4, D on 6, and BM ranks on 8/10:
P computed 1373 prompt tokens and generated one token; before D, 10 objects /
180 MiB existed in rank 1 DRAM and L1 was empty. D's request logged **0 L1,
10 L2** hits, read 180 MiB from BM, restored 1280 tokens, computed the remaining
93 tokens, and answered `ORCHID`. Its local prefix-cache count stayed zero.
Pool data still passes through temporary L1 buffers on the restoration path.
See `before_decode.json`, `p_store_event.json`, `proxy_notification.json`,
`summary.json`, and the P/D metric snapshots in the output directory.

## TP8, workload, cancellation, and failure checks

The same gate supports `--tp-size 8`. It collects the actual completion events
for all eight P ranks before checking L2 completion and empty L1, then forwards
the last event to the unchanged proxy. The TP8 run restored the same 1280 tokens
from 80 objects / 180 MiB. Tested configuration changes from TP1:

- P devices `0,1,2,3,4,5,6,7`; D devices `8,9,10,11,12,13,14,15`.
- Both engines: `--tensor-parallel-size 8 --max-num-seqs 4`.
- MP service: `--max-gpu-workers 16 --l2-store-policy skip_l1`.
- BM rank 0 uses device 8; pool member rank 1 uses device 10. Both reserve
  `pool_gb: 4`; adapter `slot_size_mb: 4` gives 1024 slots on the selected rank.
  Qwen3-8B TP8/chunk128 payloads occupy 2.25 MiB per slot.

For **ordinary ungated routing**, set P telemetry directly to port **25768**
and restart P. These checks use the unmodified upstream proxy:

```bash
python examples/memfabric/pd_workload.py --phase serial --output /your/private/serial
python examples/memfabric/pd_workload.py --phase shared --concurrency 4 --output /your/private/shared
python examples/memfabric/pd_workload.py --phase concurrent --concurrency 4 --output /your/private/concurrent
python examples/memfabric/pd_workload.py --phase concurrent --records 192 --concurrency 4 --timeout 60 --output /your/private/long
```

Across TP1 and TP8, 82 ordinary requests passed password-dependent answer
checks, including repeats, extensions, sub-chunk prompts, eviction, and four
outstanding requests. TP1 used engine `max-num-seqs=1`; TP8 used 4. A small
pool can evict KV before D reads it, causing recomputation. Under concurrent
load, service logs also showed L1/L2 mixed hits and L1-only hits. Therefore
`skip_l1`, a correct answer, or vLLM's external-transfer metric alone does not
prove that each ordinary request read exclusively from BM. Use the gate for
that specific claim and inspect each request's service prefetch log.

Start D with `--enable-log-requests` for cancellation evidence:

```bash
python examples/memfabric/cancel_smoke.py --decoder-log /your/private/decode.log --output /your/private/cancel
```

This closes three streams after decoding starts, matches each ID to D's abort
log, checks storage cleanup, then verifies a normal follow-up answer. The tested
vLLM version does not increment its completed-request abort counter on this
explicit cancellation path, so that counter is recorded but not used as proof.

The following **stops the specified experimental pool member**. Use an idle
experimental service, a fresh `SAFFRON` prefix, and a verified member PID:

```bash
python examples/memfabric/peer_loss_smoke.py --member-pid MEMBER_PID --output /your/private/peer-loss
# Use --existing-pool for a service with unrelated cached prefixes.
# Use --member-signal SIGKILL to test abrupt exit instead of SIGINT.
```

If departure is not detected, the check fails before attempting stale reads.
After a detected departure it verifies full recomputation and zero new BM reads.
The group must then be restarted together; the volatile index is not recovered.

With devices 0 and 2 free, the following check owns its own pool member and
tests departure **after lookup has pinned an object, before load starts**:

```bash
python examples/memfabric/locked_peer_loss_smoke.py --output /your/private/locked-peer-loss
```

The measured run returned a load miss without changing the destination, released
the pin on unlock, preserved both allocator references, rejected subsequent
stores, and closed cleanly. It does not inject a failure during a DMA operation.

After `peer_loss_smoke.py`, keep the failed service and D running to check
continued degradation under L1 pressure:

```bash
python examples/memfabric/degraded_workload.py --output /your/private/degraded
```

On TP1 with a 4 GiB L1 and LRU eviction, all 24 distinct long prompts passed.
D recomputed 88551 tokens with zero additional BM reads/writes. The highest
observed L1 usage was 4227858432 bytes, below its 4294967296-byte capacity;
eight decreases in object count demonstrated eviction. Failed writes left no
outstanding pins or store tasks. This is a bounded fallback check, not a
long-duration availability test.

## Inspect actual SDMA tasks

With other experiments stopped, run a member on device 2 and port 28584:

```bash
python examples/memfabric/pool_member.py --device 2 --rank 1 --store-url tcp://127.0.0.1:28584
```

In another terminal:

```bash
msprof --output=/your/private/trace --runtime-api=on --task-time=l1 --ai-core=off \
  --sys-interconnection-profiling=on --sys-interconnection-freq=50 --storage-limit=200MB \
  python examples/memfabric/profile_copy.py --device 0 --output /your/private/copy
```

The final measured run completed 3479 pairs plus one source-overwrite check,
each copying 16 MiB in each direction. CANN recorded exactly **6960 SDMA_SQE**
tasks, matching the 6960 BM copies. Both processes exited cleanly and payloads
matched. This is device execution evidence for the adapter's host/pool copy
primitive, not a vLLM throughput result or proof of a particular HCCS link.
