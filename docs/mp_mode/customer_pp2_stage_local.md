# Customer two-host PP2 candidate

This branch pairs with LMCache branch `ks/2p1d-pp-local-luv-20261109`, based on
LMCache `3e11e10812b448142c6cdb177a3058652bd826fa` and this plugin's
`fbabeecc79ba79663db6c7bb8555b9dcc649a125` baseline. Both halves are required.
It is a local candidate, not an upstream release or device-validated result.

The customer Prefill engine is TP16/PP2 across two hosts, with one complete
pipeline stage and one independent LMCache L1 server on each host. D remains
DP4/TP4/PP1 with Mooncake. The plugin implements server-local piece identities,
reader counts, writer selection, local-cache PP handshake and NPU-only index
pinning. It reuses upstream min-prefix aggregation and lock cleanup.

## Activation

After building/installing the paired sources in the private container venv,
export this in **both P model terminals and both cache terminals**, before
starting any Python process:

```bash
export LMCACHE_ASCEND_PP_STAGE_LOCAL=1
export PYTHONUNBUFFERED=1
```

In the existing P-side MultiConnector configuration, retain the Mooncake child
and use this LMCache child (replace P1/P2 addresses with the real internal IPs):

```json
{
  "kv_connector": "AscendPPStageLocalConnector",
  "kv_connector_module_path": "lmcache_ascend.integration.vllm.pp_stage_local",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "lmcache.mp.server_urls": ["tcp://P1_IP:5565", "tcp://P2_IP:5565"],
    "lmcache.mp.autostart": false,
    "lmcache.mp.nonblocking_lookup_status": false
  }
}
```

Both model nodes use the **same ordered list**: front then rear. Neither URL may
be loopback. Each server binds a reachable interface; it accesses only its local
NPU handles. Use independent local L1 storage; shared L2 is outside this mode.

The outer MultiConnector retains the customer's producer role, Mooncake's
producer role and original TP16/PP2 -> DP4/TP4 metadata. Keep MTP3, 41/37 layers
and DSA CP; do not replace them with DCP/PCP. The candidate rejects other
TP/PP/DP/DCP/PCP combinations and CUDA devices. Normal upstream connectors keep
their original rejection, rather than silently enabling unsupported layouts.

## NPU index staging

With explicit activation the plugin changes only
`NpuDeviceSpec.create_cache_context` to lazily construct `PinnedNpuCacheContext`.
The CPU and CUDA context classes are untouched. The small index tensor uses
`pin_memory(device="npu")`; the inherited NPU allocator/event contract must keep
its source alive until the asynchronous copy completes. No reused host buffer,
model-kernel edit or extra device synchronization is introduced.

## CPU verification and remaining runtime gate

In the plugin checkout run:

```bash
export KS_LMCACHE_SOURCE=/absolute/path/to/paired/LMCache
python -S -B -X utf8 tests/ks/test_pp_stage_local_cpu.py
```

The tests execute actual constructor/lookup source and plugin policy, replacing
framework objects, network replies and device allocations with explicit CPU
fixtures. They cover all 32 workers, replicated and sharded KV, two-stage
prefix intersection, incomplete replies, misses, cancellation, tail locks and
NPU-only factory dispatch. They do not exercise a model, native IPC or NPU.

Next runtime gate: build with the customer's matching CANN/hcomm; verify imports
and the selected context on both cache processes; verify each worker registers
with its own stage's server and layer layout; perform real cold STORE and warm
RETRIEVE with complete D generation; validate KV/output correctness and lengths;
then benchmark against a same-environment OFF. CPU success is not speedup proof.
