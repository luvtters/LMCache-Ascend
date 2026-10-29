# Independent native batch primitive

This probe uses SDK `BigMemory.copy_data_batch` on original split-NHD NPU KV
planes. There is no gather/scatter scratch and no production connector change.
CPU checks and PLAN_ONLY output do not prove native batch or inference support.

Use the verified personal container and matching core/Ascend sources on
PYTHONPATH. Set private endpoints and paths explicitly; start a fresh CPU coordinator:

```bash
python examples/memfabric/direct_coordination.py --bind "$COORD_BIND" --port "$COORD_PORT" \
  --directory "$PRIVATE_COORD_DIR" --run-id "$FRESH_RUN_ID"
```

Launch three foreground roles concurrently, each in its own process/environment:
pool/rank0, P/rank1, D/rank2. Each process masks its
single physical card with `ASCEND_RT_VISIBLE_DEVICES`, then uses logical device0.
The outer executor must verify this mapping and save PID/process group and wait.

```bash
ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL_DEVICE" python examples/memfabric/batch_direct_probe.py \
  --role "$ROLE" --run-id "$FRESH_RUN_ID" --layers 4 \
  --store-url "$STORE_URL" --coord-url "$COORD_URL" \
  --output-dir "$PRIVATE_RESULT_DIR" --run --allow-npu
```

Use matching endpoint/run arguments with each role's selected physical device.
Roles may run on different hosts; HTTP contains metadata only and paths are local
to each host. Default without `--run` is PLAN_ONLY. Runtime imports require the
existing SDK1.2.1/torch_npu/CANN environment; no installation is performed.

First finish L4 (2MiB); then use a fresh coordinator/session/group for `--layers
36` (18MiB). Source pages `[3,1]` map to target pages `[4,2]`; each plane has six
64-token pages and BF16 8x128 values per token. The plan contains 4L distinct
plane/page segments. P zeros its original pages only after batch and wait have
completed. D compares every element of every written page against an independent
CPU-index pattern and checks every untouched page remains -79. CPU verification
copies occur after the native direct transfer; HTTP coordination never holds KV.

Each role writes raw `events-ROLE.jsonl`, the full segment/plane metadata and
`result-ROLE.json`. Requested bytes, native return code, confirmed completed
bytes, wait and close are separate fields/events. A failure retains its arrays
and tensors, writes FAIL, and immediately exits without Python/native finalizers;
the executor must terminate/release the entire failed fixed group and never reuse
it. A stuck native wait/close needs the executor's independent stopping policy.

`NATIVE_BATCH_PRIMITIVE_PASS_PROCESS_EXIT_PENDING` still requires outer real
wait/exit and device-release evidence. It is not a model inference result and
does not mean LMCache has integrated batch transfers. No automatic restart.

CPU check: `python examples/memfabric/check_batch_direct_probe_cpu.py`.

## DEVICE selection and selected validation, 2026-10-30

HOST remains the default. Select `--pool-memory-type device` explicitly and use
a matching BMSession wrapper supporting DEVICE. Each DEVICE rank declares
**1 GiB DRAM plus 1 GiB HBM**, not a pure-HBM allocation. The target is the owner's
DEVICE peer pointer; its LOCAL_DEVICE mapping must become nonzero. HOST does
not add this DEVICE mapping requirement. The 7 CPU checks cover this boundary.

The original HOST probe passed same-host L4 and L36. Frozen DEVICE probe
`fe14a5f` passed one independent cross-host L36 group: P and D on one host, an
independent owner on another. Each direction submitted one 144-segment native
batch covering 18 MiB, flags0/stream0. Every local pointer was distinct and
checked against its retained tensor plane. All 72 K/V planes passed written-page,
untouched-page and full-content checks (28,311,552 elements), without scratch.
Native export/import logs confirmed owner DEVICE and separate 1 GiB DRAM/HBM
segments; configuration alone was not used as proof.

P, D and owner exited naturally 0 after wait/leave/destroy/uninit. The executor
and isolated 75-second deadline guard exited 0; the guard cancelled after all
role waits. The CPU coordinator received planned SIGINT and exited -2. Final
readback found the three selected devices released. Native logs retain early
mapping retries before readiness; these were not described as absent.

This proves an independent original-page batch primitive, not production
LMCache batch integration, model inference, multi-request IO aggregation,
asynchronous DMA, overlap, link-byte attribution, performance or long-term
stability. Cross-host HOST/DRAM failures remain unresolved.
