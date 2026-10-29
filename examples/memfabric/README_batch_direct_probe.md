# Independent native batch primitive

This probe uses SDK `BigMemory.copy_data_batch` on original split-NHD NPU KV
planes. There is no gather/scatter scratch and no production connector change.
CPU checks and PLAN_ONLY output do not prove native batch or inference support.

Run on 12 (`192.168.0.50`) inside the verified personal container, with the
matching private core/Ascend sources on PYTHONPATH. Start a fresh CPU coordinator:

```bash
python examples/memfabric/direct_coordination.py --bind 192.168.0.50 --port 29615 \
  --directory /workspace/d00896497/batch_probe/coord --run-id batch-l4-attempt01
```

Launch three foreground roles concurrently, each in its own process/environment:
pool physical6/rank0, P physical2/rank1, D physical4/rank2. Each process masks its
single physical card with `ASCEND_RT_VISIBLE_DEVICES`, then uses logical device0.
The outer executor must verify this mapping and save PID/process group and wait.

```bash
ASCEND_RT_VISIBLE_DEVICES=6 python examples/memfabric/batch_direct_probe.py \
  --role pool --run-id batch-l4-attempt01 --layers 4 \
  --output-dir /workspace/d00896497/batch_probe/results --run --allow-npu
```

Use the same arguments with physical2/role p and physical4/role d. Default store
is `tcp://192.168.0.50:28715`; coordinator is `http://192.168.0.50:29615`. Both are
overridable. Default without `--run` is PLAN_ONLY. Runtime imports require the
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
