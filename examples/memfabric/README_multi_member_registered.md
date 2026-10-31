# World3 registered NPU -> HOST -> separate NPU diagnostic

This is a new, independent synthetic three-member primitive. It does not modify
`registered_host_rdma_probe.py`, LMCache, adapters, worker-direct or the SDK.
Base revision: `d1fd5791be1747717c77a9476574b6ec0230fd79`.
Target SDK remains MemFabric 1.2.1 fixed source `106f783`.

## Fixed contract and causal order

All ranks request the same `create2(id=0, local_dram_size=1073741824,
max_dram_size=1073741824, local_hbm_size=0, max_hbm_size=0,
data_op_type=DEVICE_RDMA, enable_56bits_gva=True, flags=0)`, world size 3,
`MF_HYBM_RDMA_SWAP_SPACE_SIZE=0`, and explicit logical `--device 0`.
The caller must independently verify each process's physical device mask.

1. Rank 0 owns the HOST pool; rank 1 is the only writer; rank 2 is the only reader.
   All three publish and compare the entire creation/transfer contract before
   device initialization, and agree on rank 0's HOST GVA after join.
2. Writer allocates a real contiguous NPU uint8 Tensor containing A and publicly
   registers its actual pointer. Reader independently allocates and registers a
   different Tensor in a separate process. Owner validates its SDK HOST HVA
   against `/proc/self/maps` before any CPU access.
3. Writer synchronously calls `L2G(writer_tensor_ptr, owner_HOST_GVA, n, 0)`.
   Owner independently compares every byte against A, then acknowledges it.
4. Writer fills its source Tensor with 51, synchronizes, copies it to CPU for
   full sentinel verification, and publishes the causal completion metadata.
5. Reader resets and fully verifies sentinel 90, then synchronously calls
   `G2L(owner_HOST_GVA, reader_tensor_ptr, n, 0)`, synchronizes and compares all A.
6. Only after reader A is verified does owner write and independently verify B.
   Reader resets/verifies sentinel again and reads/compares all B.
7. All roles publish and observe all three payload-done events before teardown.
   Registered Tensor pointers are explicitly unregistered before leave/destroy;
   Python Tensor objects remain alive throughout all local SDK teardown.

HTTP carries only configuration, pointers, byte counts, identities, errors and
completion metadata. No payload, Tensor, base64 or buffer dump is published.
Initial test-pattern upload and final CPU content inspection are verification
operations; no intermediate local HBM pool or HOST payload staging is used by
the cross-member L2G/G2L calls.

Missing peers time out; any published peer failure aborts subsequent waits.
Partial or nonzero native calls fail closed. Every raw public SDK return is
logged. Normal DEVICE_RDMA `handle.wait` is a no-op and is not used as completion
evidence. Parent execution must retain all three real process waits, full raw
stdout/stderr and independent resource-release evidence.

Public `unregister=0` is not proof that lower native cleanup emitted no warning:
the known `nativeHalHostUnregisterEx ret8` asymmetry must remain in raw logs and
be reported separately. This probe neither changes the SDK nor suppresses native
logs. Payload pass, local cleanup pass, external process exit and device release
are separate evidence classes.

## CPU check and default plan

Run in this checkout from its repository root:

```bash
python examples/memfabric/check_multi_member_registered_cpu.py
python examples/memfabric/multi_member_registered_probe.py \
  --rank 0 --device 0 --payload-bytes 64 \
  --store-url tcp://115.120.90.25:19631 \
  --coord-url http://115.120.90.25:29631 \
  --run-id world3mix01 --output /tmp/world3mix01-owner-plan.json
```

Without `--allow-npu`, the probe writes `PLAN_ONLY` and never imports torch,
torch_npu or MemFabric. Output files use exclusive creation; each retry requires
a fresh run ID, coordination directory, ports and result directory.
Permitted payloads are exactly 64, 1048576 and 18874368 bytes.

The seven CPU checks exercise CLI limits, identical world3 creation contracts,
exclusive PLAN_ONLY output with no native imports, full-content and peer
identity rejection, all three causal role flows with local bytearrays and real
localhost HTTP metadata, missing-peer timeout, and explicit peer failure.
They prove **CPU_CONTROL_PATH_ONLY**, not NPU registration, RDMA, cross-host
content completion, native cleanup or device release.

## Three-party native command parameters

These are commands for a newly authorized device run, not a launch receipt.
Use the already verified personal container and existing private SDK environment
on each host; `python` below refers to that environment. Before execution check
the current time/window, exact container identity/mounts, physical device masks,
free device ownership, SDK path and a newly enabled independent deadline guard.
Do not use these commands to revive an expired run.

Example placement A: owner on 90.25 physical 8; writer on 137 physical 8;
reader on 137 physical 10. All commands have the same run ID, store URL, NIC
port, creation contract and payload size. Use the independently verified
mask-to-logical mapping; these examples assume mask value 8 or 10 exposes that
physical device as logical 0. Raw logs must be separate per rank.

On owner 90.25, start metadata coordination in the confirmed private run folder:

```bash
python examples/memfabric/direct_coordination.py \
  --bind 0.0.0.0 --port 29631 --directory "$RUN_DIR/coord" \
  --run-id world3mix01 >"$RUN_DIR/coord.log" 2>&1 &
coord_pid=$!
```

Start the three ranks promptly in separate terminals/processes. Each command
has a finite outer timeout in addition to SDK and coordination timeouts:

```bash
# 90.25 owner; retain owner_pid and wait result in the launching shell.
ASCEND_RT_VISIBLE_DEVICES=8 timeout --signal=TERM --kill-after=10s 480s \
  python -u examples/memfabric/multi_member_registered_probe.py \
  --rank 0 --device 0 --payload-bytes 64 --init-timeout 180 --timeout 180 \
  --device-nic tcp://127.0.0.1:10005 \
  --store-url tcp://115.120.90.25:19631 \
  --coord-url http://115.120.90.25:29631 --run-id world3mix01 \
  --output "$RUN_DIR/rank0.json" --allow-npu >"$RUN_DIR/rank0.log" 2>&1 &
owner_pid=$!
wait "$owner_pid"
owner_rc=$?
printf '%s\n' "$owner_rc" >"$RUN_DIR/rank0.wait.rc"
```

```bash
# 137 writer.
ASCEND_RT_VISIBLE_DEVICES=8 timeout --signal=TERM --kill-after=10s 480s \
  python -u examples/memfabric/multi_member_registered_probe.py \
  --rank 1 --device 0 --payload-bytes 64 --init-timeout 180 --timeout 180 \
  --device-nic tcp://127.0.0.1:10005 \
  --store-url tcp://115.120.90.25:19631 \
  --coord-url http://115.120.90.25:29631 --run-id world3mix01 \
  --output "$RUN_DIR/rank1.json" --allow-npu >"$RUN_DIR/rank1.log" 2>&1 &
writer_pid=$!
wait "$writer_pid"
writer_rc=$?
printf '%s\n' "$writer_rc" >"$RUN_DIR/rank1.wait.rc"
```

```bash
# 137 reader, separate process and physical card from writer.
ASCEND_RT_VISIBLE_DEVICES=10 timeout --signal=TERM --kill-after=10s 480s \
  python -u examples/memfabric/multi_member_registered_probe.py \
  --rank 2 --device 0 --payload-bytes 64 --init-timeout 180 --timeout 180 \
  --device-nic tcp://127.0.0.1:10005 \
  --store-url tcp://115.120.90.25:19631 \
  --coord-url http://115.120.90.25:29631 --run-id world3mix01 \
  --output "$RUN_DIR/rank2.json" --allow-npu >"$RUN_DIR/rank2.log" 2>&1 &
reader_pid=$!
wait "$reader_pid"
reader_rc=$?
printf '%s\n' "$reader_rc" >"$RUN_DIR/rank2.wait.rc"
```

Read all three wait receipts and JSON reports before terminating the matching
coordination PID in its original shell. Preserve logs even for failures and
outer-timeout exits. Require zero rank exits plus payload/local cleanup reports;
then independently verify process/container/device release. The 480-second
timeout is an attempt bound, not permission to overrun a resource cutoff.
The independent deadline guard must enforce the earlier actual release time.

For placement B, place writer on 137 and reader on 90.25 physical 10 while owner
remains 90.25 physical 8. For placement C, place owner on 137 physical 12,
writer on 90.25 physical 8 and reader on 137 physical 10; all three store URLs
must then use 137's reachable owner address. No code change is needed.
Each placement is a fresh group. Never select 137 physical 14 during this
window. Start at 64 bytes; proceed to 1 MiB then 18 MiB only after the prior
attempt's true exits and release are inspected.

## Evidence boundary

At authoring time only the seven CPU checks have run. World3 native creation,
three-member register/import, all three physical placements, real RDMA full
payload completion, native process exits and release remain **NOT_VERIFIED**.
No source/CPU result is a model TP3, production-worker, concurrent-copy,
throughput or service-speedup result. Cross-host control uses causal events;
`mono_ns` and copy durations may be compared only within their own process.
