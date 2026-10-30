# DRAM pool validation with two logical nodes on one physical host

Validation date: 2026-10-30. This is a selected functional record. Original
service logs, HTTP exchanges, counters, worker JSONL and actual process waits
remain in the private project archive. No performance improvement is claimed.

## Scope

Two containers had different network and IPC namespaces, independent private
runtime directories, and read-only access to the existing Qwen3-8B model.
The pool explicitly selected SDK HOST/DRAM; no fallback to DEVICE/HBM was used.
Native BM exports still identified the same physical server on both sides.

This validates software role placement and real NPU/DRAM operations across
logical nodes. It does not establish physical inter-server DRAM mapping, HCCS
traffic, or a fix for the earlier physical inter-server HOST failures.

Selected production revisions were:

- L2 route: LMCache `fa2e6ff`, LMCache-Ascend `137038a`.
- Worker route: LMCache `5b565d2`, LMCache-Ascend `1e5ef32`.
- Two-sequence diagnostics: example files from `4485055`, isolated from the
  production data-path revisions.

## Accepted groups

| Group | Placement | Evidence |
|---|---|---|
| L2 full-content smoke | CPU MemoryObj on A, DRAM owner on B | Entire 18 MiB payload matched; delete and protection counters returned to zero; both roles naturally exited 0 |
| L2 Qwen3-8B TP1 and TP4 | P/D/service on A, DRAM owner on B | 1,373 input tokens; D restored 1,280, computed 93, local prefix 0; all prefetch blocks came from L2; 180 MiB per direction |
| Worker Qwen3-8B TP1 and TP4 | P/D on A, DRAM service on B | Cold and restored answers both VELM42; 1,364 input tokens; restored 1,280, computed 84, local prefix 0; cold reference had no BM copy |
| Worker split P/D TP1 | P on A, D and DRAM service on B | Same strong context/token/source checks; original D process and trace were on B |
| Worker split P/D TP2 | Two P shards on A, two D shards and DRAM service on B | Same strong context/token/source checks and actual per-shard restore completion |
| Worker TP4, 500 repeated retrievals | P/D on A, DRAM service on B; max-num-seqs=1 | 500 successful HTTP responses; external 640,000, computed 42,000, local prefix 0; D completed 94,371,840,000 bytes |
| Reverse L2 smoke and Qwen3-8B TP1 | P/D/service on B, DRAM owner on A | Entire 18 MiB smoke matched; model 1,373/1,280/93/0, ten L2-only prefetch blocks, 180 MiB per direction; controlled lifecycle accepted |
| Worker split P/D TP4 | Four P shards and DRAM service on A, four D shards on B | Strong cold/restored context checks; each direction completed 180 MiB across four actual workers |
| Worker TP4 mixed requests, max-num-seqs=2 | P/D on A, DRAM service on B | Warm/partial and warm/cold pairs had successful HTTP identities in the same D scheduler step |
| Worker TP4, another 500 retrievals, max-num-seqs=2 | P/D on A, DRAM service on B | All responses correct; 94,371,840,000 completed D bytes; 750 same-step two-request scheduling records in the two-client phase |
| TP1 production-context half-chunk restore, no model | P/D on A, DRAM service on B | Original NPU test recorded successful retrieval Future, restored second half and unchanged skipped/other pages; no independent completed-byte counter |

Worker TP4 used four P and four D shards, with 10 completed copies of
4.5 MiB per shard in each direction. Native return, BM completion and
gather/scatter completion were checked against the original worker identity.
Ordinary L1 remained empty, no L2 adapter was configured on this worker route,
and final pins, pending writes and quarantine were zero.

The repeated group used 250 serial requests followed by 250 requests with two
HTTP client threads. Each D shard completed 5,000 copies of 4.5 MiB, matched
to its original native and scatter records. P added no copy. Every 50-request
checkpoint had empty ordinary L1/no L2 adapters and zero pins, pending writes
and quarantine. This max-num-seqs=1 group does not prove two-request engine
scheduling. The independent max-num-seqs=2 group had the same HTTP token/source
counts and per-shard completed bytes, with original scheduler records strictly
linked to successful complete API IDs (including the installed engine's
optional eight-hex internal suffix). Its 750 records are scheduling steps,
not 750 additional requests or distinct request pairs.

The mixed pairs used warm 1,373 / partial 1,237 / cold 1,113 input tokens and
10 / 5 / 0 shared chunks. Pair metric deltas were respectively
2,610/1,920/690/0 and 2,486/1,280/1,206/0 for prompt/external/compute/local
prefix. D completed 270 MiB and 180 MiB across four shards. Same-step positive
token assignments were independently linked to both successful HTTP responses.

The half-chunk context group selected 128 cached tokens but restored only the
last 64. Its 18 MiB field is the requested payload, not an independently
measured completed-byte counter. No full tensor artifact was retained for an
independent offline content comparison; its content evidence is the original
NPU test's comparison result, with native logs and actual role waits retained.

## Lifecycle and retained failures

The first L2 model group completed inference but failed lifecycle acceptance:
the example's 18-second per-role shutdown grace expired and the supervisor
sent SIGKILL to the service. That failed group was retained.

A subsequent configuration used a 180-second total shutdown budget, which the
existing harness rejected before model initialization. It is a configuration
failure, not an NPU/model result. CPU admission checks were corrected in the
private executor to reject values above the harness's existing 90-second bound.

Fresh L2 TP1 and TP4 groups used the existing supported configuration:

```json
{"stop_timeout": 60, "shutdown_timeout": 90}
```

Their total controlled shutdowns took 31.51 and 42.36 seconds respectively.
Supervisors and outer driver/member processes returned 0; service waits were
130 after planned SIGINT, with no SIGKILL or SIGTERM. This is a harness budget
adjustment, not a production adapter, BM or SDK change. It does not resolve
every historical shutdown failure.

This archive changes the example supervisor and campaign defaults to the same
60/90-second values. Explicit overrides remain supported. The real groups used
explicit 60/90 settings in frozen runtime copies; those copies were not edited
while running. Seven focused L2 CPU checks passed after observing the old
default fail the updated budget check. The CPU campaign fixture now uses a
relative test-only deadline so its before-deadline branch remains testable
after the archived hardware window; live resource policy is not changed.

The first two-sequence diagnostic bundle omitted its matching stock gate.
That old gate incorrectly required the scheduler PID to have TP worker
identity, rejected the notification, and prevented both follow-up probes from
starting. Both gates returned 1; the mixed group's proxy eventually returned
-9 during controlled shutdown, so lifecycle also failed. These groups were
retained as failed harness attempts, not DRAM transfer results. Fresh groups
with all matching diagnostic helpers passed the independent runtime audit.

Before starting a staged scheduling bundle, check its actual gate file rather
than only testing a canonical source checkout:

```bash
python examples/memfabric/cross_host_20261030/check_staged_gate_cpu.py \
  --gate /private/staged/examples/memfabric/direct_pd_gate.py
```

This CPU-only check requires all eight real TP4 worker identities, separately
accepts a scheduler PID, and rejects missing workers or incomplete worker
identity. The old deployed bundle reproduced its identity error; the matched
bundle passed. These checks do not initialize NPU/BM or prove runtime transfer.

Worker gates returned 0 naturally. P/D/proxy returned 0 after planned shutdown;
services returned 130 after planned SIGINT; trace collectors returned -2 after
planned SIGINT. These actual waits are retained separately rather than being
reported as universally natural zero exits.

Every retry had a new group and output directory. Independent supervisors
continued the finite queue without relying on the interactive conversation;
an independent full-container-ID deadline guard bounded resource use.

The completed campaign retained 112 original task waits before stopping its
two new containers. Final readback at 12:03:47 Beijing time confirmed both
containers stopped with PID 0, no NPU processes, the original container still
running, and the deadline guard cancelled/exited. Idle-container Docker exit
codes are distinct from the previously collected experiment-role waits.

## Remaining boundaries

Physical inter-server HOST/DRAM, real-model TP8, production raw-page batch
integration, asynchronous DMA/compute overlap and long-term stability are not
established by these groups. Two concurrent HTTP clients alone also do not
prove that the D engine scheduled two requests in one step.
