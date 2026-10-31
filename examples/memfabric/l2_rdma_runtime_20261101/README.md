# Experimental L2 runtime entry archive (2026-11-01)

These four source files archive the executed experimental startup wrappers and
official vLLM benchmark entry. They are not production-module changes or a
single-command reproduction bundle. No private configuration, request data,
runtime logs, receipts or credentials are included.

- `normal_model.py`: foreground supervisor and experimental service entry for
  same-host P/D with an externally managed pool member. It retains `--config`
  and `--plan-only`; a plan is not execution or resource authorization.
- `l2_config.py`: external configuration parsing, L2 configuration construction
  and source-audit helpers, not a supplied runnable campaign configuration.
- `install_reservation_trace.py`: the executed observational reservation wrapper;
  it preserves the original reserve-write result and does not introduce a new
  allocator, retry policy or registration cache.
- `official_run.py`: sequential fill/revisit through official `vllm bench serve`,
  with original benchmark outputs, status snapshots and request-length checks.

The source contents are preserved without refactoring. The first three files
depend on the original runtime/core interfaces; they do not vendor that runtime
or the model. The benchmark's raw JSON can contain official default extra metric
columns; this does not change which columns the accompanying report presents.

## Method and validation boundaries

The normal service uses default L2 store/prefetch policies and normal prefix
caching, without manual CPU-L1 clearing or an extra P-to-D cache-drain barrier.
Users must retain and inspect their actual resolved configuration: the wrapper
accepts external settings and is not proof of the settings used by a future run.

The recorded successful combination pinned/unpinned the original L1 arena on
logical device 0 and used logical device 2 for BM. This is the tested mapping,
not a universal MemFabric SDK restriction. Device visibility, logical/physical
mapping, IPC access and the selected contexts must be verified for each run.

The archived supervisor has a default 30-second service shutdown allowance and
a shared 90-second shutdown budget for the local group. These are bounded
shutdown allowances, not proof of clean SDK teardown. One real group lost its
service to SIGKILL while requests were running; the supervisor recorded no signal
send for that role, and the source was unknown. That failure was retained and
the queue stopped without restarting it. Planned SIGINT returns, native failures
and unplanned termination must remain distinct in any new result record.

The separate `bm_stage_trace.py` hook had only CPU fake-path checks and no real
device validation in this campaign, so it is intentionally not published in
this archive. API-stage timing must not replace normal-service performance.

## Requirements before using these entries

The user must provide a new private configuration, confirmed model/tokenizer
paths, the original compatible runtime and environment, valid device mappings,
an external pool/coordinator with its own real waits, and an independent deadline
and resource-release guard. Store master/owner launchers and benchmark dataset
generation are not included here. No internal endpoint or personal configuration
is supplied.

`official_run.py` preserves the campaign's historical cutoff defaults. For a new
authorized window explicitly pass both `--stop-new-at` and `--stop-at` with the
new date, time and timezone; do not rely on or extend the old defaults. Its client
timeout does not replace the independent release guard. The supervisor similarly
needs external resource-window enforcement.

The presence of these scripts means neither that execution has been scheduled
nor that devices are available or authorized. `--plan-only` checks only a plan;
this directory cannot by itself reproduce the complete distributed experiment.

See [the real-service validation checkpoint](../REGISTERED_L2_RDMA_VALIDATION_20261101.md)
for the observed functionality and unresolved teardown failures. This archive was
checked by CPU source compilation and whitespace review only; no model test was
repeated to prepare it.
