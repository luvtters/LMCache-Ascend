# Cross-host PP startup patches

## Revised customer procedure

Use `switch-to-official.sh` with all three patch files in this directory. Source
the customer's `/opt/ks-luv-pp/env.sh` first. The script resolves the actual
top-level package paths without importing the device frameworks, checks all
three target files, reverses patches 0001 and 0002 if present, then applies 0003.
It preserves unrelated source edits and skips steps already completed. It also
handles files previously modified by the equivalent Python replacement helper.

Patch 0003 contains only the production change from merged official
[PR #18521](https://github.com/vllm-project/vllm-ascend/pull/18521), commit
`7b5ad8160e5f1eb6eda6954d0659d1a0e65b1122`. It changes
`vllm_ascend/patch/platform/patch_multiproc_executor.py`, updating the cached
headless CLI executor reference from the existing Ascend executor adapter.
It does not require editing the core `serve.py` file.

Keep `DYNAMIC_EPLB=True`, dynamic EPLB enabled in the additional configuration,
and `--headless` on node 1. No model configuration change or compilation is
required. Apply source changes before starting new model processes.

The separate receive-buffer change in 0002 is no longer a mandatory customer
prerequisite: the recorded original communication mode also had successful
device runs, so its necessity and its relationship to the observed failure
were not established. The old `apply.sh` and patches 0001/0002 are retained for
review and reversal; do not use `apply.sh` for the revised procedure.

## Original patch bundle

These are project-specific framework patches distributed alongside the LMCache
Ascend integration. They modify the two framework source files listed below;
they do not modify LMCache runtime code.

| Patch | Target module and function | Change |
| --- | --- | --- |
| `0001-headless-executor.patch` | `vllm/entrypoints/cli/serve.py::run_headless()` | Import the adapted `MultiprocExecutor` at the headless instantiation site. |
| `0002-eplb-receive-buffers.patch` | `vllm_ascend/eplb/eplb_updator.py::EplbUpdator.warm_up_eplb()` | Allocate and retain a separate receive tensor for each peer during warm-up. |

Recorded target revisions:

- Core: `568afb3a13806beb53bb2e6bd518269357b237c0`.
- Ascend plugin: `f2f74a16c3c50a76f4349d807918e83edec1e35c`.

Run `bash apply.sh` inside each customer container, with both patch files beside
the script. It checks both patches before applying either, skips changes already
present, and stops on a source mismatch. Target repositories are
`/vllm-workspace/vllm` and `/vllm-workspace/vllm-ascend`.

No compilation or dependency installation is required. Start new model processes
to load the changed Python code. This script does not start or stop services.

The patch files were checked against recorded original sources, matched the
recorded repaired sources, and passed reverse checks. Customer device readiness
and KV correctness require validation in the customer environment.

To review or revert a patch, use `git diff -- <target-file>` or
`git apply --reverse <absolute-patch-path>` in its target repository.
