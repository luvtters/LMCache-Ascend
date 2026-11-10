# Cross-host PP startup patches

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
