# Registered L2 RDMA validation checkpoint (2026-11-01)

This sanitized checkpoint records real allocator and model functionality.
Internal endpoints, identities and raw logs remain in private records. The
[2026-10-31 checkpoint](REGISTERED_RDMA_VALIDATION_20261031.md) remains valid
as a record of that attempt, including HAL ret8 / public -1 and zero completed
L2 payload bytes. The later successful path is separate evidence, not a rewrite
of that failure.

## Executed sources and topology

The executed core baseline was `5b565d2`. Ascend used the existing cross-host
snapshot with three source files overlaid from `25ceb46`; this is not evidence
that the complete Ascend tree was that revision. The allocator and MemFabric
SDK 1.2.1 were unchanged. No new allocator or pointer-registration cache was
introduced.

The Qwen3-8B TP1 P and D engines ran on the same host. The independent HOST
DRAM pool ran on another host. This is real cross-host L2 storage, while P and
D themselves were not split across hosts in this checkpoint. Logical device
numbers below are per-process mappings, not physical-device identifiers.

The successful service path kept L1 pin/unpin on logical device 0 and BM
communication on logical device 2, separating the required device contexts.
Both mappings must be visible and verified. A `device_id: 0` standalone example
does not describe this service arrangement. The campaign's startup/scheduling
wrapper is an experimental execution aid, not a production allocator or SDK
fix, and does not establish that every service launch has this context setup.

## Real allocator and request results

- The real service MixedSHM allocator passed the 18 MiB A/B content gate using
  actual MemoryObjs. Restored bytes matched both patterns. This is distinct
  from the standalone paged-allocator gate documented earlier.
- The allocator/object checks recorded refcount 1 after the completed operation,
  zero final lookup pins and passing memory accounting checks. These are the
  observed object-lifetime and allocator checks, not proof of absence of every
  native resource leak.
- A real Qwen3-8B TP1 request had 1373 input tokens, 1280 external/restored
  tokens, 93 computed tokens and zero local-prefix tokens. The returned content
  was ORCHID. The traced single request completed 180 MiB of payload transfer;
  the additional functional group recorded 540 MiB. Preserve the
  two groups separately rather than treating their totals as a throughput test.
- Actual pin/unpin and the synchronous BM calls were observed under the
  separated logical-0/logical-2 contexts. This does not substitute for actual
  per-request L1/L2 source accounting in a natural-cache performance campaign.

These gates establish the recorded MixedSHM and TP1 functional paths only.
They do not establish long-duration reliability, capacity eviction, automatic
recovery, or an end-to-end performance advantage.

## Exit evidence and remaining failures

The successful functional group used planned SIGINT shutdown. Actual waits
recorded service exit 130, pool-member exit 0 and coordinator exit -2; no
SIGKILL was needed in that group. The recorded signals are not rewritten as
all-zero natural exits. The SDK `free sq_cq` return -8 warning remains unresolved;
successful payload and planned shutdown do not demonstrate warning-free teardown.

In the separate Mooncake Store comparison, NPU SDK 0.3.11.post1 service and
owner teardown produced SIGSEGV and SIGABRT respectively. Loading that SDK
without a model or LMCache also reproduced SIGABRT; its dependency teardown
stack is a crash site, not an established original corruption cause. These
failures are not MemFabric payload failures and no SDK teardown fix is claimed.
The historical MemFabric pool-member SIGSEGV and default-SDMA cross-host DRAM
failure remain separate unresolved records; explicit DEVICE_RDMA success does
not erase them.

## Comparison status

The normal-service comparison uses the existing Mooncake Store L2 backend
(`mooncake-transfer-engine-npu` 0.3.11.post1) and MemFabric L2 with SDK 1.2.1.
Three natural-cache cases formed complete paired measurements; the long-output
revisit failed and the remaining queue stopped. See the
[sanitized comparison](REGISTERED_L2_RDMA_COMPARISON_20261101.md) for measured
values, exclusions and unresolved failures. Standalone
API-stage observations, controlled read-pool checks and worker-direct results
must not replace the normal-service comparison or its actual source accounting.
