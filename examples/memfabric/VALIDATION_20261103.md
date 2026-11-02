# Cross-host L2 validation summary

This selected validation summary covers compatibility L2 only. It separates model correctness, controlled-read performance, transfer-level measurements, and process shutdown. The canonical short- and long-output sequences and a sequential TP8 case completed; experiment resources were released after raw evidence was retrieved.

## Fixed source versions

- LMCache core: `57066a5e995891edc3318a5740f0fb087082a038`.
- LMCache-Ascend: `0964c2d685a6dcebad8230052f20912ad1358004`.

The candidate above was exercised in dedicated TP1, TP2 and TP4 runs. Earlier per-object-registration and intermediate whole-arena runs retain their own source identifiers and are not retroactively attributed to the final candidate. Native extensions were reused; their exact build commit was not independently established.

## Functional evidence

Cross-host Host-DRAM L2 functional runs completed for TP1, TP2, and TP4. Each direction transferred 180 MiB between the inference side and a separate host-memory pool. The TP4 strict record covers four KV ranks, ten 4.5 MiB objects per rank, 40 objects total, and 188,743,680 bytes per direction. Its read path reported zero L1 objects and 40 L2 objects. Per-rank store/load keys, chunk counts, and byte counts matched. The test also recorded a model response and separate external-transfer and local-compute token counts.

A sequential TP8 case then reused the same eight inference devices: P completed its stores, exited, and received eight matching KV IPC-unregistration acknowledgements before D started. The service and remote Host-DRAM pool remained alive across the handoff. Native service logs recorded 80 stores and 80 loads, each 2,359,296 bytes, with eight KV ranks and ten objects per rank: 188,743,680 bytes per direction. The prefetch log reported zero L1 and 80 L2 objects. D answered the request and restored 1,280 prompt tokens, computed 93 locally, and reported zero local-prefix hits. This is not a concurrent TP8 1P1D run, and model P/D remained on the same physical host.

Functionality and lifecycle are separate results. Planned signal-driven shutdowns and SDK close warnings were retained in the records; a successful request does not imply warning-free shutdown.

In the sequential TP8 run, the real P/D parent waits were zero, service exited with 130 after the recorded SIGINT, the remote member exited zero, and the coordinator's parent collected a zero wait. Model shutdown still logged internal forced process cleanup, leaked semaphore/shared-memory warnings, and the SDK `free sq_cq` warning. Experiment device containers were subsequently stopped deliberately, with container exit 137 retained separately from model-process results; deadline guards also exited and resource queries confirmed release.

## Implementation scope

The candidate adds a separate stable host-registration size to the memory descriptor while preserving the previous registration-size field semantics. The Ascend L1-arena path is opt-in and defaults off. It reuses registration for objects fully contained within the verified arena, retains object references while work is in flight, leaves external objects on the established per-object path, rejects partial overlap, and avoids freeing ranges when completion is unknown. The same candidate supports optional on-demand registration in 32 MiB segments. These implementation properties do not by themselves establish an end-to-end model speedup.

## Performance status and limits

Raw transfer tests used 1, 18 and 64 MiB payloads. In the 18 MiB comparison, persistent registration brought MemFabric steady-state transfer time to about 0.89-0.93 ms, close to the approximately 0.87 ms TE reference, while the per-object path took about 10.93 ms including repeated registration and unregistration. Segmented SDK batch-copy tests also reduced per-segment dispatch overhead at higher segment counts. These are transfer-level observations, not a complete model-serving result.

An earlier approximately 2x request-latency gap was confounded by different model logging settings. Reverse-order and logging-control observations support treating that gap as configuration-sensitive. The completed short-output canonical ABBA sequence used ten requests per independent run: Store proxy response medians were 376.741 and 394.514 ms; MemFabric medians were 347.591 and 382.963 ms. These are full non-streaming response times, not proxy TTFT. Both backends were in the same latency range. This small controlled-read sample does not establish a general advantage.

The separate 128-output-token ABBA sequence also used ten requests per run. In the first pair, Store/MemFabric full-response medians were 5842.630/5945.584 ms and D-worker mean TPOT was 44.934/45.486 ms. In the second pair, they were 6226.768/6975.909 ms and 46.806/54.268 ms. MemFabric D-prefetch means remained approximately 14.5 ms in both runs, compared with Store's 15.6 ms. The slower second MemFabric run affected its whole TPOT sample, so it was retained rather than discarded as an outlier. The remaining decode difference is unexplained; these measurements do not establish a stable end-to-end advantage. Cold-start latency, worker metrics, and end-to-end proxy latency remain separate.

Both remote pools had 1 GiB configured capacity and each request's D leg completed 180 MiB of L2 reads. The baseline used a run-scoped adapter around the existing NativeConnector/C++ Store client, with explicit stable-L1 registration and transport-device selection, plus completion-only logging. It was not an unmodified default Store launch. Independent native owner queries established remote placement. Warm measurements amortize registration on both sides; startup and registration costs are reported separately. Background activity on other allowed hosts/devices was not fully isolated, and allowed CPU/memory-node masks do not prove NUMA page placement.

## Reproducibility requirements

1. Use the same CANN and model logging environment for both sides. Explicitly set and record the agreed `ASCEND_GLOBAL_LOG_LEVEL`, `ASCEND_SLOG_PRINT_TO_STDOUT`, `HCCL_INTRA_ROCE_ENABLE`, and `TORCH_DEVICE_BACKEND_AUTOLOAD` values for service and model processes. Verify the effective environment after process launch from an allowlisted capture; do not infer it from the launch script alone. Logging levels must be normalized before comparing request latency because verbose device logs can change observed service time.
2. Separate cold and warm requests. Before a cold request, capture that the local L1 object count and in-flight queues are empty and that the owner has the expected remote objects. For warm requests, preserve request-by-request cache state and identify L1 hits, L2 hits, and misses from service records. Do not combine cold first-use setup with steady-state percentiles.
3. Keep the transfer protocol, payload bytes, object count, direction, concurrency, memory backing, and setup/teardown boundaries identical. Report setup and steady-state copy costs separately, and preserve every sample including outliers.
4. Validate remote placement and content through independent owner-side queries and full payload checks. A script-level PASS alone is insufficient evidence of remote data placement or complete model-path correctness.
5. Retain source revisions, launch settings, raw service and metric records, owner query results, process signals, and actual wait results for each run. Keep planned shutdowns and SDK warnings visible.

## Not established

- A natural-cache production-throughput conclusion or a read-pool-versus-recomputation result.
- A general performance advantage for either backend.
- Concurrent TP8 1P1D, model P/D on different hosts, production batch integration, or worker-direct success.
- Warning-free SDK shutdown or automatic multi-owner L2 capacity aggregation.
