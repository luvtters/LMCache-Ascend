# MemFabric L2 explicit failure candidate

Adapter code: `137038a`, based on the experimental selected-pool adapter
`23dba90`. Paired LMCache core: `fa2e6ff`.

Lookup/load on a poisoned session or an observed departed peer raise explicitly.
A BM copy exception retains its buffer reference in quarantine until BM close.
Capacity exhaustion continues to return a failed store with zero bytes while
the adapter remains healthy.

## Selected validation, 2026-10-30

- Four CPU checks passed using the actual adapter, native bitmap/eventfd path,
  and a fake BM session: peer departure, existing poison, copy failure with
  quarantine lifetime, and locked-capacity failure without poison.
- Paired real-NPU Qwen3-8B TP1/TP4 normal gates passed: 1,373/1,280/93/0
  prompt/external/computed/local-prefix tokens; 180 MiB per direction.
- A successful gate followed by controlled idle-member departure caused decoder
  HTTP 500 and an explicit `RemoteRequestError` naming the departed peer. Store
  and load counters did not increase beyond 188,743,680 bytes; pins remained zero.
  Probe, driver, member, prefiller, and decoder API processes exited zero; service
  exit 130 followed the supervisor's controlled stop. EngineCore nevertheless
  failed closed with `EngineDeadError`; request-level recovery is not claimed.

Deploy the matching ZMQ client/server core together. The ordinary success path
remains intact. gRPC, a total request deadline, native DMA hangs, and performance
are outside this result. Internal raw logs, addresses, and credentials are not
included in this archive.
