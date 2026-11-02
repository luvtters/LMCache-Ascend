# TP1 Qwen3-8B native-core L2 model gate

This is one controlled correctness request through the official core P/D proxy.
It is not a performance comparison. The script does not patch core, host pinning,
allocator or transfer implementations. Core remains official `6913259`.

Copy `l2_model_gate.example.json` to a fresh private case configuration and set
the actual communication IPv4 in `device_nic` (`tcp://IPv4:port`), store URL,
fresh join group and coordination URL. Each rank uses its own unused NIC port.
The default configuration fixes P=8, D=10, service=[8,10,12], service BM logical=2.
Choose an independently available physical device for the pool member; its process-local device is logical0. Choose one
matching protocol pair in the L2 adapter and `pool_member` configuration:

| Protocol | `data_op_type` / member `transport` | `enable_56bits_gva` | Member CLI flag |
|---|---|---:|---|
| SDMA | `SDMA` | `false` | omitted |
| Device RDMA | `DEVICE_RDMA` | `true` | `--enable-56bits-gva` |

The gate rejects other protocols and mismatched GVA settings. The controller
also rejects a member transport that differs from the L2 adapter, and checks
that service status reports the configured protocol. Start the existing
`pool_coordination.py` and `pool_member.py` entry points under the caller's
bounded supervisor. Member and service must join concurrently.

After sourcing the existing private environment, inspect the role command plan:

```bash
python examples/memfabric/l2_model_gate.py --config /private/case.json
```

The caller launches these roles as tracked process groups, redirects each to
`case_root/<role>.log`, records actual wait return codes, and owns deadline cleanup:

```bash
python examples/memfabric/l2_model_gate.py --config /private/case.json --role service
# Wait for service /healthcheck, then start P and D in separate tracked groups:
python examples/memfabric/l2_model_gate.py --config /private/case.json --role p
python examples/memfabric/l2_model_gate.py --config /private/case.json --role d
# Wait for both /health endpoints:
python examples/memfabric/l2_model_gate.py --config /private/case.json --role proxy
# Wait for proxy /v1/models, then the single-request gate:
python examples/memfabric/l2_model_gate.py --config /private/case.json --role gate
```

The gate starts its telemetry relay before making its only inference call to the
official proxy. P telemetry is held until native `skip_l1` commits all objects and
automatically drains L1. The relay then notifies the unchanged official proxy.
There is no manual L1 clearing. Each complete Qwen3-8B BF16 128-token chunk is
18 MiB. The gate checks actual prompt tokens against object count and byte totals,
D external/local-compute token metrics, exactly one D prefetch record with zero
L1 hits, matching D request ID and the literal ORCHID answer.

Successful `gate/summary.json` reports function pass only. Record service, P, D,
proxy, member and coordinator wait return codes separately. Unknown exits,
forced kills or peer cleanup faults remain lifecycle failures even if the answer
and byte checks passed. Do not restart after external termination.

The service explicitly establishes main-thread logical0 before calling official
CLI. Original native pin-memory code remains unchanged. If native host registration
chooses an unsuitable device/context or fails, preserve its original error and
record this model path as NOT_VERIFIED; do not copy historical hostpin monkeypatches.

## Optional vLLM shutdown comparison

For an explicit process-shutdown comparison, add a top-level `shutdown_timeout`
integer to the case JSON. The gate validates it as a nonnegative integer and
passes `--shutdown-timeout N` to both P and D. Value `0` selects vLLM's abort
behavior; a positive value gives vLLM that many seconds to drain before aborting.
When the field is absent, the generated P/D argv is unchanged and vLLM receives
no shutdown-timeout option. The outer supervisor must still wait for each role
and record its actual exit status; this setting does not replace the group
deadline or forced-cleanup record.

## Additional requests on the resident service

After the strict gate succeeds, keep the same service and P/D processes running
and execute the source-accounting cases once in that fresh case directory:

```bash
python examples/memfabric/l2_basic_cases.py --config /private/case.json
```

The script sends repeat, partial-prefix, short-cold and two same-prefix requests,
then a multi-chunk cold request, its repeat after completed L2 writes, and two
different-prefix concurrent requests. It records original HTTP responses,
tokenizer inputs, metrics, service status samples and the service log additions.
It starts no service and performs no manual cache clearing. The Qwen3-8B fixture
requires chunk128 and a 1 GiB pool with 32 MiB slots; one full chunk holds 18 MiB.

The first four cases check answers and source accounting, and retain the observed
hit counts rather than requiring every request to hit. In particular, a same-key
request may recompute after an L1 reservation conflict even when L2 lookup hits.
The multi-chunk cold/repeat and different-prefix cases additionally assert their
expected restore/compute totals. A script success does not establish concurrent
BM copies, a performance benefit or clean process shutdown. Preserve every role's
real wait result separately.
