# TP1 Qwen3-8B native-core L2 model gate

This is one controlled correctness request through the official core P/D proxy.
It is not a performance comparison. The script does not patch core, host pinning,
allocator or transfer implementations. Core remains official `6913259`.

Copy `l2_model_gate.example.json` to a fresh private case configuration and set
the actual communication IPv4 in `device_nic` (`tcp://IPv4:port`), store URL,
fresh join group and coordination URL. Each rank uses its own unused NIC port.
The default configuration fixes P=8, D=10, service=[8,10,12], service BM logical=2.
The external pool member uses physical14 / process-local logical0. The same
DEVICE_RDMA / 56-bit GVA / group / pool settings must be used by service and member.
Start the existing `pool_coordination.py` and `pool_member.py` entry points under
the caller's bounded supervisor. Member and service must join concurrently.

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
