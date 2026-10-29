#!/usr/bin/env python3
"""Container-only night experiment control; plans by default, no SSH.

start --kind synthetic|capacity|C|B|A --case NAME --run launches persistent supervisors.
status is read-only. stop --run sends only recorded-child SIGINT, without
automatic TERM/KILL escalation. New fixed BM groups always need new cases.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import subprocess
import sys
import time


ROOT = PurePosixPath(os.environ.get("MF_EXPERIMENT_ROOT", "/path/to/private/memfabric-worker-direct"))
CORE = ROOT / "src/LMCache"
ASCEND = ROOT / "src/LMCache-Ascend"
RUNTIME = ROOT / "runtime"
PY = os.environ.get("MF_EXPERIMENT_PYTHON", sys.executable)
ENV = ROOT / "env.sh"
MODEL = PurePosixPath(os.environ.get("MF_MODEL", "/path/to/read-only/Qwen3-8B"))
BEGIN = datetime.fromisoformat("2026-10-29T01:20:00+08:00").timestamp()
NEW_WORK_END = datetime.fromisoformat("2026-10-29T05:30:00+08:00").timestamp()
PORTS = dict(service=25595, http=28195, bm=28695, p=28401, d=28402,
             proxy=28400, proxy_telemetry=25808, gate_telemetry=25809,
             p_counter=29401, d_counter=29402)


def case_path(name):
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) is None:
        raise ValueError("Use a simple unique case name")
    return ROOT / name


def build_plan(kind, name, trace=False, pool_gb=1, diagnostic=False):
    if type(pool_gb) is not int or pool_gb < 1:
        raise ValueError("pool_gb must be a positive integer")
    if diagnostic and (kind != "C" or trace):
        raise ValueError("diagnostic is only supported for C and is mutually exclusive with trace")
    if kind == "capacity" and pool_gb != 1:
        raise ValueError("capacity uses a fixed 1 GiB pool with two 512 MiB slots")
    case = case_path(name)
    jobs, configs = {}, {}
    store_url = f"tcp://127.0.0.1:{PORTS['bm']}"
    prefix = f"{RUNTIME}:{ASCEND}:{CORE}"

    def job(role, device, argv, **extra):
        env = dict(ASCEND_RT_VISIBLE_DEVICES=str(device), OMP_NUM_THREADS="1",
                   TOKENIZERS_PARALLELISM="false", VLLM_WORKER_MULTIPROC_METHOD="spawn",
                   TORCH_DEVICE_BACKEND_AUTOLOAD="0", LMCACHE_REQUEST_TELEMETRY_TYPE="noop",
                   LMCACHE_REQUEST_TELEMETRY_ENDPOINT="")
        env.update(extra)
        jobs[role] = dict(case=str(case), role=role, argv=[str(value) for value in argv],
                          env=env, env_script=str(ENV), pythonpath_prefix=prefix)

    service = [PY, "-u", RUNTIME / "service_entry.py", "server", "--host", "127.0.0.1",
               "--port", PORTS["service"], "--http-host", "127.0.0.1",
               "--http-port", PORTS["http"], "--chunk-size", "128", "--l1-size-gb", "1",
               "--eviction-policy", "LRU", "--supported-transfer-mode", "lmcache_driven",
               "--max-gpu-workers", "2"]
    direct = kind in ("synthetic", "capacity", "C")
    if direct:
        module = dict(module_path="lmcache_ascend.v1.distributed.memfabric_direct_server",
                      config=dict(device_id=0, rank_id=0, world_size=3, pool_gb=pool_gb,
                                  slot_size_mb=512 if kind == "capacity" else 32,
                                  init_timeout=300, store_url=store_url))
        configs["server_module.json"] = module
        service += ["--server-module", json.dumps(module)]
        job("service", 14, service)
    else:
        if kind == "B":
            adapter = dict(type="plugin", module_path="lmcache_ascend.v1.distributed.memfabric_l2_adapter",
                           class_name="MemFabricL2Adapter", adapter_params=dict(
                               device_id=2, world_size=2, pool_gb=pool_gb, storage_rank=1,
                               slot_size_mb=32, store_url=store_url))
            configs["l2.json"] = adapter
            service += ["--l2-store-policy", "skip_l1", "--l2-prefetch-policy", "default",
                        "--l2-adapter", json.dumps(adapter)]
        job("service", "8,10,12,14", service, TORCH_DEVICE_BACKEND_AUTOLOAD="1")
        if kind == "B":
            job("member", 14, [PY, "-u", ASCEND / "examples/memfabric/pool_member.py",
                                "--device", "0", "--rank", "1", "--world-size", "2",
                                "--pool-gb", str(pool_gb), "--store-url", store_url])
    if kind in ("synthetic", "capacity"):
        if kind == "capacity":
            common = [PY, "-u", RUNTIME / "pool_cases.py", "--mode", "context",
                      "--coord-dir", case / "synthetic", "--run-id", "objects01", "--device", "0",
                      "--ascend-source", ASCEND, "--core-source", CORE,
                      "--server-url", f"tcp://127.0.0.1:{PORTS['service']}", "--store-url", store_url,
                      "--pool-gb", "1", "--timeout", "180", "--run", "--allow-npu"]
        else:
            common = [PY, "-u", ASCEND / "examples/memfabric/validate_direct_kv.py",
                  "--mode", "context", "--layout", "split-nhd", "--block-size", "64",
                  "--num-blocks", "6", "--source-blocks", "3,1", "--target-blocks", "4,2",
                  "--skip-blocks", "1", "--device", "0", "--ascend-source", ASCEND,
                  "--core-source", CORE, "--coord-dir", case / "synthetic", "--run-id", "skip01",
                  "--store-url", store_url, "--server-url", f"tcp://127.0.0.1:{PORTS['service']}",
                  "--pool-gb", str(pool_gb), "--timeout", "180", "--allow-npu"]
        for role, device in (("p", 8), ("d", 10)):
            job(role, device, [*common, "--role", role])
    else:
        for role, rank, device in (("p", 1, 8), ("d", 2, 10)):
            extra_config = {"lmcache.mp.host": "tcp://127.0.0.1", "lmcache.mp.port": PORTS["service"],
                            "lmcache.mp.mp_transfer_mode": "auto" if direct else "lmcache_driven",
                            "lmcache.mp.mq_timeout": 180.0}
            connector = dict(kv_connector="MemFabricDirectConnector" if direct else "LMCacheMPConnector",
                             kv_role="kv_both" if role == "p" else "kv_consumer",
                             kv_connector_extra_config=extra_config)
            extra_env = {"LMCACHE_MP_TRANSFER_MODE": "auto" if direct else "lmcache_driven"}
            if direct:
                connector["kv_connector_module_path"] = (
                    "traced_direct_connector" if trace else
                    "diagnostic_direct_connector" if diagnostic else "counted_direct_connector")
                extra_config["lmcache.mp.direct"] = dict(
                    device_id=0, rank_id=rank, world_size=3, pool_gb=pool_gb,
                    init_timeout=300, store_url=store_url)
                if trace:
                    extra_env.update(MF_DIRECT_TRACE_DIR=str(case / "worker-traces"),
                                     MF_DIRECT_TRACE_ROLE=role.upper())
                else:
                    extra_env.update(MF_DIRECT_COUNTER_ROLE=role.upper(),
                                     MF_DIRECT_COUNTER_PORT=str(PORTS[role + "_counter"]))
            if role == "p":
                extra_env.update(LMCACHE_REQUEST_TELEMETRY_TYPE="fastapi",
                                 LMCACHE_REQUEST_TELEMETRY_ENDPOINT=(
                                     f"http://127.0.0.1:{PORTS['gate_telemetry']}/api/v1/telemetry"))
            configs[role + ".json"] = connector
            argv = [PY, "-m", "vllm.entrypoints.cli.main", "serve", MODEL,
                    "--served-model-name", "mf-night", "--tensor-parallel-size", "1", "--seed", "1024",
                    "--dtype", "bfloat16", "--max-model-len", "4096", "--max-num-batched-tokens", "1024",
                    "--max-num-seqs", "4", "--block-size", "128", "--gpu-memory-utilization", "0.7",
                    "--enforce-eager", "--no-enable-prefix-caching", "--host", "127.0.0.1",
                    "--port", PORTS[role], "--kv-transfer-config", json.dumps(connector)]
            job(role, device, argv, **extra_env)
        job("proxy", 14, [PY, "-u", CORE / "examples/disagg_prefill_mp/disagg_proxy_server.py",
                            "--host", "127.0.0.1", "--port", PORTS["proxy"],
                            "--prefiller-host", "127.0.0.1", "--prefiller-port", PORTS["p"],
                            "--decoder-host", "127.0.0.1", "--decoder-port", PORTS["d"],
                            "--telemetry-port", PORTS["proxy_telemetry"]])
    needed_ports = [PORTS["service"], PORTS["http"]]
    if kind != "A":
        needed_ports.append(PORTS["bm"])
    if kind not in ("synthetic", "capacity"):
        needed_ports += [PORTS[key] for key in ("p", "d", "proxy", "proxy_telemetry", "gate_telemetry")]
        if kind == "C" and not trace:
            needed_ports += [PORTS["p_counter"], PORTS["d_counter"]]
    return dict(kind=kind, case=str(case), trace=trace, configs=configs, jobs=jobs,
                ports=needed_ports, model_mutation=False, start_waits_for_model=False)


def load_case(name):
    case = Path(str(case_path(name))).resolve()
    root = Path(str(ROOT)).resolve()
    if case.parent != root:
        raise ValueError("Case must be an immediate child of the experiment root")
    plan = json.loads((case / "plan.json").read_text(encoding="utf-8"))
    configs = plan["configs"]
    pool_gb = (configs.get("server_module.json", {}).get("config", {}).get("pool_gb")
               or configs.get("l2.json", {}).get("adapter_params", {}).get("pool_gb") or 1)
    diagnostic = (configs.get("p.json", {}).get("kv_connector_module_path") == "diagnostic_direct_connector")
    expected = build_plan(plan["kind"], name, plan["trace"], pool_gb=pool_gb, diagnostic=diagnostic)
    if plan != expected:
        raise ValueError("Saved plan differs from this fixed experiment entry")
    for role, spec in plan["jobs"].items():
        saved = json.loads((case / "specs" / (role + ".json")).read_text(encoding="utf-8"))
        if saved != spec:
            raise ValueError("Saved supervisor spec differs: " + role)
        record_path = case / (role + ".process.json")
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("role") != role or record.get("case") != str(case) or record.get("argv") != spec["argv"]:
                raise ValueError("Child record does not match saved spec: " + role)
    return case, plan


def status(name):
    import supervisor
    case, plan = load_case(name)
    result = dict(case=str(case), kind=plan["kind"], roles={})
    for role in plan["jobs"]:
        record_path = case / (role + ".process.json")
        record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else None
        current = supervisor.proc_state(record["child_pid"]) if record else None
        item = dict(process=record, current=current,
                    running=bool(current and current["state"] != "Z"),
                    identity_matches=bool(current and current["proc_start_ticks"] == record["proc_start_ticks"]))
        for suffix in ("exit.json", "supervisor-exit.json"):
            path = case / (role + "." + suffix)
            if path.exists():
                item[suffix] = json.loads(path.read_text(encoding="utf-8"))
        result["roles"][role] = item
    return result


def start(plan):
    if not BEGIN <= time.time() < NEW_WORK_END:
        raise RuntimeError("Outside the explicit new-experiment window")
    case = Path(plan["case"])
    if case.exists():
        raise FileExistsError("Fresh case required; never overwrite or resume a poisoned BM group")
    for required in (Path(PY), Path(str(ENV)), Path(str(RUNTIME / "supervisor.py"))):
        if not required.is_file():
            raise FileNotFoundError(str(required))
    for port in plan["ports"]:
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                listener.bind(("127.0.0.1", port))
            except OSError as error:
                raise RuntimeError(f"Experiment port {port} is unavailable: {error}") from error
    case.mkdir(parents=True)
    (case / "specs").mkdir()
    (case / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    for name, config in plan["configs"].items():
        (case / name).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    launched = {}
    inherited = os.environ.copy()
    for key in ("LMCACHE_REQUEST_TELEMETRY_TYPE", "LMCACHE_REQUEST_TELEMETRY_ENDPOINT"):
        inherited.pop(key, None)
    for role, spec in plan["jobs"].items():
        spec_path = case / "specs" / (role + ".json")
        spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        with (case / (role + ".supervisor.log")).open("xb") as log:
            child = subprocess.Popen([PY, str(RUNTIME / "supervisor.py"), "run", "--spec", str(spec_path)],
                                     env=inherited, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        launched[role] = dict(supervisor_pid=child.pid, spec=str(spec_path))
        (case / "launched.json").write_text(json.dumps(launched, indent=2) + "\n", encoding="utf-8")
    return dict(status="SUPERVISORS_STARTED_NOT_YET_READY", case=str(case), launched=launched)


def stop(name, roles, wait_seconds):
    import supervisor
    case, plan = load_case(name)
    chosen = roles or [role for role in ("proxy", "p", "d", "service", "member") if role in plan["jobs"]]
    if set(chosen) - plan["jobs"].keys():
        raise ValueError("Unknown role")
    sent = []
    phases = [chosen] if roles else [
        [role for role in ("proxy", "p", "d") if role in chosen],
        [role for role in ("service",) if role in chosen],
        [role for role in ("member",) if role in chosen],
    ]
    for phase in phases:
        # Preserve the parent wait() observer; only its child is signalled.
        for role in phase:
            if (case / (role + ".process.json")).exists():
                supervisor.stop(str(case), role)
                sent.append(role)
        deadline = time.monotonic() + wait_seconds
        while True:
            result = status(name)
            alive = [role for role in phase if result["roles"][role]["running"]]
            if not alive:
                break
            if time.monotonic() >= deadline:
                return dict(signalled=sent, still_running=alive, forced_signal_sent=False,
                            unrequested_roles=[role for role in chosen if role not in sent], state=result)
            time.sleep(.2)
    return dict(signalled=sent, still_running=[], forced_signal_sent=False, state=status(name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "stop"))
    parser.add_argument("--kind", choices=("synthetic", "capacity", "C", "B", "A"), default="C")
    parser.add_argument("--case", required=True)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--pool-gb", type=int, default=1)
    parser.add_argument("--roles", help="Comma-separated subset for an explicit shutdown order")
    parser.add_argument("--wait-seconds", type=float, default=30)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.action == "status":
        result = status(args.case)
    elif args.action == "start":
        plan = build_plan(args.kind, args.case, args.trace, pool_gb=args.pool_gb, diagnostic=args.diagnostic)
        result = start(plan) if args.run else dict(mode="PLAN_ONLY", plan=plan)
    else:
        case, plan = load_case(args.case)
        roles = args.roles.split(",") if args.roles else None
        result = stop(args.case, roles, args.wait_seconds) if args.run else dict(
            mode="PLAN_ONLY", case=str(case), roles=roles or list(plan["jobs"]), signal="SIGINT")
    print(json.dumps(result, indent=2))
    return 2 if result.get("still_running") else 0


if __name__ == "__main__":
    raise SystemExit(main())
