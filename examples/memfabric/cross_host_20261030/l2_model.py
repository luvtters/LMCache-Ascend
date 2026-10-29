"""Foreground supervisor for same-host P/D/service and a remote BM rank1.

The other host's member is controlled externally. This supervisor never SSHes,
starts a member, changes model files, or restarts a failed attempt.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request

from l2_config import adapter_config, model_expectations, read_config, write_json


def commands(cfg):
    py, ports = cfg["python"], cfg["ports"]
    l2 = adapter_config(cfg)
    service = [py, "-u", str(Path(__file__).resolve()), "--config", cfg["config_path"],
               "--role", "service"]
    connector = dict(kv_connector="LMCacheMPConnector", kv_role="kv_both",
                     kv_connector_extra_config={"lmcache.mp.host": "tcp://127.0.0.1",
                        "lmcache.mp.port": ports["rpc"], "lmcache.mp.mp_transfer_mode": "lmcache_driven"})
    engine = [py, "-m", "vllm.entrypoints.cli.main", "serve", cfg["model_path"],
         "--served-model-name", cfg.get("served_model_name", "mf-cross-host"),
         "--tensor-parallel-size", str(cfg["tp_size"]), "--seed", "1024", "--dtype", "bfloat16",
         "--max-model-len", "32768", "--max-num-batched-tokens", "1024", "--max-num-seqs", "4",
         "--gpu-memory-utilization", str(cfg.get("gpu_memory_utilization", 0.7)),
         "--block-size", "128", "--enforce-eager", "--no-enable-prefix-caching",
         "--host", "127.0.0.1", "--kv-transfer-config", json.dumps(connector)]
    proxy = [py, "-u", str(Path(cfg["core_path"]) / "examples/disagg_prefill_mp/disagg_proxy_server.py"),
         "--host", "127.0.0.1", "--port", str(ports["proxy"]),
         "--prefiller-host", "127.0.0.1", "--prefiller-port", str(ports["p"]),
         "--decoder-host", "127.0.0.1", "--decoder-port", str(ports["d"]),
         "--telemetry-port", str(ports["telemetry"])]
    gate = [py, "-u", str(Path(__file__).with_name("l2_gate.py")), "--config", cfg["config_path"]]
    return dict(service=service, p=engine + ["--port", str(ports["p"])],
                d=engine + ["--port", str(ports["d"])], proxy=proxy, gate=gate), l2


def service_main(cfg):
    # NPU IPC import and BM main-thread device selection match the existing
    # service wrapper; BM itself still owns its executor thread and set_device.
    import torch
    import torch_npu  # noqa: F401
    from lmcache.cli.main import main
    logical = cfg["service_bm_device"]
    torch.npu.set_device(logical)
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtGetPhyDevIdByLogicDevId.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    physical = ctypes.c_int(-1)
    code = acl.aclrtGetPhyDevIdByLogicDevId(logical, ctypes.byref(physical))
    assert code == 0 and physical.value == cfg["local_devices"]["service"][logical]
    print("SERVICE_TRANSPORT_DEVICE " + json.dumps(dict(logical=logical, physical=physical.value)), flush=True)
    p = cfg["ports"]
    main(["server", "--host", "127.0.0.1", "--port", str(p["rpc"]),
          "--http-host", "127.0.0.1", "--http-port", str(p["http"]),
          "--chunk-size", "128", "--l1-size-gb", str(cfg.get("l1_size_gb", 8)),
          "--eviction-policy", "LRU", "--supported-transfer-mode", "lmcache_driven",
          "--max-gpu-workers", str(2 * cfg["tp_size"]), "--prometheus-port", str(p["metrics"]),
          "--l2-store-policy", "skip_l1", "--l2-prefetch-policy", "default",
          "--l2-adapter", json.dumps(adapter_config(cfg))])


def supervise(cfg, plan_only=False):
    cmds, l2 = commands(cfg)
    devices = cfg["local_devices"]
    tp = cfg["tp_size"]
    if len(devices["p"]) != tp or len(devices["d"]) != tp:
        raise ValueError("Each P/D device list must have tp_size physical devices")
    if set(devices["p"]) & set(devices["d"]):
        raise ValueError("P and D physical devices must be disjoint")
    if not (set(devices["p"]) | set(devices["d"])) <= set(devices["service"]):
        raise ValueError("Service must see every local P/D device for NPU IPC")
    bm_device = devices["service"][cfg["service_bm_device"]]
    if bm_device in devices["p"] + devices["d"]:
        raise ValueError("Use a separate authorized BM physical device")
    plan = dict(commands=cmds, l2=l2, expected=model_expectations(tp), devices=devices,
                remote_member="EXTERNAL; start near service, not after READY",
                status="PLAN_ONLY_NOT_EXECUTED" if plan_only else "STARTING")
    if plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    out = Path(cfg["case_root"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "manifest.json", dict(config=cfg, **plan))
    for port in [cfg["store_port"], *cfg["ports"].values()]:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    procs, records = {}, {}
    stop = False

    def requested(signum, frame):
        nonlocal stop
        stop = True
        write_json(out / "supervisor_signal.json", dict(signal=signum, unix_ns=time.time_ns()))

    signal.signal(signal.SIGTERM, requested)
    signal.signal(signal.SIGINT, requested)

    def launch(name, role_devices, extra=None):
        env = dict(os.environ)
        for key in ("LMCACHE_REQUEST_TELEMETRY_TYPE", "LMCACHE_REQUEST_TELEMETRY_ENDPOINT"):
            env.pop(key, None)
        explicit = dict(ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, role_devices)),
                OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", TORCH_DEVICE_BACKEND_AUTOLOAD="1",
                PYTHONPATH=os.pathsep.join([cfg["core_path"], cfg["ascend_path"], env.get("PYTHONPATH", "")]))
        explicit.update(extra or {})
        env.update(explicit)
        argv = cmds[name]
        if cfg.get("env_file"):
            argv = ["bash", "-c", 'source "$1"; shift; exec "$@"', "l2-env", cfg["env_file"],
                    "env", *[key + "=" + value for key, value in explicit.items()], *argv]
        with (out / (name + ".log")).open("xb") as stream:
            proc = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        procs[name] = proc
        records[name] = dict(pid=proc.pid, argv=argv, explicit_env=explicit, started_unix_ns=time.time_ns())
        write_json(out / "processes.json", records)

    def ready(port, path, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if stop:
                raise InterruptedError("Supervisor stop requested")
            failed = {n: p.poll() for n, p in procs.items() if p.poll() is not None}
            if failed:
                raise RuntimeError(("Child exited before readiness", failed))
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d%s" % (port, path), timeout=2) as response:
                    if response.status == 200:
                        return
            except (OSError, ValueError):
                pass
            time.sleep(0.5)
        raise TimeoutError((port, path))

    exit_code = 1
    try:
        launch("service", devices["service"])
        ready(cfg["ports"]["http"], "/healthcheck", cfg.get("service_timeout", 200))
        telemetry = "http://127.0.0.1:%d/api/v1/telemetry" % cfg["ports"]["gate"]
        launch("p", devices["p"], dict(LMCACHE_REQUEST_TELEMETRY_TYPE="fastapi",
                                       LMCACHE_REQUEST_TELEMETRY_ENDPOINT=telemetry))
        launch("d", devices["d"])
        ready(cfg["ports"]["p"], "/health", cfg.get("model_timeout", 600))
        ready(cfg["ports"]["d"], "/health", cfg.get("model_timeout", 600))
        launch("proxy", [bm_device])
        ready(cfg["ports"]["proxy"], "/v1/models", 30)
        launch("gate", [bm_device])
        deadline = time.monotonic() + cfg.get("gate_timeout", 240)
        while procs["gate"].poll() is None:
            if stop or time.monotonic() > deadline:
                raise TimeoutError("Gate interrupted or timed out")
            failed = {n: p.poll() for n, p in procs.items() if n != "gate" and p.poll() is not None}
            if failed:
                raise RuntimeError(("Role exited while gate running", failed))
            time.sleep(0.2)
        exit_code = procs["gate"].wait()
        if exit_code:
            raise RuntimeError(("Gate failed", exit_code))
    except BaseException:
        write_json(out / "failure.json", dict(traceback=traceback.format_exc(), unix_ns=time.time_ns()))
        exit_code = 1
    finally:
        # Keep service/BM alive until P/D and their native transfer contexts exit.
        for name in ("gate", "proxy", "d", "p", "service"):
            if name not in procs:
                continue
            proc = procs[name]
            sent = []
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                sent.append("SIGINT")
            try:
                result = proc.wait(timeout=cfg.get("stop_timeout", 45))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                sent.append("SIGKILL")
                result = proc.wait()
            records[name].update(wait_returncode=result, signals_sent=sent,
                                 waited_unix_ns=time.time_ns())
            write_json(out / "processes.json", records)
        write_json(out / "supervisor_result.json", dict(gate_exit=exit_code, roles=records,
                     lifecycle="REVIEW_ACTUAL_WAIT_CODES_AND_REMOTE_MEMBER_SEPARATELY"))
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--role", choices=["supervisor", "service"], default="supervisor")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    cfg = read_config(args.config)
    cfg["config_path"] = str(Path(args.config).resolve())
    for entry in (cfg["core_path"], cfg["ascend_path"]):
        sys.path.insert(0, entry)
    if args.role == "service":
        service_main(cfg)
    else:
        raise SystemExit(supervise(cfg, args.plan_only))
