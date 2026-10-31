"""Foreground supervisor for same-host P/D/service and a remote BM rank1.

The other host's member is controlled externally. This supervisor never SSHes,
starts a member, changes model files, or restarts a failed attempt.
"""
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request

from l2_config import adapter_config as original_adapter_config, model_expectations, read_config, write_json


def adapter_config(cfg):
    if "l2_config" in cfg:
        if not isinstance(cfg["l2_config"], dict):
            raise ValueError("l2_config must be a dictionary")
        return cfg["l2_config"]
    return original_adapter_config(cfg)

def validate_devices(cfg):
    visible = cfg["local_devices"]["service"]
    for key, value in (("service_context_device", cfg.get("service_context_device", cfg.get("service_pin_device", 0))), ("host_pin_device", cfg.get("host_pin_device", 0))):
        if type(value) is not int or not 0 <= value < len(visible):
            raise ValueError(key + " must index service visible devices")


def shutdown_budget(cfg):
    """One shared monotonic budget, shorter than the outer 120-second grace."""
    per_role = float(cfg.get("stop_timeout", 18))
    total = float(cfg.get("shutdown_timeout", 90))
    service = float(cfg.get("service_stop_timeout", 30))
    if not math.isfinite(service) or not 0 < service <= 90:
        raise ValueError("service_stop_timeout must be finite and in (0, 90] seconds")
    if not 0 < per_role <= 90 or not 0 < total <= 90:
        raise ValueError("stop_timeout and shutdown_timeout must be in (0, 90] seconds")
    return per_role, total, service


def classify_wait(returncode, signals_sent):
    """Preserve native wait codes; only our own SIGINT explains 130/-2."""
    if "SIGKILL" in signals_sent:
        return "FAILED_FORCED_KILL"
    if returncode == 0:
        return "NATURAL_ZERO" if not signals_sent else "CONTROLLED_ZERO"
    if returncode in (130, -2) and "SIGINT" in signals_sent:
        return "CONTROLLED_SIGINT"
    if returncode is None:
        return "FAILED_WAIT_NOT_COMPLETED"
    return "FAILED_UNEXPECTED_EXIT"


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
         "--max-model-len", str(cfg.get("max_model_len", 8192)), "--max-num-batched-tokens", "1024", "--max-num-seqs", str(cfg.get("max_num_seqs", 2)),
         "--gpu-memory-utilization", str(cfg.get("gpu_memory_utilization", 0.7)),
         "--block-size", "128", "--enforce-eager",
         "--host", "127.0.0.1", "--kv-transfer-config", json.dumps(connector)]
    if cfg.get("kv_cache_memory_bytes") is not None:
        engine += ["--kv-cache-memory-bytes", str(cfg["kv_cache_memory_bytes"])]
    proxy = [py, "-u", str(Path(cfg["core_path"]) / "examples/disagg_prefill_mp/disagg_proxy_server.py"),
         "--host", "127.0.0.1", "--port", str(ports["proxy"]),
         "--prefiller-host", "127.0.0.1", "--prefiller-port", str(ports["p"]),
         "--decoder-host", "127.0.0.1", "--decoder-port", str(ports["d"]),
         "--telemetry-port", str(ports["telemetry"])]
    gate = cfg["benchmark_command"]
    if not isinstance(gate, list) or not gate or not all(isinstance(x, str) for x in gate):
        raise ValueError("benchmark_command must be a nonempty explicit argv list")
    roles = dict(service=service, p=engine + ["--port", str(ports["p"])],
                 d=engine + ["--port", str(ports["d"])], proxy=proxy, gate=gate)
    return roles, l2


def service_main(cfg):
    # NPU IPC import and BM main-thread device selection match the existing
    # service wrapper; BM itself still owns its executor thread and set_device.
    import faulthandler
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    import torch
    import torch_npu  # noqa: F401
    from lmcache.cli.main import main
    validate_devices(cfg)
    logical = cfg.get("service_context_device", cfg.get("service_pin_device", 0))
    torch.npu.set_device(logical)
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtGetPhyDevIdByLogicDevId.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    physical = ctypes.c_int(-1)
    code = acl.aclrtGetPhyDevIdByLogicDevId(logical, ctypes.byref(physical))
    assert code == 0 and physical.value == cfg["local_devices"]["service"][logical]
    print("SERVICE_ALLOCATION_DEVICE " + json.dumps(dict(logical=logical, physical=physical.value)), flush=True)
    from lmcache.v1.platform.devices.npu.pin_memory import NpuPinMemoryBackend
    def wrap_host_registration(original, operation):
        def wrapped(self, *args, **kwargs):
            before = torch.npu.current_device()
            try:
                torch.npu.set_device(cfg.get("host_pin_device", 0))
                result = original(self, *args, **kwargs)
                try:
                    print("L1_ACTUAL_ACL_" + operation + " " + json.dumps(dict(
                        pointer=args[0] if args else kwargs.get("ptr"),
                        bytes=args[1] if len(args) > 1 else kwargs.get("size"),
                        success=result, logical_device=torch.npu.current_device(),
                        bm_logical_device=cfg["service_bm_device"])), flush=True)
                except Exception:
                    pass
                return result
            finally:
                torch.npu.set_device(before)
        return wrapped
    NpuPinMemoryBackend.pin_memory = wrap_host_registration(NpuPinMemoryBackend.pin_memory, "PIN")
    NpuPinMemoryBackend.unpin_memory = wrap_host_registration(NpuPinMemoryBackend.unpin_memory, "UNPIN")
    import runpy
    runpy.run_path(cfg["reservation_trace_path"])["install_reservation_trace"]()
    from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
    from lmcache.v1.platform.torch_ops.mem_alloc import _pinned_ptr_registry, _shm_registry
    original_allocate = MixedMemoryAllocator.__init__
    def trace_allocation(self, *args, **kwargs):
        original_allocate(self, *args, **kwargs)
        ptr=self.buffer.data_ptr()
        print('L1_ARENA_REGISTRATION '+json.dumps(dict(pointer=ptr,bytes=self.buffer.numel(),
            pin_logical_device=torch.npu.current_device(),bm_logical_device=cfg['service_bm_device'],
            pin_present=ptr in _pinned_ptr_registry,shm_present=ptr in _shm_registry)),flush=True)
    MixedMemoryAllocator.__init__ = trace_allocation
    p = cfg["ports"]
    main(["server", "--host", "127.0.0.1", "--port", str(p["rpc"]),
          "--http-host", "127.0.0.1", "--http-port", str(p["http"]),
          "--chunk-size", "128", "--l1-size-gb", str(cfg.get("l1_size_gb", 8)),
          "--eviction-policy", "LRU", "--supported-transfer-mode", "lmcache_driven",
          "--max-gpu-workers", str(2 * cfg["tp_size"]), "--prometheus-port", str(p["metrics"]),
          "--l2-store-policy", cfg.get("l2_store_policy", "default"), "--l2-prefetch-policy", "default",
          "--l2-adapter", json.dumps(adapter_config(cfg))])


def supervise(cfg, plan_only=False):
    validate_devices(cfg)
    per_role_stop, shutdown_total, service_stop = shutdown_budget(cfg)
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
            # Closed HTTP clients may leave TIME_WAIT after the preceding group.
            # Match the real servers' address reuse without accepting a listener.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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
                ASCEND_VISIBLE_DEVICES=",".join(map(str, role_devices)),
                OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", TORCH_DEVICE_BACKEND_AUTOLOAD="1",
                L2_SOURCE_PATH=os.pathsep.join([cfg["core_path"], cfg["ascend_path"]]))
        explicit.update(extra or {})
        env.update(explicit)
        argv = cmds[name]
        if cfg.get("env_file"):
            argv = ["bash", "-c", 'source "$1"; shift; export PYTHONPATH="$L2_SOURCE_PATH${PYTHONPATH:+:$PYTHONPATH}"; exec "$@"', "l2-env", cfg["env_file"],
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
    gate_exit = None
    followup_exit = None
    try:
        launch("service", devices["service"])
        ready(cfg["ports"]["http"], "/healthcheck", cfg.get("service_timeout", 200))
        telemetry = "http://127.0.0.1:%d/api/v1/telemetry" % cfg["ports"]["telemetry"]
        launch("p", devices["p"], dict(LMCACHE_REQUEST_TELEMETRY_TYPE="fastapi",
                                       LMCACHE_REQUEST_TELEMETRY_ENDPOINT=telemetry))
        launch("d", devices["d"])
        ready(cfg["ports"]["p"], "/health", cfg.get("model_timeout", 600))
        ready(cfg["ports"]["d"], "/health", cfg.get("model_timeout", 600))
        launch("proxy", [bm_device])
        ready(cfg["ports"]["proxy"], "/v1/models", 30)
        launch("gate", [bm_device])
        deadline = time.monotonic() + cfg.get("gate_timeout", 1800)
        while procs["gate"].poll() is None:
            if stop or time.monotonic() > deadline:
                raise TimeoutError("Gate interrupted or timed out")
            failed = {n: p.poll() for n, p in procs.items() if n != "gate" and p.poll() is not None}
            if failed:
                raise RuntimeError(("Role exited while gate running", failed))
            time.sleep(0.2)
        gate_exit = procs["gate"].wait()
        exit_code = gate_exit
        if exit_code:
            raise RuntimeError(("Gate failed", exit_code))
        if "followup" in cmds:
            launch("followup", [bm_device])
            deadline = time.monotonic() + cfg.get("followup_timeout", 90)
            while procs["followup"].poll() is None:
                if stop or time.monotonic() > deadline:
                    raise TimeoutError("Follow-up interrupted or timed out")
                # Negative probes may intentionally make D fail closed. Retain
                # its actual exit in the common lifecycle record below.
                time.sleep(.2)
            followup_exit = procs["followup"].wait()
            if followup_exit:
                raise RuntimeError(("Follow-up failed", followup_exit))
    except BaseException:
        write_json(out / "failure.json", dict(traceback=traceback.format_exc(), unix_ns=time.time_ns()))
        exit_code = 1
    finally:
        shutdown_start = time.monotonic()
        shutdown_deadline = shutdown_start + shutdown_total
        # Reserve a final bounded interval to reap any forcibly stopped groups.
        graceful_deadline = shutdown_deadline - min(5, shutdown_total / 4)
        lifecycle_errors = []

        signal_events = []

        def send(proc, signum, sent):
            event = dict(pid=proc.pid, signal=signal.Signals(signum).name, unix_ns=time.time_ns(), monotonic=time.monotonic(), poll_before=proc.poll(), outcome="attempt")
            signal_events.append(event)
            try:
                os.killpg(proc.pid, signum)
                event["outcome"] = "syscall_succeeded_delivery_not_verified"
                sent.append(signal.Signals(signum).name)
            except ProcessLookupError:
                event["outcome"] = "not_found"

        # Keep service/BM alive until P/D and their native transfer contexts exit.
        for name in ("followup", "gate", "proxy", "d", "p", "service"):
            if name not in procs:
                continue
            proc = procs[name]
            sent = []
            if proc.poll() is None:
                send(proc, signal.SIGINT, sent)
            try:
                result = proc.wait(timeout=min(service_stop if name == "service" else per_role_stop, max(0, graceful_deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                send(proc, signal.SIGKILL, sent)
                try:
                    result = proc.wait(timeout=min(1, max(0, shutdown_deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    result = None
            records[name].update(wait_returncode=result, signals_sent=sent,
                                 wait_completed=result is not None, waited_unix_ns=time.time_ns())
            write_json(out / "processes.json", records)
        # All remaining groups have already received SIGKILL. Reap using the
        # same deadline; an unreaped process remains an explicit failure.
        for name, proc in procs.items():
            if not records[name]["wait_completed"]:
                try:
                    result = proc.wait(timeout=max(0, shutdown_deadline - time.monotonic()))
                    records[name].update(wait_returncode=result, wait_completed=True,
                                         waited_unix_ns=time.time_ns())
                except subprocess.TimeoutExpired:
                    pass
            classification = classify_wait(records[name]["wait_returncode"], records[name]["signals_sent"])
            records[name]["exit_classification"] = classification
            if classification.startswith("FAILED"):
                lifecycle_errors.append(dict(role=name, classification=classification,
                    wait_returncode=records[name]["wait_returncode"], signals_sent=records[name]["signals_sent"]))
        if lifecycle_errors:
            exit_code = 1
        write_json(out / "processes.json", records)
        write_json(out / "supervisor_result.json", dict(gate_exit=gate_exit, followup_exit=followup_exit,
                     supervisor_exit=exit_code, roles=records, lifecycle_errors=lifecycle_errors,
                     signal_events=signal_events,
                     shutdown_budget_seconds=shutdown_total,
                     shutdown_elapsed_seconds=time.monotonic() - shutdown_start,
                     lifecycle="FAILED" if lifecycle_errors else "LOCAL_ROLES_ACCEPTED_REMOTE_MEMBER_REQUIRES_SEPARATE_WAIT"))
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
