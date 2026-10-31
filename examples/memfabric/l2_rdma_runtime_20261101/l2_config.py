"""External configuration for isolated two-host compatibility-L2 acceptance."""
import ipaddress
import json
from pathlib import Path
import re

MIB = 1 << 20
DEFAULT_PORTS = dict(rpc=25555, http=28080, metrics=29090, p=28001,
                     d=28002, proxy=28000, telemetry=25768, gate=25769)


def read_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("run_root", "run_id", "core_path", "ascend_path", "communication_ip"):
        if not cfg.get(key):
            raise ValueError("Missing configuration: " + key)
    ip = ipaddress.ip_address(cfg["communication_ip"])
    if ip.is_loopback or ip.is_unspecified or ip.version != 4:
        raise ValueError("communication_ip must be an explicit non-loopback IPv4 address")
    if Path(cfg["run_id"]).name != cfg["run_id"] or cfg["run_id"] in (".", ".."):
        raise ValueError("run_id must be one fresh directory component")
    cfg.setdefault("python", "python3")
    cfg.setdefault("store_port", 28580)
    cfg.setdefault("pool_gb", 1)
    cfg.setdefault("slot_size_mb", 32)
    cfg.setdefault("tp_size", 1)
    cfg.setdefault("driver_devices", [8])
    cfg.setdefault("driver_bm_device", 0)
    cfg.setdefault("service_bm_device", 2)
    cfg.setdefault("local_devices", dict(p=[8], d=[10], service=[8, 10, 12]))
    cfg["ports"] = dict(DEFAULT_PORTS, **cfg.get("ports", {}))
    cfg["store_url"] = "tcp://%s:%d" % (cfg["communication_ip"], cfg["store_port"])
    cfg["case_root"] = str(Path(cfg["run_root"]) / cfg["run_id"])
    if cfg["tp_size"] not in (1, 2, 4, 8):
        raise ValueError("Supported planned Qwen3-8B tensor parallel sizes: 1/2/4/8")
    if any(type(cfg[k]) is not int or cfg[k] <= 0 for k in ("pool_gb", "slot_size_mb")):
        raise ValueError("Pool and slot sizes must be positive integers")
    if cfg["pool_gb"] * 1024 % cfg["slot_size_mb"]:
        raise ValueError("slot_size_mb must divide pool_gb * 1024")
    ports = [cfg["store_port"], *cfg["ports"].values()]
    if len(ports) != len(set(ports)) or any(type(p) is not int or not 0 < p < 65536 for p in ports):
        raise ValueError("Ports must be distinct valid integer ports")
    return cfg


def adapter_config(cfg, driver=False):
    return dict(type="plugin", module_path="lmcache_ascend.v1.distributed.memfabric_l2_adapter",
                class_name="MemFabricL2Adapter", adapter_params=dict(
                    device_id=cfg["driver_bm_device"] if driver else cfg["service_bm_device"],
                    world_size=2, storage_rank=1, pool_gb=cfg["pool_gb"],
                    slot_size_mb=cfg["slot_size_mb"], store_url=cfg["store_url"]))


def model_expectations(tp_size):
    # Candidate geometry, checked against actual native snapshots by the gate.
    # No TP>1 validation is implied by this calculation.
    return dict(prompt_tokens=1373, external_tokens=1280, computed_tokens=93,
                prefix_tokens=0, objects=10 * tp_size,
                object_bytes=18 * MIB // tp_size, total_bytes=180 * MIB)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
                          encoding="utf-8")


def prefetch_audit(text, request_id, tp_size):
    pattern = re.compile(
        r"Prefetch request completed \(L1\+L2\): (\d+)/(\d+) retained keys "
        r"\((\d+) L1, (\d+) L2\).*?external_request_id=([^,]+),")
    normalize = lambda value: value.removeprefix("chatcmpl-")
    rows, unknown, other = [], [], []
    for line in text.splitlines():
        if "Prefetch request completed" not in line:
            continue
        match = pattern.search(line)
        if match is None:
            unknown.append(line)
            continue
        external = normalize(match[5])
        public = normalize(request_id)
        row = dict(retained=int(match[1]), requested=int(match[2]), l1=int(match[3]),
                   l2=int(match[4]), request_id=match[5], raw=line)
        (rows if external == public or external.startswith(public + "-") else other).append(row)
    expected = model_expectations(tp_size)["objects"]
    passed = (not unknown and not other and len(rows) == 1
              and all(r["l1"] == 0 and r["l2"] == r["retained"] == r["requested"]
                      == expected for r in rows))
    return dict(status="PASS" if passed else "FAIL", request_id=request_id,
                expected_records=1, expected_total_l2_keys=expected, rows=rows,
                unparsed_prefetch_records=unknown, other_request_records=other)


# Explicit DEVICE_RDMA/JOIN options; retain the original geometry and source auditor.
_original_adapter_config = adapter_config
def adapter_config(cfg, driver=False):
    value = _original_adapter_config(cfg, driver)
    for key in ('data_op_type', 'device_nic', 'enable_56bits_gva', 'join_coord_url', 'join_group_id'):
        value['adapter_params'][key] = cfg[key]
    return value
