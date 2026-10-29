"""Small CPU checks of configuration, real command plans and source audit gates.

These checks do not import native SDKs, emulate BM, or prove device correctness.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from l2_config import adapter_config, model_expectations, prefetch_audit, read_config
from l2_model import commands, supervise


class Checks(unittest.TestCase):
    def config(self):
        raw = dict(run_root="/private/runs", run_id="new-attempt", core_path="/private/core",
                   ascend_path="/private/ascend", communication_ip="192.168.0.223",
                   model_path="/private/models/Qwen3-8B")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            cfg = read_config(path)
        cfg["config_path"] = "/private/config.json"
        return cfg

    def test_remote_rank_and_plan(self):
        cfg = self.config()
        config = adapter_config(cfg)["adapter_params"]
        self.assertEqual((config["world_size"], config["storage_rank"], config["device_id"]), (2, 1, 2))
        self.assertEqual(config["store_url"], "tcp://192.168.0.223:28580")
        plan, _ = commands(cfg)
        self.assertEqual(plan["p"][plan["p"].index("--tensor-parallel-size") + 1], "1")
        self.assertNotIn("pool_member.py", json.dumps(plan))

    def test_tp_geometry_is_dynamic(self):
        for tp in (1, 2, 4, 8):
            expected = model_expectations(tp)
            self.assertEqual(expected["objects"], 10 * tp)
            self.assertEqual(expected["objects"] * expected["object_bytes"], expected["total_bytes"])

    def test_device_ipc_scope_rejected(self):
        cfg = self.config()
        cfg["local_devices"]["service"] = [12]
        with self.assertRaisesRegex(ValueError, "every local P/D"):
            supervise(cfg, plan_only=True)

    def test_non_loopback_required(self):
        for ip in ("127.0.0.1", "0.0.0.0"):
            with tempfile.TemporaryDirectory() as tmp:
                cfg = self.config()
                cfg["communication_ip"] = ip
                path = Path(tmp) / "bad.json"
                path.write_text(json.dumps(cfg), encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_config(path)

    def test_actual_source_audit_rejects_missing_mixed_extra(self):
        def line(keys=10, l1=0, request="chatcmpl-abc-0-seq"):
            return (f"Prefetch request completed (L1+L2): {keys}/{keys} retained keys "
                    f"({l1} L1, {keys-l1} L2) in 1.0 ms "
                    f"(external_request_id={request}, prefetch_request_id=1)")
        self.assertEqual(prefetch_audit(line(), "chatcmpl-abc", 1)["status"], "PASS")
        self.assertEqual(prefetch_audit(line(80), "chatcmpl-abc", 8)["status"], "PASS")
        for bad in ("", line(l1=1), line() + "\n" + line(),
                    line(request="other"), "Prefetch request completed UNRECOGNIZED"):
            self.assertEqual(prefetch_audit(bad, "chatcmpl-abc", 1)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
