"""One pure CPU check: fake endpoints only; no Ascend/Torch/native import."""

import importlib.util
import json
from pathlib import Path
import sys
import types
import urllib.error
import urllib.request


def main():
    stub_name = "lmcache_ascend.integration.vllm.memfabric_direct_connector"
    stub = types.ModuleType(stub_name)
    stub.MemFabricDirectConnector = type("FakeConnector", (), {})
    sys.modules[stub_name] = stub
    spec = importlib.util.spec_from_file_location(
        "counter_cpu_check", Path(__file__).with_name("counted_direct_connector.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeTransfer:
        def __init__(self):
            self.size = 18 * 1024 * 1024
            self.buffer = types.SimpleNamespace(
                device=types.SimpleNamespace(type="npu"), data_ptr=lambda: 4096)

        def gather(self, ids):
            return ids

        def scatter(self, ids, skip_tokens=0):
            if ids == "fail":
                raise RuntimeError("fake scatter failure")
            return skip_tokens

    class FakeSession:
        def copy_npu_to_global(self, src, dst, size):
            return "stored"

        def copy_global_to_npu(self, src, dst, size):
            if src == -1:
                raise RuntimeError("fake uncertain copy")
            return "loaded"

        def copy(self, src, dst, size, *, store):
            raise AssertionError("CPU original must not be called")

    try:
        module.install_counters(session_type=FakeSession,
                                transfer_type=FakeTransfer, port=0, role="D")
        port = module._server.server_address[1]
        transfer, session = FakeTransfer(), FakeSession()
        size = transfer.size
        assert transfer.gather([1]) == [1]
        assert session.copy_npu_to_global(4096, 8192, size) == "stored"
        assert session.copy_global_to_npu(8192, 4096, size) == "loaded"
        assert transfer.scatter([2], skip_tokens=64) == 64
        for call in (
            lambda: session.copy_global_to_npu(-1, 4096, size),
            lambda: session.copy_global_to_npu(8192, 0, size),
            lambda: transfer.scatter("fail"),
            lambda: session.copy(1, 2, size, store=True),
        ):
            try:
                call()
            except RuntimeError:
                pass
            else:
                raise AssertionError("Expected fake error")
        url = f"http://127.0.0.1:{port}/counters"
        with urllib.request.urlopen(url, timeout=2) as response:
            result = json.load(response)
        assert result["pid"] > 0 and result["session_id"]
        assert result["role"] == "D"
        assert result["cpu_copy_attempts"] == 1
        assert result["operations"]["L2G"]["completed_bytes"] == size
        assert result["operations"]["G2L"]["completed_bytes"] == size
        assert result["operations"]["G2L"]["completed_calls"] == 1
        assert result["operations"]["G2L"]["failed_calls"] == 2
        assert result["operations"]["scatter"]["failed_calls"] == 1
        assert result["operations"]["scatter"]["completed_bytes"] == 0
        assert all(item["in_flight"] == 0 for item in result["operations"].values())
        try:
            urllib.request.urlopen(urllib.request.Request(url, method="POST"), timeout=2)
        except urllib.error.HTTPError as error:
            assert error.code == 501
        else:
            raise AssertionError("Counter mutation must not be supported")
        assert not any(name in sys.modules for name in ("torch", "torch_npu", "acl"))
        print(json.dumps(dict(status="PASS", evidence="CPU fakes only",
                              snapshot=result), indent=2))
    finally:
        module.close_counter_server()


if __name__ == "__main__":
    main()
