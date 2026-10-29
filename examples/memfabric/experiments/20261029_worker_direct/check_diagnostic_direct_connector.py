"""CPU check with the real MessagingFuture class extracted without imports."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import types
import typing
import urllib.request


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    runtime = Path(__file__).parent
    core_source = os.environ.get("MF_CORE_SOURCE")
    if not core_source:
        raise ValueError("Set MF_CORE_SOURCE to the fixed LMCache source checkout")
    source = Path(core_source) / "lmcache/v1/multiprocess/futures.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(item for item in tree.body
                if isinstance(item, ast.ClassDef) and item.name == "MessagingFuture")
    waits = next(item for item in node.body
                 if isinstance(item, ast.FunctionDef) and item.name == "wait")
    assert any(isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
               and item.func.attr == "wait" for item in ast.walk(waits))
    assert not any(isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
                   and item.func.attr == "sleep" for item in ast.walk(node))
    namespace = dict(threading=threading, Generic=typing.Generic,
                     T=typing.TypeVar("T"), cast=typing.cast,
                     LMCacheTimeoutError=TimeoutError,
                     lmcache_deprecate=lambda _message: lambda method: method,
                     logger=types.SimpleNamespace(exception=lambda *_a, **_k: None))
    extracted = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(extracted)
    exec(compile(extracted, str(source), "exec"), namespace)
    Future = namespace["MessagingFuture"]
    assert isinstance(Future().__dict__, dict)
    stub_name = "lmcache_ascend.integration.vllm.memfabric_direct_connector"
    stub = types.ModuleType(stub_name)
    stub.MemFabricDirectConnector = type("FakeConnector", (), {})
    sys.modules[stub_name] = stub
    counted = load_file("counted_direct_connector", runtime / "counted_direct_connector.py")
    diagnostic = load_file("diagnostic_direct_connector", runtime / "diagnostic_direct_connector.py")

    class Control:
        def __init__(self):
            self.next = None

        def submit(self, method, payload):
            if method == "submit_failure":
                raise OSError("fake submit error")
            return self.next

    class Context:
        tokens_per_chunk = 128

        def _retrieve(self, key, block_ids, event, blocks_in_chunk, skip_first_n_tokens):
            if block_ids == "fail":
                raise RuntimeError("fake retrieve error")
            future = Future()
            future.set_result(block_ids != "miss")
            return future

    def reply(ok=True):
        return types.SimpleNamespace(success=True,
            payload=json.dumps(dict(ok=ok)).encode())

    try:
        diagnostic.install_diagnostics(context_type=Context, control_type=Control)
        port = counted._start_counter_server(0)
        control = Control()
        future = control.next = Future()
        returned = control.submit("lookup", {})
        assert returned is future
        value = reply()
        future.set_result(value)
        assert future.result(timeout=0) is value
        assert future.result(timeout=0) is value

        late = control.next = Future()
        assert control.submit("acquire", {}) is late
        try:
            late.result(timeout=0)
        except TimeoutError:
            pass
        else:
            raise AssertionError("Pending future must timeout")
        assert diagnostic.diagnostic_snapshot()["rpc"]["acquire"]["observed_futures"] == 0
        late_value = reply()
        late.set_result(late_value)
        assert late.result(timeout=0) is late_value
        assert late.result(timeout=0) is late_value

        rejected = control.next = Future()
        assert control.submit("release", {}) is rejected
        rejection = reply(False)
        rejected.set_result(rejection)
        assert rejected.result(timeout=0) is rejection

        failed = control.next = Future()
        control.submit("status", {})
        error = TimeoutError("stored RPC timeout, not an Event.wait timeout")
        failed.set_exception(error)
        for _ in range(2):
            try:
                failed.result(timeout=0)
            except TimeoutError as caught:
                assert caught is error
            else:
                raise AssertionError("Stored exception must propagate unchanged")
        try:
            control.submit("submit_failure", {})
        except OSError:
            pass
        else:
            raise AssertionError("Submit exception must propagate")

        context, key = Context(), types.SimpleNamespace(start=0, end=1280)
        assert context._retrieve(key, [1], None, 1, 0).result(timeout=0) is True
        assert context._retrieve(key, "miss", None, 1, 0).result(timeout=0) is False
        try:
            context._retrieve(key, "fail", None, 1, 0)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Retrieve error must propagate")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/counters", timeout=2) as response:
            result = json.load(response)
        rpc = result["rpc"]
        assert rpc["lookup"]["result_returned_calls"] == 2
        assert rpc["lookup"]["first_success_calls"] == 1
        assert rpc["acquire"]["result_timeout_calls"] == 1
        assert rpc["acquire"]["observed_futures"] == 1
        assert rpc["acquire"]["first_success_calls"] == 1
        assert rpc["release"]["metadata_failures"] == 1
        assert rpc["other"]["observed_exception_futures"] == 1
        assert rpc["other"]["result_timeout_calls"] == 0
        assert rpc["other"]["submit_failed_calls"] == 1
        assert result["retrieve"]["started_calls"] == 3
        assert result["retrieve"]["requested_chunks"] == 30
        assert result["retrieve"]["restored_chunks"] == 10
        assert result["retrieve"]["miss_calls"] == 1
        assert result["retrieve"]["failed_calls"] == 1
        assert all(stats["active_result_calls"] == 0 for stats in rpc.values())
        assert not any(name in sys.modules for name in ("torch", "torch_npu", "acl"))
        print(json.dumps(dict(status="PASS", evidence="CPU only; actual Future class AST",
                              identity_preserved=True, event_wait_without_sleep=True,
                              snapshot=result), indent=2))
    finally:
        counted.close_counter_server()


if __name__ == "__main__":
    main()
