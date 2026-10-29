# SPDX-License-Identifier: Apache-2.0
"""Experiment-only counters around the unchanged synchronous direct path.

Select ``kv_connector_module_path=counted_direct_connector`` and set
``MF_DIRECT_COUNTER_ROLE=P|D`` plus a distinct ``MF_DIRECT_COUNTER_PORT``
for each worker. GET /counters on 127.0.0.1 reads cumulative in-memory totals.
There is no reset API and no per-copy file or network operation.
"""

import atexit
from functools import wraps
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time
import uuid

from lmcache_ascend.integration.vllm.memfabric_direct_connector import (
    MemFabricDirectConnector as _DirectConnector,
)


_lock = threading.RLock()
_installed = False
_server = None
_server_thread = None
_session_id = uuid.uuid4().hex
_started_at_ns = time.time_ns()
_role = None
_scratch = {}
_operations = {}
_cpu_copy_attempts = 0


def _empty_operation():
    return dict(started_calls=0, completed_calls=0, failed_calls=0,
                in_flight=0, completed_bytes=0, completed_ns=0, failed_ns=0)


def snapshot():
    """Copy metadata only; this function never reads a payload or calls NPU."""
    with _lock:
        return dict(
            schema_version=1, pid=os.getpid(), session_id=_session_id,
            role=_role, started_at_ns=_started_at_ns,
            monotonic_ns=time.monotonic_ns(),
            operations={name: dict(values) for name, values in _operations.items()},
            cpu_copy_attempts=_cpu_copy_attempts,
            scratch_allocations=len(_scratch),
            scratch_bytes=sum(item["bytes"] for item in _scratch.values()),
            completion_semantics="BM flags=0 synchronous successful return; "
                                 "gather/scatter synchronize before return",
        )


def _measure(name, function, *, size=0):
    with _lock:
        item = _operations[name]
        item["started_calls"] += 1
        item["in_flight"] += 1
    started = time.perf_counter_ns()
    try:
        result = function()
    except BaseException:
        elapsed = time.perf_counter_ns() - started
        with _lock:
            item["failed_calls"] += 1
            item["failed_ns"] += elapsed
            item["in_flight"] -= 1
        # Failure is not evidence that DMA ended; do not count these bytes.
        raise
    elapsed = time.perf_counter_ns() - started
    with _lock:
        item["completed_calls"] += 1
        item["completed_bytes"] += size
        item["completed_ns"] += elapsed
        item["in_flight"] -= 1
    return result


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/counters", "/health"):
            self.send_error(404)
            return
        body = json.dumps(snapshot(), separators=(",", ":")).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        pass


def close_counter_server():
    """Stop only the CPU HTTP helper, never the transfer context or BM."""
    global _server, _server_thread
    server, thread = _server, _server_thread
    _server = _server_thread = None
    if server is not None:
        server.shutdown()
        server.server_close()
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=2)


def _start_counter_server(port):
    global _server, _server_thread
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever,
                              kwargs={"poll_interval": 0.1},
                              name="memfabric-counter-http", daemon=True)
    _server, _server_thread = server, thread
    thread.start()
    return server.server_address[1]


def install_counters(*, session_type=None, transfer_type=None, port=None, role=None):
    """Install once in the worker factory, after normal package initialization.

    Explicit types/port are used by the pure CPU check; production calls pass
    no arguments and import the existing real classes here.
    """
    global _installed, _role
    if _installed:
        return
    if session_type is None:
        from lmcache_ascend.v1.distributed.memfabric_bm import BMSession
        session_type = BMSession
    if transfer_type is None:
        from lmcache_ascend.v1.distributed.memfabric_direct_transfer import NpuDirectTransfer
        transfer_type = NpuDirectTransfer
    if role is None:
        role = os.environ["MF_DIRECT_COUNTER_ROLE"]
    if role not in ("P", "D"):
        raise ValueError("MF_DIRECT_COUNTER_ROLE must be P or D")
    if port is None:
        port = int(os.environ["MF_DIRECT_COUNTER_PORT"])
        if not 1 <= port <= 65535:
            raise ValueError("MF_DIRECT_COUNTER_PORT must be in [1,65535]")
    # Bind first: a port conflict must fail startup, not silently lose counters.
    _start_counter_server(port)
    _role = role
    for name in ("L2G", "G2L", "gather", "scatter"):
        _operations[name] = _empty_operation()

    original_init = transfer_type.__init__

    @wraps(original_init)
    def initialize(transfer, *args, **kwargs):
        original_init(transfer, *args, **kwargs)
        if transfer.buffer.device.type != "npu":
            raise RuntimeError("Counters require an actual NPU scratch endpoint")
        with _lock:
            _scratch[int(transfer.buffer.data_ptr())] = dict(bytes=int(transfer.size))

    transfer_type.__init__ = initialize
    for method_name in ("gather", "scatter"):
        original = getattr(transfer_type, method_name)

        def wrap_stage(function, stage):
            @wraps(function)
            def stage_call(transfer, *args, **kwargs):
                # Production methods synchronize before return. Partial scatter
                # is counted as one call, not as a full-buffer byte transfer.
                return _measure(stage, lambda: function(transfer, *args, **kwargs))
            return stage_call

        setattr(transfer_type, method_name, wrap_stage(original, method_name))

    for method_name, direction in (("copy_npu_to_global", "L2G"),
                                   ("copy_global_to_npu", "G2L")):
        original = getattr(session_type, method_name)

        def wrap_copy(function, copy_direction):
            @wraps(function)
            def copy(session, src, dst, size):
                def invoke():
                    local = int(src if copy_direction == "L2G" else dst)
                    with _lock:
                        info = _scratch.get(local)
                    if info is None or type(size) is not int or size != info["bytes"]:
                        raise RuntimeError("BM endpoint/size is not the registered NPU scratch")
                    # Existing BMSession._copy checks the synchronous flags=0
                    # return code. Only this successful return counts bytes.
                    return function(session, src, dst, size)
                return _measure(copy_direction, invoke, size=size)
            return copy

        setattr(session_type, method_name, wrap_copy(original, direction))

    def forbidden_cpu_copy(session, src, dst, size, *, store):
        global _cpu_copy_attempts
        with _lock:
            _cpu_copy_attempts += 1
        raise RuntimeError("CPU BM copy is forbidden in this direct-only experiment")

    session_type.copy = forbidden_cpu_copy
    _installed = True
    atexit.register(close_counter_server)


class MemFabricDirectConnector(_DirectConnector):
    """Production direct connector with opt-in worker-only memory counters."""

    def create_lmcache_mp_worker_adapter(self, **adapter_kwargs):
        install_counters()
        return super().create_lmcache_mp_worker_adapter(**adapter_kwargs)
