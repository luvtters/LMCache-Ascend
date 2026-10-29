# SPDX-License-Identifier: Apache-2.0
"""Experiment-only external connector tracing real direct-worker calls.

Put this directory on PYTHONPATH, select module ``traced_direct_connector``,
and set MF_DIRECT_TRACE_DIR plus MF_DIRECT_TRACE_ROLE=P or D. No production
source is modified. Only metadata, pointers, and byte counts are recorded.
"""

from functools import wraps
import json
import os
from pathlib import Path
import threading
import time
from direct_coordination import host_identity

from lmcache_ascend.integration.vllm.memfabric_direct_connector import (
    MemFabricDirectConnector as _DirectConnector,
)


_installed = False
_lock = threading.RLock()
_buffers = {}


def emit(kind, phase, **fields):
    """Append one per-worker record after validating explicit trace settings."""
    directory = os.environ["MF_DIRECT_TRACE_DIR"]
    role = os.environ["MF_DIRECT_TRACE_ROLE"]
    if role not in ("P", "D"):
        raise ValueError("MF_DIRECT_TRACE_ROLE must be P or D")
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    record = dict(kind=kind, phase=phase, role=role, pid=os.getpid(),
                  host_id=host_identity(), session_id=os.environ.get("MF_DIRECT_SESSION_ID"),
                  thread_id=threading.get_ident(), mono_ns=time.monotonic_ns(),
                  **fields)
    data = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    with _lock:
        fd = os.open(path / f"{role}-{os.getpid()}.jsonl",
                     os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)


def install_trace():
    """Patch only the direct worker's BM and native transfer entry points."""
    global _installed
    if _installed:
        return
    # Called from the worker factory after normal package initialization;
    # never import Ascend from Python sitecustomize or an import hook.
    from lmcache_ascend.v1.distributed.memfabric_bm import BMSession
    from lmcache_ascend.v1.distributed.memfabric_direct_transfer import NpuDirectTransfer

    original_init = NpuDirectTransfer.__init__

    @wraps(original_init)
    def initialize(transfer, *args, **kwargs):
        original_init(transfer, *args, **kwargs)
        pointer = int(transfer.buffer.data_ptr())
        info = dict(local_ptr=pointer, bytes=int(transfer.size),
                    device=str(transfer.buffer.device),
                    memory_type=transfer.buffer.device.type,
                    shape=list(transfer.buffer.shape))
        if info["memory_type"] != "npu":
            raise RuntimeError("Trace requires a real NPU scratch allocation")
        _buffers[pointer] = info
        emit("npu_scratch", "complete", **info)
        native = transfer.ops.multi_layer_block_kv_transfer
        if not getattr(native, "_memfabric_trace", False):
            @wraps(native)
            def native_transfer(*native_args, **native_kwargs):
                object_pointers = [int(value) for value in native_args[1]]
                if not object_pointers or any(p not in _buffers for p in object_pointers):
                    raise RuntimeError("Native direct object pointer is not traced NPU scratch")
                details = dict(object_pointers=object_pointers,
                               device=str(native_args[3]),
                               direction=str(native_args[4]))
                emit("native_transfer", "begin", **details)
                try:
                    result = native(*native_args, **native_kwargs)
                except BaseException as exc:
                    emit("native_transfer", "error", error=repr(exc), **details)
                    raise
                emit("native_transfer", "returned", **details)
                return result
            native_transfer._memfabric_trace = True
            transfer.ops.multi_layer_block_kv_transfer = native_transfer

    NpuDirectTransfer.__init__ = initialize

    for method_name, direction in (("gather", "gather"), ("scatter", "scatter")):
        original = getattr(NpuDirectTransfer, method_name)

        def wrap_stage(function, stage):
            @wraps(function)
            def stage_call(transfer, *args, **kwargs):
                details = dict(direction=stage, local_ptr=int(transfer.buffer.data_ptr()),
                               bytes=int(transfer.size), device=str(transfer.buffer.device))
                emit("kv_stage", "begin", **details)
                try:
                    result = function(transfer, *args, **kwargs)
                except BaseException as exc:
                    emit("kv_stage", "error", error=repr(exc), **details)
                    raise
                # The production stage synchronizes before it returns. A
                # native 'returned' event alone is not completion evidence.
                emit("kv_stage", "complete", **details)
                return result
            return stage_call

        setattr(NpuDirectTransfer, method_name, wrap_stage(original, direction))

    for method_name, direction in (("copy_npu_to_global", "L2G"),
                                   ("copy_global_to_npu", "G2L")):
        original = getattr(BMSession, method_name)

        def wrap_copy(function, copy_direction):
            @wraps(function)
            def copy(session, src, dst, size):
                local = int(src if copy_direction == "L2G" else dst)
                info = _buffers.get(local)
                if info is None or info["memory_type"] != "npu" or size != info["bytes"]:
                    raise RuntimeError("BM endpoint is not the registered NPU scratch")
                details = dict(direction=copy_direction, src=int(src), dst=int(dst),
                               local_ptr=local, bytes=int(size),
                               device=info["device"], memory_type=info["memory_type"],
                               bm_rank=int(session.rank_id))
                emit("bm_copy", "begin", **details)
                try:
                    result = function(session, src, dst, size)
                except BaseException as exc:
                    emit("bm_copy", "error", error=repr(exc), **details)
                    raise
                # BMSession uses flags=0 and returns after successful completion.
                emit("bm_copy", "complete", **details)
                return result
            return copy

        setattr(BMSession, method_name, wrap_copy(original, direction))

    def forbidden_cpu_copy(session, src, dst, size, *, store):
        emit("cpu_bm_copy", "forbidden", src=int(src), dst=int(dst),
             bytes=int(size), direction="H2G" if store else "G2H")
        raise RuntimeError("CPU BM copy is forbidden in this direct-only validation")

    BMSession.copy = forbidden_cpu_copy
    _installed = True
    emit("trace_installed", "complete", cpu_bm_copy="fail_fast")


class MemFabricDirectConnector(_DirectConnector):
    """Use the production connector with worker-only diagnostic wrappers."""

    def create_lmcache_mp_worker_adapter(self, **adapter_kwargs):
        install_trace()
        return super().create_lmcache_mp_worker_adapter(**adapter_kwargs)
