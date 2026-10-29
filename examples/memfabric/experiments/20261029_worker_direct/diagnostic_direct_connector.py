# SPDX-License-Identifier: Apache-2.0
"""Separate, opt-in worker diagnostics; the formal counted module is unchanged.

The original Future object is retained. Only its instance ``result`` callable
is timed; cancellation, callbacks, wait/query, responses and exceptions remain
owned by the production implementation. No polling sleep or transport is added.
"""

from functools import wraps
import json
import threading
import time

import counted_direct_connector as _counted


_lock = threading.RLock()
_installed = False
_base_snapshot = None
_rpc = {}
_retrieve = dict(started_calls=0, returned_calls=0, restored_calls=0,
                 miss_calls=0, failed_calls=0, in_flight=0,
                 requested_chunks=0, restored_chunks=0,
                 returned_ns=0, failed_ns=0, total_ns=0)


def _empty_rpc():
    return dict(submit_calls=0, submitted_futures=0, submit_failed_calls=0,
                submit_ns=0, result_calls=0, active_result_calls=0,
                result_returned_calls=0, result_failed_calls=0,
                result_timeout_calls=0, result_wait_ns=0,
                returned_wait_ns=0, failed_wait_ns=0, timeout_wait_ns=0,
                observed_futures=0, observed_exception_futures=0,
                first_result_calls=0, submit_to_first_result_ns=0,
                first_success_calls=0, submit_to_first_success_ns=0,
                metadata_successes=0, metadata_failures=0, metadata_unknown=0)


def _metadata_outcome(value):
    """Inspect a reply without decoding it on behalf of the caller."""
    success = getattr(value, "success", None)
    if success is False:
        return "metadata_failures"
    if success is True:
        try:
            payload = json.loads(value.payload)
        except (AttributeError, TypeError, ValueError, UnicodeError):
            return "metadata_unknown"
        if isinstance(payload, dict) and payload.get("ok") is True:
            return "metadata_successes"
        if isinstance(payload, dict) and payload.get("ok") is False:
            return "metadata_failures"
    return "metadata_unknown"


def diagnostic_snapshot():
    result = _base_snapshot()
    with _lock:
        result["rpc"] = {name: dict(stats) for name, stats in _rpc.items()}
        result["retrieve"] = dict(_retrieve)
    result["diagnostic_scope"] = (
        "worker process only; result wait and first-result observation latency; "
        "retrieve owner execution excludes scheduler and owner-queue waiting; "
        "RPC timings are nested inside retrieve, not additive to it")
    return result


def _observe_future(future, group, submitted_ns):
    # The real MessagingFuture has an instance dict and no __slots__. Refuse
    # another Future shape rather than replacing identity with a proxy object.
    if not isinstance(getattr(future, "__dict__", None), dict):
        raise TypeError("Diagnostic requires a MessagingFuture instance dictionary")
    if future.__dict__.get("_mf_direct_diagnostic_result"):
        return future
    original_result = future.result
    state = dict(observed=False, returned=False)

    @wraps(original_result)
    def result(*args, **kwargs):
        with _lock:
            stats = _rpc[group]
            stats["result_calls"] += 1
            stats["active_result_calls"] += 1
        started = time.perf_counter_ns()
        try:
            value = original_result(*args, **kwargs)
        except BaseException as error:
            elapsed = time.perf_counter_ns() - started
            # A timeout from Event.wait is not a completion. Do not use query()
            # here: the response can arrive just after that timeout. A stored
            # exception has the exact object identity in exception_.
            stored_exception = getattr(future, "exception_", None) is error
            pending_timeout = isinstance(error, TimeoutError) and not stored_exception
            with _lock:
                stats["active_result_calls"] -= 1
                stats["result_wait_ns"] += elapsed
                if pending_timeout:
                    stats["result_timeout_calls"] += 1
                    stats["timeout_wait_ns"] += elapsed
                else:
                    stats["result_failed_calls"] += 1
                    stats["failed_wait_ns"] += elapsed
                if stored_exception and not state["observed"]:
                    state["observed"] = True
                    stats["observed_futures"] += 1
                    stats["observed_exception_futures"] += 1
            raise
        observed_ns = time.perf_counter_ns()
        elapsed = observed_ns - started
        with _lock:
            stats["active_result_calls"] -= 1
            stats["result_wait_ns"] += elapsed
            stats["returned_wait_ns"] += elapsed
            stats["result_returned_calls"] += 1
            if not state["returned"]:
                state["returned"] = True
                stats["first_result_calls"] += 1
                stats["submit_to_first_result_ns"] += observed_ns - submitted_ns
            if not state["observed"]:
                state["observed"] = True
                stats["observed_futures"] += 1
                outcome = _metadata_outcome(value)
                stats[outcome] += 1
                if outcome == "metadata_successes":
                    stats["first_success_calls"] += 1
                    stats["submit_to_first_success_ns"] += observed_ns - submitted_ns
        return value

    future.result = result
    future.__dict__["_mf_direct_diagnostic_result"] = True
    return future


def install_diagnostics(*, context_type=None, control_type=None):
    """Patch only after worker creation; explicit classes serve the CPU check."""
    global _installed, _base_snapshot
    if _installed:
        return
    if context_type is None or control_type is None:
        from lmcache_ascend.v1.distributed.memfabric_direct import (
            MemFabricDirectTransferContext, _Control,
        )
        context_type = context_type or MemFabricDirectTransferContext
        control_type = control_type or _Control
    for group in ("lookup", "acquire", "release", "other"):
        _rpc[group] = _empty_rpc()
    original_submit = control_type.submit

    @wraps(original_submit)
    def submit(control, method, payload):
        group = method if method in ("lookup", "acquire", "release") else "other"
        started = time.perf_counter_ns()
        with _lock:
            stats = _rpc[group]
            stats["submit_calls"] += 1
        try:
            future = original_submit(control, method, payload)
        except BaseException:
            with _lock:
                stats["submit_failed_calls"] += 1
                stats["submit_ns"] += time.perf_counter_ns() - started
            raise
        with _lock:
            stats["submitted_futures"] += 1
            stats["submit_ns"] += time.perf_counter_ns() - started
        return _observe_future(future, group, started)

    control_type.submit = submit
    original_retrieve = context_type._retrieve

    @wraps(original_retrieve)
    def retrieve(context, key, block_ids, event, blocks_in_chunk,
                 skip_first_n_tokens):
        # Count the requested aligned range without invoking _blocks twice or
        # making any RPC. Actual restored operation count comes from result_.
        size = context.tokens_per_chunk
        requested = max(0, (key.end - key.start) // size) if size > 0 else 0
        with _lock:
            _retrieve["started_calls"] += 1
            _retrieve["in_flight"] += 1
            _retrieve["requested_chunks"] += requested
        started = time.perf_counter_ns()
        try:
            value = original_retrieve(context, key, block_ids, event,
                                      blocks_in_chunk, skip_first_n_tokens)
        except BaseException:
            elapsed = time.perf_counter_ns() - started
            with _lock:
                _retrieve["failed_calls"] += 1
                _retrieve["failed_ns"] += elapsed
                _retrieve["total_ns"] += elapsed
                _retrieve["in_flight"] -= 1
            raise
        elapsed = time.perf_counter_ns() - started
        with _lock:
            _retrieve["returned_calls"] += 1
            _retrieve["returned_ns"] += elapsed
            _retrieve["total_ns"] += elapsed
            _retrieve["in_flight"] -= 1
            if getattr(value, "result_", None) is True:
                _retrieve["restored_calls"] += 1
                _retrieve["restored_chunks"] += requested
            elif getattr(value, "result_", None) is False:
                _retrieve["miss_calls"] += 1
        return value

    context_type._retrieve = retrieve
    _base_snapshot = _counted.snapshot
    _counted.snapshot = diagnostic_snapshot
    _installed = True


class MemFabricDirectConnector(_counted.MemFabricDirectConnector):
    """Separate diagnostic run; select module diagnostic_direct_connector."""

    def create_lmcache_mp_worker_adapter(self, **adapter_kwargs):
        _counted.install_counters()
        install_diagnostics()
        return super().create_lmcache_mp_worker_adapter(**adapter_kwargs)
