"""Experimental service-only observer; call before lmcache.cli.main.main()."""


def install_reservation_trace():
    import functools
    import json
    import time

    from lmcache.v1.distributed.error import L1Error
    from lmcache.v1.distributed.l1_manager import L1Manager

    original = L1Manager.reserve_write

    @functools.wraps(original)
    def traced(self, keys, is_temporary, layout_desc, tag=""):
        result = original(self, keys, is_temporary, layout_desc, tag=tag)
        # Observation must never turn a successfully returned operation into
        # an exception. Preserve the returned dictionary and every object.
        try:
            failures = [
                {"key": repr(key), "error": error.name,
                 "reason": "l1_contended" if error == L1Error.KEY_NOT_WRITABLE
                 else "l1_oom" if error == L1Error.OUT_OF_MEMORY else "other"}
                for key, (error, obj) in result.items()
                if error != L1Error.SUCCESS or obj is None
            ]
            if failures:
                print("L1_RESERVE_WRITE_FAILURE " + json.dumps({
                    "unix_ns": time.time_ns(), "writer_tag": tag,
                    "manager_id": getattr(self, "_l1_manager_id", None),
                    "requested": len(keys), "failed": len(failures),
                    "failures": failures,
                }, default=str), flush=True)
        except Exception:
            pass
        return result

    L1Manager.reserve_write = traced
