# SPDX-License-Identifier: Apache-2.0
"""Opt-in MPCacheServer metadata/control module for worker-direct BM copies.

Configure via --server-module with this module_path and explicit config fields:
device_id, rank_id (storage rank), world_size, pool_gb, slot_size_mb, store_url.
Optional init_timeout extends BM startup/join waiting for cold-loading workers;
it defaults to 90 seconds and does not change copy-operation timeouts.
Only metadata crosses the JSON control interface. The BM pool owns payloads;
the server's CPU L1 allocator is never used by this module.
"""

import json
import threading

from lmcache.v1.multiprocess.engine_module import DiscoverableModule
from lmcache.v1.multiprocess.ext_server_module import server_module_handler

from lmcache_ascend.v1.distributed.memfabric_pool_index import PoolIndex


class MemFabricDirectServerModule(DiscoverableModule):
    module_name = "memfabric_direct"

    @classmethod
    def create(cls, build_ctx):
        return cls(build_ctx.engine_context, dict(build_ctx.config_for(cls)))

    def __init__(self, context, config, *, session_factory=None):
        fields = {"device_id", "rank_id", "world_size", "pool_gb", "slot_size_mb", "store_url"}
        if not fields <= set(config) or set(config) - fields - {"init_timeout"}:
            raise ValueError("MemFabric direct config requires %s and optional init_timeout" % sorted(fields))
        init_timeout = config.get("init_timeout", 90)
        if type(init_timeout) is not int or not 1 <= init_timeout <= 3600:
            raise ValueError("BM init_timeout must be an integer in [1, 3600] seconds")
        numeric = [config[k] for k in fields - {"store_url"}]
        if (any(type(v) is not int for v in numeric) or config["device_id"] < 0
                or config["pool_gb"] < 1 or config["slot_size_mb"] < 1
                or not isinstance(config["store_url"], str) or not config["store_url"]):
            raise ValueError("Invalid BM device, rank, capacity or store URL")
        self._ctx = context
        self._config = dict(config)
        self._index = PoolIndex(pool_bytes=config["pool_gb"] << 30,
                                slot_size=config["slot_size_mb"] << 20,
                                storage_rank=config["rank_id"],
                                world_size=config["world_size"], ready=False)
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._owner_thread = threading.Thread(target=self._run_pool,
                                               args=(session_factory,),
                                               name="memfabric-pool-owner", daemon=True)
        # BM join may wait for workers, which need the server API to register.
        # Never wait for join on the server construction/handler thread.
        self._owner_thread.start()

    @property
    def context(self):
        return self._ctx

    def _run_pool(self, session_factory):
        session = None
        try:
            if session_factory is None:
                from lmcache_ascend.v1.distributed.memfabric_bm import BMSession

                session_factory = BMSession
            config = self._config
            optional = {"init_timeout": config["init_timeout"]} if "init_timeout" in config else {}
            session = session_factory(device_id=config["device_id"],
                                      rank_id=config["rank_id"],
                                      world_size=config["world_size"],
                                      pool_gb=config["pool_gb"],
                                      store_url=config["store_url"], **optional)
            self._index.mark_ready()
            while True:
                if session.peer_departed.is_set():
                    self._index.invalidate("peer rank %s left" % session.departed_rank)
                status = self._index.status()
                if self._stop.is_set() and not (status["pending_writes"] or status["total_pins"]):
                    break
                self._wake.wait(0.05)
                self._wake.clear()
        except BaseException as exc:
            self._index.invalidate("pool startup/owner failed: %s" % exc)
        finally:
            if session is not None:
                # close/wait/unmap must run on the BM owner thread. No timeout
                # is proof that active device access has stopped.
                try:
                    session.close()
                except BaseException as exc:
                    self._index.invalidate("pool close failed: %s" % exc)

    def _call(self, operation, payload):
        try:
            if not isinstance(payload, bytes):
                raise ValueError("Expected UTF-8 JSON bytes")
            request = json.loads(payload)
            if not isinstance(request, dict):
                raise ValueError("Expected a JSON object")
            # Strict method signatures reject payload/tensor fields: this
            # endpoint intentionally has no KV serialization or L1 path.
            result = getattr(self._index, operation)(**request)
            return json.dumps(dict(ok=True, **result), separators=(",", ":")).encode("utf-8")
        except (ValueError, TypeError, RuntimeError, UnicodeError) as exc:
            return json.dumps(dict(ok=False, error=str(exc)), separators=(",", ":")).encode("utf-8")
        finally:
            self._wake.set()

    @server_module_handler("memfabric_direct.register")
    def register(self, payload: bytes) -> bytes:
        return self._call("register", payload)

    @server_module_handler("memfabric_direct.lookup")
    def lookup(self, payload: bytes) -> bytes:
        return self._call("lookup", payload)

    @server_module_handler("memfabric_direct.prepare_store")
    def prepare_store(self, payload: bytes) -> bytes:
        return self._call("prepare_store", payload)

    @server_module_handler("memfabric_direct.commit_store")
    def commit_store(self, payload: bytes) -> bytes:
        return self._call("commit_store", payload)

    @server_module_handler("memfabric_direct.abort_store")
    def abort_store(self, payload: bytes) -> bytes:
        return self._call("abort_store", payload)

    @server_module_handler("memfabric_direct.acquire")
    def acquire(self, payload: bytes) -> bytes:
        return self._call("acquire", payload)

    @server_module_handler("memfabric_direct.release")
    def release(self, payload: bytes) -> bytes:
        return self._call("release", payload)

    @server_module_handler("memfabric_direct.status")
    def status(self, payload: bytes) -> bytes:
        return self._call("status", payload)

    @server_module_handler("memfabric_direct.invalidate")
    def invalidate(self, payload: bytes) -> bytes:
        return self._call("invalidate", payload)

    def report_status(self):
        return self._index.status()

    def close(self):
        self._index.close()
        self._stop.set()
        self._wake.set()
        # Retain the BM session while any writer or read lease could still be
        # in flight. abort/release remain available to explicitly drain them.
        self._owner_thread.join()
