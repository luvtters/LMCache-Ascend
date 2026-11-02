#!/usr/bin/env python3
"""Model-free real NPU gate through the native MemFabric L1 service.

This connects to an already-running LMCache MP service. It does not start a
service, a model, or an NPU session unless ``--allow-npu`` is supplied.
"""

import argparse
import json
from pathlib import Path
import secrets
import sys
import traceback


def _emit(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def _read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("gate config must be a JSON object")
    required = ("server_url", "session_id", "bm", "core_source", "ascend_source")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"gate config is missing: {', '.join(missing)}")
    if not isinstance(config["bm"], dict):
        raise ValueError("bm must be a BMSession constructor mapping")
    if config["bm"].get("rank_id") != 1 or config["bm"].get("world_size") != 2:
        raise ValueError("worker BM configuration must use rank_id=1, world_size=2")
    if (not config["bm"].get("join_coord_url")
            or config["bm"].get("join_group_id") != config["session_id"]):
        raise ValueError("bm JOIN coordinator and join_group_id must match session_id")
    return config


def _synthetic_kv(torch, *, layers, blocks, block_size, heads, head_size, device):
    """Build format-16 Qwen-style per-layer (K, V) NHD page tuples."""
    result = []
    expected = []
    shape = (blocks, block_size, heads, head_size)
    for layer in range(layers):
        planes = []
        for plane in range(2):
            values = torch.arange(
                blocks * block_size * heads * head_size,
                dtype=torch.int32,
                device=device,
            )
            values = ((values + layer * 97 + plane * 31) % 251 - 125)
            tensor = values.to(torch.bfloat16).reshape(shape)
            planes.append(tensor)
        result.append(tuple(planes))
        expected.append(tuple(tensor.clone() for tensor in planes))
    return result, expected


def run(config: dict) -> dict:
    core_source = Path(config["core_source"]).resolve()
    ascend_source = Path(config["ascend_source"]).resolve()
    sys.path[:0] = [str(ascend_source), str(core_source)]

    import torch
    import torch_npu  # noqa: F401

    from lmcache.utils import EngineType
    from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
    from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
    from lmcache.v1.platform import synchronize_device
    from lmcache_ascend.integration.vllm.memfabric_l1_worker import (
        BMAffineWorker,
        MemFabricL1TransferContext,
    )

    timeout = float(config.get("timeout_seconds", 180))
    if timeout <= 0:
        raise ValueError("timeout_seconds must be positive")
    session_id = config["session_id"]
    model_name = config.get("model_name", "qwen3-8b-l1-worker-gate")
    device_id = int(config["bm"]["device_id"])
    device = torch.device(f"npu:{device_id}")
    bm_config = dict(config["bm"])
    bm_config["session_id"] = session_id

    bm_worker = None
    client = None
    transfer = None
    registered = False
    request_ids: set[str] = set()
    result = {"status": "FAILED", "session_id": session_id}
    try:
        bm_worker = BMAffineWorker(bm_config)
        client = RequestClientFactory.create(config["server_url"])
        transfer = MemFabricL1TransferContext(
            secrets.randbits(63) or 1, client, bm_worker
        )
        chunk_size = int(client.get_chunk_size().result(timeout=timeout))
        configured_chunk = config.get("chunk_size")
        if configured_chunk is not None and int(configured_chunk) != chunk_size:
            raise ValueError(
                f"configured chunk_size={configured_chunk}, service returned {chunk_size}"
            )
        block_size = int(config.get("block_size", chunk_size))
        if block_size <= 0 or chunk_size % block_size:
            raise ValueError("service chunk size must be divisible by block_size")
        blocks_per_chunk = chunk_size // block_size
        num_layers = int(config.get("num_layers", 2))
        num_heads = int(config.get("num_heads", 8))
        head_size = int(config.get("head_size", 128))
        if min(num_layers, num_heads, head_size) <= 0:
            raise ValueError("layer and head dimensions must be positive")

        kv_caches, expected = _synthetic_kv(
            torch,
            layers=num_layers,
            blocks=blocks_per_chunk,
            block_size=block_size,
            heads=num_heads,
            head_size=head_size,
            device=device,
        )
        synchronize_device(device)
        transfer.register(
            {f"layer-{i}": layer for i, layer in enumerate(kv_caches)},
            model_name,
            world_size=1,
            blocks_in_chunk=blocks_per_chunk,
            mq_timeout=timeout,
            layout_hints={"kv_layout": "NHD"},
            engine_type=EngineType.VLLM,
            lmcache_chunk_tokens=chunk_size,
        )
        registered = True
        _emit(
            "REGISTERED",
            instance_id=transfer._instance_id,
            engine_kv_format=str(transfer._engine_kv_format),
            chunk_size=chunk_size,
            block_size=block_size,
            blocks_per_chunk=blocks_per_chunk,
            layers=num_layers,
            payload_bytes=sum(
                tensor.numel() * tensor.element_size()
                for layer in kv_caches
                for tensor in layer
            ),
        )

        token_ids = list(range(1, chunk_size + 1))
        block_ids = [list(range(blocks_per_chunk))]
        store_request_id = f"mf-l1-store-{secrets.token_hex(8)}"
        store_key = IPCCacheServerKey.from_token_ids(
            model_name=model_name,
            world_size=1,
            worker_id=0,
            token_ids=token_ids,
            start=0,
            end=chunk_size,
            request_id=store_request_id,
        )
        request_ids.add(store_request_id)
        store_future = transfer.submit_store(
            store_request_id,
            store_key,
            {f"layer-{i}": layer for i, layer in enumerate(kv_caches)},
            block_ids,
            event=None,
            blocks_in_chunk=blocks_per_chunk,
        )
        if not store_future.result(timeout=timeout):
            raise RuntimeError("service store did not complete")
        _emit("STORE_COMPLETE", request_id=store_request_id)

        retrieve_request_id = f"mf-l1-retrieve-{secrets.token_hex(8)}"
        request_ids.add(retrieve_request_id)
        lookup_key = IPCCacheServerKey.from_token_ids(
            model_name=model_name,
            world_size=1,
            worker_id=None,
            token_ids=token_ids,
            start=0,
            end=chunk_size,
            request_id=retrieve_request_id,
            num_kv_readers=1,
        )
        client.lookup(lookup_key, 1).result(timeout=timeout)
        hits = client.wait_prefetch_status(
            retrieve_request_id, timeout
        ).result(timeout=timeout + 1)
        if hits != 1:
            raise RuntimeError(f"native service prefetch returned {hits!r} chunks, expected 1")
        _emit("NATIVE_L1_PREFETCH_HIT", request_id=retrieve_request_id, chunks=hits)

        retrieve_key = IPCCacheServerKey.from_token_ids(
            model_name=model_name,
            world_size=1,
            worker_id=0,
            token_ids=token_ids,
            start=0,
            end=chunk_size,
            request_id=retrieve_request_id,
            num_kv_readers=1,
        )
        for layer in kv_caches:
            for tensor in layer:
                tensor.zero_()
        synchronize_device(device)
        retrieve_future = transfer.submit_retrieve(
            retrieve_request_id,
            retrieve_key,
            {f"layer-{i}": layer for i, layer in enumerate(kv_caches)},
            block_ids,
            event=None,
            blocks_in_chunk=blocks_per_chunk,
        )
        if not retrieve_future.result(timeout=timeout):
            raise RuntimeError("service retrieve reported a cache miss")
        synchronize_device(device)
        synchronized = True
        for layer_index, (actual_layer, expected_layer) in enumerate(
            zip(kv_caches, expected, strict=True)
        ):
            for plane_index, (actual, expected_tensor) in enumerate(
                zip(actual_layer, expected_layer, strict=True)
            ):
                if not torch.equal(actual, expected_tensor):
                    raise RuntimeError(
                        f"retrieved KV mismatch at layer={layer_index} plane={plane_index}"
                    )
        # The successful retrieve converted this lookup reference into an IO
        # pin, and its commit released the pin; the lookup range is done.
        _emit(
            "RETRIEVE_CONTENT_MATCH",
            request_id=retrieve_request_id,
            chunks=hits,
            synchronized=synchronized,
            tensors=num_layers * 2,
        )
        result.update(
            status="PASS_MODEL_FREE_NPU_SERVICE_GATE",
            layers=num_layers,
            tokens_per_chunk=chunk_size,
            payload_bytes=sum(
                tensor.numel() * tensor.element_size()
                for layer in expected
                for tensor in layer
            ),
            native_l1_hits=hits,
            retrieved_content_equal=True,
        )
        return result
    except BaseException:
        _emit("GATE_FAILED", error=traceback.format_exc())
        raise
    finally:
        # Finish every tracked request session before unregistering the engine
        # layout: END_SESSION resolves the registered attention descriptor, so
        # it must run while the layout is still registered.
        for request_id in request_ids:
            try:
                if client is None:
                    break
                client.end_session(request_id).result(timeout=timeout)
            except Exception as exc:
                if result.get("status", "").startswith("PASS"):
                    result["status"] = "FAILED_SESSION_CLEANUP"
                _emit("END_SESSION_UNCONFIRMED", request_id=request_id, error=repr(exc))
        if registered and transfer is not None:
            try:
                future = transfer.unregister()
                if future is not None:
                    future.result(timeout=timeout)
            except Exception as exc:
                result["status"] = "FAILED_RELEASE_UNCONFIRMED"
                result["release_error"] = repr(exc)
                _emit("WORKER_CLOSE_UNCONFIRMED", error=repr(exc))
            finally:
                try:
                    transfer.close()
                except Exception as exc:
                    result["status"] = "FAILED_RELEASE_UNCONFIRMED"
                    result["release_error"] = repr(exc)
                    _emit("WORKER_CONTEXT_CLOSE_FAILED", error=repr(exc))
            if bm_worker._closed:
                _emit("WORKER_UNREGISTERED_AND_CLOSED")
            else:
                result["status"] = "FAILED_RESOURCES_RETAINED"
                _emit("WORKER_RESOURCES_RETAINED", reason="completion was uncertain")
        elif transfer is not None:
            try:
                transfer.close()
            except Exception:
                pass
        elif bm_worker is not None:
            try:
                bm_worker.close()
            except Exception as exc:
                result["status"] = "FAILED_BM_CLOSE_UNCONFIRMED"
                result["release_error"] = repr(exc)
                _emit("BM_CLOSE_UNCONFIRMED", error=repr(exc))
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                if result.get("status", "").startswith("PASS"):
                    result["status"] = "FAILED_CLIENT_CLOSE"
                _emit("CLIENT_CLOSE_FAILED", error=repr(exc))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--allow-npu",
        action="store_true",
        help="perform the real NPU and service-RPC gate",
    )
    args = parser.parse_args()
    config = _read_config(args.config)
    config_dir = args.config.resolve().parent
    for name in ("core_source", "ascend_source"):
        source = Path(config[name])
        if not source.is_absolute():
            config[name] = str((config_dir / source).resolve())
    if not args.allow_npu:
        _emit(
            "NOT_STARTED",
            reason="real NPU/service actions require --allow-npu",
            server_url=config["server_url"],
            session_id=config["session_id"],
        )
        return 0
    result = run(config)
    _emit("GATE_RESULT", **result)
    return 0 if result.get("status") == "PASS_MODEL_FREE_NPU_SERVICE_GATE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
