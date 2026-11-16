# SPDX-License-Identifier: Apache-2.0
"""Adapt registered stage-local KV tensors to the existing Ascend block operator."""

from collections import defaultdict
from math import prod

import torch

from lmcache_ascend import c_ops


def physical_planes(cache):
    if isinstance(cache, (tuple, list)):
        return tuple(cache)
    if cache.ndim == 5 and cache.shape[0] == 2:
        return cache.unbind(0)
    if cache.ndim == 5 and cache.shape[1] == 2:
        return cache.unbind(1)
    return (cache,)


class RegisteredBlockTransfer:
    """One packed chunk in local LMCache storage, with one table per vLLM group."""

    def __init__(self):
        self.load_stream = torch.npu.Stream()
        self.store_stream = torch.npu.Stream()

    def register(self, config, caches, block_sizes, chunk_size):
        self.groups = []
        self.row_bytes = 0
        self.chunk_size = chunk_size
        seen = set()
        devices = {p.device for v in caches.values() for p in physical_planes(v)}
        if len(devices) != 1:
            raise ValueError("Registered stage KV must reside on one device")
        self.device = devices.pop()
        if self.device != self.store_stream.device:
            raise ValueError("Registered KV device differs from the worker stream device")
        for index, group in enumerate(config.kv_cache_groups):
            bs = block_sizes[index]
            if bs <= 0 or chunk_size <= 0 or chunk_size % bs:
                raise ValueError("LMCache chunk size must be a multiple of every logical block size")
            layouts = defaultdict(list)
            for name in sorted(group.layer_names):
                if name not in caches:
                    continue  # Another PP stage owns this layer.
                if name in seen:
                    raise ValueError(f"Layer belongs to multiple KV groups: {name}")
                seen.add(name)
                planes = physical_planes(caches[name])
                if not 1 <= len(planes) <= 4:
                    raise ValueError(f"Unsupported plane count for {name}")
                if planes[0].ndim < 3 or planes[0].shape[0] <= 0:
                    raise ValueError(f"Invalid KV block dimensions for {name}")
                nb = planes[0].shape[0]
                widths, strides = [], []
                for plane in planes:
                    if plane.ndim < 3 or tuple(plane.shape[:2]) != (nb, bs):
                        raise ValueError(f"Expected dense [block, token, ...] KV plane for {name}")
                    width = prod(plane.shape[2:]) * plane.element_size()
                    stride = plane.stride(0) * plane.element_size()
                    if (not plane[0].is_contiguous() or width <= 0
                            or stride < bs * width or stride % 32):
                        raise ValueError(f"Unsupported token layout or block stride for {name}")
                    widths.append(width)
                    strides.append(stride)
                layouts[nb, tuple(widths), tuple(strides)].append(planes)
            for (nb, widths, strides), layers in layouts.items():
                row = sum(widths)
                if (row * bs) % 32:
                    raise ValueError("Packed KV page length must be aligned to 32 bytes")
                desc = c_ops.PageBufferShapeDesc()
                desc.nl, desc.nb, desc.bs = len(layers), nb, bs
                desc.kv_size, desc.nh, desc.hs, desc.element_size = 1, 1, row, 1
                desc.block_stride_elems = strides[0]
                desc.num_planes = len(widths)
                desc.plane_slot_bytes, desc.plane_block_stride_bytes = widths, strides
                ptrs = torch.tensor([p.data_ptr() for layer in layers for p in layer],
                                    device=self.device, dtype=torch.long)
                self.groups.append((index, self.row_bytes, ptrs, desc))
                self.row_bytes += row * len(layers)
        if not self.groups or seen != set(caches):
            raise ValueError(f"KV registration has unmapped layers: {sorted(set(caches) - seen)}")
        # The first STORE may be submitted on a different thread and stream.
        self.registered = torch.npu.Event()
        self.registered.record()

    def _copy(self, obj, start, end, tables, direction, cached_tokens=0):
        span = end - start
        raw = obj.tensor
        if (span != self.chunk_size or raw is None or not raw.is_contiguous()
                or raw.numel() * raw.element_size() != span * self.row_bytes):
            raise ValueError("Invalid packed LMCache chunk length")
        for group, offset, ptrs, desc in self.groups:
            bs = desc.bs
            if start % bs:
                raise ValueError("Unaligned KV transfer start")
            blocks = tables[group][start // bs:end // bs]
            if len(blocks) != span // bs or any(b <= 0 or b >= desc.nb for b in blocks):
                raise ValueError(f"Incomplete or invalid physical blocks for group {group}")
            skip = max(0, min(cached_tokens, end) - start)
            if skip % bs:
                raise ValueError("APC prefix ends inside a destination block")
            if skip == span:
                continue
            ids = torch.tensor(blocks, dtype=torch.long, device=self.device)
            c_ops.multi_layer_block_kv_transfer(
                ptrs, [raw.data_ptr() + span * offset], ids, self.device, direction,
                desc, span, c_ops.EngineKVFormat.NL_X_NP_X_NB_BS_ONE_HS, skip // bs,
            )

    def to_gpu(self, obj, start, end, **kwargs):
        torch.npu.current_stream().wait_event(self.registered)
        self._copy(obj, start, end, kwargs["block_tables"], c_ops.TransferDirection.H2D,
                   kwargs.get("vllm_cached_tokens", 0))

    def batched_to_gpu(self, objs, starts, ends, **kwargs):
        with torch.npu.stream(self.load_stream):
            try:
                for obj, start, end in zip(objs, starts, ends, strict=True):
                    self.to_gpu(obj, start, end, **kwargs)
            finally:
                self.load_stream.synchronize()

    def batched_from_gpu(self, objs, starts, ends, **kwargs):
        with torch.npu.stream(self.store_stream):
            self.store_stream.wait_event(self.registered)
            self.store_stream.wait_event(kwargs["ordering_event"])
            try:
                for obj, start, end in zip(objs, starts, ends, strict=True):
                    self._copy(obj, start, end, kwargs["block_tables"], c_ops.TransferDirection.D2H)
            finally:
                self.store_stream.synchronize()
