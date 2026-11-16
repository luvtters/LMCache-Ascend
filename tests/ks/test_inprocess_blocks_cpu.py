# SPDX-License-Identifier: Apache-2.0
"""Exercise the registered transfer with real CPU pages and a byte-copy operator."""

import ctypes
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch


def copy_pages(ptrs, addresses, ids, device, direction, desc, tokens, fmt, skip):
    assert device.type == "cpu" and fmt == 17
    row = sum(desc.plane_slot_bytes)
    for layer in range(desc.nl):
        for token in range(skip * desc.bs, tokens):
            block, slot = divmod(token, desc.bs)
            cursor = addresses[0] + (layer * tokens + token) * row
            for plane, width in enumerate(desc.plane_slot_bytes):
                address = ptrs[layer * desc.num_planes + plane].item()
                address += ids[block].item() * desc.plane_block_stride_bytes[plane]
                address += slot * width
                dst, src = (cursor, address) if direction == "store" else (address, cursor)
                ctypes.memmove(dst, src, width)
                cursor += width


class TransferTests(unittest.TestCase):
    def setUp(self):
        op = NS(PageBufferShapeDesc=NS, EngineKVFormat=NS(NL_X_NP_X_NB_BS_ONE_HS=17),
                TransferDirection=NS(H2D="load", D2H="store"),
                multi_layer_block_kv_transfer=copy_pages)
        stream = Mock(device=torch.device("cpu"))
        npu = NS(Stream=lambda: stream, stream=lambda _: nullcontext(), Event=Mock,
                 current_stream=lambda: stream)
        patch_npu = patch.object(torch, "npu", npu, create=True)
        patch_npu.start()
        self.addCleanup(patch_npu.stop)
        path = Path(__file__).resolve().parents[2] / "lmcache_ascend/v1/npu_connector/inprocess_blocks.py"
        spec = importlib.util.spec_from_file_location("block_transfer_cpu", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"lmcache_ascend": NS(c_ops=op)}):
            spec.loader.exec_module(module)
        self.backend = module.RegisteredBlockTransfer()
        self.layers = {
            "L.1": (torch.arange(5 * 2 * 16, dtype=torch.int16).reshape(5, 2, 16),
                    torch.arange(5 * 2 * 16, dtype=torch.uint8).reshape(5, 2, 16)),
            "L.2": torch.arange(5 * 4 * 8, dtype=torch.uint8).reshape(5, 4, 8),
        }
        self.config = NS(kv_cache_groups=[NS(layer_names=["L.1"]), NS(layer_names=["L.2"])])
        self.stream = stream

    def test_mixed_planes_distinct_group_tables_and_apc(self):
        transfer = self.backend
        transfer.register(self.config, self.layers, [2, 4], 8)
        source = ((1, 2, 3, 4), (4, 3))
        target = ((4, 3, 2, 1), (1, 2))
        obj = NS(tensor=torch.empty(8 * (32 + 16 + 8), dtype=torch.uint8))
        event = Mock()
        transfer.batched_from_gpu([obj], [0], [8], block_tables=source, ordering_event=event)
        self.stream.wait_event.assert_any_call(event)
        before = {}
        for name, cache in self.layers.items():
            planes = cache if isinstance(cache, tuple) else (cache,)
            before[name] = [plane.clone() for plane in planes]
            for plane in planes:
                plane.zero_()
        transfer.batched_to_gpu([obj], [0], [8], block_tables=target, vllm_cached_tokens=4)
        for group, (name, cache) in enumerate(self.layers.items()):
            planes = cache if isinstance(cache, tuple) else (cache,)
            for plane, original in zip(planes, before[name]):
                for i, (src, dst) in enumerate(zip(source[group], target[group])):
                    expected = original[src] if i >= len(target[group]) // 2 else torch.zeros_like(plane[dst])
                    self.assertTrue(torch.equal(plane[dst], expected))
        self.assertEqual(self.stream.synchronize.call_count, 2)

    def test_bad_chunk_and_bad_table(self):
        with self.assertRaisesRegex(ValueError, "multiple"):
            self.backend.register(self.config, self.layers, [2, 4], 6)
        self.backend.register(self.config, self.layers, [2, 4], 8)
        obj = NS(tensor=torch.empty(8 * self.backend.row_bytes, dtype=torch.uint8))
        with self.assertRaisesRegex(ValueError, "invalid physical blocks"):
            self.backend.batched_from_gpu([obj], [0], [8], block_tables=((0, 1, 2, 3), (1, 2)),
                                          ordering_event=Mock())
        self.stream.synchronize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
