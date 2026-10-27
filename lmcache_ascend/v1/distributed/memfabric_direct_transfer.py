# SPDX-License-Identifier: Apache-2.0
"""Worker-owned NPU scratch storage for direct BM transfers.

Only control metadata and block IDs originate on the host. KV payloads remain
in engine NPU pages, this NPU allocation, and the BM pool. No IPC ownership is
transferred: closing this helper never closes the worker's KV tensors.
"""


class NpuDirectTransfer:
    def __init__(self, kv_caches, tokens_per_chunk, blocks_per_chunk,
                 layout_hints=None, engine_type=None):
        import torch
        from lmcache.utils import EngineType
        from lmcache.v1.gpu_connector import utils
        import lmcache_ascend.c_ops as ops

        if tokens_per_chunk <= 0 or blocks_per_chunk <= 0:
            raise ValueError("Chunk geometry must be positive")
        self.torch, self.ops = torch, ops
        self.format, self.kv = utils.normalize_kv_and_discover_format(
            list(kv_caches.values()), engine_type or EngineType.VLLM,
            layout_hints=layout_hints)
        self.device = utils.get_device(self.kv)
        if self.device.type != "npu":
            raise ValueError("MemFabric direct requires actual NPU KV tensors")
        torch.npu.set_device(self.device)
        self.block_size = utils.get_block_size(self.kv, self.format)
        if tokens_per_chunk != self.block_size * blocks_per_chunk:
            raise ValueError("Direct chunks must contain complete engine blocks")
        # This first direct path is intentionally limited to split K/V GQA.
        if utils.get_kv_size(self.kv, self.format) != 2:
            raise ValueError("MemFabric direct currently supports split K/V GQA only")
        self.blocks_per_chunk = blocks_per_chunk
        self.tokens_per_chunk = tokens_per_chunk
        layers = utils.get_num_layers(self.kv, self.format)
        width = (utils.get_num_heads(self.kv, self.format)
                 * utils.get_head_size(self.kv, self.format))
        dtype = utils.get_dtype(self.kv, self.format)
        self.num_blocks = utils.get_num_blocks(self.kv, self.format)
        self.shape_desc = utils.make_page_buffer_shape_desc(
            self.kv, self.format, layer_idx=0, num_layers_in_group=layers,
            num_blocks=self.num_blocks, block_size=self.block_size,
            block_stride_elems=utils.resolve_block_stride_and_log_layout(
                self.kv, self.format, layer_idx=0, group_idx=0))
        self.buffer = torch.empty((2, layers, tokens_per_chunk, width),
                                  dtype=dtype, device=self.device)
        self.pointers = torch.tensor(utils.get_group_data_ptrs(
            self.kv, self.format, list(range(layers))), dtype=torch.int64,
            device=self.device)
        self.size = self.buffer.numel() * self.buffer.element_size()
        self.layout = {"shape": list(self.buffer.shape),
                       "dtype": str(dtype), "format": "2LTD"}

    def synchronize(self):
        self.torch.npu.synchronize(self.device)
        self._inflight_ids = None

    def _transfer(self, block_ids, *, gather, skip_tokens=0):
        if len(block_ids) != self.blocks_per_chunk:
            raise ValueError("Expected exactly one complete chunk of block IDs")
        if any(type(i) is not int or not 0 <= i < self.num_blocks for i in block_ids):
            raise ValueError("Invalid engine block ID")
        if not 0 <= skip_tokens <= self.tokens_per_chunk or skip_tokens % self.block_size:
            raise ValueError("Direct retrieve skip must be block aligned")
        ids = self.torch.tensor(block_ids, dtype=self.torch.int64, device=self.device)
        self._inflight_ids = ids
        # The direction names describe page->object / object->page. Passing
        # a true NPU object pointer bypasses native HostStage allocations.
        direction = self.ops.TransferDirection.D2H if gather else self.ops.TransferDirection.H2D
        try:
            self.ops.multi_layer_block_kv_transfer(
                self.pointers, [self.buffer.data_ptr()], ids, self.device,
                direction, self.shape_desc, self.tokens_per_chunk, self.format,
                skip_tokens // self.block_size)
        finally:
            # A native call may enqueue work before raising. Keep IDs and
            # scratch storage alive until synchronization also succeeds.
            self.synchronize()

    def gather(self, block_ids):
        self._transfer(block_ids, gather=True)

    def scatter(self, block_ids, skip_tokens=0):
        self._transfer(block_ids, gather=False, skip_tokens=skip_tokens)
