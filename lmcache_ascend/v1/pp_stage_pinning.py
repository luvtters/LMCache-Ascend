# SPDX-License-Identifier: Apache-2.0
"""NPU-only block-index staging for the stage-local MP deployment."""

# Standard
import array

# Third Party
import torch

# First Party
from lmcache.v1.platform.devices.npu.cache_context import NpuCacheContext


class PinnedNpuCacheContext(NpuCacheContext):
    """Retain native NPU transfers, pinning only their small host index source."""

    def stage_block_ids(
        self, block_ids_per_group: list[list[int]]
    ) -> list[torch.Tensor]:
        """Stage unchanged group indices using the torch-npu host allocator.

        The allocator must retain the source until its asynchronous copy ends.
        No reusable host buffer is introduced; existing stream/event ordering
        and the parent's device staging-buffer ownership remain unchanged.

        Args:
            block_ids_per_group: Engine block IDs in native group order.

        Returns:
            Non-overlapping views, including empty groups, of the device buffer.

        Raises:
            ValueError: Indices exceed the parent's staging-buffer capacity.
            RuntimeError: NPU host pinning or the copy fails.
        """
        offsets = [0]
        flat = array.array("q")
        for blocks in block_ids_per_group:
            flat.extend(blocks)
            offsets.append(len(flat))
        total = offsets[-1]
        if total > self.block_ids_buffer_.shape[0]:
            raise ValueError("Block IDs exceed the pre-allocated staging buffer")
        if total:
            source = torch.frombuffer(flat, dtype=torch.long).pin_memory(device="npu")
            self.block_ids_buffer_[:total].copy_(source, non_blocking=True)
        return [
            self.block_ids_buffer_[offsets[i] : offsets[i + 1]]
            for i in range(len(block_ids_per_group))
        ]
