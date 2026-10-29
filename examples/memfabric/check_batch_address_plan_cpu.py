#!/usr/bin/env python3
"""Narrow CPU content and address checks; no SDK or accelerator imports."""
import unittest

from batch_address_plan import plan_segments


class AddressPlanChecks(unittest.TestCase):
    def planes(self, token_stride=3):
        return [dict(component=c, layer=l, base_ptr=1000 + (c * 2 + l) * 1000,
                     offset_bytes=7, block_stride_bytes=16,
                     token_stride_bytes=token_stride, token_bytes=3, num_blocks=5)
                for c in (0, 1) for l in range(2)]

    def test_reverse_pages_unaligned_logical_range_matches_independent_content(self):
        planes = self.planes(token_stride=4)
        blocks = [3, 1, 4]
        segments = plan_segments(planes, blocks, block_size=3, token_start=1,
                                 token_count=6, pool_base=10000, pool_bytes=72)
        source, destination, expected = {}, {}, bytearray()
        for plane in planes:
            for block in range(5):
                for token in range(3):
                    for byte in range(3):
                        address = plane["base_ptr"] + 7 + block * 16 + token * 4 + byte
                        source[address] = (plane["component"] * 79 + plane["layer"] * 37
                                           + block * 13 + token * 3 + byte) % 256
            for logical in range(1, 7):
                page, token = divmod(logical, 3)
                expected.extend(source[plane["base_ptr"] + 7 + blocks[page] * 16 + token * 4 + b]
                                for b in range(3))
        for segment in segments:
            for byte in range(segment["size"]):
                address = segment["dst"] + byte
                self.assertNotIn(address, destination)
                destination[address] = source[segment["src"] + byte]
        self.assertEqual(bytes(destination[a] for a in range(10000, 10072)), expected)
        self.assertEqual(sum(s["size"] for s in segments), 72)

    def test_contiguous_tokens_merge_but_reversed_pages_do_not(self):
        segments = plan_segments(self.planes(), [3, 1], block_size=3,
                                 token_start=0, token_count=6, pool_base=10000, pool_bytes=72)
        self.assertEqual(len(segments), 8)
        self.assertEqual({s["size"] for s in segments}, {9})
        self.assertEqual(len({s["src"] for s in segments}), 8)

    def test_bad_table_range_layout_or_capacity_rejected(self):
        options = dict(block_size=3, token_start=0, token_count=6, pool_base=10000, pool_bytes=72)
        for override in ({"token_start": 1}, {"pool_bytes": 71}):
            with self.assertRaises(ValueError):
                plan_segments(self.planes(), [3, 1], **dict(options, **override))
        with self.assertRaises(ValueError):
            plan_segments(self.planes(), [5, 1], **options)
        with self.assertRaises(ValueError):
            plan_segments(self.planes(token_stride=2), [3, 1], **options)


if __name__ == "__main__":
    unittest.main(verbosity=2)
