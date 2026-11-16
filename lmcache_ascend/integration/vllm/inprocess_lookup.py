# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side lookup state carried over vLLM's existing step messages."""

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import count
import struct
import sys


@dataclass
class LookupRequest:
    sequence: int
    request_id: str
    # Long uint32 prompts travel as little-endian byte planes. Short prompts
    # and wide multimodal IDs keep the original integer representation.
    tokens: list[int] | tuple[bytes, ...]
    configs: dict | None

    @property
    def token_count(self) -> int:
        return len(self.tokens[0]) if isinstance(self.tokens, tuple) else len(self.tokens)

    def owner_tokens(self) -> list[int]:
        if isinstance(self.tokens, list):
            return self.tokens
        count = self.token_count
        words = bytearray(4 * count)
        for i, plane in enumerate(self.tokens):
            lane = i if sys.byteorder == "little" else 3 - i
            words[lane::4] = plane
        return memoryview(words).cast("I").tolist()


def _snapshot(token_ids):
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, Iterator):
        token_ids = list(token_ids)
    if not hasattr(token_ids, "__len__") or len(token_ids) < 256:
        return list(token_ids)
    try:
        words = struct.pack(f"<{len(token_ids)}I", *token_ids)
    except (struct.error, OverflowError):
        return list(token_ids)
    planes = [words[i::4] for i in range(4)]
    while len(planes) > 1 and planes[-1].count(0) == len(token_ids):
        planes.pop()
    return tuple(planes)


class StepLookupClient:
    """One pinned lookup per admission attempt, answered by all worker ranks."""

    def __init__(self, world_size: int):
        self.world_size = world_size
        self.sequence = count()
        self.active: dict[str, int] = {}
        self.unsent: dict[str, LookupRequest] = {}
        self.answers: dict[int, dict[int, int]] = {}

    def supports_producer_reuse(self) -> bool:
        return True

    def lookup_cache(self, lookup_id: str) -> int | None:
        if lookup_id not in self.active:
            return -1
        ranks = self.answers[self.active[lookup_id]]
        return min(ranks.values()) if len(ranks) == self.world_size else None

    def lookup(self, token_ids, lookup_id: str, request_configs=None) -> int | None:
        if lookup_id not in self.active:
            tokens = _snapshot(token_ids)
            seq = next(self.sequence)
            self.active[lookup_id] = seq
            self.answers[seq] = {}
            self.unsent[lookup_id] = LookupRequest(seq, lookup_id, tokens, request_configs)
        return self.lookup_cache(lookup_id)

    def take_requests(self) -> list[LookupRequest]:
        requests = list(self.unsent.values())
        self.unsent.clear()
        return requests

    def receive(self, replies: dict[int, dict[int, int]]) -> None:
        for seq, ranks in replies.items():
            if seq not in self.answers:
                continue  # Reply to an already cancelled admission attempt.
            for rank, tokens in ranks.items():
                if not 0 <= rank < self.world_size or tokens < 0:
                    raise ValueError("Invalid rank or token count in worker lookup reply")
                prior = self.answers[seq].setdefault(rank, tokens)
                if prior != tokens:
                    raise ValueError("Conflicting replies for the same rank and lookup")

    def clear_lookup_status(self, lookup_id: str) -> None:
        self.unsent.pop(lookup_id, None)
        self.answers.pop(self.active.pop(lookup_id, None), None)

    def close(self) -> None:
        self.active.clear()
        self.unsent.clear()
        self.answers.clear()
