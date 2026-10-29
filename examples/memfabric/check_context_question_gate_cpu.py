#!/usr/bin/env python3
"""CPU preconditions/default routing; no model HTTP or device operations."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import direct_context_question_gate as gate
from make_context_question_request import request_body


class ContextQuestionChecks(unittest.TestCase):
    def test_default_is_original_gate_with_original_frozen_options(self):
        with patch.object(gate.stock, "run") as original, patch.object(gate, "run_custom") as custom:
            gate.main(["--output", "unused", "--expected-prompt-tokens", "1373",
                       "--expected-copy-bytes", "188743680"])
        original.assert_called_once()
        custom.assert_not_called()
        self.assertEqual(original.call_args.args[0].expected_prompt_tokens, 1373)
        self.assertEqual(original.call_args.args[0].expected_copy_bytes, 188743680)

    def test_custom_request_discards_frozen_token_and_byte_expectations(self):
        args, extra, custom = gate.parse(["--output", "unused", "--request-json", "seed.json",
            "--expected-answer", "VELM42", "--cold-reference", "--session-id", "cpu-case",
            "--p-trace-url", "http://P/snapshot", "--d-trace-url", "http://D/snapshot",
            "--expected-prompt-tokens", "1373", "--expected-copy-bytes", "188743680"])
        self.assertTrue(custom and extra.cold_reference)
        self.assertIsNone(args.expected_prompt_tokens)
        self.assertIsNone(args.expected_copy_bytes)

    def test_unique_header_code_not_system_or_tail(self):
        body = request_body()
        gate.validate_request(body, "VELM42")
        self.assertEqual(sum(message["content"].count("VELM42") for message in body["messages"]), 1)
        self.assertNotIn("VELM42", body["messages"][0]["content"])
        self.assertNotIn("VELM42", body["messages"][-1]["content"].split("End of entries.")[-1])
        self.assertNotIn("ORCHID", str(body))
        duplicate = deepcopy(body)
        duplicate["messages"][-1]["content"] += " VELM42"
        with self.assertRaisesRegex(AssertionError, "exactly once"):
            gate.validate_request(duplicate, "VELM42")
        wrong_system = deepcopy(body)
        wrong_system["messages"][-1]["content"] = wrong_system["messages"][-1]["content"].replace("VELM42", "EMPTY")
        wrong_system["messages"][0]["content"] += " VELM42"
        with self.assertRaisesRegex(AssertionError, "system"):
            gate.validate_request(wrong_system, "VELM42")

    def test_decoded_first_chunk_must_contain_whole_code_without_later_repetition(self):
        gate.verify_code_chunks("header Identifier: VELM42", "filler only", "VELM42")
        for first, later in (("header VELM", "42 tail"), ("VELM42", "tail VELM42")):
            with self.assertRaises(AssertionError):
                gate.verify_code_chunks(first, later, "VELM42")

    def test_tp4_capacity_rejected_before_producer_is_started(self):
        self.assertEqual(gate.check_capacity({"free_slots": 64}, 1500, 128, 4), 11)
        self.assertEqual(gate.check_capacity({"free_slots": 64}, 2048, 128, 4), 16)
        with self.assertRaisesRegex(ValueError, "shorten it before starting P"):
            gate.check_capacity({"free_slots": 64}, 2176, 128, 4)

    def test_cold_compute_then_fresh_baseline_produces_only_cached_request_delta(self):
        def metrics(prompt, compute, external):
            return "\n".join([
                'vllm:prompt_tokens_total{model="test"} ' + str(prompt),
                'vllm:prompt_tokens_by_source_total{source="local_compute"} ' + str(compute),
                'vllm:prompt_tokens_by_source_total{source="local_cache_hit"} 0',
                'vllm:prompt_tokens_by_source_total{source="external_kv_transfer"} ' + str(external)])
        before, cold, restored = metrics(0, 0, 0), metrics(1500, 1500, 0), metrics(3000, 1592, 1408)
        self.assertEqual(gate.metric_deltas(before, cold), dict(prompt=1500, compute=1500, external=0, local_prefix=0))
        self.assertEqual(gate.metric_deltas(cold, restored), dict(prompt=1500, compute=92, external=1408, local_prefix=0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
