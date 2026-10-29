#!/usr/bin/env python3
"""CPU-only follow-up contracts; the HTTP fixture is local and carries no KV."""
from copy import deepcopy
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import direct_followups as follow


class FollowupChecks(unittest.TestCase):
    def base(self):
        return dict(model="mf-direct", max_tokens=8, temperature=0,
                    chat_template_kwargs={"enable_thinking": False}, messages=[
                        dict(role="system", content="Reply with exactly ORCHID."),
                        dict(role="user", content="\n".join("Record " + str(i) + ": ORCHID" for i in range(64)))])

    def test_variants_preserve_first_half_without_mutating_gate(self):
        original = self.base()
        before = deepcopy(original)
        variants = follow.variants(original)
        self.assertEqual(original, before)
        self.assertEqual(variants["warm"], original)
        self.assertEqual(variants["partial"]["messages"][-1]["content"].splitlines()[:32],
                         original["messages"][-1]["content"].splitlines()[:32])
        self.assertNotEqual(variants["partial"]["messages"][-1], original["messages"][-1])
        self.assertNotEqual(variants["cold"]["messages"][0], original["messages"][0])
        self.assertEqual(follow.common_chunks([1, 2, 3, 4], [1, 2, 8, 4], 2), 1)
        self.assertEqual(follow.common_chunks([1, 2, 3], [1, 9, 3], 2), 0)

    def test_dry_run_performs_no_http_and_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "request.json").write_text(json.dumps(self.base()), encoding="utf-8")
            with patch.object(follow, "http", side_effect=AssertionError("unexpected HTTP")), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                follow.main(["--gate-dir", str(root), "--decoder", "http://unused",
                    "--service-http", "http://unused", "--trace-url", "http://unused/snapshot",
                    "--session-id", "cpu-case", "--output", str(root / "new-output")])
            self.assertEqual(json.loads(output.getvalue())["status"], "DRY_RUN")
            self.assertFalse((root / "new-output").exists())

    def test_raw_http_body_is_preserved_byte_for_byte(self):
        payload = b"\x00\xff\r\ncomplete-response-body\n"

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                raw, metadata = follow.http("http://127.0.0.1:" + str(server.server_port), root)
                self.assertEqual(raw, payload)
                self.assertEqual((root / "response.body.bin").read_bytes(), payload)
                self.assertEqual(metadata["response_bytes"], len(payload))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def trace(self, copies):
        records = []
        for shard in (0, 1):
            common = dict(role="D", worker_shard=shard, tp_size=2, pid=10 + shard,
                host_id="cpu-host", session_id="cpu-case", local_ptr=100 + shard,
                bytes=9 << 20, thread_id=7, memory_type="npu")
            records.append(dict(common, kind="npu_scratch", phase="complete", mono_ns=10))
            for index in range(copies):
                for phase, stamp in (("begin", 110 + index * 2), ("complete", 111 + index * 2)):
                    records.append(dict(common, kind="bm_copy", phase=phase, mono_ns=stamp,
                        direction="G2L", bm_rank=3 + shard, src=1000 + index, dst=100 + shard))
        return dict(host_id="cpu-host", run_id="cpu-case", mono_ns=100 if not copies else 200, records=records)

    def test_per_rank_actual_bytes_partial_and_zero_transfer_cold_case(self):
        args = SimpleNamespace(tp_size=2, session_id="cpu-case")
        before = self.trace(0)
        for copies in (0, 3):
            result = follow.trace_counts(args, before, self.trace(copies), copies)
            self.assertEqual(result["0"]["completed_bytes"], copies * (9 << 20))
            self.assertEqual(result["1"]["completed_copies"], copies)
        wrong = self.trace(3)
        wrong["records"][-1]["bm_rank"] = 99
        with self.assertRaises(AssertionError):
            follow.trace_counts(args, before, wrong, 3)
        missing = self.trace(3)
        missing["records"].pop()
        with self.assertRaisesRegex(AssertionError, "in flight"):
            follow.trace_counts(args, before, missing, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
