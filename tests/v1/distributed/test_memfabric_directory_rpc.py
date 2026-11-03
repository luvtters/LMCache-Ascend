# SPDX-License-Identifier: Apache-2.0
"""Two independent HTTP clients and the real volatile directory; CPU only."""
import http.client
import json
from pathlib import Path
import runpy
import tempfile
import threading
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
coord = runpy.run_path(str(ROOT / "examples/memfabric/pool_coordination.py"))
DirectoryClient = runpy.run_path(str(
    ROOT / "lmcache_ascend/v1/distributed/memfabric_directory_rpc.py"))["DirectoryClient"]


class DirectoryRPCTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.geometry = dict(storage_rank=1, pool_gb=1, slot_size_mb=1024)
        self.contract = dict(world_size=2, **self.geometry, store_url="tcp://owner:28580")
        self.start(self.tmp.name)

    def start(self, directory, port=0):
        self.server = coord["make_server"]("127.0.0.1", port, directory, "run",
                                           shared_directory=self.geometry)
        self.thread = threading.Thread(target=self.server.serve_forever, args=(0.01,))
        self.thread.start()
        self.stop = self.add_server_cleanup(self.server, self.thread)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def add_server_cleanup(self, server, thread):
        def stop():
            if thread.is_alive():
                server.shutdown()
                server.server_close()
                thread.join()
        self.addCleanup(stop)
        return stop

    def client(self, rank, **changes):
        return DirectoryClient(self.url, "run", rank, {**self.contract, **changes})

    def test_cross_service_publication_hides_uncommitted_slot_and_checks_pin(self):
        writer, reader = self.client(0), self.client(1)
        self.assertEqual(writer.epoch, reader.epoch)
        layout = [[[32]], ["uint8"], "binary"]
        slot = writer.call("reserve", key="k", size=32, layout=layout)
        self.assertIsNone(reader.call("pin", key="k"))
        with self.assertRaisesRegex(ValueError, "writer"):
            reader.call("commit", key="k", slot=slot)
        writer.call("commit", key="k", slot=slot)
        entry = reader.call("pin", key="k", hint=layout[:2])
        self.assertEqual(entry, dict(slot=slot, size=32, layout=layout))
        self.assertEqual(reader.call("load", key="k"), entry)
        self.assertIsNone(writer.call("load", key="k"))
        reader.call("unpin", key="k")
        self.assertEqual(writer.call("status")["total_pins"], 0)
        # Existing JOIN events remain available at the same endpoint.
        event = dict(name="bm-joined-0", epoch=writer.epoch)
        coord["publish"](self.url, "run", event["name"], event)
        self.assertEqual(coord["wait_event"](self.url, "run", event["name"], 1), event)

    def test_contract_failures_do_not_claim_rank_or_replace_first_store(self):
        with self.assertRaisesRegex(RuntimeError, "geometry"):
            self.client(0, slot_size_mb=32)
        with self.assertRaisesRegex(RuntimeError, "geometry"):
            self.client(0, world_size=True)
        with self.assertRaises(RuntimeError):
            DirectoryClient(self.url, "other-group", 0, self.contract)
        self.client(0)
        with self.assertRaisesRegex(RuntimeError, "already claimed"):
            self.client(0)
        with self.assertRaisesRegex(RuntimeError, "store_url"):
            self.client(1, store_url="tcp://wrong:28580")
        self.assertEqual(self.client(1).call("status")["objects"], 0)

    def test_eviction_notifies_only_the_publisher_on_its_next_successful_call(self):
        publisher, peer = self.client(0), self.client(1)
        first, second = [], []
        publisher.on_retired, peer.on_retired = first.append, second.append
        layout = [[[32]], ["uint8"], "binary"]
        slot = publisher.call("reserve", key="old", size=32, layout=layout)
        publisher.call("commit", key="old", slot=slot)
        replacement = peer.call("reserve", key="new", size=32, layout=layout)
        peer.call("commit", key="new", slot=replacement)
        self.assertEqual(second, [])
        with self.assertRaises(ValueError):
            publisher.call("commit", key="missing", slot=slot)
        self.assertEqual(first, [])
        publisher.call("status")
        publisher.call("delete", key="new")
        self.assertEqual(first, [[["old", 32]]])
        self.assertEqual(second, [])
        peer.call("status")
        self.assertEqual(second, [[["new", 32]]])

    def test_invalid_retirements_and_callback_failures_are_fatal(self):
        client = self.client(0)
        def reject(batch):
            raise ValueError("consumer rejected batch")
        client.on_retired = reject
        for batch, error in (([["k", True]], "invalid"), ([["k", 32]], "failed to apply")):
            with patch.object(client, "_post", return_value=dict(
                    epoch=client.epoch, result=None, retired=batch)):
                with self.assertRaisesRegex(RuntimeError, error):
                    client.call("status")

    def test_status_that_exceeds_response_limit_is_a_fatal_error(self):
        self.stop()
        self.geometry["slot_size_mb"] = self.contract["slot_size_mb"] = 512
        self.start(Path(self.tmp.name) / "large-status")
        writer, reader = self.client(0), self.client(1)
        layout = [[[4]], ["uint8"], "bin"]
        for key in ("a", "b"):
            slot = writer.call("reserve", key=key, size=4, layout=layout, cache_salt=key * 34000)
            writer.call("commit", key=key, slot=slot)
        with self.assertRaisesRegex(RuntimeError, "HTTP 500"):
            reader.call("status")

    def test_restart_refuses_existing_run_and_fences_old_epoch_on_new_server(self):
        old = self.client(0)
        port = self.server.server_port
        self.stop()
        with self.assertRaises(FileExistsError):
            coord["make_server"]("127.0.0.1", port, self.tmp.name, "run",
                                 shared_directory=self.geometry)
        with self.assertRaises(RuntimeError):
            old.call("status")
        self.start(Path(self.tmp.name) / "another-root", port)
        new = self.client(0)
        self.assertNotEqual(old.epoch, new.epoch)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            old.call("delete", key="k")

    def test_transport_rejects_invalid_envelopes_and_directory_rejects_raw_addresses(self):
        client = self.client(0)
        with self.assertRaises(ValueError):
            client.call("reserve", key="k", size=4, layout=[[[4]], ["uint8"], "bin"], offset=0)
        with self.assertRaises(ValueError):
            client.call("__getattribute__", key="k")
        call = dict(rank=1, epoch=client.epoch, claim=client._claim, op="status", args={})
        for payload, code in ((json.dumps(call).encode(), 409), (b"[]", 400),
                              (b"{" + b" " * 65536 + b"}", 400)):
            conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
            try:
                conn.request("POST", "/runs/run/directory/call", payload,
                             {"Content-Type": "application/json"})
                self.assertEqual(conn.getresponse().status, code)
            finally:
                conn.close()
        self.assertEqual(client.call("status")["free_slots"], 1)

    def test_lost_mutation_reply_is_never_replayed(self):
        writer, observer = self.client(0), self.client(1)
        responses = []
        def drop_reply(handler, code, value):
            responses.append((code, value))
        with patch.object(self.server.RequestHandlerClass, "respond", drop_reply):
            with self.assertRaisesRegex(RuntimeError, "outcome unknown"):
                writer.call("reserve", key="k", size=4, layout=[[[4]], ["uint8"], "bin"])
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0][0], 200)
        self.assertEqual(observer.call("status")["pending_writes"], 1)


if __name__ == "__main__":
    unittest.main()
