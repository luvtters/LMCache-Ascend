# SPDX-License-Identifier: Apache-2.0
"""CPU TP transactions and worker routing; no NPU/BM execution evidence."""

from concurrent.futures import ThreadPoolExecutor
import types
import unittest

from test_memfabric_direct import DIRECT, Client, PoolIndex, Session, Transfer, key


class TpPoolTests(unittest.TestCase):
    def pool(self, tp=2, slots=8):
        pool = PoolIndex(pool_bytes=slots * 64, slot_size=64,
                         storage_rank=0, world_size=1 + 2 * tp, tp_size=tp)
        self.layout = {"shape": [2, 36, 128, 1024 // tp],
                       "dtype": "bf16", "num_heads": 8 // tp}
        for rank in range(1, 1 + 2 * tp):
            pool.register(rank=rank, member_id=str(rank), shard=(rank - 1) % tp,
                          tp_size=tp, layout=self.layout, size=32)
        self.assertTrue(pool.status()["registration_ready"])
        return pool

    def owner(self, pool, rank):
        return dict(epoch=pool.epoch, rank=rank, member_id=str(rank))

    def prepare(self, pool, name, shard):
        return pool.prepare_store(**self.owner(pool, shard + 1), key=name,
                                  size=32, layout=self.layout, shard=shard,
                                  tp_size=pool.tp_size)

    def store(self, pool, name, shard):
        item = self.prepare(pool, name, shard)
        if item["write_id"]:
            pool.commit_store(**self.owner(pool, shard + 1), write_id=item["write_id"])
        return item["descriptor"]

    def test_out_of_order_commit_missing_shard_and_contiguous_prefix(self):
        for tp in (2, 4, 8):
            with self.subTest(tp=tp):
                pool = self.pool(tp, slots=tp * 3)
                for shard in reversed(range(tp)):
                    self.store(pool, "a", shard)
                    expected = 1 if shard == 0 else 0
                    for reader in range(tp):
                        self.assertEqual(len(pool.lookup(keys=["a"], shard=reader,
                            tp_size=tp)["descriptors"]), expected)
                for shard in range(tp - 1):
                    self.store(pool, "b", shard)
                for shard in range(tp):
                    self.store(pool, "c", shard)
                for reader in range(tp):
                    self.assertEqual(len(pool.lookup(keys=["a", "b", "c"],
                        shard=reader, tp_size=tp)["descriptors"]), 1)
                self.store(pool, "b", tp - 1)
                self.assertEqual(len(pool.lookup(keys=["a", "b", "c"],
                    tp_size=tp)["descriptors"]), 3)

    def test_scheduler_pins_all_shards_and_each_reader_owns_its_lease(self):
        pool = self.pool(slots=4)
        descriptors = [self.store(pool, "a", s) for s in range(2)]
        pool.lookup(keys=["a"], tp_size=2, request_id="schedule")
        self.assertEqual(pool.status()["total_pins"], 2)
        leases = [pool.acquire(**self.owner(pool, 3 + s), descriptor=d)
                  for s, d in enumerate(descriptors)]
        self.assertEqual(pool.status()["total_pins"], 4)
        pool.release(epoch=pool.epoch, request_id="schedule", dma_complete=True)
        with self.assertRaisesRegex(ValueError, "another TP shard"):
            pool.acquire(**self.owner(pool, 3), descriptor=descriptors[1])
        with self.assertRaisesRegex(RuntimeError, "stale read lease"):
            pool.release(**self.owner(pool, 4), lease_id=leases[0]["lease_id"], dma_complete=True)
        for s, lease in enumerate(leases):
            pool.release(**self.owner(pool, 3 + s), lease_id=lease["lease_id"], dma_complete=True)
        self.assertEqual(pool.status()["total_pins"], 0)

    def test_eviction_removes_whole_unpinned_group(self):
        pool = self.pool(slots=2)
        for s in range(2):
            self.store(pool, "a", s)
        pool.lookup(keys=["a"], tp_size=2, request_id="pin")
        with self.assertRaisesRegex(RuntimeError, "full"):
            self.prepare(pool, "b", 0)
        pool.release(epoch=pool.epoch, request_id="pin", dma_complete=True)
        self.store(pool, "b", 0)
        self.assertEqual(pool.status()["objects"], 1)
        self.assertEqual(pool.lookup(keys=["a"], tp_size=2)["descriptors"], [])
        self.store(pool, "b", 1)
        self.assertEqual(len(pool.lookup(keys=["b"], tp_size=2)["descriptors"]), 1)

    def test_layout_rank_and_tp_mismatch_rejected(self):
        pool = self.pool()
        with self.assertRaisesRegex(ValueError, "actual KV layouts"):
            pool.register(rank=1, member_id="1", shard=0, tp_size=2,
                          layout={"wrong": 1}, size=32)
        with self.assertRaisesRegex(ValueError, "real TP shard"):
            pool.register(rank=2, member_id="2", shard=0, tp_size=2,
                          layout=self.layout, size=32)
        with self.assertRaises(ValueError):
            pool.lookup(keys=["a"], tp_size=1)

    def test_missing_commit_or_aborted_shard_stays_invisible(self):
        pool = self.pool()
        first, late = self.prepare(pool, "a", 0), self.prepare(pool, "a", 1)
        pool.commit_store(**self.owner(pool, 2), write_id=late["write_id"])
        self.assertEqual(pool.lookup(keys=["a"], tp_size=2)["descriptors"], [])
        pool.abort_store(**self.owner(pool, 1), write_id=first["write_id"], dma_complete=False)
        self.assertEqual(pool.status()["quarantined_slots"], 1)
        self.assertEqual(pool.lookup(keys=["a"], tp_size=2)["descriptors"], [])
        pool.abort_store(**self.owner(pool, 1), write_id=first["write_id"], dma_complete=True)
        self.store(pool, "a", 0)
        self.assertEqual(len(pool.lookup(keys=["a"], tp_size=2)["descriptors"]), 1)

    def test_layout_keys_share_prefix_across_workers_but_isolate_layout(self):
        a, b = key(worker=0), key(worker=1, request="other")
        a.world_size = b.world_size = 2
        layout = {"num_heads": 4, "head_size": 128}
        self.assertEqual(DIRECT.chunk_keys(a, 128, layout), DIRECT.chunk_keys(b, 128, layout))
        self.assertNotEqual(DIRECT.chunk_keys(a, 128, layout),
                            DIRECT.chunk_keys(a, 128, dict(layout, num_heads=2)))
        with self.assertRaises(ValueError):
            DIRECT.chunk_keys(a, 128)


class TpWorkerTests(unittest.TestCase):
    def test_real_strategy_shard_routes_bm_rank_and_full_group_lookup(self):
        client = Client()
        client.index = PoolIndex(pool_bytes=1 << 30, slot_size=32 << 20,
                                storage_rank=0, world_size=5, tp_size=2)
        contexts = []
        for base in (1, 3):
            for shard in (0, 1):
                strategy = types.SimpleNamespace(tp_size=2, kv_worker_id=shard,
                    pp_size=1, dcp_size=1, n_servers=1, vllm_world_size=2, mla_only=False)

                def factory(*args, shard=shard, **kwargs):
                    transfer = Transfer(*args, **kwargs)
                    transfer.device.index = shard
                    transfer.layout = {"format": "test-NPU-layout", "num_heads": 4}
                    return transfer

                ctx = DIRECT.create_worker_transfer_context(parallel_strategy=strategy,
                    instance_id=base * 10 + shard, req_client=client, tokens_per_chunk=128,
                    mq_timeout=2, session_factory=Session, transfer_factory=factory,
                    extra_config={"lmcache.mp.direct": dict(tp_size=2, device_ids=[0, 1],
                        total_kv_heads=8, rank_id=base, world_size=5, pool_gb=1,
                        store_url="test")})
                contexts.append(ctx)
                self.addCleanup(ctx.close)
                self.assertEqual(ctx.config["rank_id"], base + shard)
                self.assertEqual(ctx.shard, shard)
        with ThreadPoolExecutor(max_workers=4) as executor:
            jobs = [executor.submit(ctx.register, {}, "qwen3", 2, 1, 2) for ctx in contexts]
            for job in jobs:
                job.result()
        for s, ctx in enumerate(contexts[:2]):
            object_key = key(worker=s)
            object_key.world_size = 2
            self.assertTrue(ctx.submit_store("P", object_key, {}, [[0]], None, 1).result())
            self.assertEqual(len(client.index.lookup(keys=DIRECT.chunk_keys(object_key,
                128, ctx.transfer.layout), tp_size=2)["descriptors"]), s)
        backend = DIRECT.MemFabricDirectSchedulerBackend(req_clients={"test": client},
            parallel_strategy=strategy, tokens_per_chunk=128, mq_timeout=2)
        object_key = key(worker=None)
        object_key.world_size = 2
        lookup = backend.create_lookup(object_key, tp_size=2, tokens_per_chunk=128,
                                       timeout=2, blocking_lookup=True)
        lookup.submit()
        lookup.on_scheduler_check()
        self.assertEqual(lookup.get_result(), 128)
        self.assertEqual(client.index.status()["total_pins"], 2)
        for s, ctx in enumerate(contexts[2:]):
            object_key.worker_id = s
            self.assertTrue(ctx.submit_retrieve("D", object_key, {}, [[0]], None, 1).result())
        backend.end_session(object_key.request_id)
        self.assertEqual(client.index.status()["total_pins"], 0)


if __name__ == "__main__":
    unittest.main()
