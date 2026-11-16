# SPDX-License-Identifier: Apache-2.0
"""Two spawned CPU workers. Pipes carry control and observations only."""
from dataclasses import asdict
import multiprocessing as mp
import os
from pathlib import Path
import traceback
from types import SimpleNamespace as NS

import torch

import inprocess_cpu_fixture as cpu
import test_inprocess_lifecycle_cpu as src


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def worker(rank, conn, gate, entered, waiting):
    torch.set_num_threads(1)
    engine = cpu.engine_for(rank, gate=gate, entered=entered)
    child = cpu.connector(engine)
    original_wait = engine._store_cv.wait_for
    def observed_wait(predicate):
        waiting.set()
        return original_wait(predicate)
    engine._store_cv.wait_for = observed_wait
    info = dict(rank=rank, pid=os.getpid(), parent=os.getppid(),
                device=str(child._lmcache_engine.kv_caches["cpu"].device), width=rank+1)
    conn.send(info)
    source = None
    try:
        while True:
            op, args = conn.recv()
            try:
                if op == "store":
                    req, tokens, value = args
                    source = cpu.tensor(rank, value, len(tokens))
                    engine.store(tokens, req_id=req, source=source)
                    result = cpu.snapshot(engine)
                elif op == "reuse_source":
                    source.fill_(-7)
                    result = source.flatten().tolist()
                elif op == "drain":
                    engine.wait_for_pending_stores(set(args))
                    result = cpu.snapshot(engine)
                elif op == "state":
                    result = cpu.snapshot(engine)
                elif op == "hit":
                    result = engine.lookup(args[0])
                elif op == "read":
                    req, tokens = args
                    dest = torch.full((len(tokens), rank+1), -1, dtype=torch.int32)
                    mask = engine.retrieve(tokens, req_id=req, destination=dest)
                    result = {"mask": mask.tolist(), "values": dest.flatten().tolist()}
                elif op == "step":
                    packet, finished = args
                    packet["queries"] = [src.lookup.LookupRequest(**q) for q in packet["queries"]]
                    meta = cpu.api.AscendStepMetadata(**packet)
                    child.step = meta
                    child.handle_preemptions(meta)
                    child.start_load_kv(NS(attn_metadata=None))
                    sends, _ = child.get_finished(set(finished))
                    replies = child.build_connector_worker_meta()
                    result = dict(rank=rank, sends=sorted(sends or ()),
                                  replies=replies.replies if replies else {})
                elif op == "close":
                    engine.close()
                    result = cpu.snapshot(engine)
                    require(not any(result["refs"]) and result["pins"] == 0
                            and not result["thread_alive"], "live resources after close")
                else:
                    raise ValueError(op)
                conn.send((True, result))
                if op == "close":
                    break
            except Exception:
                conn.send((False, traceback.format_exc()))
    finally:
        gate.set()
        engine.close()
        conn.close()


class Endpoint:
    def __init__(self, ctx, rank):
        self.gate, self.entered, self.waiting = (ctx.Event() for _ in range(3))
        self.gate.set()
        self.pipe, child = ctx.Pipe()
        self.process = ctx.Process(target=worker, args=(rank, child, self.gate, self.entered, self.waiting))
        self.process.start()
        child.close()
        self.closed = False

    def read(self):
        if not self.pipe.poll(30):
            raise TimeoutError(f"worker {self.process.pid} did not reply")
        return self.pipe.recv()

    def result(self):
        ok, result = self.read()
        if not ok:
            raise RuntimeError(result)
        return result

    def send(self, op, *args):
        self.pipe.send((op, args))

    def call(self, op, *args):
        self.send(op, *args)
        return self.result()

    def hold(self):
        self.entered.clear()
        self.waiting.clear()
        self.gate.clear()

    def stop(self):
        result = self.call("close")
        self.process.join(10)
        require(self.process.exitcode == 0, "worker failed to exit normally")
        self.closed = True
        return result


class Workers:
    def __enter__(self):
        self.nodes = []
        self.sched = cpu.connector()
        self.client = self.sched._lmcache_engine.lookup_client
        try:
            for rank in range(2):
                self.nodes.append(Endpoint(mp.get_context("spawn"), rank))
            self.info = [node.read() for node in self.nodes]
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        errors = []
        for node in self.nodes:
            node.gate.set()
        for node in self.nodes:
            try:
                if node.process.is_alive() and not node.closed:
                    node.stop()
            except Exception as exc:
                errors.append(str(exc))
            finally:
                if node.process.is_alive():
                    node.process.terminate()
                    node.process.join(5)
                node.pipe.close()
        if errors:
            raise RuntimeError("worker shutdown failed: " + ", ".join(errors))

    def outcome(self):
        return [dict(info, exitcode=node.process.exitcode, joined=node.closed)
                for info, node in zip(self.info, self.nodes)]

    def packet(self, **kwargs):
        return asdict(self.sched.build_connector_meta(src.step(**kwargs)))

    def finish_packet(self, req, status="FINISHED"):
        delay, _ = self.sched.request_finished_all_groups(NS(request_id=req, status=status), ([1],))
        require(delay, "scheduler failed to hold blocks")
        return self.packet(finished=[req])

    def seed(self, rank, req, tokens, value):
        self.nodes[rank].call("store", req, tokens, value)
        self.nodes[rank].call("drain", req)

    def query(self, req, tokens):
        require(self.client.lookup(tokens, req) is None, "fresh lookup unexpectedly ready")
        packet = self.packet()
        counts = []
        for rank, node in enumerate(self.nodes):
            reply = node.call("step", packet, [])["replies"]
            counts.append(next(iter(reply.values()))[rank])
            self.sched.update_connector_output(NS(kv_connector_worker_meta=cpu.api.WorkerLookups(reply)))
            if rank == 0:
                require(self.client.lookup_cache(req) is None, "scheduler accepted only PP0")
        return counts, self.client.lookup_cache(req)

    def finish(self, req, status="FINISHED"):
        packet = self.finish_packet(req, status)
        return [n.call("step", packet, [req]) for n in self.nodes]

    def cases(self):
        return [
            ("双进程启动和初始状态", self.startup),
            ("相同 key 两段不同宽度与内容回读", self.same_key),
            ("一段短命中与共同连续前缀", self.short_hit),
            ("仅 PP0 有 key 时零外部命中", self.miss),
            ("分段读 pin 及独立释放", self.pins),
            ("异步 STORE 发布前不可见且其他段继续", self.publish),
            ("单段完成不能释放两段的源块", self.barrier),
            ("Abort 同时撤销查询、保留未完成 STORE", self.abort),
            ("重复 finish 跨空调度步只释放一次", self.replay),
            ("Preemption 在旧页复用前等待 STORE", self.preempt),
            ("同一请求两次异步 STORE 完成计数", self.multi_store),
            ("依次关闭 Engine、回收资源和进程", self.shutdown),
        ]

    def startup(self):
        require(len({i["pid"] for i in self.info}) == 2 and all(i["parent"] == os.getpid()
                for i in self.info), "expected two independent child PIDs")
        states = [n.call("state") for n in self.nodes]
        require(all(s["keys"] == s["pins"] == 0 for s in states), "cache not empty")
        return {"workers": self.info, "states": states}

    def same_key(self):
        tokens = [2, 22, 23, 24]
        reads = []
        for rank in (0, 1):
            self.seed(rank, "values", tokens, 100 * (rank + 1))
        for rank in (0, 1):
            value = 100 * (rank+1)
            r = self.nodes[rank].call("read", "values", tokens)
            require(all(r["mask"]) and r["values"] == cpu.tensor(rank, value, 4).flatten().tolist(),
                    "wrong stage payload")
            reads.append(r)
        return {"tokens": tokens, "readback": reads}

    def short_hit(self):
        t = [3, 31, 32, 33]
        self.seed(0, "short", t, 31)
        self.seed(1, "short", t[:2], 32)
        # A long query forces the actual packed wire form across both pipes.
        counts, common = self.query("short", t + list(range(256)))
        self.finish("short")
        require(counts == [4, 2] and common == 2, "short stage overclaimed")
        return {"stage_tokens": counts, "common": common}

    def miss(self):
        t = [4, 41]
        self.seed(0, "miss", t, 40)
        counts, common = self.query("miss", t)
        self.finish("miss")
        require(counts == [2, 0] and common == 0, "stage cache leaked")
        return {"stage_tokens": counts, "common": common}

    def pins(self):
        for rank in (0, 1):
            self.seed(rank, "pins", [5, 51], 51)
        self.query("pins", [5, 51])
        def pins():
            return [n.call("state")["pins"] for n in self.nodes]
        before = pins()
        packet = self.finish_packet("pins")
        self.nodes[0].call("step", packet, ["pins"])
        middle = pins()
        self.nodes[1].call("step", packet, ["pins"])
        after = pins()
        require([before, middle, after] == [[1, 1], [0, 1], [0, 0]], "pin cross-talk")
        return dict(before=before, after_pp0=middle, after_both=after)

    def hold_store(self, req, tokens, value):
        late = self.nodes[1]
        late.hold()
        late.call("store", req, tokens, value)
        require(late.entered.wait(5), "copy gate was not reached")
        return late

    def publish(self):
        t = [6, 61]
        late = self.hold_store("publish", t, 60)
        try:
            self.seed(0, "publish", t, 61)
            before = [n.call("hit", t) for n in self.nodes]
            pending = late.call("state")["pending"]
        finally:
            late.gate.set()
        late.call("drain", "publish")
        after = [n.call("hit", t) for n in self.nodes]
        require(before == [2, 0] and after == [2, 2] and pending == {"publish": 1}, "early publication")
        return dict(while_held=before, pending=pending, after=after)

    def aggregate(self, sends):
        # Run the pinned vLLM 0.26 aggregator body with small output objects.
        if not hasattr(self, "agg"):
            root = Path(os.environ.get("KS_VLLM_SOURCE", ""))
            file = root / "vllm/distributed/kv_transfer/kv_connector/utils.py"
            self.aggregate_fn = src.method(file, "KVOutputAggregator", "aggregate", KVConnectorOutput=NS)
            self.agg = NS(_expected_finished_count=2, _send_remaining_count={}, _recv_remaining_count={})
        outputs = [NS(kv_connector_output=NS(finished_sending=s, finished_recving=None,
                    expected_finished_count=2, kv_connector_stats=None, kv_connector_worker_meta=None,
                    kv_cache_events=None, invalid_block_ids=set())) for s in sends]
        out = self.aggregate_fn(self.agg, outputs)
        return sorted(out.kv_connector_output.finished_sending or ())

    def barrier(self):
        req = "barrier"
        self.seed(0, req, [7, 71], 70)
        late = self.hold_store(req, [7, 71], 71)
        try:
            first = [r["sends"] for r in self.finish(req)]
            partial = self.aggregate(first)
        finally:
            late.gate.set()
        late.call("drain", req)
        second = [n.call("step", self.packet(), [])["sends"] for n in self.nodes]
        final = self.aggregate(second)
        require(first == [[req], []] and second == [[], [req]] and partial == [] and final == [req],
                "premature or missing release")
        return dict(first=first, released_first=partial, second=second, released_second=final)

    def abort(self):
        req, t = "abort", [8, 81]
        for rank in (0, 1):
            self.seed(rank, req, t, 80)
        self.query(req, t)
        late = self.hold_store(req, [8, 82], 88)
        try:
            first = [r["sends"] for r in self.finish(req, cpu.ABORT)]
            states = [n.call("state") for n in self.nodes]
            pending = states[1]["pending"]
        finally:
            late.gate.set()
        late.call("drain", req)
        final = late.call("step", self.packet(), [])["sends"]
        pins = [s["pins"] for s in states]
        require(first == [[req], []] and final == [req] and pins == [0, 0] and pending == {req:1}
                and self.client.lookup_cache(req) == -1, "abort released wrong resource")
        return dict(pins=pins, pending=pending, finish=first, late_finish=final)

    def replay(self):
        req = "repeat"
        packet = self.finish_packet(req)
        actual = [n.call("step", packet, [req])["sends"] for n in self.nodes]
        first = self.aggregate(actual)
        idle = [n.call("step", self.packet(), [])["sends"] for n in self.nodes]
        repeat = [n.call("step", self.packet(), [req])["sends"] for n in self.nodes]
        again = self.aggregate(repeat)
        replay = [n.call("step", packet, [req])["sends"] for n in self.nodes]
        require(first == [req] and idle == [[], []] and repeat == [[], []] and again == [],
                f"duplicate block release: first={first} idle={idle} replay={repeat} released={again}")
        require(replay == [[], []] and not self.aggregate(replay), "replayed finish metadata")
        return dict(first=first, idle=idle, repeat=repeat, old_packet=replay, released_again=again)

    def preempt(self):
        late = self.hold_store("preempt", [10, 101], 100)
        try:
            packet = self.packet(preempted=["preempt"])
            late.send("step", packet, [])
            require(late.waiting.wait(5), "preemption did not reach the STORE drain")
            require(not late.pipe.poll(), "preemption returned while old pages are in use")
            other = self.nodes[0].call("step", packet, [])
        finally:
            late.gate.set()
        result = late.result()
        reused = late.call("reuse_source")
        read = late.call("read", "preempt", [10, 101])
        require(reused == [-7]*4 and read["values"] == [100,101,102,103]
                and not late.call("state")["pending"], "preempt lost STORE")
        return dict(pp0_returned=other["rank"], pp1_wait_observed=True,
                    pp1_returned=result["rank"], reused_page=reused, readback=read)

    def multi_store(self):
        req = "two-writes"
        late = self.hold_store(req, [11, 111], 110)
        try:
            pending = late.call("store", req, [11, 112], 120)["pending"]
            first = [r["sends"] for r in self.finish(req)]
        finally:
            late.gate.set()
        late.call("drain", req)
        second = late.call("step", self.packet(), [])["sends"]
        hits = [late.call("hit", t) for t in ([11, 111], [11, 112])]
        require(pending == {req:2} and first == [[req], []] and second == [req] and hits == [2,2],
                "multiple writes lost or released early")
        return dict(pending=pending, first=first, final=second, hits=hits)

    def shutdown(self):
        first, second = self.nodes
        first_exit = first.stop()
        self.seed(1, "still-alive", [12, 121], 120)
        sibling = second.call("read", "still-alive", [12, 121])
        require(sibling["values"] == [120,121,122,123], "peer close affected live Engine")
        second_exit = second.stop()
        return dict(after_pp0_exit=sibling, engine_states=[first_exit,second_exit], workers=self.outcome())
