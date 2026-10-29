"""CPU-only repeat-probe contracts; no model, SDK or device initialization."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import direct_repeat_probe as probe


class RepeatChecks(unittest.TestCase):
    def test_plan_only_has_no_http_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(probe.follow, "http", side_effect=AssertionError("HTTP")), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                probe.main(["--gate-dir", str(root), "--decoder", "http://unused",
                    "--service-http", "http://unused", "--trace-url", "http://unused/snapshot",
                    "--session-id", "cpu", "--output", str(root / "out")])
            self.assertEqual(json.loads(output.getvalue())["status"], "PLAN_ONLY")
            self.assertFalse((root / "out").exists())

    def test_schedule_500_has_250_serial_and_250_at_most_two_threads(self):
        active = maximum = 0
        lock = threading.Lock()
        rows = []
        def send(index):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(.0001)
            with lock:
                rows.append(index)
                active -= 1
            return index
        marks = []
        result = probe.schedule(send, lambda n, c: marks.append((n, c)), probe.Budget(10))
        self.assertEqual(len(result), 500)
        self.assertEqual(rows[:250], list(range(250)))
        self.assertEqual(set(rows), set(range(500)))
        self.assertLessEqual(maximum, 2)
        self.assertEqual(marks, [(n, 1 if n <= 250 else 2) for n in range(50, 501, 50)])

    def test_worker_exception_stops_without_later_checkpoint(self):
        marks = []
        def send(index):
            if index == 51:
                raise RuntimeError("HTTP500")
            return index
        with self.assertRaisesRegex(RuntimeError, "HTTP500"):
            probe.schedule(send, lambda n, c: marks.append(n), probe.Budget(10))
        self.assertEqual(marks, [50])

    def test_expired_absolute_budget_is_failure(self):
        with patch.object(probe.time, "monotonic", side_effect=[10, 12]):
            with self.assertRaises(TimeoutError):
                probe.Budget(1).remaining()

    def test_usage_answer_and_ids_are_checked(self):
        usage = dict(prompt_tokens=1364, completion_tokens=5, total_tokens=1369)
        response = dict(id="actual-id", usage=usage,
                        choices=[dict(message=dict(content="VELM42"))])
        self.assertEqual(probe.validate_response(response, usage, "VELM42"), "actual-id")
        bad = deepcopy(response); bad["usage"]["prompt_tokens"] -= 1
        with self.assertRaises(AssertionError):
            probe.validate_response(bad, usage, "VELM42")
        bad = deepcopy(response); bad["choices"][0]["message"]["content"] = "ORCHID"
        with self.assertRaises(AssertionError):
            probe.validate_response(bad, usage, "VELM42")

    def test_native_return_and_scatter_must_both_complete(self):
        common = dict(role="D", pid=20, thread_id=1, mono_ns=110,
                      local_ptr=99, bytes=18 << 20)
        events = [dict(common, kind="kv_stage", phase="complete", direction="scatter"),
                  dict(common, kind="native_transfer", phase="begin",
                       object_pointers=[99], direction="False"),
                  dict(common, kind="native_transfer", phase="returned",
                       object_pointers=[99], direction="False")]
        before = dict(mono_ns=100)
        self.assertEqual(probe.native_counts(before, dict(records=events), 1, 99),
                         dict(native_returned=1, scatter_completed=1))
        with self.assertRaises(AssertionError):
            probe.native_counts(before, dict(records=events[:-1]), 1, 99)

    def tp4_trace(self, copies=1):
        records = []
        for shard in range(4):
            common = dict(role='D', worker_shard=shard, tp_size=4, pid=20+shard,
                host_id='cpu', session_id='cpu', local_ptr=99+shard,
                bytes=(18 << 20)//4, thread_id=1, memory_type='npu')
            records.append(dict(common, kind='npu_scratch', phase='complete', mono_ns=10))
            for i in range(copies):
                for phase in ['begin', 'complete']:
                    records.append(dict(common, kind='bm_copy', phase=phase, mono_ns=110+i,
                        direction='G2L', bm_rank=5+shard, src=1000+i, dst=99+shard))
                for phase in ['begin', 'returned']:
                    records.append(dict(common, kind='native_transfer', phase=phase,
                        mono_ns=110+i, object_pointers=[99+shard], direction='False'))
                records.append(dict(common, kind='kv_stage', phase='complete',
                    mono_ns=110+i, direction='scatter'))
        return dict(host_id='cpu', run_id='cpu', mono_ns=100, records=records)

    def test_tp4_each_shard_completes_and_missing_one_is_rejected(self):
        args = SimpleNamespace(tp_size=4, session_id='cpu')
        before, after = self.tp4_trace(0), self.tp4_trace()
        counts = probe.native_counts_by_shard(args, before, after, 1)
        self.assertEqual(set(counts), {'0', '1', '2', '3'})
        self.assertTrue(all(r['native_returned'] == r['scatter_completed'] == 1
                            for r in counts.values()))
        copies = probe.follow.trace_counts(args, before, after, 1)
        self.assertTrue(all(r['completed_bytes'] == (18 << 20)//4 for r in copies.values()))
        after['records'].pop()  # Only shard3 scatter completion is missing.
        with self.assertRaises(AssertionError):
            probe.native_counts_by_shard(args, before, after, 1)

    def test_tp4_wrong_rank_pid_or_pointer_is_rejected(self):
        args = SimpleNamespace(tp_size=4, session_id='cpu')
        for field, value in [('bm_rank', 99), ('pid', 99), ('local_ptr', 999)]:
            before, after = self.tp4_trace(0), self.tp4_trace()
            bm = next(r for r in after['records'] if r['kind'] == 'bm_copy')
            bm[field] = value
            with self.assertRaises(AssertionError):
                probe.follow.trace_counts(args, before, after, 1)
        for field, value in [('worker_shard', 99), ('pid', 99), ('object_pointers', [999])]:
            before, after = self.tp4_trace(0), self.tp4_trace()
            native = next(r for r in after['records'] if r['kind'] == 'native_transfer')
            native[field] = value
            with self.assertRaises(AssertionError):
                probe.native_counts_by_shard(args, before, after, 1)

    def complete_http_contract(self, tp_size):
        usage = dict(prompt_tokens=1364, completion_tokens=5, total_tokens=1369)
        count = traces = 0
        lock = threading.Lock()
        def fake_http(url, directory, body=None, timeout=1, request_id=None):
            nonlocal count, traces
            if url.endswith('/tokenize'):
                value = dict(tokens=list(range(1364)), count=1364)
            elif url.endswith('/completions'):
                with lock:
                    count += 1
                    number = count
                value = dict(id='response-' + str(number), usage=usage,
                             choices=[dict(message=dict(content='VELM42'))])
            elif url.endswith('/status'):
                value = dict(chunk_size=128, world_size=1+2*tp_size, epoch='fixed', objects=10*tp_size,
                    total_pins=0, pending_writes=0, quarantined_slots=0,
                    is_healthy=True, transfer_ready=True, registration_ready=True,
                    pool_memory_type='device', storage_manager=dict(l2_adapters=[],
                        l1_manager=dict(total_object_count=0, memory_used_bytes=0)))
            elif url.endswith('/metrics'):
                raw = 'vllm:prompt_tokens_total ' + str(1364 * count) + '\n'
                raw += ''.join('vllm:prompt_tokens_by_source_total{source="' + source + '"} '
                    + str(size * count) + '\n' for source, size in
                    [('local_compute', 84), ('external_kv_transfer', 1280), ('local_cache_hit', 0)])
                return raw.encode(), {}
            else:
                traces += 1
                records = []
                for shard in range(tp_size):
                    common = dict(role='D', worker_shard=shard, tp_size=tp_size, pid=20+shard,
                        host_id='cpu', session_id='cpu', local_ptr=99+shard, bytes=(18 << 20)//tp_size,
                        thread_id=1, memory_type='npu')
                    records.append(dict(common, kind='npu_scratch', phase='complete', mono_ns=10))
                    for i in range(count * 10):
                        for phase in ['begin', 'complete']:
                            records.append(dict(common, kind='bm_copy', phase=phase, mono_ns=110+i,
                                direction='G2L', bm_rank=1+tp_size+shard, src=1000+i, dst=99+shard))
                        for phase in ['begin', 'returned']:
                            records.append(dict(common, kind='native_transfer', phase=phase,
                                mono_ns=110+i, object_pointers=[99+shard], direction='False'))
                        records.append(dict(common, kind='kv_stage', phase='complete',
                            mono_ns=110+i, direction='scatter'))
                value = dict(host_id='cpu', run_id='cpu', mono_ns=100, records=records)
            return json.dumps(value).encode(), dict(started_mono_ns=1, returned_mono_ns=2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'cached_response').mkdir()
            (root/'request.json').write_text(json.dumps(dict(temperature=0, model='cpu', messages=[])))
            (root/'cached_response/response.body.bin').write_text(json.dumps(dict(
                id='reference', usage=usage, choices=[dict(message=dict(content='VELM42'))])))
            args = SimpleNamespace(output=root/'out', duration_limit=30, timeout=1,
                gate_dir=root, decoder='http://d', service_http='http://s', trace_url='http://t/snapshot',
                session_id='cpu', expected_answer='VELM42', expected_pool_memory_type='device',
                tp_size=tp_size, chunk_size=128)
            with patch.object(probe.follow, 'http', side_effect=fake_http):
                result = probe.run(args)
            self.assertEqual(count, 500)
            self.assertEqual(traces, 2)
            self.assertEqual(result['metrics'], dict(prompt=682000, external=640000,
                                                     compute=42000, local_prefix=0))
            self.assertEqual(result['clients']['pending'], 0)
            self.assertEqual(set(result['d_bm_copies']), {str(s) for s in range(tp_size)})
            for s in range(tp_size):
                self.assertEqual(result['d_bm_copies'][str(s)]['completed_copies'], 5000)
                self.assertEqual(result['d_bm_copies'][str(s)]['completed_bytes'], 5000 * (18 << 20)//tp_size)
            if tp_size == 1:
                self.assertEqual(result['d_native'], dict(native_returned=5000, scatter_completed=5000))
            else:
                self.assertEqual(set(result['d_native']), {str(s) for s in range(tp_size)})
                self.assertTrue(all(r['native_returned'] == r['scatter_completed'] == 5000
                                    for r in result['d_native'].values()))
            self.assertEqual(len(result['checkpoints']), 10)

    def test_complete_http_contract_500_without_devices(self):
        self.complete_http_contract(1)

    def test_complete_tp4_http_contract_500_without_devices(self):
        self.complete_http_contract(4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
