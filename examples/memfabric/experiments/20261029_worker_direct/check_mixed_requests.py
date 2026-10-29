"""Targeted CPU checks; fake HTTP calls only, no device or remote imports."""
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
import threading
import time
from types import SimpleNamespace

import mixed_requests as case


def snapshot(objects=40):
    def metrics(compute=0, external=0):
        return '\n'.join(['vllm:num_requests_running{engine="0"} 0',
                          'vllm:num_requests_waiting{engine="0"} 0',
                          f'vllm:prompt_tokens_by_source_total{{source="local_compute"}} {compute}',
                          f'vllm:prompt_tokens_by_source_total{{source="external_kv_transfer"}} {external}',
                          'vllm:prompt_tokens_by_source_total{source="local_cache_hit"} 0'])

    def counter(pid):
        return dict(pid=pid, session_id=str(pid), cpu_copy_attempts=0,
                    operations={name: dict(in_flight=0, started_calls=0, completed_calls=0,
                                          completed_bytes=0, failed_calls=0) for name in case.OPERATIONS},
                    retrieve=dict(in_flight=0),
                    rpc={name: dict(active_result_calls=0) for name in ('lookup', 'acquire', 'release', 'other')})
    return dict(endpoints={name: dict(http_status=200) for name in ('d_metrics', 'p_metrics', 'counter', 'p_counter', 'status')},
                bodies=dict(d_metrics=metrics(), p_metrics=metrics()),
                json=dict(counter=counter(12), p_counter=counter(11),
                          status=dict(objects=objects, pool_bytes=2 << 30, slot_size=32 << 20,
                                      free_slots=64 - objects, evicted_objects=0,
                                      total_pins=0, pending_writes=0, quarantined_slots=0,
                                      l1=dict(read_locked_count=0, write_locked_count=0,
                                              temporary_count=0, staging_object_count=0))))


def response(role):
    return dict(http_status=200, response=dict(usage=dict(prompt_tokens=1373, completion_tokens=1 if role == 'p' else 4),
                                              choices=[dict(text='OR' if role == 'p' else 'ORCHID')]))


def main():
    checks = {}

    class Tokenizer:
        def encode(self, prompt, **_kwargs):
            values = [0] * 1373
            values[17] = int(re.search(r'Record (\d{3})', prompt)[1])
            return values

    old = [f'Record {index} CPU synthetic' for index in ('000', '100', '200', '300')]
    new, manifest = case.prepare_prompts(old, Tokenizer())
    checks['six_distinct_first_chunks'] = len(manifest['variants']) == 6 and manifest['first_complete_chunks_distinct']
    checks['new_prompts_not_old'] = len(new) == 2 and all(prompt not in old for prompt in new)
    before, after = snapshot(), snapshot(60)
    after['bodies']['p_metrics'] = after['bodies']['p_metrics'].replace('source="local_compute"} 0', 'source="local_compute"} 2746')
    after['bodies']['d_metrics'] = after['bodies']['d_metrics'].replace('source="local_compute"} 0', 'source="local_compute"} 744').replace('source="external_kv_transfer"} 0', 'source="external_kv_transfer"} 10240')
    after['json']['p_counter']['operations']['L2G'].update(completed_bytes=20 * case.CHUNK_BYTES, started_calls=20, completed_calls=20)
    after['json']['counter']['operations']['G2L'].update(completed_bytes=80 * case.CHUNK_BYTES, started_calls=80, completed_calls=80)
    requests = dict(p_responses=[response('p'), response('p')], d_responses=[response('d') for _ in range(8)],
                    client_http_interval_overlap_observed=True)
    checks['exact_mixed_window_passes'] = case.mixed_checks(before, after, requests)['passed'] is True
    wrong_bytes = deepcopy(after)
    wrong_bytes['json']['counter']['operations']['G2L']['completed_bytes'] -= 1
    checks['wrong_bytes_rejected'] = case.mixed_checks(before, wrong_bytes, requests)['passed'] is False
    missing = deepcopy(before)
    del missing['json']['p_counter']['rpc']
    checks['unknown_producer_rpc_not_idle'] = case.idle_state(missing)['known'] is False
    reset = deepcopy(after)
    reset['json']['p_counter']['session_id'] = 'reset'
    checks['producer_counter_reset_rejected'] = case.mixed_checks(before, reset, requests)['passed'] is False
    evicted = deepcopy(after)
    evicted['json']['status']['evicted_objects'] = 1
    checks['eviction_rejected'] = case.mixed_checks(before, evicted, requests)['passed'] is False
    inflight = deepcopy(after)
    inflight['json']['counter']['retrieve']['in_flight'] = 1
    checks['inflight_not_idle'] = case.idle_state(inflight)['idle'] is False

    original_post = case.post
    first_calls = threading.Barrier(3)
    d_seen = 0
    lock = threading.Lock()

    def fake_post(_args, role, _prompt, _maximum):
        nonlocal d_seen
        with lock:
            first_decoder = role == 'd' and d_seen == 0
            if role == 'd':
                d_seen += 1
        if role == 'p' or first_decoder:
            first_calls.wait(2)
        # Some Windows monotonic clocks tick at about 15 ms; keep this fake
        # transport span wider than one tick so the interval assertion is real.
        time.sleep(.05)
        return response(role)

    case.post = fake_post
    try:
        launched = case.run_mixed_http(SimpleNamespace(), old, new)
    finally:
        case.post = original_post
    checks['barrier_two_writes_eight_reads'] = len(launched['p_responses']) == 2 and len(launched['d_responses']) == 8
    checks['actual_client_interval_overlap'] = launched['client_http_interval_overlap_observed'] is True and bool(launched['overlaps'])
    checks['no_dma_overlap_claim'] = launched['native_dma_overlap_verified'] is False
    checks['no_device_modules_imported'] = not any(name in sys.modules for name in ('torch', 'torch_npu', 'transformers', 'acl'))
    result = dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
                  boundary='CPU fake tokenizer, counters and barrier only; no HTTP/NPU/model verification')
    Path(__file__).with_name('mixed_cpu_check.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
