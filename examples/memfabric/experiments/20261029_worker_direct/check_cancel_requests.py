"""Small CPU-only parser/identity/real-loopback-close checks; no device imports."""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import cancel_requests as case


def snapshot():
    text = '\n'.join([
        'vllm:num_requests_running{engine="0"} 0',
        'vllm:num_requests_waiting{engine="0"} 0',
        'vllm:generation_tokens_total{engine="0"} 10',
        'vllm:request_success_total{finished_reason="abort",engine="0"} 0',
        *[f'vllm:prompt_tokens_by_source_total{{source="{source}"}} 0'
          for source in case.bench_d.TOKEN_SOURCES]])
    operations = {name: dict(in_flight=0, completed_bytes=0, completed_calls=0,
                            started_calls=0, failed_calls=0) for name in case.OPERATIONS}
    counter = dict(pid=42, session_id="one", operations=operations, cpu_copy_attempts=0,
                   retrieve=dict(in_flight=0, started_calls=0),
                   rpc={name: dict(active_result_calls=0) for name in ("lookup", "acquire", "release", "other")})
    status = dict(total_pins=0, pending_writes=0, quarantined_slots=0,
                  l1=dict(read_locked_count=0, write_locked_count=0, staging_object_count=0, temporary_count=0))
    return dict(endpoints={name: dict(http_status=200) for name in ("d_metrics", "p_metrics", "status", "counter")},
                bodies=dict(d_metrics=text, p_metrics=text), json=dict(status=status, counter=counter))


def main():
    checks = {}
    before = snapshot()
    checks["known_idle"] = case.idle_state(before)["idle"] is True
    missing = deepcopy(before)
    missing["bodies"]["d_metrics"] = ''
    checks["missing_request_gauges_unknown"] = case.idle_state(missing)["known"] is False
    missing_diag = deepcopy(before)
    del missing_diag["json"]["counter"]["rpc"]
    checks["missing_rpc_gauges_unknown"] = case.idle_state(missing_diag)["known"] is False
    active = deepcopy(before)
    active["json"]["counter"]["operations"]["G2L"]["in_flight"] = 1
    checks["active_copy_not_idle"] = case.idle_state(active)["idle"] is False
    after = deepcopy(before)
    after["json"]["counter"]["operations"]["G2L"].update(completed_bytes=18, completed_calls=1, started_calls=1)
    after["bodies"]["d_metrics"] = after["bodies"]["d_metrics"].replace('} 10', '} 11').replace('engine="0"} 0\nvllm:prompt', 'engine="0"} 1\nvllm:prompt')
    evidence = case.window_evidence(before, after)
    checks["actual_bytes_not_token_estimate"] = evidence["actual_transfer"]["read_bytes"] == 18 and evidence["actual_transfer"]["tokens_used_to_estimate_bytes"] is False
    checks["no_native_errors"] = evidence["no_new_native_errors"] is True
    checks["abort_and_generation_deltas"] = evidence["abort_count"] == 1 and evidence["generated_tokens"] == 1
    reset = deepcopy(after)
    reset["json"]["counter"]["session_id"] = 'two'
    checks["counter_reset_not_accepted"] = case.window_evidence(before, reset)["actual_transfer"]["counter_same_session"] is False
    failed = deepcopy(after)
    failed["json"]["counter"]["operations"]["G2L"]["failed_calls"] = 1
    checks["native_failure_observed"] = case.window_evidence(before, failed)["no_new_native_errors"] is False
    checks["metric_labels_and_nan"] = (case.metric_sum('vllm:request_success_total{engine="0",finished_reason="abort"} 2', 'vllm:request_success_total', {'finished_reason': 'abort'}) == 2
                                       and case.metric_sum('vllm:num_requests_running NaN', 'vllm:num_requests_running') is None)

    closed = threading.Event()
    payloads = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def do_POST(self):
            payloads.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
            self.wfile.flush()
            self.connection.settimeout(2)
            try:
                if self.connection.recv(1) == b'':
                    closed.set()
            except ConnectionResetError:
                closed.set()
            self.close_connection = True

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response = case.cancel_http(f'http://127.0.0.1:{server.server_port}/v1/completions',
                                    dict(prompt='CPU check', stream=True, ignore_eos=True, max_tokens=128), .005, 2)
        checks['real_loopback_connection_close'] = closed.wait(2)
        checks['headers_only_http_200'] = (response['http_status'] == 200 and response['application_body_bytes_read'] == 0
                                           and response['response_close_called'] and response['connection_close_called'])
        checks['literal_payload_and_close_timing'] = (payloads[0]['ignore_eos'] is True and response['closed_monotonic_ns'] >= response['headers_received_monotonic_ns'])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
    result = dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
                  boundary='CPU parser and local-loopback transport only; no inference or NPU validation')
    Path(__file__).with_name('cancel_cpu_check.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
