#!/usr/bin/env python3
"""CPU HTTP metadata diagnostics only: no KV, BM, ACL, or accelerator imports."""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import direct_coordination as coordination

CASES = ('immutable', 'complete', 'failure', 'timeout', 'callback', 'echo')
FORBIDDEN = ('torch', 'torch_npu', 'memfabric', 'acl', 'aclrt')


def cpu_identity():
    boot_path = Path('/proc/sys/kernel/random/boot_id')
    return dict(hostname=socket.gethostname(), host_id=coordination.host_identity(),
                boot_id=boot_path.read_text().strip() if boot_path.exists() else None,
                pid=os.getpid(), uid=os.getuid() if hasattr(os, 'getuid') else None,
                forbidden_modules=[name for name in sys.modules
                    if name.split('.')[0] in FORBIDDEN],
                device_boundary='CPU TCP/HTTP metadata only; no NPU access requested')


class Evidence:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.path = self.directory / 'http.jsonl'

    def record(self, **record):
        record.update(wall_time_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(),
                      hostname=socket.gethostname(), pid=os.getpid())
        with self.lock, self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')

    def result(self, value):
        value['identity'] = cpu_identity()
        (self.directory / 'result.json').write_text(
            json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(value, ensure_ascii=False), flush=True)


def body_record(data):
    return dict(bytes=len(data), base64=base64.b64encode(data).decode('ascii'),
                utf8=data.decode('utf-8', errors='replace'))


def request(evidence, url, value=None, timeout=5):
    data = None if value is None else json.dumps(value).encode('utf-8')
    req = urllib.request.Request(url, data=data,
                                 headers={'Content-Type': 'application/json'})
    record = dict(direction='outbound', method=req.get_method(), url=url,
                  request_headers=list(req.header_items()),
                  request_body=body_record(data or b''), started_ns=time.monotonic_ns())
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            response = opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as error:
            response = error
            record['http_exception'] = dict(type=type(error).__name__, message=str(error))
        with response:
            raw = response.read()
            status = response.code
            record.update(status=status, response_headers=list(response.headers.items()),
                          response_body=body_record(raw))
        return status, json.loads(raw)
    except Exception as error:
        record['exception'] = dict(type=type(error).__name__, message=str(error),
                                   traceback=traceback.format_exc())
        raise
    finally:
        record['finished_ns'] = time.monotonic_ns()
        evidence.record(**record)


def trace_helpers(evidence):
    def http_json(url, data=None, timeout=5):
        status, value = request(evidence, url, data, timeout)
        if status >= 400:
            raise urllib.error.HTTPError(url, status, json.dumps(value), None, None)
        return value
    coordination.http_json = http_json


def attempt(run_id, case):
    return run_id + '-' + case


def distinct_boots(first, second):
    return bool(first['boot_id']) and bool(second['boot_id']) and first['boot_id'] != second['boot_id']


def serve(args, evidence):
    # Each production handler retains its own exact prefix, directory and lock.
    handlers = {}
    for case in CASES:
        run = attempt(args.run_id, case)
        temporary = coordination.make_server('127.0.0.1', 0, evidence.directory / 'events', run)
        handlers[run] = temporary.RequestHandlerClass
        temporary.server_close()
    base = handlers[attempt(args.run_id, 'immutable')]

    class Handler(base):
        def send_response(self, code, message=None):
            self.response_status = code
            self.response_headers = []
            return super().send_response(code, message)

        def send_header(self, key, value):
            self.response_headers.append((key, value))
            return super().send_header(key, value)

        def respond(self, code, value):
            raw = json.dumps(value).encode('utf-8')
            try:
                return super().respond(code, value)
            finally:
                evidence.record(direction='inbound', method=self.command, path=self.path,
                    remote_address=list(self.client_address),
                    request_line=self.requestline, request_headers=list(self.headers.items()),
                    request_body=body_record(getattr(self, 'request_body', b'')),
                    status=code, response_headers=getattr(self, 'response_headers', []),
                    response_body=body_record(raw), started_ns=self.started_ns,
                    finished_ns=time.monotonic_ns())

        def route(self):
            self.started_ns = time.monotonic_ns()
            self.request_body = b''
            if self.command == 'POST':
                try:
                    size = int(self.headers.get('Content-Length', 0))
                    if not 0 < size <= 65536:
                        return self.respond(400, {'error': 'metadata body must be 1..65536 bytes'})
                    self.request_body = self.rfile.read(size)
                    self.rfile = io.BytesIO(self.request_body)
                except ValueError as error:
                    return self.respond(400, {'error': str(error)})
            if self.path == '/identity' and self.command == 'GET':
                return self.respond(200, dict(identity=cpu_identity(), run_id=args.run_id))
            if self.path == '/echo' and self.command == 'POST':
                return self.respond(200, json.loads(self.request_body))
            if self.path == '/callback' and self.command == 'POST':
                try:
                    value = json.loads(self.request_body)
                    if (value != dict(run_id=args.run_id, callback_url=args.peer_url)
                            or not args.peer_url):
                        return self.respond(400, {'error': 'callback must match configured peer and run'})
                    event = dict(name='completed', sender=cpu_identity(), kind='cpu-callback',
                                 run_id=args.run_id)
                    accepted = coordination.publish(args.peer_url,
                        attempt(args.run_id, 'callback'), 'completed', event, timeout=5)
                    return self.respond(200, dict(callback=accepted, event=event))
                except Exception as error:
                    evidence.record(event='callback_error', exception_type=type(error).__name__,
                                    message=str(error), traceback=traceback.format_exc())
                    return self.respond(502, {'error': str(error), 'type': type(error).__name__})
            parts = self.path.split('/')
            run = urllib.parse.unquote(parts[2]) if len(parts) > 2 else ''
            selected = handlers.get(run, base)
            method = selected.do_GET if self.command == 'GET' else selected.do_POST
            return method(self)

        def do_GET(self):
            return self.route()

        def do_POST(self):
            return self.route()

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

    server = coordination.ThreadingHTTPServer((args.bind, args.port), Handler)
    server.daemon_threads = True
    stop_reason = []

    def stop(reason):
        if not stop_reason:
            stop_reason.append(reason)
            threading.Thread(target=server.shutdown, daemon=True).start()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda number, frame: stop('signal-' + str(number)))
    timer = threading.Timer(args.duration, stop, args=('duration',))
    timer.daemon = True
    timer.start()
    evidence.record(event='listening', bind=args.bind, port=server.server_port,
                    run_id=args.run_id, duration=args.duration, identity=cpu_identity())
    print(json.dumps(dict(event='listening', bind=args.bind, port=server.server_port,
                          run_id=args.run_id)), flush=True)
    try:
        server.serve_forever(poll_interval=.1)
    finally:
        timer.cancel()
        server.server_close()
        evidence.result(dict(status='STOPPED', reason=stop_reason, run_id=args.run_id,
                             boundary='Listener lifecycle only; client report determines cases'))


def probe(args, evidence):
    end = time.monotonic() + args.deadline
    checks = []

    def check(name, condition, detail=None):
        checks.append(dict(name=name, passed=bool(condition), detail=detail))
        if not condition:
            raise AssertionError(name + ': ' + repr(detail))

    def call(url, value=None):
        remaining = end - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('total client deadline exceeded')
        return request(evidence, url, value, timeout=min(5, remaining))

    def waiting(run, name, timeout):
        remaining = end - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('total client deadline exceeded')
        return coordination.wait_event(args.peer_url, run, name, min(timeout, remaining))

    report = dict(status='FAIL', run_id=args.run_id, checks=checks,
        boundary='CPU TCP/HTTP metadata only. BM/SDMA/HCCS/DRAM/KV/inference NOT_VERIFIED',
        local_url=args.local_url, peer_url=args.peer_url)
    try:
        status, remote = call(args.peer_url + '/identity')
        check('peer_identity', status == 200 and remote['run_id'] == args.run_id, remote)
        status, local = call(args.local_url + '/identity')
        check('local_identity', status == 200 and local['run_id'] == args.run_id, local)
        report.update(peer=remote['identity'], local_listener=local['identity'])
        check('cpu_imports', not remote['identity']['forbidden_modules'] and
              not local['identity']['forbidden_modules'] and not cpu_identity()['forbidden_modules'])
        if args.require_distinct_hosts:
            check('distinct_hosts', distinct_boots(remote['identity'], local['identity']))
        run = attempt(args.run_id, 'immutable')
        url = coordination.event_url(args.peer_url, run, 'metadata')
        value = dict(name='metadata', marker=args.run_id, payload='first')
        status, result = call(url, value)
        check('first_201', status == 201 and result == dict(accepted=True, duplicate=False), result)
        status, result = call(url, value)
        check('duplicate_200', status == 200 and result == dict(accepted=True, duplicate=True), result)
        status, result = call(url, dict(value, payload='conflict'))
        check('conflict_409', status == 409, result)
        status, result = call(coordination.event_url(args.peer_url, run))
        check('immutable_not_overwritten', status == 200 and result['events']['metadata'] == value, result)
        for suffix in ('-old', 'x'):
            unknown = run + suffix
            status, result = call(coordination.event_url(args.peer_url, unknown))
            check('unknown_get_' + suffix, status == 404, result)
            status, result = call(coordination.event_url(args.peer_url, unknown, 'metadata'), value)
            check('unknown_post_' + suffix, status == 400, result)
        # Use actual publish and wait_event helpers, with raw transport evidence above.
        complete_run = attempt(args.run_id, 'complete')
        complete = dict(name='completed', marker=args.run_id)
        coordination.publish(args.peer_url, complete_run, 'completed', complete,
                             timeout=min(5, max(.01, end-time.monotonic())))
        check('wait_complete', waiting(complete_run, 'completed', 2) == complete)
        try:
            waiting(attempt(args.run_id, 'timeout'), 'absent', .2)
        except TimeoutError as error:
            evidence.record(event='expected_timeout', exception_type=type(error).__name__,
                            message=str(error), traceback=traceback.format_exc())
            check('wait_timeout', 'timed out waiting for absent' in str(error), str(error))
        else:
            check('wait_timeout', False)
        failure_run = attempt(args.run_id, 'failure')
        failed = dict(name='failed-intentional', reason='expected negative CPU test')
        coordination.publish(args.peer_url, failure_run, failed['name'], failed,
                             timeout=min(5, max(.01, end-time.monotonic())))
        try:
            waiting(failure_run, 'absent', 2)
        except RuntimeError as error:
            evidence.record(event='expected_peer_failure', exception_type=type(error).__name__,
                            message=str(error), traceback=traceback.format_exc())
            check('wait_failure', 'peer failed:' in str(error) and failed['reason'] in str(error), str(error))
        else:
            check('wait_failure', False)
        status, result = call(args.peer_url + '/callback',
                              dict(run_id=args.run_id, callback_url=args.local_url))
        check('peer_to_local_callback', status == 200 and result['callback']['accepted'], result)
        event = coordination.wait_event(args.local_url, attempt(args.run_id, 'callback'),
                                        'completed', min(2, max(.01, end-time.monotonic())))
        check('callback_complete', event == result['event'] and
              event['sender']['host_id'] == remote['identity']['host_id'], event)
        metadata = dict(run_id=args.run_id, kind='metadata-not-kv', text='0123456789abcdef' * 3800)
        check('echo_size_bounded', len(json.dumps(metadata).encode('utf-8')) <= 65536)
        status, echoed = call(args.peer_url + '/echo', metadata)
        check('metadata_roundtrip_exact', status == 200 and echoed == metadata)
        report['status'] = 'PASS_CPU_HTTP_ONLY'
        return 0
    except Exception as error:
        report['exception'] = dict(type=type(error).__name__, message=str(error),
                                   traceback=traceback.format_exc())
        return 1
    finally:
        evidence.result(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('server', 'client'), required=True)
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=29635)
    parser.add_argument('--peer-url', default='')
    parser.add_argument('--local-url', default='')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=300)
    parser.add_argument('--deadline', type=float, default=90)
    parser.add_argument('--require-distinct-hosts', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run_id or any(not (c.isalnum() or c in '-_') for c in args.run_id):
        parser.error('run-id must contain only alphanumeric, hyphen or underscore')
    if not 0 < args.duration <= 300 or not 0 < args.deadline <= 180:
        parser.error('duration must be 0..300 seconds; deadline 0..180 seconds')
    for url in (args.peer_url, args.local_url):
        if url and (urllib.parse.urlsplit(url).scheme != 'http' or
                    urllib.parse.urlsplit(url).path not in ('', '/')):
            parser.error('URLs must be plain HTTP base URLs')
    args.peer_url = args.peer_url.rstrip('/')
    args.local_url = args.local_url.rstrip('/')
    if not args.run:
        print(json.dumps(dict(status='PLAN_ONLY', role=args.role, run_id=args.run_id,
                              output=str(args.output), identity=cpu_identity())))
        return 0
    if not args.peer_url or (args.role == 'client' and not args.local_url):
        parser.error('--run requires --peer-url and client also requires --local-url')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be a new or empty directory for this attempt')
    evidence = Evidence(args.output)
    trace_helpers(evidence)
    if args.role == 'server':
        serve(args, evidence)
        return 0
    return probe(args, evidence)


if __name__ == '__main__':
    raise SystemExit(main())
