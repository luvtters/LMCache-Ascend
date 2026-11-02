#!/usr/bin/env python3
"""CPU metadata rendezvous and host-local trace snapshots; never carries KV."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


def host_identity():
    boot = Path('/proc/sys/kernel/random/boot_id')
    return socket.gethostname() + ':' + (boot.read_text().strip() if boot.exists() else 'local')


def http_json(url, data=None, timeout=5):
    request = urllib.request.Request(url, data=None if data is None else
        json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout) as response:
        return json.load(response)


def event_url(base, run_id, name=None):
    url = base.rstrip('/') + '/runs/' + urllib.parse.quote(run_id, safe='') + '/events'
    return url if name is None else url + '/' + urllib.parse.quote(name, safe='')


def publish(base, run_id, name, payload, timeout=5):
    return http_json(event_url(base, run_id, name), payload, timeout)


def wait_event(base, run_id, name, timeout):
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('timed out waiting for ' + name)
        events = http_json(event_url(base, run_id), timeout=min(5, remaining))['events']
        failures = [value for key, value in events.items() if key.startswith('failed-')]
        if failures:
            raise RuntimeError('peer failed: ' + json.dumps(failures))
        if name in events:
            return events[name]
        time.sleep(min(.02, max(0, deadline - time.monotonic())))


def make_server(bind, port, directory, run_id, trace_dir=None):
    """One explicit attempt per server; durable immutable events reject overwrite."""
    directory = Path(directory) / run_id
    directory.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    prefix = '/runs/' + urllib.parse.quote(run_id, safe='') + '/events'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, code, value):
            data = json.dumps(value).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/snapshot' and trace_dir is not None:
                records = []
                for path in sorted(Path(trace_dir).glob('[PD]-*.jsonl')):
                    # A single append can be observed mid-write; retain only complete lines.
                    data = path.read_text(encoding='utf-8')
                    for line in data.splitlines(keepends=True):
                        if line.endswith('\n') and line.strip():
                            record = json.loads(line)
                            if record.get('session_id') == run_id:
                                if record.get('host_id') != host_identity():
                                    return self.respond(409, {'error': 'trace belongs to another host'})
                                records.append(record)
                counters = {}
                for record in records:
                    if record['kind'] == 'bm_copy':
                        key = record['role'] + '_shard' + str(record.get('worker_shard', 0)) + '_' + record['direction']
                        item = counters.setdefault(key, dict(started=0, completed=0, failed=0,
                                                              completed_bytes=0,
                                                              worker_shard=record.get('worker_shard', 0),
                                                              bm_rank=record.get('bm_rank'),
                                                              pid=record.get('pid')))
                        if record['phase'] == 'begin':
                            item['started'] += 1
                        elif record['phase'] == 'complete':
                            item['completed'] += 1
                            item['completed_bytes'] += record['bytes']
                        elif record['phase'] == 'error':
                            item['failed'] += 1
                return self.respond(200, dict(host_id=host_identity(), run_id=run_id,
                    mono_ns=time.monotonic_ns(), records=records,
                    trace_derived_counters=counters,
                    counter_boundary='Derived from raw completion trace; no payload access'))
            if self.path != prefix:
                return self.respond(404, {'error': 'unknown attempt or endpoint'})
            with lock:
                events = {path.stem: json.loads(path.read_text(encoding='utf-8'))
                          for path in directory.glob('*.json')}
            self.respond(200, dict(run_id=run_id, events=events))

        def do_POST(self):
            name = urllib.parse.unquote(self.path[len(prefix) + 1:])
            if (not self.path.startswith(prefix + '/') or not name or
                    any(not (char.isalnum() or char in '-_') for char in name)):
                return self.respond(400, {'error': 'invalid event or attempt'})
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 65536:
                    raise ValueError('metadata body must be 1..65536 bytes')
                value = json.loads(self.rfile.read(size))
                if not isinstance(value, dict) or value.get('name') != name:
                    raise ValueError('event must be an object with matching name')
                path = directory / (name + '.json')
                with lock:
                    if path.exists():
                        if json.loads(path.read_text(encoding='utf-8')) != value:
                            return self.respond(409, {'error': 'immutable event conflict'})
                        return self.respond(200, dict(accepted=True, duplicate=True))
                    temporary = path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')
                    os.replace(temporary, path)
                self.respond(201, dict(accepted=True, duplicate=False))
            except (ValueError, UnicodeError) as error:
                self.respond(400, {'error': str(error)})

    return ThreadingHTTPServer((bind, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=29595)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--trace-dir', type=Path)
    args = parser.parse_args()
    if not args.run_id or any(not (c.isalnum() or c in '-_') for c in args.run_id):
        parser.error('invalid run-id')
    server = make_server(args.bind, args.port, args.directory, args.run_id, args.trace_dir)
    print(json.dumps(dict(event='listening', bind=args.bind, port=server.server_port,
                          host_id=host_identity(), run_id=args.run_id)), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
