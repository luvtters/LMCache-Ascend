#!/usr/bin/env python3
"""CPU metadata rendezvous and host-local trace snapshots; never carries KV."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import runpy
import signal
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


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


class SharedDirectory:
    """One volatile slot directory and two fixed adapter identities for one run."""

    def __init__(self, geometry):
        common = {'pool_gb', 'slot_size_mb'}
        placement = set(geometry) - common
        if (not common <= geometry.keys() or placement not in ({'storage_rank'}, {'storage_ranks'})
                or any(type(geometry[k]) is not int for k in common)):
            raise ValueError('invalid shared L2 pool geometry')
        pool_mb, slot_mb = geometry['pool_gb'] * 1024, geometry['slot_size_mb']
        if pool_mb <= 0 or slot_mb <= 0 or pool_mb % slot_mb:
            raise ValueError('invalid two-rank shared L2 pool geometry')
        owners = geometry.get('storage_ranks', [])
        if 'storage_ranks' in geometry:
            if (type(owners) is not list or owners != [0, 1]
                    or any(type(owner) is not int for owner in owners)):
                raise ValueError('storage_ranks must be [0, 1]')
        elif type(geometry['storage_rank']) is not int or geometry['storage_rank'] not in (0, 1):
            raise ValueError('storage_rank must be 0 or 1')
        # Standalone CPU coordinator. Importing lmcache_ascend activates platform
        # hooks, so load only its stdlib directory implementation from this checkout.
        source = Path(__file__).resolve().parents[2] / 'lmcache_ascend/v1/distributed/memfabric_directory.py'
        self.pool = runpy.run_path(str(source))['PoolDirectory'](
            pool_mb // slot_mb, slot_mb << 20, owner_ranks=owners)
        self.geometry = dict(world_size=2, **geometry)
        self.store_url = None
        self.epoch = uuid.uuid4().hex
        self.claims = {}
        self.lock = threading.Lock()

    def handle(self, method, request):
        rank = request.get('rank')
        if type(rank) is not int or rank not in (0, 1):
            raise ValueError('shared L2 requires adapter rank 0 or 1')
        with self.lock:
            if method == 'claim':
                contract = request.get('contract')
                if (set(request) != {'contract', 'rank'} or not isinstance(contract, dict)
                        or set(contract) != self.geometry.keys() | {'store_url'}):
                    raise ValueError('invalid shared L2 claim')
                if (any(type(contract[k]) is not type(v) or contract[k] != v
                        for k, v in self.geometry.items())
                        or any(type(owner) is not int for owner in contract.get('storage_ranks', []))):
                    return 409, {'error': 'shared L2 geometry contract mismatch'}
                store_url = contract['store_url']
                if not isinstance(store_url, str) or not store_url:
                    raise ValueError('invalid shared L2 store_url')
                if self.store_url is not None and store_url != self.store_url:
                    return 409, {'error': 'shared L2 store_url contract mismatch'}
                if rank in self.claims:
                    return 409, {'error': 'shared L2 adapter rank already claimed'}
                self.store_url = store_url
                self.claims[rank] = uuid.uuid4().hex
                return 200, dict(epoch=self.epoch, claim=self.claims[rank])
            if (set(request) != {'rank', 'epoch', 'claim', 'op', 'args'}
                    or not isinstance(request['op'], str) or not isinstance(request['args'], dict)):
                raise ValueError('invalid shared L2 call')
            if (request['epoch'] != self.epoch or rank not in self.claims
                    or request['claim'] != self.claims[rank]):
                return 409, {'error': 'stale shared L2 epoch or invalid rank claim'}
            try:
                # apply is the directory's closed operation and argument whitelist.
                result = self.pool.apply(rank, request['op'], request['args'])
            except ValueError as error:
                return 422, dict(epoch=self.epoch, error=str(error))
            return 200, dict(epoch=self.epoch, result=result, retired=self.pool.take_retired(rank))


def make_server(bind, port, directory, run_id, trace_dir=None, *, shared_directory=None):
    """One explicit attempt per server; durable immutable events reject overwrite."""
    directory = Path(directory) / run_id
    shared = None
    if shared_directory is not None:
        if not run_id or any(not (c.isalnum() or c in '-_') for c in run_id):
            raise ValueError('invalid shared L2 run-id')
        shared = SharedDirectory(shared_directory)
    directory.mkdir(parents=True, exist_ok=shared is None)
    lock = threading.Lock()
    prefix = '/runs/' + urllib.parse.quote(run_id, safe='') + '/events'
    shared_prefix = '/runs/' + urllib.parse.quote(run_id, safe='') + '/directory/'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, code, value):
            data = json.dumps(value).encode('utf-8')
            if self.path.startswith(shared_prefix) and len(data) > 65536:
                # The operation may already have consumed retirement events.
                # A recoverable (422) rejection would let the caller keep using stale state.
                code, data = 500, b'{"error":"shared L2 directory response exceeds 65536 bytes"}'
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
            if shared is not None and self.path in (shared_prefix + 'claim', shared_prefix + 'call'):
                try:
                    size = int(self.headers.get('Content-Length', 0))
                    if (self.headers.get_content_type() != 'application/json' or not 0 < size <= 65536
                            or self.headers.get('Transfer-Encoding')):
                        raise ValueError('shared L2 requires a JSON metadata body of 1..65536 bytes')
                    raw = self.rfile.read(size)
                    if len(raw) != size:
                        raise ValueError('incomplete shared L2 metadata')
                    value = json.loads(raw)
                    if not isinstance(value, dict):
                        raise ValueError('shared L2 metadata must be an object')
                except (ValueError, RecursionError) as error:
                    return self.respond(400, {'error': str(error)})
                try:
                    code, result = shared.handle(self.path[len(shared_prefix):], value)
                except ValueError as error:
                    return self.respond(400, {'error': str(error)})
                return self.respond(code, result)
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
    parser.add_argument('--shared-l2', action='store_true',
                        help='create a volatile shared L2 directory for two adapter ranks in a fresh run')
    placement = parser.add_mutually_exclusive_group()
    placement.add_argument('--storage-rank', type=int, choices=(0, 1))
    placement.add_argument('--storage-ranks', type=int, choices=(0, 1), nargs=2)
    parser.add_argument('--pool-gb', type=int)
    parser.add_argument('--slot-size-mb', type=int)
    args = parser.parse_args()
    if not args.run_id or any(not (c.isalnum() or c in '-_') for c in args.run_id):
        parser.error('invalid run-id')
    owner = (dict(storage_ranks=args.storage_ranks) if args.storage_ranks is not None
             else dict(storage_rank=args.storage_rank))
    geometry = dict(**owner, pool_gb=args.pool_gb, slot_size_mb=args.slot_size_mb)
    if args.shared_l2:
        if any(v is None for v in geometry.values()):
            parser.error('--shared-l2 requires owner rank(s), --pool-gb and --slot-size-mb')
    elif any(v is not None for v in geometry.values()):
        parser.error('shared L2 geometry requires --shared-l2')
    signal.signal(signal.SIGINT, signal.default_int_handler)
    server = make_server(args.bind, args.port, args.directory, args.run_id, args.trace_dir,
                         shared_directory=geometry if args.shared_l2 else None)
    print(json.dumps(dict(event='listening', bind=args.bind, port=server.server_port,
                          host_id=host_identity(), run_id=args.run_id)), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
