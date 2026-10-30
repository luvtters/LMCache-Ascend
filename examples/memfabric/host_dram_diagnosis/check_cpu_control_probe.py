#!/usr/bin/env python3
"""Short local CPU positive/negative checks. No SSH or accelerator modules."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import cpu_control_probe as probe

PROBE = Path(__file__).with_name('cpu_control_probe.py')


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def execute(*args):
    return subprocess.run([sys.executable, str(PROBE), *map(str, args)],
                          capture_output=True, text=True, timeout=20)


def main():
    assert not probe.distinct_boots(dict(boot_id='same', hostname='container-a'),
                                    dict(boot_id='same', hostname='container-b'))
    assert not probe.distinct_boots(dict(boot_id=None), dict(boot_id='other'))
    assert probe.distinct_boots(dict(boot_id='kernel-a'), dict(boot_id='kernel-b'))
    with tempfile.TemporaryDirectory(prefix='cpu-control-') as directory:
        root = Path(directory)
        ports = [free_port(), free_port()]
        while ports[0] == ports[1]:
            ports[1] = free_port()
        urls = ['http://127.0.0.1:' + str(port) for port in ports]
        run_id = 'local-' + str(time.time_ns())
        planned = execute('--role', 'server', '--run-id', run_id, '--output', root/'plan')
        assert planned.returncode == 0 and 'PLAN_ONLY' in planned.stdout, planned.stderr
        assert not (root/'plan').exists(), 'PLAN_ONLY wrote files'
        processes = []
        try:
            for index in range(2):
                processes.append(subprocess.Popen([sys.executable, str(PROBE),
                    '--role', 'server', '--bind', '127.0.0.1', '--port', str(ports[index]),
                    '--peer-url', urls[1-index], '--run-id', run_id,
                    '--output', str(root/('server' + str(index))), '--duration', '8', '--run'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            for url in urls:
                deadline = time.monotonic() + 8
                while True:
                    try:
                        with opener.open(url + '/identity', timeout=.2) as response:
                            assert response.status == 200
                        break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise
                        time.sleep(.05)
            for index in range(2):
                output = root/('client' + str(index))
                result = execute('--role', 'client', '--peer-url', urls[1-index],
                    '--local-url', urls[index], '--run-id', run_id, '--output', output, '--run')
                report = json.loads((output/'result.json').read_text(encoding='utf-8'))
                assert result.returncode == 0, report
                assert report['status'] == 'PASS_CPU_HTTP_ONLY'
                assert len(report['checks']) == 18, report['checks']
                assert all(check['passed'] for check in report['checks'])
                records = [json.loads(line) for line in
                           (output/'http.jsonl').read_text(encoding='utf-8').splitlines()]
                assert {200, 201, 400, 404, 409} <= {r.get('status') for r in records}
                assert any(r.get('event') == 'expected_timeout' for r in records)
                assert any(r.get('event') == 'expected_peer_failure' for r in records)
            # Reused run cannot silently count immutable duplicate as a first publication.
            reused = root/'reused'
            result = execute('--role', 'client', '--peer-url', urls[1], '--local-url', urls[0],
                             '--run-id', run_id, '--output', reused, '--run')
            report = json.loads((reused/'result.json').read_text(encoding='utf-8'))
            assert result.returncode == 1 and report['exception']['type'] == 'AssertionError'
            assert 'first_201' in report['exception']['message']
            # A one-host run must fail the optional two-physical-host gate.
            same = root/'same-host'
            result = execute('--role', 'client', '--peer-url', urls[1], '--local-url', urls[0],
                             '--run-id', run_id, '--output', same,
                             '--require-distinct-hosts', '--run')
            report = json.loads((same/'result.json').read_text(encoding='utf-8'))
            assert result.returncode == 1 and 'distinct_hosts' in report['exception']['message']
        finally:
            for process in processes:
                _, stderr = process.communicate(timeout=12)
                assert process.returncode == 0, stderr.decode(errors='replace')
        for index in range(2):
            report = json.loads((root/('server' + str(index))/'result.json').read_text(encoding='utf-8'))
            assert report['status'] == 'STOPPED' and report['reason']
            assert not report['identity']['forbidden_modules']
            with socket.socket() as sock:
                assert sock.connect_ex(('127.0.0.1', ports[index])) != 0
        # Preserve raw transport exception in the failure report, with no fake PASS.
        failed = root/'unreachable'
        result = execute('--role', 'client', '--peer-url', urls[1], '--local-url', urls[0],
                         '--run-id', run_id, '--output', failed, '--deadline', '2', '--run')
        report = json.loads((failed/'result.json').read_text(encoding='utf-8'))
        assert result.returncode == 1 and report['status'] == 'FAIL'
        assert report['exception']['type'] == 'URLError' and report['exception']['traceback']
        print(json.dumps(dict(status='PASS_LOCAL_CPU_ONLY', directions=2,
            contract_checks_per_direction=18, negative_cases=['reused attempt',
            'same host', 'unreachable peer'], lifecycle='duration clean exit and closed ports',
            boundary='Local loopback only. Two physical hosts and BM/NPU NOT_VERIFIED')))


if __name__ == '__main__':
    main()
