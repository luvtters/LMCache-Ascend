"""Check the actual staged gate's scheduler/worker identity contract on CPU."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys


def check(path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('staged_gate', path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    records = [dict(kind='trace_installed', phase='complete', role=role,
                    worker_shard=shard, tp_size=4, vllm_worker_id=shard,
                    host_id='cpu-'+role, pid=1000+offset+shard)
               for role, offset in [('P', 0), ('D', 100)] for shard in range(4)]
    expected = gate.worker_roster(records, 4)
    records.append(dict(kind='scheduler_step', phase='begin', role='D',
                        host_id='cpu-D', pid=9000))
    assert gate.worker_roster(records, 4) == expected
    # The scheduler exception must not hide missing real workers or identity.
    for broken in [records[1:], deepcopy(records)]:
        if len(broken) == len(records):
            del broken[0]['tp_size']
        try:
            gate.worker_roster(broken, 4)
        except AssertionError:
            continue
        raise AssertionError('Invalid real-worker identity was accepted')
    assert not any(name in sys.modules for name in ['torch', 'torch_npu', 'memfabric'])
    return dict(status='PASS_CPU_GATE_IDENTITY_ONLY', staged_gate=str(path),
                real_worker_shards=len(expected), negative_contracts=2,
                evidence_boundary='No NPU, service, BM or inference validation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.gate.resolve())))
