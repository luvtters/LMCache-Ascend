#!/usr/bin/env python3
"""Validate matched BM / classic TE results and disclose remaining differences."""
import argparse
import json
from pathlib import Path
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def percentile(samples, fraction):
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load(path, backend):
    data = json.loads(path.read_text(encoding='utf-8'))
    config = data['configuration']
    require(data['status'] == 'PASS', f'{path}: status must be PASS')
    require(config['check_only'] is False, f'{path}: check-only is not a measurement')
    require(config['backend'] == backend, f'{path}: wrong backend')
    require(config['protocol'] in (('tcp', 'shm', 'ascend-roce', 'ascend-hccs-relay')
                                   if backend == 'te' else ('BM SDMA',)),
            f'{path}: unsupported backend protocol')
    sizes = [int(value) * 1048576 for value in config['sizes_mib'].split(',')]
    require(len(sizes) == 3 and len(set(sizes)) == 3 and min(sizes) > 0,
            f'{path}: expected three distinct positive sizes')
    for key in ('iterations', 'repeats', 'batch', 'outstanding', 'submission_threads_per_process'):
        require(type(config[key]) is int and config[key] > 0, f'{path}: invalid {key}')
    require(type(config['warmup']) is int and config['warmup'] >= 0, f'{path}: invalid warmup')
    require(data['worker_exit_codes'] == [0, 0], f'{path}: workers did not exit normally')
    checks = {row['size_bytes']: row for row in data['correctness']}
    require(set(checks) == set(sizes) and all(all(row.get(key) is True for key in
            ('remote_write_checked', 'overwritten_source_readback', 'independent_remote_read'))
            for row in checks.values()), f'{path}: incomplete correctness evidence')
    if config.get('schema_version', 1) >= 2:
        config['_observed_cpu_affinity'] = [data[side + '_resources']['cpu_affinity']
                                             for side in ('initiator', 'target')]
    require(all(config[key] == 1 for key in ('batch', 'outstanding', 'submission_threads_per_process')),
            f'{path}: only single-copy, single-submission-thread measurements are supported')
    groups, seen = {}, set()
    for row in data['measurements']:
        size, direction, repeat = row['size_bytes'], row['direction'], row['repeat']
        key = (size, direction, repeat)
        require(size in sizes and direction in ('write', 'read') and type(repeat) is int
                and 0 <= repeat < config['repeats'], f'{path}: unexpected measurement {key}')
        require(key not in seen, f'{path}: duplicate measurement {key}')
        seen.add(key)
        samples = row['samples_ns']
        require(len(samples) == config['iterations'] and all(type(n) is int and n > 0 for n in samples),
                f'{path}: invalid samples for {key}')
        require(row['errors'] == 0 and row['successful_copies'] == len(samples)
                and row['successful_bytes'] == size * len(samples)
                and row['completed_api_ns'] == sum(samples), f'{path}: inconsistent success accounting {key}')
        groups.setdefault((size, direction), []).extend(samples)
    expected = {(size, direction, repeat) for size in sizes for direction in ('write', 'read')
                for repeat in range(config['repeats'])}
    require(seen == expected, f'{path}: missing measurement cells or repeats')
    return config, sorted(sizes), groups


def summarize(bm_path, te_path):
    bm, bm_sizes, bm_groups = load(bm_path, 'bm')
    te, te_sizes, te_groups = load(te_path, 'te')
    require(bm_sizes == te_sizes, 'BM and TE payload sizes differ')
    for key in ('warmup', 'iterations', 'repeats', 'batch', 'outstanding', 'submission_threads_per_process'):
        require(bm[key] == te[key], f'BM and TE measurement configuration differs: {key}')
    require(bm.get('schema_version', 1) == te.get('schema_version', 1),
            'Do not mix legacy and new benchmark schema; rerun both sides')
    if bm.get('schema_version', 1) >= 2:
        for key in ('host_id', 'machine', 'platform', 'cpu_affinity_request', 'numa_node_request',
                    '_observed_cpu_affinity', 'working_set', 'timing', 'endpoints'):
            require(bm[key] == te[key], f'BM and TE environment differs: {key}')
    te_name = {'tcp': 'TE TCP loopback', 'shm': 'TE SHM', 'ascend-roce': 'TE Ascend Direct, RoCE requested',
               'ascend-hccs-relay': 'TE Ascend Direct, HCCS relay requested'}[te['protocol']]
    lines = ['# BM / TE transfer results', '',
             f'Same-host CPU DRAM: BM SDMA versus {te_name}. This is a path comparison; '
             'it does not isolate identical HCCS links or pure backend overhead.', '',
             f"Warmup={bm['warmup']}, iterations={bm['iterations']}, repeats={bm['repeats']}; "
             'batch=1, outstanding=1, Python submission threads/process=1.', '',
             '| MiB | Direction | Backend | n | Mean us | p50 us | p95 us | Effective GB/s | BM/TE mean time |',
             '| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for size in bm_sizes:
        for direction in ('write', 'read'):
            ratio = statistics.mean(bm_groups[size, direction]) / statistics.mean(te_groups[size, direction])
            for name, groups in (('BM SDMA', bm_groups), (te_name, te_groups)):
                samples = groups[size, direction]
                lines.append(f'| {size / 1048576:g} | {direction} | {name} | {len(samples)} | '
                             f'{statistics.mean(samples) / 1000:.3f} | {percentile(samples, .5) / 1000:.3f} | '
                             f'{percentile(samples, .95) / 1000:.3f} | {size * len(samples) / sum(samples):.3f} | '
                             f'{ratio:.3f} |')
    lines += ['', 'Statistics pool all individual completed-copy samples across repeats. '
              'Effective GB/s = total successful payload bytes / total synchronous API time. '
              'Per-repeat bandwidth and percentiles are not averaged. BM/TE mean time < 1 means BM has lower mean time.', '']
    for name, config in (('BM', bm), ('TE', te)):
        lines.append(f"- {name}: registration={config['registration']}; allocator={config['allocator']}; "
                     f"remote allocator={config['remote_allocator']}; NUMA={config['numa']}; "
                     f"working set={config['working_set']}.")
    lines += ['', '## Residual differences', '',
              '- Target memory: BM-managed pool versus the TE allocator; address ownership is different.',
              '- Registration: the compatibility BM wrapper has no explicit preregistration; TE registers explicitly '
              '(HIXL relay skips native Host registration). Internal BM registration cost is not established.',
              '- Physical NUMA placement of the BM pool remains NOT_VERIFIED even with matching process policies.',
              '- Backend thread roles, hardware offload and native/Python binding layers differ; '
              'matching submission count does not make them identical.',
              '- Fixed hot buffers and single outstanding copy do not establish streaming or inference performance.',
              f"- BM route evidence: {bm.get('route_evidence', 'legacy receipt; inspect original logs')}.",
              f"- TE route evidence: {te.get('route_evidence', 'legacy receipt; inspect original logs')}."]
    if bm.get('schema_version', 1) == 1:
        lines += ['- Legacy receipt: host identity, requested policy and per-worker resources were not fully recorded.']
    lines += ['', 'Registration, allocation and NUMA differences limit attribution. '
              'These transfer measurements do not establish inference speedup.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bm', type=Path, required=True)
    parser.add_argument('--te', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = summarize(args.bm, args.te)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f'Invalid benchmark inputs: {exc}\n')
    args.output.write_text(report, encoding='utf-8')


if __name__ == '__main__':
    main()
