import unittest
import os
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch
import transfer_bench as bench
import summarize_transfer_bench as report


class BenchTests(unittest.TestCase):
    def test_success_bytes_and_quantiles(self):
        result = bench.summarize([10, 20, 30, 40], 1024)
        self.assertEqual(result['successful_bytes'], 4096)
        self.assertEqual(result['p50_ns'], 25)
        self.assertAlmostEqual(result['p95_ns'], 38.5)

    def test_guard_before_native_import(self):
        args = bench.parser().parse_args(['--backend', 'bm'])
        with self.assertRaises(ValueError):
            bench.validate(args)

    def test_completion_error_is_not_a_sample(self):
        def failed():
            raise RuntimeError('completion failed')
        with self.assertRaises(RuntimeError):
            bench.measure(failed, 1)

    def test_classic_te_copy_uses_required_four_argument_api(self):
        class Engine:
            def __init__(self):
                self.calls = []

            def transfer_sync_write(self, session, local, remote, size):
                self.calls.append(('write', session, local, remote, size))
                return 0

            def transfer_sync_read(self, session, local, remote, size):
                self.calls.append(('read', session, local, remote, size))
                return 0

        for protocol in ['tcp', 'ascend-roce', 'ascend-hccs-relay']:
            with self.subTest(protocol=protocol):
                endpoint = bench.Endpoint.__new__(bench.Endpoint)
                endpoint.a = SimpleNamespace(backend='te', protocol=protocol)
                endpoint.buf = SimpleNamespace(ptr=1024)
                endpoint.peer = {'session': 'peer:1234', 'address': 2048}
                endpoint.engine = Engine()
                for direction in ['write', 'read']:
                    endpoint.copy(direction, 4096)
                self.assertEqual(endpoint.engine.calls,
                                 [(direction, 'peer:1234', 1024, 2048, 4096)
                                  for direction in ['write', 'read']])

    def test_te_native_type_error_propagates_without_retry(self):
        for direction in ['write', 'read']:
            with self.subTest(direction=direction):
                error = TypeError('native transfer failure')
                method = Mock(side_effect=error)
                endpoint = bench.Endpoint.__new__(bench.Endpoint)
                endpoint.a = SimpleNamespace(backend='te', protocol='ascend-roce')
                endpoint.buf = SimpleNamespace(ptr=1024)
                endpoint.peer = {'session': 'peer:1234', 'address': 2048}
                endpoint.engine = SimpleNamespace(transfer_sync_write=method, transfer_sync_read=method)
                with self.assertRaises(TypeError) as raised:
                    endpoint.copy(direction, 4096)
                self.assertIs(raised.exception, error)
                method.assert_called_once_with('peer:1234', 1024, 2048, 4096)

    def test_native_state_records_loaded_engine_realpath_without_import(self):
        with tempfile.TemporaryDirectory() as folder:
            engine_file = Path(folder) / 'engine.so'
            for loaded in [None, SimpleNamespace(__file__=str(engine_file))]:
                with self.subTest(loaded=loaded is not None):
                    with patch.dict(bench.sys.modules, {'mooncake.engine': loaded}), \
                            patch.object(Path, 'read_text', return_value=''):
                        state = bench.native_state()
                    expected = str(engine_file.resolve()) if loaded else None
                    self.assertEqual(state['mooncake_engine_realpath'], expected)

    def test_ascend_requires_opt_in_before_native_import(self):
        args = bench.parser().parse_args(['--backend', 'te', '--protocol', 'ascend-roce'])
        with self.assertRaisesRegex(ValueError, 'allow-npu'):
            bench.validate(args)

    def test_ascend_memory_and_timeout_contract(self):
        flags = ['--backend', 'te', '--protocol', 'ascend-roce',
                 '--allow-npu', '--devices', '8,10']
        args = bench.parser().parse_args(flags)
        bench.validate(args)
        self.assertEqual(args.allocator, 'acl-host')
        args = bench.parser().parse_args(flags + ['--allocator', 'posix'])
        with self.assertRaisesRegex(ValueError, 'acl-host'):
            bench.validate(args)
        args = bench.parser().parse_args(flags + ['--timeout', '10'])
        with self.assertRaisesRegex(ValueError, 'timeout'):
            bench.validate(args)
        args = bench.parser().parse_args(flags + ['--te-thread-pool-size', '17'])
        with self.assertRaisesRegex(ValueError, 'thread'):
            bench.validate(args)

    def test_cpu_route_does_not_inherit_fast_or_shm_switches(self):
        args = bench.parser().parse_args(['--backend', 'te'])
        bench.validate(args)
        dirty = {'MC_USE_TENT': '1', 'MC_FORCE_SHM': '1', 'MC_FORCE_HCA': '1',
                 'ASCEND_ENABLE_USE_FABRIC_MEM': '1', 'HCCL_INTRA_ROCE_ENABLE': '1',
                 'ASCEND_LOCAL_COMM_RES': '{"version":"1.3"}'}
        with patch.dict(os.environ, dirty):
            bench.configure_environment(args)
            self.assertEqual(os.environ['MC_USE_TENT'], '0')
            self.assertEqual(os.environ['MC_FORCE_TCP'], '1')
            self.assertFalse('MC_FORCE_SHM' in os.environ, 'MC_FORCE_SHM must be unset')
            self.assertFalse('MC_FORCE_HCA' in os.environ, 'MC_FORCE_HCA must be unset')
            self.assertFalse('ASCEND_ENABLE_USE_FABRIC_MEM' in os.environ, 'Fabric Memory must be unset')
            self.assertFalse('HCCL_INTRA_ROCE_ENABLE' in os.environ, 'RoCE switch must be unset')
            self.assertIsNone(os.environ.get('ASCEND_LOCAL_COMM_RES'))
            self.assertIsNone(bench.configuration(args)['ascend_local_comm_res_requested'])

    def test_ascend_local_comm_res_is_explicit_or_unset(self):
        for protocol in ['ascend-roce', 'ascend-hccs-relay']:
            for requested in ['auto', '1.2', '1.3']:
                with self.subTest(protocol=protocol, requested=requested):
                    flags = ['--backend', 'te', '--protocol', protocol,
                             '--allow-npu', '--devices', '8,10']
                    if requested != 'auto':
                        flags += ['--ascend-local-comm-res', requested]
                    args = bench.parser().parse_args(flags)
                    bench.validate(args)
                    with patch.dict(os.environ, {'ASCEND_LOCAL_COMM_RES': '{"version":"9.9"}'}):
                        bench.configure_environment(args)
                        actual = os.environ.get('ASCEND_LOCAL_COMM_RES')
                    if requested == 'auto':
                        self.assertIsNone(actual)
                    else:
                        self.assertEqual(json.loads(actual), {'version': requested})
                    self.assertEqual(bench.configuration(args)['ascend_local_comm_res_requested'], requested)

    def test_explicit_local_comm_res_requires_ascend_protocol(self):
        for backend, protocol in [('bm', 'tcp'), ('te', 'tcp'), ('te', 'shm')]:
            with self.subTest(backend=backend, protocol=protocol):
                args = bench.parser().parse_args(['--backend', backend, '--protocol', protocol,
                                                  '--ascend-local-comm-res', '1.2'])
                with self.assertRaisesRegex(ValueError, 'only supported for Ascend'):
                    bench.validate(args)

    def test_shm_requires_real_library_and_shared_allocator(self):
        args = bench.parser().parse_args(['--backend', 'te', '--protocol', 'shm'])
        with self.assertRaisesRegex(ValueError, 'shm-library'):
            bench.validate(args)

    def test_cpu_affinity_ranges(self):
        self.assertEqual(bench.cpu_list('2-4,7'), [2, 3, 4, 7])
        with self.assertRaises(ValueError):
            bench.cpu_list('4-2')

    def test_ascend_sync_disables_presence_based_async_switch(self):
        for protocol in ['ascend-roce', 'ascend-hccs-relay']:
            args = bench.parser().parse_args(['--backend', 'te', '--protocol', protocol,
                                              '--allow-npu', '--devices', '8,10'])
            bench.validate(args)
            inherited = {'ASCEND_USE_ASYNC_TRANSFER': '1', 'ASCEND_ENABLE_USE_FABRIC_MEM': '1',
                         'ASCEND_GLOBAL_RESOURCE_CONFIG': '{"protocol_desc":"roce:8"}',
                         'HCCL_INTRA_ROCE_ENABLE': '1'}
            with self.subTest(protocol=protocol), patch.dict(os.environ, inherited):
                bench.configure_environment(args)
                self.assertFalse('ASCEND_USE_ASYNC_TRANSFER' in os.environ, 'async switch must be unset')
                self.assertFalse('ASCEND_GLOBAL_RESOURCE_CONFIG' in os.environ, 'global route override must be unset')
                self.assertFalse('ASCEND_ENABLE_USE_FABRIC_MEM' in os.environ, 'Fabric Memory must be unset')
                if protocol == 'ascend-roce':
                    self.assertEqual(os.environ['HCCL_INTRA_ROCE_ENABLE'], '1')
                else:
                    self.assertFalse('HCCL_INTRA_ROCE_ENABLE' in os.environ, 'RoCE switch must be unset')


class SummaryTests(unittest.TestCase):
    def synthetic(self, backend, protocol='tcp'):
        args = bench.parser().parse_args(['--backend', backend, '--protocol', protocol,
                                          '--warmup', '0', '--iterations', '2', '--repeats', '1'])
        args.allocator = 'te-shm' if protocol == 'shm' else 'posix'
        config = bench.configuration(args)
        config['host_id'] = 'SYNTHETIC_TEST_ONLY'
        rows, correctness = [], []
        for size in [1048576, 18 * 1048576, 64 * 1048576]:
            correctness.append(dict(size_bytes=size, remote_write_checked=True,
                                    overwritten_source_readback=True, independent_remote_read=True))
            for direction in ['read', 'write']:
                rows.append(dict(size_bytes=size, direction=direction, repeat=0,
                                 **bench.summarize([1000, 2000], size)))
        return dict(status='PASS', configuration=config, measurements=rows,
                    correctness=correctness, worker_exit_codes=[0, 0],
                    initiator_resources={'cpu_affinity': [0]}, target_resources={'cpu_affinity': [0]})

    def compare(self, bm, te):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / n for n in ['synthetic_bm.json', 'synthetic_te.json']]
            for path, data in zip(paths, [bm, te]):
                path.write_text(json.dumps(data), encoding='utf-8')
            return report.summarize(*paths)

    def test_shm_result_lists_residual_differences(self):
        text = self.compare(self.synthetic('bm'), self.synthetic('te', 'shm'))
        self.assertIn('TE SHM', text)
        self.assertIn('Residual differences', text)

    def test_cross_host_or_cpu_policy_is_not_a_paired_comparison(self):
        for field, value in [('host_id', 'another-host'), ('cpu_affinity_request', [7]),
                             ('numa_node_request', 4)]:
            bm, te = self.synthetic('bm'), self.synthetic('te')
            te['configuration'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.compare(bm, te)

    def test_bad_correctness_or_exit_is_rejected(self):
        bm, te = self.synthetic('bm'), self.synthetic('te')
        te['worker_exit_codes'] = [0, 1]
        with self.assertRaises(ValueError):
            self.compare(bm, te)
        te['worker_exit_codes'] = [0, 0]
        te['correctness'][0]['independent_remote_read'] = False
        with self.assertRaises(ValueError):
            self.compare(bm, te)


if __name__ == '__main__':
    unittest.main()
