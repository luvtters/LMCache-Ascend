#!/usr/bin/env python3
"""Two-process CPU DRAM transfer benchmark. BM requires explicit NPU opt-in."""
import argparse
import ctypes
import ctypes.util
import importlib.metadata
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['te', 'bm'], required=True)
    p.add_argument('--protocol', choices=['tcp', 'shm', 'ascend-roce', 'ascend-hccs-relay'], default='tcp')
    p.add_argument('--allocator', choices=['auto', 'posix', 'acl-host'], default='auto')
    p.add_argument('--alignment', type=int, default=4096, help='POSIX allocator alignment in bytes')
    p.add_argument('--shm-library', type=Path, help='Built libte_shm_bridge.so; CPU only')
    p.add_argument('--acl-library', default='libascendcl.so')
    p.add_argument('--cpu-affinity', help='Same CPU list for both workers, e.g. 96-119')
    p.add_argument('--numa-node', type=int, help='Linux MPOL_BIND before allocation; failure is fatal')
    p.add_argument('--te-thread-pool-size', type=int, default=1, help='Ascend Direct executor threads')
    p.add_argument('--ascend-buffer-pool', default='4:8', help='HCCS relay count:MiB')
    p.add_argument('--ascend-local-comm-res', choices=['auto', '1.2', '1.3'], default='auto',
                   help='Ascend ADXL LocalCommRes version; auto leaves the option unset')
    p.add_argument('--te-transfer-timeout-ms', type=int, default=10000)
    p.add_argument('--host', default='127.0.0.1', help='Same-host TE metadata/RPC address')
    p.add_argument('--dry-run', action='store_true', help='Validate and print plan without native imports')
    p.add_argument('--sizes-mib', default='1,18,64')
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--output', type=Path, required=False)
    p.add_argument('--timeout', type=float, default=300)
    p.add_argument('--allow-npu', action='store_true')
    p.add_argument('--devices', help='Two physical devices in 8..15, e.g. 8,9')
    p.add_argument('--bm-module', type=Path)
    p.add_argument('--store-url', default='tcp://127.0.0.1:19867')
    return p


def validate(a):
    a.sizes = [int(s) * 1048576 for s in a.sizes_mib.split(',')]
    if not a.sizes or min(a.sizes) <= 0:
        raise ValueError('sizes must be positive integer MiB')
    if a.warmup < 0 or a.iterations < 1 or a.repeats < 1 or a.timeout <= 0:
        raise ValueError('invalid iteration or timeout configuration')
    if a.alignment < ctypes.sizeof(ctypes.c_void_p) or a.alignment & (a.alignment - 1):
        raise ValueError('alignment must be a power of two, at least pointer size')
    if a.numa_node is not None and not 0 <= a.numa_node < 64:
        raise ValueError('numa-node must be in 0..63')
    if a.cpu_affinity:
        cpu_list(a.cpu_affinity)
    ascend = a.backend == 'te' and a.protocol.startswith('ascend-')
    if not ascend and a.ascend_local_comm_res != 'auto':
        raise ValueError('--ascend-local-comm-res is only supported for Ascend Direct')
    if a.allocator == 'auto':
        a.allocator = 'acl-host' if ascend else 'te-shm' if a.protocol == 'shm' else 'posix'
    if ascend and a.allocator != 'acl-host':
        raise ValueError('Ascend Direct requires the acl-host allocator')
    if a.backend == 'te' and a.protocol == 'shm':
        if not a.shm_library or not a.shm_library.is_file():
            raise ValueError('SHM requires --shm-library pointing to the CPU bridge')
        if a.allocator != 'te-shm':
            raise ValueError('SHM requires its shared-memory allocator (use --allocator auto)')
    if a.backend == 'bm' and a.protocol != 'tcp':
        raise ValueError('--protocol selects TE routes; BM remains SDMA')
    if a.backend == 'bm' or ascend or a.allocator == 'acl-host':
        if not a.allow_npu or not a.devices:
            raise ValueError('Device path requires --allow-npu and --devices before native imports')
        a.device_ids = [int(s) for s in a.devices.split(',')]
        if len(a.device_ids) != 2 or len(set(a.device_ids)) != 2 or any(d not in range(8, 16) for d in a.device_ids):
            raise ValueError('Device path requires two distinct physical devices in 8..15')
    if ascend:
        if not 1 <= a.te_thread_pool_size <= 16 or a.te_transfer_timeout_ms < 1:
            raise ValueError('invalid Ascend thread or timeout setting')
        if int(a.timeout) <= 2 * a.te_transfer_timeout_ms / 1000:
            raise ValueError('outer timeout must exceed twice the Ascend transfer timeout')
        parts = a.ascend_buffer_pool.split(':')
        if a.protocol.endswith('relay') and (len(parts) != 2 or any(not n.isdigit() or int(n) < 1 for n in parts)):
            raise ValueError('ascend-buffer-pool must be positive count:MiB')
    if a.backend == 'bm':
        if not a.allow_npu or not a.devices or not a.bm_module:
            raise ValueError('BM requires --allow-npu, --devices and --bm-module before native imports')
        a.device_ids = [int(s) for s in a.devices.split(',')]
        if len(a.device_ids) != 2 or len(set(a.device_ids)) != 2 or any(d not in range(8, 16) for d in a.device_ids):
            raise ValueError('BM requires two distinct physical devices in 8..15')
        if not a.bm_module.is_file():
            raise ValueError('BM module must be an existing compatible source file')
        if max(a.sizes) > 1 << 30:
            raise ValueError('BM payload exceeds the fixed 1 GiB peer pool')


def cpu_list(value):
    result = set()
    for item in value.split(','):
        endpoints = item.split('-')
        if len(endpoints) not in (1, 2):
            raise ValueError('invalid CPU list')
        first, last = int(endpoints[0]), int(endpoints[-1])
        if first < 0 or last < first:
            raise ValueError('invalid CPU range')
        result.update(range(first, last + 1))
    return sorted(result)


def configure_environment(a):
    for key in ('MC_FORCE_TCP', 'MC_FORCE_HCA', 'MC_FORCE_SHM', 'MC_INTRANODE_NVLINK',
                'MC_FORCE_MNNVL', 'MC_CUSTOM_TOPO_JSON', 'ASCEND_ENABLE_USE_FABRIC_MEM', 'ASCEND_BUFFER_POOL',
                'ASCEND_GLOBAL_RESOURCE_CONFIG', 'ASCEND_LOCAL_COMM_RES',
                'ASCEND_AUTO_CONNECT', 'ASCEND_USE_SHORT_CONNECTION',
                'ASCEND_CONNECT_TIMEOUT', 'ASCEND_USE_ASYNC_TRANSFER', 'ASCEND_THREAD_POOL_SIZE',
                'HCCL_INTRA_ROCE_ENABLE'):
        os.environ.pop(key, None)
    os.environ.update(PYTHONNOUSERSITE='1', TORCH_DEVICE_BACKEND_AUTOLOAD='0', MC_USE_TENT='0',
                      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if a.backend == 'te' and a.protocol == 'tcp':
        os.environ['MC_FORCE_TCP'] = '1'
    if a.backend == 'te' and a.protocol.startswith('ascend-'):
        # This fixed TE version enables async by getenv presence, even for "0".
        # Keep ASCEND_USE_ASYNC_TRANSFER unset after clearing inherited flags.
        os.environ.update(ASCEND_THREAD_POOL_SIZE=str(a.te_thread_pool_size),
                          ASCEND_BUFFER_POOL=a.ascend_buffer_pool if a.protocol.endswith('relay') else '0:0',
                          ASCEND_CONNECT_TIMEOUT=str(a.te_transfer_timeout_ms),
                          ASCEND_TRANSFER_TIMEOUT=str(a.te_transfer_timeout_ms),
                          MC_TRANSFER_TIMEOUT=str(int(a.timeout)))
        if a.protocol.endswith('roce'):
            os.environ['HCCL_INTRA_ROCE_ENABLE'] = '1'
        if a.ascend_local_comm_res != 'auto':
            os.environ['ASCEND_LOCAL_COMM_RES'] = json.dumps({'version': a.ascend_local_comm_res})
    if hasattr(a, 'device_ids'):
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = ','.join(map(str, a.device_ids))


def apply_cpu_policy(a):
    if a.cpu_affinity:
        if not hasattr(os, 'sched_setaffinity'):
            raise RuntimeError('CPU affinity requires Linux')
        os.sched_setaffinity(0, cpu_list(a.cpu_affinity))
    if a.numa_node is not None:
        lib = ctypes.CDLL(ctypes.util.find_library('numa') or 'libnuma.so.1', use_errno=True)
        lib.set_mempolicy.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong]
        lib.set_mempolicy.restype = ctypes.c_int
        mask = ctypes.c_ulong(1 << a.numa_node)
        if lib.set_mempolicy(2, ctypes.byref(mask), 64) != 0:
            raise OSError(ctypes.get_errno(), 'set_mempolicy(MPOL_BIND) failed')


def resource_state(buf):
    state = {'cpu_affinity': sorted(os.sched_getaffinity(0)), 'native_threads': {},
             'buffer_numa_mapping': 'NOT_VERIFIED', 'buffer_alignment_mod_65536': buf.ptr % 65536}
    for path in Path('/proc/self/task').glob('*/comm'):
        try:
            name = path.read_text().strip()
            state['native_threads'][name] = state['native_threads'].get(name, 0) + 1
        except FileNotFoundError:
            pass
    status = Path('/proc/self/status').read_text().splitlines()
    state['allowed_resources'] = [line for line in status if line.startswith(('Cpus_allowed_list:', 'Mems_allowed_list:'))]
    for line in Path('/proc/self/maps').read_text().splitlines():
        region = line.split()[0]
        begin, end = [int(x, 16) for x in region.split('-')]
        if begin <= buf.ptr < end:
            entries = Path('/proc/self/numa_maps')
            if entries.exists():
                state['buffer_numa_mapping'] = next((x for x in entries.read_text().splitlines()
                                                     if x.split()[0] == format(begin, 'x')), 'NOT_VERIFIED')
            break
    state['numa_scope'] = 'mapping containing buffer; not a per-buffer page count; BM pool not verified'
    return state


def quantile(values, q):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize(samples, size):
    total = sum(samples)
    return dict(samples_ns=samples, mean_ns=statistics.mean(samples),
                p50_ns=quantile(samples, .5), p95_ns=quantile(samples, .95),
                successful_copies=len(samples), successful_bytes=size * len(samples),
                errors=0, completed_api_ns=total,
                effective_GB_s=size * len(samples) / total if total else None)


def measure(copy, count):
    samples = []
    for _ in range(count):
        start = time.perf_counter_ns()
        copy()  # Must return only after successful completion; errors propagate.
        samples.append(time.perf_counter_ns() - start)
    return samples


class Buffer:
    def __init__(self, size, allocator='posix', alignment=4096, owner=None, acl_library='libascendcl.so'):
        self.allocator, self.owner, self.size = allocator, owner, size
        if allocator == 'te-shm':
            self.ptr = owner.allocate(size)
        elif allocator == 'acl-host':
            self.acl = ctypes.CDLL(acl_library)
            self.acl.aclrtMallocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            self.acl.aclrtMallocHost.restype = ctypes.c_int
            self.acl.aclrtFreeHost.argtypes = [ctypes.c_void_p]
            self.acl.aclrtFreeHost.restype = ctypes.c_int
            pointer = ctypes.c_void_p()
            checked(self.acl.aclrtMallocHost(ctypes.byref(pointer), size), 'aclrtMallocHost')
            self.ptr = pointer.value
        else:
            self._posix_allocate(size, alignment)
        self.fill(0, size)  # First-touch outside transfer timing.

    def _posix_allocate(self, size, alignment):
        self.libc = ctypes.CDLL(None)
        if not hasattr(self.libc, 'posix_memalign'):
            raise RuntimeError('Linux/POSIX posix_memalign is required for actual transfers')
        self.libc.posix_memalign.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_size_t]
        self.libc.posix_memalign.restype = ctypes.c_int
        self.libc.free.argtypes = [ctypes.c_void_p]
        ptr = ctypes.c_void_p()
        rc = self.libc.posix_memalign(ctypes.byref(ptr), alignment, size)
        if rc:
            raise RuntimeError('posix_memalign failed: %s' % rc)
        self.ptr = ptr.value

    def fill(self, value, size):
        ctypes.memset(self.ptr, value, size)

    def check(self, value, size):
        # Chunked verification avoids a second payload-sized Python allocation.
        for offset in range(0, size, 65536):
            n = min(65536, size - offset)
            if ctypes.string_at(self.ptr + offset, n) != bytes([value]) * n:
                raise RuntimeError('payload mismatch at chunk offset %d' % offset)

    def close(self):
        if self.allocator == 'te-shm':
            self.owner.call('free', self.ptr)
            self.owner.close()
        elif self.allocator == 'acl-host':
            checked(self.acl.aclrtFreeHost(self.ptr), 'aclrtFreeHost')
        else:
            self.libc.free(self.ptr)


def checked(rc, operation):
    if rc != 0:
        raise RuntimeError('%s failed: %r' % (operation, rc))


class Endpoint:
    def __init__(self, a, rank, buf, shared_engine=None):
        self.a, self.rank, self.buf = a, rank, buf
        start = time.perf_counter_ns()
        if a.backend == 'te' and a.protocol == 'shm':
            self.engine = shared_engine
            initialized = time.perf_counter_ns()
            self.engine.call('register', buf.ptr, buf.size)
            self.meta = {'session': '%s:%d' % (a.host, self.engine.port()), 'address': buf.ptr}
        elif a.backend == 'te':
            from mooncake.engine import TransferEngine
            self.engine = TransferEngine()
            protocol = 'ascend' if a.protocol.startswith('ascend-') else 'tcp'
            checked(self.engine.initialize(a.host, 'P2PHANDSHAKE', protocol, ''), 'TE initialize')
            initialized = time.perf_counter_ns()
            checked(self.engine.register_memory(buf.ptr, buf.size), 'TE register_memory')
            self.meta = {'session': '%s:%d' % (a.host, self.engine.get_rpc_port()), 'address': buf.ptr}
        else:
            # The guard is repeated inside each spawned process, before imports.
            validate(a)
            os.environ['ASCEND_RT_VISIBLE_DEVICES'] = ','.join(map(str, a.device_ids))
            spec = importlib.util.spec_from_file_location('bench_compat_bm', a.bm_module)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.session = module.BMSession(device_id=rank, rank_id=rank, world_size=2,
                                            pool_gb=1, store_url=a.store_url)
            initialized = time.perf_counter_ns()
            self.meta = {'rank': rank}
        self.setup = {'initialize_ns': initialized - start,
                      'register_ns': time.perf_counter_ns() - initialized}
        if shared_engine:
            self.setup['initialize_ns'] = shared_engine.initialize_ns

    def connect(self, peer):
        self.peer = peer
        if self.a.backend == 'bm':
            self.peer_address = self.session.peer_address(1 - self.rank)
        elif self.a.protocol == 'shm':
            self.engine.connect(peer['session'])

    def copy(self, direction, size):
        if self.a.backend == 'te' and self.a.protocol == 'shm':
            self.engine.copy(direction, self.buf.ptr, self.peer['address'], size)
        elif self.a.backend == 'te':
            method = (self.engine.transfer_sync_write if direction == 'write'
                      else self.engine.transfer_sync_read)
            # Classic bindings may expose only these four required arguments.
            # Routing is selected during initialize/configure_environment.
            checked(method(self.peer['session'], self.buf.ptr, self.peer['address'], size),
                    'TE sync ' + direction)
        elif direction == 'write':
            self.session.copy(self.buf.ptr, self.peer_address, size, store=True)
        else:
            self.session.copy(self.peer_address, self.buf.ptr, size, store=False)

    def close(self):
        if self.a.backend == 'te' and self.a.protocol == 'shm':
            self.engine.call('unregister', self.buf.ptr)
        elif self.a.backend == 'te':
            checked(self.engine.unregister_memory(self.buf.ptr), 'TE unregister_memory')
        else:
            self.session.close()


def receive(pipe, timeout):
    if not pipe.poll(timeout):
        raise TimeoutError('control peer timeout')
    msg = pipe.recv()
    if msg.get('error'):
        raise RuntimeError(msg['error'])
    return msg


def versions():
    found = {}
    for name in ['mooncake-transfer-engine', 'mooncake-transfer-engine-npu', 'memfabric-hybrid']:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return found


def native_state():
    markers = ('libascend', 'libacl', 'libmf_smem', 'libcuda', 'libcudart',
               'libcann_hixl', 'libllm_datadist', 'libhccl', 'libhcomm')
    libraries = set()
    for line in Path('/proc/self/maps').read_text(encoding='utf-8').splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6:
            name = Path(fields[5]).name
            if any(marker in name.lower() for marker in markers):
                libraries.add(name)
    engine_file = getattr(sys.modules.get('mooncake.engine'), '__file__', None)
    return {'imported_modules': {name: name in sys.modules
                                for name in ('torch', 'torch_npu', 'memfabric_hybrid')},
            'mooncake_engine_realpath': str(Path(engine_file).resolve()) if engine_file else None,
            'accelerator_library_filenames': sorted(libraries)}


def worker(a, rank, pipe):
    # A failed completion never triggers normal buffer free or reuse. Keep native
    # objects alive until parent terminates the failed worker process.
    try:
        validate(a)
        configure_environment(a)
        apply_cpu_policy(a)
        started = time.perf_counter_ns()
        if a.allocator == 'acl-host':
            import torch
            import torch_npu  # noqa: F401
            torch.npu.set_device(rank)
        device_ready = time.perf_counter_ns()
        shared_engine = None
        if a.backend == 'te' and a.protocol == 'shm':
            from te_shm import ShmEngine
            location = 'cpu:%d' % a.numa_node if a.numa_node is not None else 'cpu:0'
            shared_engine = ShmEngine(a.shm_library, a.host, location, a.timeout)
        allocation_start = time.perf_counter_ns()
        buf = Buffer(max(a.sizes), a.allocator, a.alignment, shared_engine, a.acl_library)
        allocated = time.perf_counter_ns()
        ep = Endpoint(a, rank, buf, shared_engine)
        ep.setup['device_context_ns'] = device_ready - started
        ep.setup['allocation_first_touch_ns'] = allocated - allocation_start
        pipe.send({'ready': ep.meta})
        connect_start = time.perf_counter_ns()
        ep.connect(receive(pipe, a.timeout)['peer'])
        ep.setup['peer_control_exchange_connect_ns'] = time.perf_counter_ns() - connect_start
        if rank == 1:
            target_loop(a, ep, buf, pipe)
            return
        rows = []
        correctness = []
        for size in a.sizes:
            # Verify actual remote writes, then destroy the local source and read
            # back. BM owner validates via legal G2H, never dereferences its GVA.
            buf.fill(0x35, size)
            ep.copy('write', size)
            pipe.send({'command': 'check', 'size': size, 'value': 0x35})
            receive(pipe, a.timeout)
            buf.fill(0, size)
            ep.copy('read', size)
            buf.check(0x35, size)
            pipe.send({'command': 'prepare', 'size': size, 'value': 0xA7})
            receive(pipe, a.timeout)
            buf.fill(0, size)
            ep.copy('read', size)
            buf.check(0xA7, size)
            correctness.append({'size_bytes': size, 'remote_write_checked': True,
                                'overwritten_source_readback': True, 'independent_remote_read': True})
            if a.check_only:
                continue
            for direction in ['write', 'read']:
                for repeat in range(a.repeats):
                    if direction == 'write':
                        buf.fill(0x35, size)
                    else:
                        pipe.send({'command': 'prepare', 'size': size, 'value': 0xA7})
                        receive(pipe, a.timeout)
                        buf.fill(0, size)
                    copy = lambda: ep.copy(direction, size)
                    measure(copy, a.warmup)
                    wall = time.perf_counter_ns()
                    samples = measure(copy, a.iterations)
                    measurement_wall_ns = time.perf_counter_ns() - wall
                    stats = summarize(samples, size)
                    stats['measurement_wall_ns'] = measurement_wall_ns
                    if direction == 'write':
                        pipe.send({'command': 'check', 'size': size, 'value': 0x35})
                        receive(pipe, a.timeout)
                    else:
                        buf.check(0xA7, size)
                    rows.append(dict(size_bytes=size, direction=direction, repeat=repeat, **stats))
        pipe.send({'command': 'stop'})
        target = receive(pipe, a.timeout)
        resources = resource_state(buf)
        ep.close()
        buf.close()
        pipe.send({'result': {'correctness': correctness, 'measurements': rows,
                             'initiator_setup': ep.setup, 'target_setup': target['setup'],
                             'allocation_first_touch_ns': allocated - allocation_start,
                             'initiator_resources': resources, 'target_resources': target['resources'],
                             'initiator_affinity': sorted(os.sched_getaffinity(0)),
                             'target_affinity': target['affinity'], 'packages': versions(),
                             'initiator_native_state': native_state(),
                             'target_native_state': target['native_state'],
                             'engine_module_file': (sys.modules['mooncake.engine'].__file__
                                                    if 'mooncake.engine' in sys.modules else None),
                             'shm_bridge_file': str(a.shm_library) if shared_engine else None,
                             'total_worker_wall_ns': time.perf_counter_ns() - started}})
    except BaseException:
        try:
            pipe.send({'error': traceback.format_exc()})
        finally:
            pipe.close()
        # Do not use os._exit: successful paths must expose native exit failures.
        # Failure cleanup is owned by the parent's terminate/kill process boundary.
        while True:
            time.sleep(60)


def target_loop(a, ep, buf, pipe):
    while True:
        msg = receive(pipe, a.timeout)
        cmd = msg['command']
        if cmd == 'stop':
            resources = resource_state(buf)
            ep.close()
            buf.close()
            pipe.send({'setup': ep.setup, 'affinity': sorted(os.sched_getaffinity(0)), 'resources': resources,
                       'native_state': native_state()})
            return
        size, value = msg['size'], msg['value']
        if a.backend == 'te':
            if cmd == 'prepare':
                buf.fill(value, size)
            else:
                buf.check(value, size)
        else:
            own = ep.session.peer_address(1)
            if cmd == 'prepare':
                buf.fill(value, size)
                ep.session.copy(buf.ptr, own, size, store=True)
            else:
                buf.fill(0, size)
                ep.session.copy(own, buf.ptr, size, store=False)
                buf.check(value, size)
        pipe.send({'ok': True})


def run(a):
    os.environ['PYTHONNOUSERSITE'] = '1'
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    os.environ['MC_USE_TENT'] = '0'
    os.environ.pop('PYTHONPATH', None)
    ctx = mp.get_context('spawn')
    parents, processes = [], []
    begin = time.perf_counter_ns()
    try:
        for rank in range(2):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=worker, args=(a, rank, child))
            process.start()
            child.close()
            parents.append(parent)
            processes.append(process)
        ready = [receive(p, a.timeout)['ready'] for p in parents]
        for rank in range(2):
            parents[rank].send({'peer': ready[1 - rank]})
        # Relay only control metadata, never payload bytes.
        while True:
            msg = receive(parents[0], a.timeout)
            if 'result' in msg:
                result = msg['result']
                break
            parents[1].send(msg)
            parents[0].send(receive(parents[1], a.timeout))
        for process in processes:
            process.join(a.timeout)
            if process.exitcode != 0:
                raise RuntimeError('worker exit status %r' % process.exitcode)
        result.update(status='PASS', total_wall_ns=time.perf_counter_ns() - begin,
                      worker_pids=[p.pid for p in processes],
                      worker_exit_codes=[p.exitcode for p in processes])
        return result
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(10)
                if process.is_alive():
                    process.kill()
                    process.join()
        for pipe in parents:
            pipe.close()


def configuration(a):
    try:
        affinity = cpu_list(a.cpu_affinity) if a.cpu_affinity else None
    except ValueError:
        affinity = 'invalid request: ' + a.cpu_affinity
    allocators = {'posix': 'posix_memalign(%d), ordinary pages, first-touch memset' % a.alignment,
                  'acl-host': 'aclrtMallocHost, pinned Host, first-touch memset',
                  'te-shm': 'TE allocateSharedMemory, POSIX MAP_SHARED, first-touch memset'}
    protocol = a.protocol if a.backend == 'te' else 'BM SDMA'
    transport = 'BM SDMA' if a.backend == 'bm' else 'ascend' if a.protocol.startswith('ascend-') else a.protocol
    registration = ('compat BMSession.copy, no explicit preregistration' if a.backend == 'bm'
                    else 'TE metadata registration; HIXL skips Host registration in relay mode'
                    if a.protocol == 'ascend-hccs-relay' else 'explicit TE register_memory')
    route = ('NOT_VERIFIED; requested %s; inspect HIXL logs/counters; allocator is not memfd-backed' % a.protocol
             if a.protocol.startswith('ascend-') else
             'SHM-only native bridge, CPU memcpy' if a.protocol == 'shm' else
             'MC_FORCE_TCP=1; classic copy hint is not the route selector' if a.backend == 'te' else
             'compat wrapper requests SDMA; physical HCCS traffic not quantified')
    return {'schema_version': 2, 'backend': a.backend, 'protocol': protocol,
        'host_id': platform.node(), 'machine': platform.machine(),
        'python': sys.version, 'platform': platform.platform(), 'check_only': a.check_only,
        'warmup': a.warmup, 'iterations': a.iterations, 'repeats': a.repeats,
        'sizes_mib': a.sizes_mib, 'batch': 1, 'outstanding': 1, 'submission_threads_per_process': 1,
        'native_worker_threads': 'see worker resource snapshots; backend roles differ',
        'te_ascend_threads_requested': a.te_thread_pool_size if transport == 'ascend' else None,
        'ascend_local_comm_res_requested': a.ascend_local_comm_res if transport == 'ascend' else None,
        'endpoints': 'two same-host processes, CPU DRAM',
        'allocator': allocators.get(a.allocator, 'unresolved; validation did not finish'),
        'remote_allocator': 'BM private DRAM pool (1 GiB per rank)' if a.backend == 'bm'
                            else allocators.get(a.allocator, 'unresolved'),
        'registration': registration,
        'requested_transport': transport, 'route_evidence': route,
        'timing': 'perf_counter_ns around full synchronous completion API, Python wrapper included',
        'excluded': 'allocation, registration, connection, warmup, validation, cleanup',
        'working_set': 'fixed addresses reused; hot buffers',
        'cpu_affinity_request': affinity,
        'numa_node_request': a.numa_node,
        'numa': 'worker resource snapshots show mapping location; BM pool NOT_VERIFIED',
        'physical_devices': a.devices if a.backend == 'bm' or a.allocator == 'acl-host' else None,
        'ascend_buffer_pool': a.ascend_buffer_pool if a.protocol == 'ascend-hccs-relay' else '0:0'
                              if transport == 'ascend' else None,
        'cleanup_on_failure': 'terminate workers; no normal buffer free after failed completion'}


def main():
    a = parser().parse_args()
    try:
        validate(a)
        result = {'status': 'PLAN_ONLY', 'native_imports': False} if a.dry_run else run(a)
    except BaseException as exc:
        result = {'status': 'FAIL', 'error': str(exc)}
    result['configuration'] = configuration(a)
    text = json.dumps(result, indent=2)
    if a.output:
        a.output.write_text(text + '\n', encoding='utf-8')
    print(text)
    return 0 if result['status'] in ('PASS', 'PLAN_ONLY') else 1


if __name__ == '__main__':
    raise SystemExit(main())
