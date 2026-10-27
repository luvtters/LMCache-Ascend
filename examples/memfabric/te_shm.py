"""Small ctypes binding to the fixed-version, SHM-only TE bridge."""
import ctypes
import time


class ShmEngine:
    def __init__(self, library, name, location, timeout):
        begin = time.perf_counter_ns()
        self.lib = ctypes.CDLL(str(library))
        pointer = ctypes.c_void_p
        signatures = {
            'create': ([ctypes.c_char_p, ctypes.c_char_p], pointer),
            'allocate': ([pointer, ctypes.c_size_t], pointer),
            'register': ([pointer, pointer, ctypes.c_size_t], ctypes.c_int),
            'rpc_port': ([pointer], ctypes.c_int),
            'connect': ([pointer, ctypes.c_char_p], ctypes.c_int),
            'copy': ([pointer, pointer, ctypes.c_uint64, ctypes.c_size_t,
                      ctypes.c_int, ctypes.c_int], ctypes.c_int),
            'unregister': ([pointer, pointer], ctypes.c_int),
            'free': ([pointer, pointer], ctypes.c_int),
            'destroy': ([pointer], ctypes.c_int),
            'last_error': ([], ctypes.c_char_p),
        }
        for name_, (arguments, result) in signatures.items():
            method = getattr(self.lib, 'te_shm_' + name_)
            method.argtypes, method.restype = arguments, result
        self.handle = self.lib.te_shm_create(name.encode(), location.encode())
        if not self.handle:
            self.error('create')
        self.timeout_ms = int(timeout * 1000)
        self.initialize_ns = time.perf_counter_ns() - begin

    def error(self, operation):
        message = self.lib.te_shm_last_error()
        raise RuntimeError('TE SHM %s: %s' %
                           (operation, message.decode(errors='replace') if message else 'unknown error'))

    def call(self, operation, *args):
        result = getattr(self.lib, 'te_shm_' + operation)(self.handle, *args)
        if result != 0:
            self.error(operation)

    def allocate(self, size):
        pointer = self.lib.te_shm_allocate(self.handle, size)
        if not pointer:
            self.error('allocate')
        return pointer

    def port(self):
        port = self.lib.te_shm_rpc_port(self.handle)
        if port <= 0:
            self.error('rpc_port')
        return port

    def connect(self, peer):
        self.call('connect', peer.encode())

    def copy(self, direction, local, remote, size):
        self.call('copy', local, remote, size, int(direction == 'read'), self.timeout_ms)

    def close(self):
        if self.handle:
            self.call('destroy')
            self.handle = None
