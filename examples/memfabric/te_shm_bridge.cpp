// CPU-only ctypes bridge for classic Mooncake TE at
// f751cff2b67668ae97f1ff7e45fb2154fdbff64a. One peer and one outstanding
// transfer per session; the parent process must terminate a poisoned worker.
#include "transfer_engine.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(USE_CUDA) || defined(USE_ASCEND) || defined(USE_ASCEND_DIRECT) || \
    defined(USE_MEMFABRIC) || defined(USE_TENT)
#error "te_shm_bridge requires a separate CPU-only classic TE build"
#endif

namespace {
thread_local char last_error[2048] = {};

void error(const char* message) noexcept {
    std::snprintf(last_error, sizeof(last_error), "%s", message);
}

struct Allocation {
    size_t size;
    bool registered = false;
};

struct Session {
    mooncake::TransferEngine engine{false};
    std::string location;
    std::string peer_name;
    mooncake::SegmentHandle peer = 0;
    bool connected = false;
    bool poisoned = false;
    std::unordered_map<void*, Allocation> allocations;
    // TE keeps pointers into requests while transport work can be in flight.
    std::vector<mooncake::TransferRequest> requests;
    mooncake::BatchID batch = mooncake::INVALID_BATCH_ID;
    std::mutex mutex;
};

Session& session(void* handle) {
    if (!handle) throw std::runtime_error("null SHM session");
    auto& s = *static_cast<Session*>(handle);
    if (s.poisoned) {
        throw std::runtime_error(
            "SHM session poisoned: do not free buffers or engine; terminate worker");
    }
    return s;
}

template <typename Result, typename Fn>
Result call(Result failed, Fn&& fn) noexcept {
    last_error[0] = '\0';
    try {
        return fn();
    } catch (const std::exception& e) {
        error(e.what());
    } catch (...) {
        error("unknown native SHM exception");
    }
    return failed;
}

void require_zero(int rc, const char* operation) {
    if (rc != 0) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 std::to_string(rc));
    }
}

void require_status(const mooncake::Status& status, const char* operation) {
    if (!status.ok()) {
        throw std::runtime_error(std::string(operation) + " code=" +
                                 std::to_string(static_cast<int>(status.code())) +
                                 ": " + std::string(status.message()));
    }
}

void require_cpu_location(const char* location) {
    if (!location || std::string(location).rfind("cpu:", 0) != 0 || !location[4]) {
        throw std::runtime_error("SHM location must be cpu:<NUMA node>");
    }
    for (const char* p = location + 4; *p; ++p) {
        if (*p < '0' || *p > '9') {
            throw std::runtime_error("SHM location must be cpu:<NUMA node>");
        }
    }
}
}  // namespace

extern "C" {
const char* te_shm_last_error() noexcept { return last_error; }

void* te_shm_create(const char* name, const char* location) noexcept {
    return call<void*>(nullptr, [&]() -> void* {
        if (!name || !*name) throw std::runtime_error("SHM name is empty");
        require_cpu_location(location);
        // MC_FORCE_TCP is checked even when its value is "0" by upstream TE.
        if (std::getenv("MC_FORCE_TCP")) {
            throw std::runtime_error("unset MC_FORCE_TCP before creating SHM engine");
        }
        const char* tent = std::getenv("MC_USE_TENT");
        if (tent && std::string(tent) != "0") {
            throw std::runtime_error("MC_USE_TENT must be unset or 0 for classic SHM");
        }
        auto s = std::make_unique<Session>();
        s->location = location;
        require_zero(s->engine.init("P2PHANDSHAKE", name, name, 0), "init");
        if (!s->engine.getTransport("shm") &&
            !s->engine.installTransport("shm", nullptr)) {
            throw std::runtime_error("installTransport(shm) failed");
        }
        if (s->engine.isUsingTent() || s->engine.getTransport("tcp") ||
            s->engine.getTransport("rdma") || s->engine.getTransport("ascend")) {
            throw std::runtime_error("SHM-only engine required; another transport is installed");
        }
        return s.release();
    });
}

void* te_shm_allocate(void* handle, size_t size) noexcept {
    return call<void*>(nullptr, [&]() -> void* {
        auto& s = session(handle);
        std::lock_guard<std::mutex> lock(s.mutex);
        if (!size) throw std::runtime_error("zero-sized SHM allocation");
        void* ptr = s.engine.allocateSharedMemory(size);
        if (!ptr) throw std::runtime_error("allocateSharedMemory failed");
        try {
            s.allocations.emplace(ptr, Allocation{size});
        } catch (...) {
            s.engine.freeSharedMemory(ptr);
            throw;
        }
        return ptr;
    });
}

int te_shm_register(void* handle, void* ptr, size_t size) noexcept {
    return call<int>(-1, [&]() {
        auto& s = session(handle);
        std::lock_guard<std::mutex> lock(s.mutex);
        auto it = s.allocations.find(ptr);
        if (it == s.allocations.end() || it->second.size != size ||
            it->second.registered) {
            throw std::runtime_error("register requires an unregistered whole SHM allocation");
        }
        require_zero(s.engine.registerLocalMemory(ptr, size, s.location),
                     "registerLocalMemory");
        it->second.registered = true;
        return 0;
    });
}

int te_shm_rpc_port(void* handle) noexcept {
    return call<int>(-1, [&]() {
        auto& s = session(handle);
        std::lock_guard<std::mutex> lock(s.mutex);
        int port = s.engine.getRpcPort();
        if (port <= 0) throw std::runtime_error("invalid SHM engine RPC port");
        return port;
    });
}

int te_shm_connect(void* handle, const char* peer_session) noexcept {
    return call<int>(-1, [&]() {
        auto& s = session(handle);
        std::lock_guard<std::mutex> lock(s.mutex);
        if (!peer_session || !*peer_session) throw std::runtime_error("empty peer session");
        if (s.connected) {
            if (s.peer_name == peer_session) return 0;
            throw std::runtime_error("SHM session already connected to another peer");
        }
        // Cache the segment here. SHM maps its published buffers lazily on the
        // first transfer, so the harness must warm up before timed copies.
        auto peer = s.engine.openSegment(peer_session);
        if (static_cast<int64_t>(peer) < 0) {
            throw std::runtime_error("openSegment failed");
        }
        s.peer_name = peer_session;
        s.peer = peer;
        s.connected = true;
        return 0;
    });
}

int te_shm_copy(void* handle, void* local, uint64_t remote, size_t size,
                int is_read, int timeout_ms) noexcept {
    return call<int>(-1, [&]() {
        auto& s = session(handle);
        std::lock_guard<std::mutex> lock(s.mutex);
        auto it = s.allocations.find(local);
        if (!s.connected || it == s.allocations.end() || !it->second.registered ||
            !size || size > it->second.size || !remote || timeout_ms <= 0 ||
            (is_read != 0 && is_read != 1)) {
            throw std::runtime_error("invalid SHM copy arguments or unregistered buffer");
        }
        s.requests.resize(1);
        auto& request = s.requests.front();
        request = {};
        request.opcode = is_read ? mooncake::TransferRequest::READ
                                : mooncake::TransferRequest::WRITE;
        request.source = local;
        request.target_id = s.peer;
        request.target_offset = remote;
        request.length = size;
        s.batch = s.engine.allocateBatchID(1);
        if (s.batch == mooncake::INVALID_BATCH_ID) {
            throw std::runtime_error("allocateBatchID failed");
        }
        // Any exception or uncertain status from submission onward retains
        // the batch, request vector, allocations, segment and engine. A failed
        // copy cannot make their reclamation safe by returning to Python.
        s.poisoned = true;
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(timeout_ms);
        require_status(s.engine.submitTransfer(s.batch, s.requests), "submitTransfer");
        for (;;) {
            mooncake::TransferStatus status{};
            require_status(s.engine.getTransferStatus(s.batch, 0, status),
                           "getTransferStatus");
            if (status.s == mooncake::TransferStatusEnum::COMPLETED) {
                if (status.transferred_bytes != size) {
                    throw std::runtime_error("SHM COMPLETED with an unexpected byte count");
                }
                require_status(s.engine.freeBatchID(s.batch), "freeBatchID");
                s.batch = mooncake::INVALID_BATCH_ID;
                s.poisoned = false;
                return 0;
            }
            if (status.s == mooncake::TransferStatusEnum::FAILED ||
                status.s == mooncake::TransferStatusEnum::TIMEOUT ||
                status.s == mooncake::TransferStatusEnum::CANCELED ||
                status.s == mooncake::TransferStatusEnum::INVALID) {
                throw std::runtime_error("SHM transfer failed; terminate worker without cleanup");
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                throw std::runtime_error("SHM completion timed out; terminate worker without cleanup");
            }
            std::this_thread::yield();
        }
    });
}

int te_shm_unregister(void* handle, void* ptr) noexcept {
    return call<int>(-1, [&]() {
        auto& s = session(handle);
        std::lock_guard<std::mutex> lock(s.mutex);
        auto it = s.allocations.find(ptr);
        if (it == s.allocations.end() || !it->second.registered) {
            throw std::runtime_error("unregister requires a registered SHM allocation");
        }
        require_zero(s.engine.unregisterLocalMemory(ptr), "unregisterLocalMemory");
        it->second.registered = false;
        return 0;
    });
}

int te_shm_free(void* handle, void* ptr) noexcept {
    return call<int>(-1, [&]() {
        auto& s = session(handle);
        std::lock_guard<std::mutex> lock(s.mutex);
        auto it = s.allocations.find(ptr);
        if (it == s.allocations.end() || it->second.registered) {
            throw std::runtime_error("free requires an unregistered SHM allocation");
        }
        require_zero(s.engine.freeSharedMemory(ptr), "freeSharedMemory");
        s.allocations.erase(it);
        return 0;
    });
}

int te_shm_destroy(void* handle) noexcept {
    return call<int>(-1, [&]() {
        auto& s = session(handle);
        {
            std::lock_guard<std::mutex> lock(s.mutex);
            if (!s.allocations.empty()) {
                throw std::runtime_error("unregister and free SHM allocations before destroy");
            }
            if (s.connected) {
                require_zero(s.engine.closeSegment(s.peer), "closeSegment");
                s.connected = false;
            }
        }
        delete &s;
        return 0;
    });
}
}  // extern "C"
