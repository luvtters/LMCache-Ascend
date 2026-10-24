"""Small real-device BM check; run inside the prepared Ascend container.

python sdma_smoke.py --devices 0
python sdma_smoke.py --devices 0,2 --port 28572

Two ranks test remote-rank DRAM in both directions with SDMA only. This does
not measure bandwidth or prove a particular physical link carried the data.
"""

import argparse
import multiprocessing as mp
import time


def check(ret, operation):
    if ret != 0:
        raise RuntimeError(f"{operation} returned {ret}")


def run_rank(rank, devices, port, barrier):
    import torch
    import torch_npu  # noqa: F401
    import memfabric_hybrid as mf
    from memfabric_hybrid import bm

    device = devices[rank]
    torch.npu.set_device(device)
    mf.set_log_level(1)
    check(mf.initialize(), "mf.initialize")
    initialized = False
    handle = None
    joined = False
    try:
        config = bm.BmConfig()
        config.auto_ranking = False
        config.rank_id = rank
        config.start_store = rank == 0
        config.init_timeout = 60
        config.create_timeout = 60
        config.operation_timeout = 60
        check(bm.initialize(f"tcp://127.0.0.1:{port}", len(devices), device,
                            config), "bm.initialize")
        initialized = True
        # MemFabric 1.2.0 requires a 1 GiB-aligned DRAM segment.
        handle = bm.create2(id=0, local_dram_size=1 << 30,
                            max_dram_size=1 << 30,
                            data_op_type=bm.BmDataOpType.SDMA)
        if handle is None:
            raise RuntimeError("bm.create2 returned None")
        check(handle.join(), "join")
        joined = True
        barrier.wait(timeout=90)
        peer = (rank + 1) % len(devices)
        address = handle.peer_rank_ptr(peer, bm.BmMemType.HOST)
        if not address:
            raise RuntimeError("peer DRAM address is zero")
        source = torch.arange(1 << 20, dtype=torch.int32) + rank * 1000003
        expected = source.clone()
        target = torch.zeros_like(source)
        size = source.numel() * source.element_size()
        check(handle.copy_data(source.data_ptr(), address, size,
                               bm.BmCopyType.H2G), "H2G")
        source.fill_(-1)
        check(handle.copy_data(address, target.data_ptr(), size,
                               bm.BmCopyType.G2H), "G2H")
        if not torch.equal(target, expected):
            raise AssertionError("Host -> peer DRAM -> Host mismatch")
        npu_source = (expected + 71).to(f"npu:{device}")
        npu_target = torch.zeros_like(npu_source)
        torch.npu.synchronize(device)
        check(handle.copy_data(npu_source.data_ptr(), address + size, size,
                               bm.BmCopyType.L2G), "L2G")
        npu_source.fill_(-2)
        torch.npu.synchronize(device)
        check(handle.copy_data(address + size, npu_target.data_ptr(), size,
                               bm.BmCopyType.G2L), "G2L")
        torch.npu.synchronize(device)
        if not torch.equal(npu_target.cpu(), expected + 71):
            raise AssertionError("NPU -> peer DRAM -> NPU mismatch")
        print(f"PASS rank={rank} device={device} peer_rank={peer} "
              f"protocol=SDMA bytes_per_path={size} host=True npu=True",
              flush=True)
        barrier.wait(timeout=90)
    finally:
        if handle is not None:
            if joined:
                check(handle.leave(), "leave")
            handle.destroy()
        if initialized:
            bm.uninitialize()
        mf.uninitialize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0")
    parser.add_argument("--port", type=int, default=28570)
    args = parser.parse_args()
    devices = [int(x) for x in args.devices.split(",")]
    if not devices or len(set(devices)) != len(devices):
        parser.error("devices must be a nonempty list of distinct device IDs")
    context = mp.get_context("spawn")
    barrier = context.Barrier(len(devices))
    processes = [context.Process(target=run_rank,
                                args=(rank, devices, args.port, barrier))
                 for rank in range(len(devices))]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 240
    try:
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        codes = [p.exitcode for p in processes]
        if any(code != 0 for code in codes):
            raise RuntimeError(f"BM smoke failed or timed out: {codes}")
    finally:
        # Each child owns its BM session; abort the whole test on a failed rank.
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(10)
            if process.is_alive():
                process.kill()
                process.join()


if __name__ == "__main__":
    main()
