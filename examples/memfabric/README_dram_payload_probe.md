# Explicit raw BM payload diagnostic

Based on `f3947b68b0e3ea639b64d180281dab5b0bbf6273`.

`bm_media_probe.py --payload-bytes N` selects the actual allocation, independent
pattern, both copy lengths, completion markers and full comparison length.
Default: 1048576 bytes (1 MiB). Allowed: positive multiples of 32 through
67108864 bytes (64 MiB), including 64, 4096 and 18874368 (18 MiB).
Both ranks must agree on the payload contract before selecting a device.

The 32-byte multiple is this diagnostic's explicit restriction, not a claimed
SDK-wide minimum. Fixed SDK source `106f783df905fe8d8e40446375faf615cf9b74fb`,
`src/hybm/csrc/data_operation/host/hybm_data_op_sdma.cpp:406-520`, writes the
requested count directly to SDMA task length fields; the non-V2 path rejects
counts above UINT32_MAX. These inspected task builders expose no stricter
length-alignment check. No SDK, medium or copy flags were changed.

Run the CPU checks with `python examples/memfabric/check_bm_media_probe_cpu.py`.
The 13 checks cover CLI bounds, mismatched rank payloads and a 64-byte real probe
control path with fake device/SDK boundaries, including last-byte corruption.
This proves Python sizing and control behavior only; native SDMA and cross-host
HOST correctness remain NOT_VERIFIED by these CPU checks.

The command still defaults to PLAN_ONLY. Actual device execution requires
`--allow-npu`, a fresh store/group, authorized device masks, both process waits,
raw native logs and resource-release checks supplied by the caller.

## Explicit CPU buffers

`--buffer-kind cpu` is restricted to `--media host --data-op-type sdma`.
The default remains `--buffer-kind npu` with L2G/G2L. CPU mode retains the source
bytearray and ordinary CPU Torch view, allocates the restore tensor on CPU,
and requests the native SDK H2G/G2H enums with unchanged copy flags 0.
It does not call Tensor.to(NPU), bypass the SDK, or fall back to another route.
Both ranks agree on buffer kind and copy types before device selection; SDK
event steps identify H2G/G2H and the following real handle.wait calls.

Fixed SDK `smem_bm_def.h:74-75` defines G2H/H2G, and
`smem_bm_entry.cpp:599-604` selects local-host memory for those endpoints.
The SDK still initializes a device context and may use internal NPU staging:
this is a CPU-buffer diagnostic, not an NPU-free communication test.
The CPU boundary tests execute both ranks, reject CPU+DEVICE/RDMA, check CPU
allocation/copy/wait ordering and reject last-byte corruption. They do not
establish native SDK or cross-host correctness.
