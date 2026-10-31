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
The 15 checks cover CLI bounds, mismatched rank payloads and a 64-byte real probe
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

## Explicit HAL DRAM NUMA request

`--dram-numa-node N` is restricted to HOST/SDMA and, for this A3 diagnostic,
nodes 0..7. The default is unset and keeps flags 0 (or 512 with
`--dram-map-host-va`). Fixed SDK `hybm_def.h:28-34` assigns node bits 0..6 and
NUMA-policy enable bit 7; the request is `(1 << 7) | N`. Node 0 and node 4
therefore request flags 128 and 132, or 640 and 644 with HOST-VA mapping.
Both ranks compare the complete create2 flags in their existing contract.

`requested_dram_numa_node` reports only the requested HAL allocation node;
`actual_dram_numa_node` remains NOT_VERIFIED. This parameter does not establish
actual page placement. A failed transfer with it cannot exclude NUMA as a cause.
