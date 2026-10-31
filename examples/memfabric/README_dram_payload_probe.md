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
The 11 checks cover CLI bounds, mismatched rank payloads and a 64-byte real probe
control path with fake device/SDK boundaries, including last-byte corruption.
This proves Python sizing and control behavior only; native SDMA and cross-host
HOST correctness remain NOT_VERIFIED by these CPU checks.

The command still defaults to PLAN_ONLY. Actual device execution requires
`--allow-npu`, a fresh store/group, authorized device masks, both process waits,
raw native logs and resource-release checks supplied by the caller.
