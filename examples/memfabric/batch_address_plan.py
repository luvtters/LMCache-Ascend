#!/usr/bin/env python3
"""CPU-only address arithmetic for paged NHD KV -> packed 2LTD pool storage.

This candidate does not import or call torch, ACL, or MemFabric. It emits direct
page addresses, never repeated references to a reused gather scratch buffer.
The caller must retain engine pages and the pool reservation until real transfer
completion. A valid plan proves neither pointer ownership nor device reachability.

Each plane is a mapping with layer/component (K=0,V=1), base_ptr,
block_stride_bytes, token_stride_bytes, token_bytes, num_blocks, and optional
offset_bytes. The offsets/strides must come from actual tensor metadata.
"""


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return value


def plan_segments(planes, block_ids, *, block_size, token_start, token_count,
                  pool_base, pool_bytes):
    """Return disjoint destination segments in canonical [2,L,T,D] byte order.

    token_start indexes the logical sequence described by block_ids. Output
    segment keys are src, dst, size, component, layer and logical_start_token.
    Adjacent tokens merge only when BOTH physical source and destination touch;
    different planes stay separate so their identities remain auditable.
    """
    for name, value, minimum in (("block_size", block_size, 1),
            ("token_start", token_start, 0), ("token_count", token_count, 1),
            ("pool_base", pool_base, 1), ("pool_bytes", pool_bytes, 1)):
        _integer(value, name, minimum)
    if not planes or not block_ids or token_start + token_count > len(block_ids) * block_size:
        raise ValueError("Planes and block table must cover the requested logical tokens")
    required = {"layer", "component", "base_ptr", "block_stride_bytes",
                "token_stride_bytes", "token_bytes", "num_blocks"}
    for plane in planes:
        if not isinstance(plane, dict) or not required <= plane.keys():
            raise ValueError("Each plane requires actual pointer, stride and layout fields")
        for name in required | {"offset_bytes"}:
            _integer(plane.get(name, 0), name,
                     0 if name in ("layer", "component", "offset_bytes") else 1)
        if (plane["component"] not in (0, 1)
                or plane["token_stride_bytes"] < plane["token_bytes"]
                or plane["block_stride_bytes"] <
                    (block_size - 1) * plane["token_stride_bytes"] + plane["token_bytes"]):
            raise ValueError("Unsupported overlapping or non-NHD plane geometry")
        for block in block_ids:
            _integer(block, "block ID")
            if block >= plane["num_blocks"]:
                raise ValueError("Block ID exceeds actual engine allocation")
    layers = len(planes) // 2
    identities = {(p["component"], p["layer"]) for p in planes}
    if (len(identities) != len(planes)
            or identities != {(c, l) for c in (0, 1) for l in range(layers)}
            or len({p["token_bytes"] for p in planes}) != 1):
        raise ValueError("Expected one K/V plane per layer with homogeneous token widths")
    width = planes[0]["token_bytes"]
    if 2 * layers * token_count * width > pool_bytes:
        raise ValueError("Packed KV payload exceeds the reserved pool range")
    segments = []
    for plane in sorted(planes, key=lambda p: (p["component"], p["layer"])):
        previous = None
        plane_dst = pool_base + (plane["component"] * layers + plane["layer"]) * token_count * width
        for relative in range(token_count):
            logical = token_start + relative
            table_index, within = divmod(logical, block_size)
            src = (plane["base_ptr"] + plane.get("offset_bytes", 0)
                   + block_ids[table_index] * plane["block_stride_bytes"]
                   + within * plane["token_stride_bytes"])
            dst = plane_dst + relative * width
            if (previous is not None and previous["src"] + previous["size"] == src
                    and previous["dst"] + previous["size"] == dst):
                previous["size"] += width
            else:
                previous = dict(src=src, dst=dst, size=width,
                    component=plane["component"], layer=plane["layer"],
                    logical_start_token=logical)
                segments.append(previous)
    return segments
