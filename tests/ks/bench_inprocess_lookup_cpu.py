# SPDX-License-Identifier: Apache-2.0
"""Paired CPU benchmark. Use the pinned vLLM source via KS_VLLM_SOURCE."""
import argparse
import ast
from collections.abc import Sequence
import json
import os
from pathlib import Path
import pickle
import platform
import statistics
import subprocess
import sys
from time import perf_counter_ns
from types import ModuleType, SimpleNamespace as NS
from typing import Generic, TypeVar, overload

import torch

import test_inprocess_lifecycle_cpu as src


BASE = "7420d75b3f6e19cc2465e949a7bf82dd13bf99f5"
SOURCE = "lmcache_ascend/integration/vllm/inprocess_lookup.py"


def load_client(text, name):
    mod = ModuleType(name)
    sys.modules[name] = mod
    exec(compile(text, SOURCE, "exec"), mod.__dict__)
    return mod


def inputs(count):
    path = Path(os.environ["KS_VLLM_SOURCE"]) / "vllm/v1/utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ConstantList"]
    ns = dict(Generic=Generic, Sequence=Sequence, T=TypeVar("T"), overload=overload)
    exec(compile(tree, str(path), "exec"), ns)
    return [ns["ConstantList"]([(i * 3571 + r * 197) % 100003 for i in range(8192)])
            for r in range(count)]


def database():
    # Use the production prefix chain with a fixed initial value and builtin hash.
    path = src.CORE / "lmcache/v1/token_database.py"
    cls = type("CPUChunkDB", (), {})
    for owner, names in [
        ("TokenDatabase", ["_canonicalize_hash_inputs", "_hash_tokens"]),
        ("ChunkedTokenDatabase", ["_get_init_hash", "_chunk_tokens", "_prefix_hash", "process_tokens"]),
    ]:
        for name in names:
            setattr(cls, name, src.method(path, owner, name, NONE_HASH=0, torch=torch,
                                          _normalize_hash_to_int=lambda h: h))
    db = cls()
    db.chunk_size, db.config, db.hash_func = 256, NS(save_unfull_chunk=True), hash
    return db


def sample(mod, prompts, db, expected):
    client = mod.StepLookupClient(32)
    t = [perf_counter_ns()]
    for i, prompt in enumerate(prompts):
        client.lookup(prompt, str(i))
    queries = client.take_requests()
    t.append(perf_counter_ns())
    blob = pickle.dumps(queries, protocol=5, buffer_callback=lambda b: len(b.raw()) < 1024**2)
    t.append(perf_counter_ns())
    received = pickle.loads(blob)
    t.append(perf_counter_ns())
    owner = [q.owner_tokens() if hasattr(q, "owner_tokens") else q.tokens for q in received]
    t.append(perf_counter_ns())
    hashes = [list(db.process_tokens(p, make_key=False)) for p in owner]
    t.append(perf_counter_ns())
    assert hashes == expected
    for i, prompt in enumerate(prompts):
        assert client.lookup(prompt, str(i)) is None
    assert not client.take_requests()
    start = perf_counter_ns()
    passive = pickle.loads(blob)
    lengths = [q.token_count if hasattr(q, "token_count") else len(q.tokens) for q in passive]
    end = perf_counter_ns()
    assert lengths == [8192] * len(prompts)
    return dict(bytes=len(blob), prepare_ms=(t[1]-t[0])/1e6, encode_ms=(t[2]-t[1])/1e6,
                decode_ms=(t[3]-t[2])/1e6, materialize_ms=(t[4]-t[3])/1e6,
                owner_hash_ms=(t[5]-t[4])/1e6, writer_ms=(t[2]-t[0])/1e6,
                writer_plus_one_owner_ms=(t[5]-t[0])/1e6, passive_decode_ms=(end-start)/1e6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    args = p.parse_args()
    old = subprocess.check_output(["git", "-C", str(src.ROOT), "show", f"{BASE}:{SOURCE}"], text=True)
    clients = [load_client(old, "lookup_before"), load_client(
        (src.ROOT / SOURCE).read_text(encoding="utf-8"), "lookup_after_")]
    db = database()
    result = dict(baseline=BASE, head=subprocess.check_output(
        ["git", "-C", str(src.ROOT), "rev-parse", "HEAD"], text=True).strip(),
        python=platform.python_version(), codec="pickle 5; local CPU; no transport",
        warmup_pairs=2, measured_pairs=15, chunk_size=256, npu_tested=False, cases={})
    prompts = inputs(32)
    for count in (1, 8, 32):
        group = prompts[:count]
        expected = [list(db.process_tokens(list(v), make_key=False)) for v in group]
        samples = [[], []]
        for pair in range(17):
            for idx in ((0, 1) if pair % 2 == 0 else (1, 0)):
                row = sample(clients[idx], group, db, expected)
                if pair >= 2:
                    samples[idx].append(row)
        result["cases"][count] = {key: dict(samples=rows, median={
            k: statistics.median(r[k] for r in rows) for k in rows[0]})
            for key, rows in zip(("before", "after"), samples)}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for n, cases in result["cases"].items():
        print(n, {k: v["median"] for k, v in cases.items()})


if __name__ == "__main__":
    main()
