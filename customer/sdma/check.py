#!/usr/bin/env python3
"""Check source selection and PP APIs without constructing a BM session."""
import importlib.metadata
import inspect
from pathlib import Path

import lmcache
import lmcache_ascend
from lmcache.v1.multiprocess.pp_layout import PPStageProfile
from memfabric_sdma import memfabric_l2_adapter as adapter
from packaging.version import Version


checkout = Path(__file__).resolve().parents[2]
assert not Path(lmcache_ascend.__file__).resolve().is_relative_to(checkout), (
    "The L2 checkout shadows the installed PP plugin; put only customer/sdma on PYTHONPATH"
)
assert Path(adapter.__file__).resolve() == checkout / "lmcache_ascend/v1/distributed/memfabric_l2_adapter.py"
assert hasattr(PPStageProfile, "cache_namespace"), "The installed core lacks native PP stage namespaces"
assert not inspect.isabstract(adapter.MemFabricL2Adapter), "L2 interface differs from this candidate"
assert Version(importlib.metadata.version("memfabric-hybrid")) >= Version("1.2.1")
print("core:", lmcache.__file__)
print("installed PP plugin:", lmcache_ascend.__file__)
print("SDMA L2:", adapter.__file__)
print("memfabric-hybrid:", importlib.metadata.version("memfabric-hybrid"))
print("IMPORTS_OK (no BM session, mapping or transfer tested)")
