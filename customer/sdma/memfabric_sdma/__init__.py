# SPDX-License-Identifier: Apache-2.0
"""Load only this checkout's L2 modules alongside the installed PP plugin.

Add customer/sdma (not the repository root) to PYTHONPATH. This namespace
does not import or replace the installed lmcache_ascend package.
"""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[3] / "lmcache_ascend/v1/distributed")]
