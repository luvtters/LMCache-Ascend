# SPDX-License-Identifier: Apache-2.0
"""NPU registration selection and legacy opt-in stage pinning."""

from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lmcache.v1.multiprocess.pp_layout import PPRegistrationInfo
    from lmcache.v1.platform.base.cache_context import BaseCacheContext
    from lmcache.v1.platform.devices.npu import NpuDeviceSpec


def _create_pinned_context(
    self: NpuDeviceSpec, *args: Any, **kwargs: Any
) -> BaseCacheContext:
    # Delay context imports until registration, after LMCache bootstrap ends.
    from lmcache_ascend.v1.pp_stage_pinning import PinnedNpuCacheContext

    return PinnedNpuCacheContext(*args, **kwargs)


def install_npu_stage_pinning() -> None:
    """Select the plugin's NPU-only context in this process, idempotently.

    Called only for explicit LMCACHE_ASCEND_PP_STAGE_LOCAL=1 activation. The
    factory class is already discovered before native ops import this plugin.
    """
    from lmcache.v1.platform.devices.npu import NpuDeviceSpec

    NpuDeviceSpec.create_cache_context = _create_pinned_context


def _create_context_for_registration(
    self: NpuDeviceSpec,
    *args: Any,
    registration_info: PPRegistrationInfo | None,
    **kwargs: Any,
) -> BaseCacheContext:
    """Use stage pinning only for the new PP registration path."""
    if registration_info is None:
        return self.create_cache_context(*args, **kwargs)
    return _create_pinned_context(self, *args, **kwargs)


def install_npu_registration_context() -> None:
    """Install the metadata selection hook without importing device contexts."""
    from lmcache.v1.platform.devices.npu import NpuDeviceSpec

    NpuDeviceSpec.create_cache_context_for_registration = (
        _create_context_for_registration
    )
