from __future__ import annotations

from dataclasses import dataclass

from negpy.infrastructure.loaders.memory import PreviewMemoryEstimate
from negpy.services.rendering.preview_cache import PreviewCacheUsage


# Shared with any other RAM-consuming background worker that needs the same floor.
MIN_RAM_RESERVE_BYTES = 512 * 1024 * 1024
_INTEGRATED_GPU_RESERVE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class PrefetchDecision:
    allowed: bool
    reason: str
    required_ram_bytes: int


def decide_prefetch(
    estimate: PreviewMemoryEstimate,
    cache: PreviewCacheUsage,
    available_ram_bytes: int | None,
    *,
    integrated_gpu: bool,
) -> PrefetchDecision:
    """Admit a prefetch only when safe LRU eviction and RAM margins can hold it."""
    if cache.entries_remaining + cache.reclaimable_entries < 1:
        return PrefetchDecision(False, "preview cache entry budget is full", 0)
    if estimate.cached_bytes > cache.bytes_remaining + cache.reclaimable_bytes:
        return PrefetchDecision(False, "preview cache byte budget is full", 0)

    working_bytes = estimate.temporary_bytes + estimate.cached_bytes
    reserve_bytes = max(MIN_RAM_RESERVE_BYTES, working_bytes // 4)
    if integrated_gpu:
        reserve_bytes += max(_INTEGRATED_GPU_RESERVE_BYTES, estimate.cached_bytes * 2)
    required_ram_bytes = working_bytes + reserve_bytes
    if available_ram_bytes is None:
        return PrefetchDecision(False, "available system memory is unknown", required_ram_bytes)
    if available_ram_bytes < required_ram_bytes:
        return PrefetchDecision(False, "available system memory is below the safety margin", required_ram_bytes)
    return PrefetchDecision(True, "memory margins are sufficient", required_ram_bytes)
