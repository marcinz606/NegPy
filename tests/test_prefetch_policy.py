from unittest.mock import MagicMock, patch
from dataclasses import replace

import numpy as np
import tifffile

from negpy.infrastructure.loaders.memory import PreviewMemoryEstimate
from negpy.services.rendering.prefetch_policy import decide_prefetch
from negpy.services.rendering.preview_cache import PreviewCacheUsage
from negpy.services.rendering.preview_manager import PreviewManager
from negpy.kernel.system.memory import _parse_vm_stat


def _cache(
    *,
    entries_remaining: int = 4,
    bytes_remaining: int = 512 * 1024 * 1024,
    reclaimable_entries: int = 0,
    reclaimable_bytes: int = 0,
) -> PreviewCacheUsage:
    return PreviewCacheUsage(
        entries=0,
        bytes_used=0,
        entries_remaining=entries_remaining,
        bytes_remaining=bytes_remaining,
        reclaimable_entries=reclaimable_entries,
        reclaimable_bytes=reclaimable_bytes,
    )


def _estimate(*, cached: int = 24 * 1024 * 1024, temporary: int = 512 * 1024 * 1024) -> PreviewMemoryEstimate:
    return PreviewMemoryEstimate(cached, temporary, (6000, 4000))


def test_prefetch_is_allowed_with_cache_and_ram_margin() -> None:
    decision = decide_prefetch(_estimate(), _cache(), 4 * 1024 * 1024 * 1024, integrated_gpu=False)

    assert decision.allowed


def test_prefetch_is_skipped_below_ram_margin() -> None:
    decision = decide_prefetch(_estimate(temporary=2 * 1024 * 1024 * 1024), _cache(), 2 * 1024 * 1024 * 1024, integrated_gpu=False)

    assert not decision.allowed
    assert "system memory" in decision.reason


def test_prefetch_is_skipped_when_cache_budget_cannot_hold_result() -> None:
    decision = decide_prefetch(_estimate(cached=64 * 1024 * 1024), _cache(bytes_remaining=32 * 1024 * 1024), None, integrated_gpu=False)

    assert not decision.allowed
    assert "cache byte" in decision.reason


def test_prefetch_is_allowed_when_a_cold_cache_entry_can_be_replaced() -> None:
    cache = _cache(
        entries_remaining=0,
        bytes_remaining=0,
        reclaimable_entries=1,
        reclaimable_bytes=64 * 1024 * 1024,
    )

    decision = decide_prefetch(_estimate(cached=64 * 1024 * 1024), cache, 4 * 1024 * 1024 * 1024, integrated_gpu=False)

    assert decision.allowed


def test_prefetch_is_skipped_when_a_full_cache_contains_only_the_active_entry() -> None:
    cache = _cache(entries_remaining=0, bytes_remaining=0)

    decision = decide_prefetch(_estimate(cached=1), cache, 4 * 1024 * 1024 * 1024, integrated_gpu=False)

    assert not decision.allowed
    assert "cache entry" in decision.reason


def test_integrated_gpu_keeps_extra_system_ram_free() -> None:
    estimate = _estimate(temporary=256 * 1024 * 1024)
    discrete = decide_prefetch(estimate, _cache(), 900 * 1024 * 1024, integrated_gpu=False)
    integrated = decide_prefetch(estimate, _cache(), 900 * 1024 * 1024, integrated_gpu=True)

    assert discrete.allowed
    assert not integrated.allowed


def test_prefetch_preflight_rejects_before_loader_decode() -> None:
    manager = PreviewManager()
    manager.load_linear_preview = MagicMock()

    with (
        patch("negpy.services.rendering.preview_manager.loader_factory.estimate_linear_preview_prefetch_memory", return_value=_estimate()),
        patch("negpy.services.rendering.preview_manager.available_system_memory_bytes", return_value=1),
    ):
        admitted = manager.prefetch_linear_preview(
            "/large.dng",
            "Adobe RGB",
            use_camera_wb=False,
            file_hash="hash",
        )

    assert not admitted
    manager.load_linear_preview.assert_not_called()


def test_prefetch_forwards_highlight_decode_identity() -> None:
    manager = PreviewManager()
    manager.load_linear_preview = MagicMock()

    with (
        patch("negpy.services.rendering.preview_manager.loader_factory.estimate_linear_preview_prefetch_memory", return_value=_estimate()),
        patch("negpy.services.rendering.preview_manager.available_system_memory_bytes", return_value=4 * 1024**3),
    ):
        admitted = manager.prefetch_linear_preview(
            "/slide.dng",
            "Adobe RGB",
            use_camera_wb=False,
            file_hash="slide",
            highlight_mode=5,
            bake_camera_wb=True,
        )

    assert admitted
    assert manager.load_linear_preview.call_args.kwargs["highlight_mode"] == 5
    assert manager.load_linear_preview.call_args.kwargs["bake_camera_wb"] is True


def test_prefetch_budgets_retained_ir_before_decode(tmp_path) -> None:
    path = tmp_path / "rgbi.tif"
    tifffile.imwrite(path, np.zeros((100, 150, 4), dtype=np.uint16), photometric="rgb", extrasamples=[0])
    manager = PreviewManager()
    manager._cache._app = replace(manager._cache._app, preview_cache_max_bytes=200_000)
    manager.load_linear_preview = MagicMock()
    with patch("negpy.services.rendering.preview_manager.available_system_memory_bytes", return_value=4 * 1024**3):
        admitted = manager.prefetch_linear_preview(str(path), "Adobe RGB", use_camera_wb=False, file_hash="rgbi")
    assert not admitted
    manager.load_linear_preview.assert_not_called()


def test_macos_available_memory_parser_uses_reclaimable_pages() -> None:
    output = """Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 10.\nPages active: 100.\nPages inactive: 20.\nPages speculative: 5.\n"""

    assert _parse_vm_stat(output) == 35 * 16384
