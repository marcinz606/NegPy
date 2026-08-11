from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from negpy.services.rendering.preview_cache import PreviewBufferCache, PreviewCacheKey
from negpy.features.rgbscan.models import RgbScanConfig
from negpy.services.rendering.preview_manager import PreviewManager
from negpy.desktop.workers.render import PreviewLoadTask, PreviewLoadWorker


def _small_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        preview_cache_max_entries=2,
        preview_cache_max_bytes=10**9,
        preview_cache_max_full_res_entries=1,
        preview_render_size=2000,
        canvas_zoom_min=0.25,
        canvas_zoom_max=8.0,
    )


def test_cache_eviction_by_count() -> None:
    c = PreviewBufferCache(_small_cfg())
    a = np.zeros((4, 4, 3), dtype=np.float32)
    b = np.ones((4, 4, 3), dtype=np.float32)
    c.put(PreviewCacheKey("h1", False, "Adobe RGB", False), a, (4, 4), {})
    c.put(PreviewCacheKey("h2", False, "Adobe RGB", False), b, (4, 4), {})
    c.put(PreviewCacheKey("h3", False, "Adobe RGB", False), a.copy(), (4, 4), {})
    assert c.get(PreviewCacheKey("h1", False, "Adobe RGB", False)) is None
    assert c.get(PreviewCacheKey("h3", False, "Adobe RGB", False)) is not None


def test_cache_skips_entry_larger_than_byte_cap() -> None:
    """An over-cap buffer must be rejected outright, not inserted and then
    evicted along with every other entry."""
    cfg = _small_cfg()
    cfg.preview_cache_max_bytes = 100
    c = PreviewBufferCache(cfg)
    small = np.zeros((2, 2, 3), dtype=np.float32)  # 48 B
    huge = np.zeros((8, 8, 3), dtype=np.float32)  # 768 B > cap
    c.put(PreviewCacheKey("small", False, "Adobe RGB", False), small, (2, 2), {})
    c.put(PreviewCacheKey("huge", False, "Adobe RGB", True), huge, (8, 8), {})
    assert c.get(PreviewCacheKey("huge", False, "Adobe RGB", True)) is None
    # The resident small entry must survive the rejected insert.
    assert c.get(PreviewCacheKey("small", False, "Adobe RGB", False)) is not None


def test_full_resolution_entries_respect_slot_budget() -> None:
    """Full-res (HQ) buffers beyond the slot budget evict oldest-first instead of
    pushing every small preview out through the byte cap."""
    cfg = _small_cfg()  # budget of 1: any new HQ entry replaces the previous one
    c = PreviewBufferCache(cfg)
    small = np.zeros((2, 2, 3), dtype=np.float32)
    full = np.zeros((4, 4, 3), dtype=np.float32)
    c.put(PreviewCacheKey("frame1", False, "Adobe RGB", False), small, (2, 2), {})
    c.put(PreviewCacheKey("frame1", False, "Adobe RGB", True), full, (4, 4), {})
    c.put(PreviewCacheKey("frame2", False, "Adobe RGB", True), full.copy(), (4, 4), {})
    assert c.get(PreviewCacheKey("frame1", False, "Adobe RGB", True)) is None
    assert c.get(PreviewCacheKey("frame2", False, "Adobe RGB", True)) is not None
    assert c.get(PreviewCacheKey("frame1", False, "Adobe RGB", False)) is not None


def test_full_resolution_budget_keeps_previous_frame() -> None:
    """With the default budget of 2, navigating A -> B -> A hits the cache for A
    (no HQ re-decode); a third frame evicts the oldest (A) first."""
    cfg = _small_cfg()
    cfg.preview_cache_max_entries = 4
    cfg.preview_cache_max_full_res_entries = 2
    c = PreviewBufferCache(cfg)
    full = np.zeros((4, 4, 3), dtype=np.float32)
    c.put(PreviewCacheKey("A", False, "Adobe RGB", True), full, (4, 4), {})
    c.put(PreviewCacheKey("B", False, "Adobe RGB", True), full.copy(), (4, 4), {})
    assert c.get(PreviewCacheKey("A", False, "Adobe RGB", True)) is not None  # navigate back: hit
    c.put(PreviewCacheKey("C", False, "Adobe RGB", True), full.copy(), (4, 4), {})
    # A was refreshed by the get above, so LRU-oldest B goes first.
    assert c.get(PreviewCacheKey("B", False, "Adobe RGB", True)) is None
    assert c.get(PreviewCacheKey("A", False, "Adobe RGB", True)) is not None
    assert c.get(PreviewCacheKey("C", False, "Adobe RGB", True)) is not None


def test_full_resolution_reput_same_key_keeps_entry() -> None:
    c = PreviewBufferCache(_small_cfg())
    full = np.zeros((4, 4, 3), dtype=np.float32)
    key = PreviewCacheKey("frame1", False, "Adobe RGB", True)
    c.put(key, full, (4, 4), {})
    c.put(key, full.copy(), (4, 4), {})
    assert c.get(key) is not None


def test_cache_bypasses_second_postprocess() -> None:
    """After first load, second load with same key must not call raw.postprocess."""
    import rawpy

    rgb = np.zeros((8, 8, 3), dtype=np.uint16)
    raw = MagicMock()
    raw.raw_type = rawpy.RawType.Flat
    raw.raw_pattern = np.zeros((2, 2), dtype=np.uint8)
    raw.sizes = SimpleNamespace(raw_height=8, raw_width=8, iheight=8, iwidth=8)
    n_calls = [0]

    def _pp(**kwargs: object) -> object:
        n_calls[0] += 1
        return rgb

    raw.postprocess = _pp

    class _Ctx:
        def __enter__(self) -> MagicMock:
            return raw

        def __exit__(self, *args: object) -> None:
            return None

    pm = PreviewManager()
    with (
        patch("negpy.services.rendering.preview_manager.loader_factory") as lf,
        patch("negpy.services.rendering.preview_manager.APP_CONFIG", _small_cfg()),
    ):
        lf.get_loader.return_value = (_Ctx(), {"color_space": "Adobe RGB"})
        out, _, _ = pm.load_linear_preview("/x.dng", file_hash="abc")
        assert n_calls[0] == 1
        pm.load_linear_preview("/x.dng", file_hash="abc")
        assert n_calls[0] == 1
        # Cold load and cache share one buffer (read-only contract) — the
        # defensive copy doubled steady RSS on HQ loads.
        hit = pm._cache.get(PreviewCacheKey("abc", False, "Adobe RGB", False))
        assert hit is not None and hit[0] is out


def test_cache_warm_task_does_not_emit_finished() -> None:
    """Prefetch jobs populate cache only — no `finished` to the UI path."""
    pm = MagicMock()
    pm.load_linear_preview.return_value = (MagicMock(), (1, 1), {})
    w = PreviewLoadWorker(pm)
    fin = MagicMock()
    w.finished.connect(fin)
    t = PreviewLoadTask(
        file_path="/n.dng",
        workspace_color_space="Adobe RGB",
        use_camera_wb=False,
        for_cache_warm=True,
        file_hash="x",
    )
    w.process(t)
    fin.assert_not_called()
    pm.load_linear_preview.assert_called_once()


def test_rgb_preview_cache_invalidates_when_companion_content_changes(tmp_path) -> None:
    paths = [tmp_path / name for name in ("r.dng", "g.dng", "b.dng")]
    for path in paths:
        path.write_bytes(b"old")

    pm = PreviewManager()

    def decode(path, *_args, **_kwargs):
        value = {str(paths[0]): 0.2, str(paths[1]): 0.4, str(paths[2]): 0.6}[path]
        return np.full((4, 4, 3), value, dtype=np.float32), (4, 4), {}

    pm.load_linear_preview = MagicMock(side_effect=decode)
    rgbscan = RgbScanConfig(enabled=True, green_path=str(paths[1]), blue_path=str(paths[2]), align=False)
    args = (str(paths[0]), rgbscan, "Adobe RGB")

    pm.load_linear_preview_rgb(*args, file_hash="same-red-hash")
    pm.load_linear_preview_rgb(*args, file_hash="same-red-hash")
    assert pm.load_linear_preview.call_count == 3  # unchanged triplet reuses the merge

    paths[1].write_bytes(b"new-green-content")
    pm.load_linear_preview_rgb(*args, file_hash="same-red-hash")

    assert pm.load_linear_preview.call_count == 6


def test_hdr_preview_resamples_a_rounding_difference_but_rejects_a_different_aspect(tmp_path) -> None:
    """Preview sizing rounds per file, so a pixel or two between bracket frames is
    ordinary. A different aspect is not — resizing a rotated or differently cropped frame
    onto the reference would merge it as if it lined up, and the full-res path raises."""
    from negpy.features.hdr.models import HdrConfig

    paths = [tmp_path / name for name in ("a.dng", "b.dng")]
    for path in paths:
        path.write_bytes(b"x")
    hdr = HdrConfig(hdr_enabled=True, hdr_paths=(str(paths[1]),), hdr_ratios=(1.0, 2.0), hdr_align=False)

    def _manager(sibling_shape):
        pm = PreviewManager()
        shapes = {str(paths[0]): (40, 60, 3), str(paths[1]): sibling_shape}

        def decode(path, *_args, **_kwargs):
            shape = shapes[path]
            return np.full(shape, 0.4, dtype=np.float32), shape[:2], {}

        pm.load_linear_preview = MagicMock(side_effect=decode)
        return pm

    merged, _, _ = _manager((41, 61, 3)).load_linear_preview_hdr(str(paths[0]), hdr, "Adobe RGB")
    assert np.asarray(merged).shape[:2] == (40, 60)

    try:
        _manager((60, 40, 3)).load_linear_preview_hdr(str(paths[0]), hdr, "Adobe RGB")
    except ValueError as e:
        assert "differ in shape" in str(e)
    else:
        raise AssertionError("a transposed frame was silently squashed onto the reference")
