"""Half-frame mode: split detection, slicing, identities, and per-half plumbing."""

import os

import numpy as np
import pytest
from unittest.mock import MagicMock

from negpy.domain.models import ExportConfig, WorkspaceConfig
from negpy.services.assets.half_frame import (
    HalfGeometry,
    base_hash,
    detect_film_crop,
    detect_gutter,
    detect_split_x,
    SPLIT_SCANS_KEY,
    diptych_configs,
    forget_split_scan,
    gap_px,
    half_hash,
    half_name,
    join_halves,
    remap_point,
    remap_workspace_config,
    remember_split_scans,
    saved_crop_rect,
    slice_for_asset,
    slice_half,
    slice_half_dimensions,
)
from negpy.features.local.models import LocalAdjustmentsConfig, LocalMask
from negpy.features.retouch.models import RetouchConfig
from negpy.services.assets.sidecar import load_or_promote, sidecar_path_for
from negpy.services.export.templating import render_export_filename


def _two_frame_scan(gutter_value: float, w: int = 400, gutter_w: int = 16) -> np.ndarray:
    rng = np.random.default_rng(0)
    h = 200
    side = (w - gutter_w) // 2
    left = 0.35 + 0.3 * rng.random((h, side, 3))
    right = 0.4 + 0.3 * rng.random((h, w - gutter_w - side, 3))
    gutter = np.full((h, gutter_w, 3), gutter_value)
    return np.concatenate([left, gutter, right], axis=1).astype(np.float32)


class TestDetectSplitX:
    def test_dark_gutter(self):
        sx = detect_split_x(_two_frame_scan(0.02))
        assert abs(sx - 0.5) < 0.03 and sx != 0.5

    def test_bright_gutter(self):
        sx = detect_split_x(_two_frame_scan(0.98))
        assert abs(sx - 0.5) < 0.03 and sx != 0.5

    def test_off_center_gutter(self):
        scan = _two_frame_scan(0.98)
        scan = np.roll(scan, 40, axis=1)  # gutter at ~0.6
        assert abs(detect_split_x(scan) - 0.6) < 0.03

    def test_no_gutter_falls_back_to_center(self):
        rng = np.random.default_rng(1)
        flat = (0.4 + 0.2 * rng.random((200, 400, 3))).astype(np.float32)
        assert detect_split_x(flat) == 0.5

    def test_in_scene_step_edge_rejected(self):
        # Bright left frame, dark right frame, no gutter: the brightness step
        # must not be mistaken for a gutter.
        rng = np.random.default_rng(2)
        left = 0.7 + 0.2 * rng.random((200, 200, 3))
        right = 0.05 + 0.1 * rng.random((200, 200, 3))
        scan = np.concatenate([left, right], axis=1).astype(np.float32)
        assert detect_split_x(scan) == 0.5

    def test_textured_vertical_feature_rejected(self):
        # A narrow bright band that varies along y (in-scene feature, not film base).
        scan = _two_frame_scan(0.5)
        h, w = scan.shape[:2]
        band = slice(w // 2 - 8, w // 2 + 8)
        scan[:, band] = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
        assert detect_split_x(scan) == 0.5

    def test_tiny_image_falls_back(self):
        assert detect_split_x(np.zeros((4, 20, 3), np.float32)) == 0.5


def _diptych_scan_with_rebate() -> np.ndarray:
    """Light bed (1.0) >> film base/rebate (0.78) > exposed frames (0.25), a wide margin
    on every side (as a scanner bed around a whole film strip would be), too wide for the
    edge search's own bounded margin -- it is real content the detector cannot tell from
    a bed this size, so it is left alone rather than trimmed on a guess."""
    img = np.full((480, 720, 3), 1.0, dtype=np.float32)
    img[80:400, 100:620] = 0.78  # film strip incl. rebate
    img[105:375, 135:585] = 0.25  # both exposed frames plus the gutter between them
    return img


def _diptych_scan_with_edge_rebate(margin_w: int = 40, w: int = 600, h: int = 300, rebate: float = 0.95, seed: int = 0) -> np.ndarray:
    """A tight diptych with a real, narrow rebate band on the left edge only, within a
    plausible sprocket margin -- the shape the edge search is built to find."""
    rng = np.random.default_rng(seed)
    img = (0.3 + 0.3 * rng.random((h, w, 3))).astype(np.float32)
    img[:, :margin_w] = rebate
    return img


def _diptych_scan_with_rebate_on_two_sides(seed: int = 1) -> np.ndarray:
    """The same rebate tone on the left and top edges, meeting at a consistent corner --
    each side is detected independently."""
    rng = np.random.default_rng(seed)
    h, w = 300, 600
    img = (0.3 + 0.3 * rng.random((h, w, 3))).astype(np.float32)
    img[:, :30] = 0.95
    img[:20, :] = 0.95
    return img


def _scene_with_a_flat_midtone_band() -> np.ndarray:
    """A locally uniform strip near one edge that sits well inside the frame's own
    tonal range rather than near its darkest or brightest tone -- real content (a calm
    sea, an overcast sky), not unexposed film, even though it is flat and contrasts
    with its immediate neighbor."""
    rng = np.random.default_rng(3)
    img = (0.15 + 0.3 * rng.random((300, 600, 3))).astype(np.float32)
    img[:, :50] = 0.55
    img[:, 400:410] = 1.0  # establishes the frame's real peak, off the flat band
    return img


class TestDetectFilmCrop:
    def test_narrow_rebate_on_one_side_is_trimmed(self):
        roi = detect_film_crop(_diptych_scan_with_edge_rebate())
        assert roi is not None
        x1, y1, x2, y2 = roi
        assert abs(x1 - 40 / 600) < 0.01
        assert (y1, x2, y2) == (0.0, 1.0, 1.0)

    def test_the_rect_survives_a_settings_round_trip_as_numbers(self):
        """Settings are JSON with default=str, so a numpy scalar would read back as a string."""
        import json

        roi = detect_film_crop(_diptych_scan_with_edge_rebate())
        assert roi is not None
        assert json.loads(json.dumps(list(roi), default=str)) == list(roi)

    def test_each_side_is_detected_independently(self):
        roi = detect_film_crop(_diptych_scan_with_rebate_on_two_sides())
        assert roi is not None
        x1, y1, x2, y2 = roi
        assert abs(x1 - 30 / 600) < 0.01
        assert abs(y1 - 20 / 300) < 0.01
        assert (x2, y2) == (1.0, 1.0)

    def test_a_scanner_bed_margin_is_too_wide_to_trust(self):
        """A margin far wider than any plausible rebate is real content the edge
        search cannot rule out, so every side is left uncropped."""
        assert detect_film_crop(_diptych_scan_with_rebate()) is None

    def test_a_flat_band_that_is_not_extremal_is_rejected(self):
        assert detect_film_crop(_scene_with_a_flat_midtone_band()) is None

    def test_no_rebate_on_any_side_returns_none(self):
        assert detect_film_crop(_two_frame_scan(0.02)) is None

    def test_tiny_image_returns_none(self):
        assert detect_film_crop(np.zeros((4, 20, 3), np.float32)) is None


def _gradient_gutter_scan(
    true_frac: float = 0.5,
    w: int = 2000,
    h: int = 800,
    gutter_w: int = 30,
    gutter_value: float = 0.95,
    decay_span_frac: float = 0.16,
    dark: bool = False,
    seed: int = 0,
) -> tuple:
    """A thin gutter with a smooth in-scene gradient (an overexposed sky, say)
    blending into the frame past one of its edges. The gutter's own edges stay
    sharp; only the scene beyond them fades gradually toward it."""
    rng = np.random.default_rng(seed)
    base = 0.6 if dark else 0.4
    true_center = int(w * true_frac)
    lo, hi = true_center - gutter_w // 2, true_center - gutter_w // 2 + gutter_w
    img = np.full((h, w), base, dtype=np.float32)
    img[:, :lo] = base - 0.05 + 0.1 * rng.random((h, lo)).astype(np.float32)
    decay_px = max(1, int(w * decay_span_frac))
    x = np.arange(w - hi, dtype=np.float32)
    img[:, hi:] = base + (gutter_value - base) * np.exp(-x / decay_px) + 0.02 * rng.random((h, w - hi)).astype(np.float32)
    img[:, lo:hi] = gutter_value
    img += 0.015 * rng.standard_normal((h, w)).astype(np.float32)
    img = np.clip(img, 0, 1)
    return np.repeat(img[:, :, None], 3, axis=2).astype(np.float32), true_center / w


class TestDetectGutter:
    """A smooth gradient blending into one side of the gutter must not pull the
    detected center toward it, and the band's own width is also measurable."""

    def test_gradient_blending_into_one_side_stays_centered(self):
        scan, true_center = _gradient_gutter_scan()
        sx, thickness = detect_gutter(scan)
        assert abs(sx - true_center) < 0.015
        assert 0.005 < thickness < 0.05

    def test_dark_gutter_with_gradient_blending(self):
        scan, true_center = _gradient_gutter_scan(gutter_value=0.05, dark=True)
        sx, _ = detect_gutter(scan)
        assert abs(sx - true_center) < 0.015

    def test_off_center_gutter_with_gradient_blending(self):
        scan, true_center = _gradient_gutter_scan(true_frac=0.42)
        sx, _ = detect_gutter(scan)
        assert abs(sx - true_center) < 0.015

    def test_thickness_matches_the_true_band_width(self):
        # a 16px gutter in a 400px scan is a 4% band
        _, thickness = detect_gutter(_two_frame_scan(0.02))
        assert abs(thickness - 0.04) < 0.01

    def test_rejection_returns_the_tuple_fallback(self):
        rng = np.random.default_rng(1)
        flat = (0.4 + 0.2 * rng.random((200, 400, 3))).astype(np.float32)
        assert detect_gutter(flat) == (0.5, 0.0)


class TestSliceHalf:
    @pytest.mark.parametrize("w", [100, 101])
    def test_halves_partition_width(self, w):
        buf = np.arange(2 * w * 3, dtype=np.float32).reshape(2, w, 3)
        h1 = slice_half(buf, 1, 0.5)
        h2 = slice_half(buf, 2, 0.5)
        assert h1.shape[1] + h2.shape[1] == w
        np.testing.assert_array_equal(np.concatenate([h1, h2], axis=1), buf)

    def test_extreme_split_never_empty(self):
        buf = np.zeros((2, 50, 3), np.float32)
        assert slice_half(buf, 1, 0.0).shape[1] == 1
        assert slice_half(buf, 2, 1.0).shape[1] == 1

    def test_slice_for_asset(self):
        buf = np.zeros((2, 100, 3), np.float32)
        assert slice_for_asset(buf, {"path": "p"}) is buf
        assert slice_for_asset(buf, {"half": 2, "split_x": 0.25}).shape[1] == 75


class TestIdentities:
    def test_hash_roundtrip(self):
        assert half_hash("abc", 1) == "abc#1"
        assert base_hash("abc#2") == "abc"
        assert base_hash("abc") == "abc"
        assert base_hash(None) is None

    def test_half_of_reads_only_a_numeric_suffix(self):
        from negpy.features.hdr.models import hdr_hash
        from negpy.features.stitch.models import stitch_hash
        from negpy.services.assets.half_frame import half_of

        assert half_of("abc#1") == 1
        assert half_of("abc#2") == 2
        assert half_of("abc") is None
        assert half_of(None) is None
        # A composite shares the separator by design, so only the suffix can decide.
        assert half_of(hdr_hash(["a", "b"])) is None
        assert half_of(stitch_hash(["a", "b"])) is None

    def test_half_name(self):
        assert half_name("IMG420.tif", 2) == "IMG420.tif [2]"

    def test_sidecar_path(self):
        assert sidecar_path_for("/a/roll.tif") == os.path.join("/a", "roll.negpy")
        assert sidecar_path_for("/a/roll.tif", 1) == os.path.join("/a", "roll.1.negpy")

    def test_export_filename_suffix(self):
        plain = render_export_filename("/x/IMG420.tif", ExportConfig())
        halved = render_export_filename("/x/IMG420.tif", ExportConfig(), half=2)
        assert "IMG420" in plain and "IMG420_2" not in plain
        assert "IMG420_2" in halved


def test_expand_half_frames(monkeypatch):
    from negpy.desktop.workers import render as render_mod

    monkeypatch.setattr("negpy.services.assets.half_frame.detect_split_x_for_file", lambda p: 0.48)
    worker = render_mod.AssetDiscoveryWorker()
    assets = [
        {"name": "a.tif", "path": "/p/a.tif", "hash": "ha"},
        {"name": "t.raw", "path": "/p/t.raw", "hash": "ht", "green_path": "/p/g.raw", "blue_path": "/p/b.raw"},
    ]
    out = worker._expand_half_frames(assets)
    assert [a["hash"] for a in out] == ["ha#1", "ha#2", "ht"]
    assert out[0]["name"] == "a.tif [1]" and out[1]["name"] == "a.tif [2]"
    assert out[0]["path"] == out[1]["path"] == "/p/a.tif"
    assert out[0]["split_x"] == out[1]["split_x"] == 0.48


def test_saved_crop_rect_reads_string_values_as_floats():
    assert saved_crop_rect(["0.04874884", 0.0, "0.9512549", 1.0]) == (0.04874884, 0.0, 0.9512549, 1.0)
    assert saved_crop_rect(None) is None
    assert saved_crop_rect([0.0, 0.0, 1.0]) is None
    assert saved_crop_rect(["x", 0.0, 1.0, 1.0]) is None


def test_expand_half_frames_reads_a_crop_saved_as_strings(monkeypatch):
    """An override saved from numpy values before they were cast still slices."""
    from negpy.desktop.workers import render as render_mod

    worker = render_mod.AssetDiscoveryWorker()
    assets = [{"name": "a.tif", "path": "/p/a.tif", "hash": "ha"}]
    overrides = {"ha": {"crop_rect": ["0.05", 0.0, "0.95", 1.0], "split_x": 0.5, "gutter_thickness": 0.0}}
    a1, a2 = worker._expand_half_frames(assets, profile={}, overrides=overrides)
    assert a1["crop_rect"] == a2["crop_rect"] == (0.05, 0.0, 0.95, 1.0)
    assert slice_half(np.zeros((100, 200, 3), np.float32), 1, a1["split_x"], a1["crop_rect"]).shape == (100, 90, 3)


def test_expand_half_frames_with_profile_applies_it_uniformly(monkeypatch):
    """A saved profile applies to every file without its own override — a file's own
    gutter (0.1, per the monkeypatch) is ignored once a roll-wide value is set."""
    from negpy.desktop.workers import render as render_mod

    monkeypatch.setattr("negpy.services.assets.half_frame.detect_split_x_for_file", lambda p: 0.1)
    worker = render_mod.AssetDiscoveryWorker()
    assets = [{"name": "a.tif", "path": "/p/a.tif", "hash": "ha"}]
    profile = {"crop_rect": [0.0, 0.0, 1.0, 1.0], "split_x": 0.6, "gutter_thickness": 0.02}
    out = worker._expand_half_frames(assets, profile=profile)
    assert out[0]["split_x"] == 0.6
    assert out[0]["gutter_thickness"] == 0.02


def test_expand_half_frames_per_file_override_wins_over_the_profile(monkeypatch):
    """The odd frame the roll-wide profile still gets wrong: a saved override for its
    own base hash wins over it."""
    from negpy.desktop.workers import render as render_mod

    monkeypatch.setattr("negpy.services.assets.half_frame.detect_split_x_for_file", lambda p: 0.9)
    worker = render_mod.AssetDiscoveryWorker()
    assets = [
        {"name": "a.tif", "path": "/p/a.tif", "hash": "ha"},
        {"name": "b.tif", "path": "/p/b.tif", "hash": "hb"},
    ]
    profile = {"crop_rect": [0.0, 0.0, 1.0, 1.0], "split_x": 0.5, "gutter_thickness": 0.0}
    overrides = {"ha": {"crop_rect": [0.05, 0.0, 0.95, 1.0], "split_x": 0.4, "gutter_thickness": 0.03}}
    out = worker._expand_half_frames(assets, profile=profile, overrides=overrides)
    a1, a2, b1, b2 = out
    assert a1["split_x"] == a2["split_x"] == 0.4
    assert a1["crop_rect"] == a2["crop_rect"] == (0.05, 0.0, 0.95, 1.0)
    assert a1["gutter_thickness"] == a2["gutter_thickness"] == 0.03
    # b.tif has no override, so it takes the profile's fixed split, not its own gutter.
    assert b1["split_x"] == b2["split_x"] == 0.5


def test_auto_detect_all_splits_worker_emits_per_file_results(monkeypatch):
    """process_auto_detect_all_splits reports one (split, thickness, crop) triple
    per path, so a big roll's detection can run off the GUI thread and still land
    as one dict."""
    from negpy.desktop.workers import render as render_mod
    from negpy.desktop.workers.render import AutoDetectAllSplitsTask

    detected = {"/p/a.tif": (0.4, 0.02, (0.05, 0.05, 0.95, 0.95)), "/p/b.tif": (0.6, 0.0, None)}
    monkeypatch.setattr("negpy.services.assets.half_frame.detect_split_and_crop_for_file", lambda p: detected[p])
    worker = render_mod.AssetDiscoveryWorker()
    results = []
    worker.splits_detected.connect(results.append)
    worker.process_auto_detect_all_splits(AutoDetectAllSplitsTask(paths=list(detected)))
    assert results == [detected]


def test_add_files_keeps_both_halves():
    from negpy.desktop.session import DesktopSessionManager
    from negpy.infrastructure.storage.repository import StorageRepository

    repo = MagicMock(spec=StorageRepository)
    repo.get_global_setting.side_effect = lambda key, default=None: default
    repo.load_file_marks.return_value = {}
    session = DesktopSessionManager(repo)

    halves = [
        {"name": "a.tif [1]", "path": "/p/a.tif", "hash": "ha#1", "half": 1, "split_x": 0.5},
        {"name": "a.tif [2]", "path": "/p/a.tif", "hash": "ha#2", "half": 2, "split_x": 0.5},
    ]
    session.add_files([], validated_info=halves)
    assert len(session.state.uploaded_files) == 2
    # Re-adding replaces in place instead of clobbering the sibling half.
    session.add_files([], validated_info=halves)
    assert [f["hash"] for f in session.state.uploaded_files] == ["ha#1", "ha#2"]


def test_load_or_promote_half_skips_path_fallback(tmp_path):
    repo = MagicMock()
    repo.load_file_settings.return_value = None
    repo.load_file_settings_by_path.return_value = ("old_hash", WorkspaceConfig())
    src = tmp_path / "roll.tif"
    src.write_bytes(b"x")

    assert load_or_promote(repo, "h#1", str(src), half=1) is None
    repo.rehome_file_settings.assert_not_called()

    assert load_or_promote(repo, "h2", str(src)) is not None
    repo.rehome_file_settings.assert_called_once()


def test_halves_measure_independent_bounds():
    from negpy.features.exposure.normalization import analyze_log_exposure_bounds

    rng = np.random.default_rng(3)
    left = 0.08 + 0.05 * rng.random((240, 160, 3))
    right = 0.5 + 0.4 * rng.random((240, 160, 3))
    gutter = np.full((240, 12, 3), 0.95)
    scan = np.concatenate([left, gutter, right], axis=1).astype(np.float32)

    sx = detect_split_x(scan)
    b1 = analyze_log_exposure_bounds(np.ascontiguousarray(slice_half(scan, 1, sx)))
    b2 = analyze_log_exposure_bounds(np.ascontiguousarray(slice_half(scan, 2, sx)))
    # Floors differ per half (dark vs bright frame). Ceils can legitimately agree:
    # both halves keep a sliver of the bright gutter at the slice boundary.
    assert not np.allclose(b1.floors, b2.floors)


class TestSliceHalfCropGutter:
    @pytest.mark.parametrize("half", [0, 1, 2])
    def test_reported_dimensions_match_full_resolution_slice(self, half):
        dimensions = (101, 203)
        geometry = {
            "split_x": 0.43,
            "crop_rect": (0.1, 0.2, 0.9, 0.8),
            "gutter_thickness": 0.07,
        }
        buf = np.empty(dimensions, dtype=np.uint8)

        actual = slice_half(buf, half, **geometry)

        assert slice_half_dimensions(dimensions, half, **geometry) == actual.shape

    def test_crop_rect_slices_only_the_cropped_region(self):
        buf = np.arange(2 * 100 * 3, dtype=np.float32).reshape(2, 100, 3)
        # crop to x 0.2..0.8 (20..80), split at 0.5 of the crop (x=50)
        left = slice_half(buf, 1, 0.5, crop_rect=(0.2, 0.0, 0.8, 1.0))
        right = slice_half(buf, 2, 0.5, crop_rect=(0.2, 0.0, 0.8, 1.0))
        assert left.shape[1] + right.shape[1] == 60
        # left covers x 20..50, right x 50..80
        np.testing.assert_array_equal(left, buf[:, 20:50])
        np.testing.assert_array_equal(right, buf[:, 50:80])

    def test_gutter_thickness_discards_a_band_at_the_split(self):
        buf = np.arange(2 * 100 * 3, dtype=np.float32).reshape(2, 100, 3)
        # gutter of 0.2 of the cropped width (20 px) around the split at 0.5
        left = slice_half(buf, 1, 0.5, crop_rect=(0.0, 0.0, 1.0, 1.0), gutter_thickness=0.2)
        right = slice_half(buf, 2, 0.5, crop_rect=(0.0, 0.0, 1.0, 1.0), gutter_thickness=0.2)
        # 10 px discarded each side of x=50 → left 0..40, right 60..100
        np.testing.assert_array_equal(left, buf[:, :40])
        np.testing.assert_array_equal(right, buf[:, 60:])

    def test_slice_for_asset_reads_crop_and_gutter(self):
        buf = np.arange(2 * 100 * 3, dtype=np.float32).reshape(2, 100, 3)
        info = {
            "half": 1,
            "split_x": 0.5,
            "crop_rect": (0.0, 0.0, 1.0, 1.0),
            "gutter_thickness": 0.2,
        }
        out = slice_for_asset(buf, info)
        np.testing.assert_array_equal(out, buf[:, :40])


class TestRemapPoint:
    """A point stays on the same physical film location across a split/crop change:
    remap into full-scan space under the old geometry, back out under the new one."""

    def test_identity_geometry_is_a_no_op(self):
        geom = HalfGeometry()
        assert remap_point(0.3, 0.7, 1, geom, geom) == pytest.approx((0.3, 0.7))
        assert remap_point(0.3, 0.7, 2, geom, geom) == pytest.approx((0.3, 0.7))

    def test_split_shift_moves_a_left_half_point_off_center(self):
        old = HalfGeometry(split_x=0.5)
        new = HalfGeometry(split_x=0.6)
        # Half-local x=1.0 sits right at the old gutter edge (full-scan x=0.5);
        # the wider new left half places that same film position further along.
        x, y = remap_point(1.0, 0.4, 1, old, new)
        assert x == pytest.approx(0.5 / 0.6)
        assert y == pytest.approx(0.4)

    def test_split_shift_leaves_the_untouched_half_alone(self):
        """Only the crop, not the split, ever moves y or the other half's x scale
        when that half's own boundary hasn't moved."""
        old = HalfGeometry(split_x=0.5)
        new = HalfGeometry(split_x=0.5, gutter_thickness=0.1)
        x, _ = remap_point(0.5, 0.2, 1, old, new)
        assert x == pytest.approx(0.5 * 0.5 / 0.45)

    def test_crop_change_round_trips_through_full_scan_space(self):
        old = HalfGeometry(crop_rect=(0.0, 0.0, 1.0, 1.0), split_x=0.5)
        new = HalfGeometry(crop_rect=(0.1, 0.1, 0.9, 0.9), split_x=0.5)
        x, y = remap_point(0.5, 0.5, 1, old, new)
        back_x, back_y = remap_point(x, y, 1, new, old)
        assert (back_x, back_y) == pytest.approx((0.5, 0.5))

    def test_whole_frame_half_zero_ignores_split(self):
        old = HalfGeometry(split_x=0.5)
        new = HalfGeometry(split_x=0.9)
        assert remap_point(0.3, 0.3, 0, old, new) == pytest.approx((0.3, 0.3))


class TestRemapWorkspaceConfig:
    def test_same_geometry_is_a_no_op_and_keeps_identity(self):
        geom = HalfGeometry(split_x=0.5)
        config = WorkspaceConfig(retouch=RetouchConfig(manual_heal_strokes=[([[0.5, 0.5]], 10.0, 0.0, 0.0)]))
        assert remap_workspace_config(config, 1, geom, geom) is config

    def test_remaps_heal_stroke_points_and_preserves_the_clone_offset(self):
        old, new = HalfGeometry(split_x=0.5), HalfGeometry(split_x=0.6)
        config = WorkspaceConfig(retouch=RetouchConfig(manual_heal_strokes=[([[0.4, 0.4], [0.42, 0.4]], 10.0, -0.02, 0.0)]))
        updated = remap_workspace_config(config, 1, old, new)
        points, size, dx, dy = updated.retouch.manual_heal_strokes[0]
        assert points != [[0.4, 0.4], [0.42, 0.4]]
        assert size == 10.0
        # The clone source stays the same film distance from the destination.
        old_src = remap_point(0.4 - 0.02, 0.4, 1, old, old)
        new_dest = remap_point(0.4, 0.4, 1, old, new)
        new_src_expected = remap_point(*old_src, 1, old, new)
        assert (points[0][0] + dx, points[0][1] + dy) == pytest.approx(new_src_expected)
        assert points[0] == pytest.approx(list(new_dest))

    def test_remaps_dust_spots_and_scratch_lines(self):
        old, new = HalfGeometry(split_x=0.5), HalfGeometry(split_x=0.6)
        config = WorkspaceConfig(retouch=RetouchConfig(manual_dust_spots=[(0.3, 0.3, 6)], scratch_lines=[(0.1, 0.1, 0.2, 0.2, 2.0)]))
        updated = remap_workspace_config(config, 1, old, new)
        assert updated.retouch.manual_dust_spots[0][:2] == pytest.approx(remap_point(0.3, 0.3, 1, old, new))
        assert updated.retouch.manual_dust_spots[0][2] == 6
        line = updated.retouch.scratch_lines[0]
        assert line[:2] == pytest.approx(remap_point(0.1, 0.1, 1, old, new))
        assert line[2:4] == pytest.approx(remap_point(0.2, 0.2, 1, old, new))
        assert line[4] == 2.0

    def test_remaps_local_mask_vertices(self):
        old, new = HalfGeometry(split_x=0.5), HalfGeometry(split_x=0.6)
        mask = LocalMask(vertices=((0.2, 0.2), (0.3, 0.2), (0.3, 0.3)), stops=0.5)
        config = WorkspaceConfig(local=LocalAdjustmentsConfig(masks=(mask,)))
        updated = remap_workspace_config(config, 1, old, new)
        assert updated.local.masks[0].vertices[0] == pytest.approx(remap_point(0.2, 0.2, 1, old, new))
        assert updated.local.masks[0].stops == 0.5

    def test_clears_a_stale_geometry_crop_rect(self):
        """crop_rect lives in post-rotation transformed-image space, not raw space, so it
        can't be remapped like the others -- a rect drawn against the old frame boundary
        has no correct position in the new one, so it's dropped rather than left wrong."""
        old, new = HalfGeometry(split_x=0.5), HalfGeometry(split_x=0.6)
        from negpy.features.geometry.models import GeometryConfig

        config = WorkspaceConfig(geometry=GeometryConfig(crop_rect=(0.1, 0.1, 0.9, 0.9), crop_from_auto=True, crop_detect_key="k"))
        updated = remap_workspace_config(config, 1, old, new)
        assert updated.geometry.crop_rect is None
        # Left alone: crop_from_auto=True + crop_rect=None is "armed", so an auto crop
        # re-detects against the new boundary on the next render rather than vanishing.
        assert updated.geometry.crop_from_auto is True
        assert updated.geometry.crop_detect_key == "k"

    def test_a_config_with_no_crop_is_untouched(self):
        old, new = HalfGeometry(split_x=0.5), HalfGeometry(split_x=0.6)
        from negpy.features.geometry.models import GeometryConfig

        config = WorkspaceConfig(geometry=GeometryConfig())
        updated = remap_workspace_config(config, 1, old, new)
        assert updated.geometry is config.geometry


class TestDiptych:
    """Half-frame mode off: a scan that carries both halves' edits renders as one image."""

    def test_half_zero_crops_without_splitting(self):
        buf = np.arange(2 * 100 * 3, dtype=np.float32).reshape(2, 100, 3)
        np.testing.assert_array_equal(slice_half(buf, 0, 0.5, crop_rect=(0.2, 0.0, 0.8, 1.0)), buf[:, 20:80])

    def test_slice_for_asset_crops_a_whole_frame_with_a_rect(self):
        buf = np.arange(2 * 100 * 3, dtype=np.float32).reshape(2, 100, 3)
        info = {"split_x": 0.5, "crop_rect": (0.2, 0.0, 0.8, 1.0), "gutter_thickness": 0.2}
        np.testing.assert_array_equal(slice_for_asset(buf, info), buf[:, 20:80])

    def test_slice_for_asset_is_a_no_op_without_a_rect(self):
        buf = np.arange(2 * 100 * 3, dtype=np.float32).reshape(2, 100, 3)
        assert slice_for_asset(buf, {"name": "x"}) is buf

    def test_gap_px_is_scale_invariant(self):
        # 20 % gutter: the two halves hold 80 % of the width, so the gap is a quarter of them.
        assert gap_px(40, 40, 0.2) == 20
        assert gap_px(400, 400, 0.2) == 200
        assert gap_px(40, 40, 0.0) == 0

    def test_join_halves_geometry_and_gap(self):
        left = np.ones((10, 6, 3), np.float32)
        right = np.full((10, 4, 3), 0.5, np.float32)
        out = join_halves(left, right, 3)
        assert out.shape == (10, 13, 3)
        np.testing.assert_array_equal(out[:, :6], left)
        assert out[:, 6:9].max() == 0.0
        np.testing.assert_array_equal(out[:, 9:], right)

    def test_join_halves_centre_pads_unequal_heights(self):
        left = np.ones((10, 4, 3), np.float32)
        right = np.ones((6, 4, 3), np.float32)
        out = join_halves(left, right, 0)
        assert out.shape == (10, 8, 3)
        # right sits in rows 2..8, black above and below
        assert out[0, 4:].max() == 0.0 and out[9, 4:].max() == 0.0
        np.testing.assert_array_equal(out[2:8, 4:], right)

    def _repo(self, rows, split=("h",)):
        repo = MagicMock()
        repo.load_file_settings_many.side_effect = lambda keys: {k: v for k, v in rows.items() if k in keys}
        repo.get_global_setting.side_effect = lambda key, default=None: list(split) if key == SPLIT_SCANS_KEY else default
        return repo

    def test_both_halves(self):
        a, b = WorkspaceConfig(), WorkspaceConfig()
        pair = diptych_configs(self._repo({"h#1": a, "h#2": b}), "h")
        assert pair == (a, b)

    def test_missing_half_copies_its_sibling(self):
        a = WorkspaceConfig()
        assert diptych_configs(self._repo({"h#2": a}), "h") == (a, a)

    def test_no_half_edits(self):
        assert diptych_configs(self._repo({}), "h") is None

    def test_a_half_is_not_a_diptych(self):
        assert diptych_configs(self._repo({"h#1": WorkspaceConfig()}), "h#1") is None

    def test_a_composite_is_not_a_diptych(self):
        """A stitch's base is its reference frame, whose half edits are not the stitch's."""
        assert diptych_configs(self._repo({"h#1": WorkspaceConfig()}), "h#stitch") is None

    def test_a_scan_the_user_never_split_is_not_a_diptych(self):
        """Half edits keyed by content hash outlive the session; the split decision rules."""
        repo = self._repo({"h#1": WorkspaceConfig(), "h#2": WorkspaceConfig()}, split=())
        assert diptych_configs(repo, "h") is None

    def test_remember_split_scans_unions_and_skips_a_known_write(self):
        repo = self._repo({}, split=("h",))
        remember_split_scans(repo, {"g", "h"})
        repo.save_global_setting.assert_called_once_with(SPLIT_SCANS_KEY, ["g", "h"])
        repo.get_global_setting.side_effect = lambda key, default=None: ["g", "h"] if key == SPLIT_SCANS_KEY else default
        repo.save_global_setting.reset_mock()
        remember_split_scans(repo, {"h"})
        repo.save_global_setting.assert_not_called()

    def test_forget_split_scan_drops_one_hash_and_skips_an_unknown_write(self):
        repo = self._repo({"h#1": WorkspaceConfig()}, split=("g", "h"))
        forget_split_scan(repo, "h")
        repo.save_global_setting.assert_called_once_with(SPLIT_SCANS_KEY, ["g"])
        repo.save_global_setting.reset_mock()
        forget_split_scan(repo, "z")
        repo.save_global_setting.assert_not_called()

    def test_export_filename_is_not_a_half(self):
        name = render_export_filename("/x/IMG420.tif", ExportConfig(), composite="DIPTYCH")
        assert name.endswith("IMG420-DIPTYCH") and "IMG420_1" not in name


class TestDiptychRender:
    """The joined render must come from two pipeline runs on two slices, not one."""

    def _worker(self, out_by_hash):
        from negpy.desktop.workers.render import RenderWorker

        worker = RenderWorker.__new__(RenderWorker)
        seen = []

        def run_pipeline(buffer, config, source_hash, **kw):
            seen.append((buffer.shape[1], config, source_hash, kw["readback_metrics"]))
            return np.full((buffer.shape[0], buffer.shape[1], 3), out_by_hash[source_hash], np.float32), {"log_bounds": source_hash}

        worker._processor = MagicMock()
        worker._processor.run_pipeline.side_effect = run_pipeline
        return worker, seen

    def _task(self, **kw):
        from negpy.desktop.workers.render import RenderTask

        return RenderTask(
            buffer=np.zeros((8, 100, 3), np.float32),
            config=WorkspaceConfig(),
            source_hash="h",
            preview_size=1000.0,
            diptych=(WorkspaceConfig(), WorkspaceConfig()),
            **kw,
        )

    def test_each_half_renders_with_its_own_config(self):
        from negpy.desktop.workers.render import RenderWorker

        worker, seen = self._worker({"h#1": 0.25, "h#2": 0.75})
        out, _ = RenderWorker._render_diptych(worker, self._task(gutter_thickness=0.2), "h")

        assert [s[0] for s in seen] == [40, 40]  # 20 % gutter discarded around the split
        assert [s[2] for s in seen] == ["h#1", "h#2"]  # distinct stage-cache identities
        assert [s[3] for s in seen] == [True, False]  # only half 1 is metered; two writebacks would fight
        assert out.shape == (8, 100, 3)  # gap restores the original width
        assert out[0, 0, 0] == 0.25 and out[0, 99, 0] == 0.75 and out[0, 50, 0] == 0.0

    def test_metrics_come_from_half_one_and_are_marked(self):
        from negpy.desktop.workers.render import RenderWorker

        worker, _ = self._worker({"h#1": 0.25, "h#2": 0.75})
        _, metrics = RenderWorker._render_diptych(worker, self._task(readback_metrics=True), "h")
        assert metrics["log_bounds"] == "h#1"
        assert metrics["diptych"] is True


class TestDiptychAsset:
    """Discovery flags the scan; every later reader takes the flag off the asset dict."""

    def _controller(self, rows, split=("ha",)):
        from negpy.desktop.controller import AppController

        ctrl = AppController.__new__(AppController)
        ctrl.session = MagicMock()
        ctrl.session.repo.load_file_settings_many.side_effect = lambda keys: {k: v for k, v in rows.items() if k in keys}
        profile = {"crop_rect": [0.1, 0.0, 0.9, 1.0], "split_x": 0.4, "gutter_thickness": 0.05}
        ctrl.session.repo.get_global_setting.side_effect = lambda key, default=None: list(split) if key == SPLIT_SCANS_KEY else profile
        ctrl._active_diptych_memo = ("", None)
        return ctrl

    def test_mark_diptychs_flags_only_scans_with_half_edits(self):
        from negpy.desktop.controller import AppController

        ctrl = self._controller({"ha#2": WorkspaceConfig()})
        assets = [
            {"path": "/p/a.tif", "hash": "ha"},
            {"path": "/p/b.tif", "hash": "hb"},
            {"path": "/p/c.tif", "hash": "hc#1", "half": 1},
        ]
        AppController._mark_diptychs(ctrl, assets)
        assert assets[0]["diptych"] is True
        assert assets[1]["diptych"] is False
        assert "diptych" not in assets[2]  # a half is never its own diptych

    def test_half_edits_alone_do_not_make_a_diptych(self):
        from negpy.desktop.controller import AppController

        # A scan the user never split with Half Frame on stays whole, however its content
        # hash was worked on in an earlier session or another folder.
        ctrl = self._controller({"ha#1": WorkspaceConfig(), "ha#2": WorkspaceConfig()}, split=())
        asset = {"path": "/p/a.tif", "hash": "ha"}
        AppController._mark_diptychs(ctrl, [asset])
        assert asset.get("diptych") is not True
        assert AppController.diptych_pair(ctrl, {"path": "/p/a.tif", "hash": "ha"}) is None

    def test_a_composite_keeps_its_primarys_half_edits_out(self):
        from negpy.desktop.controller import AppController

        # An RGB triplet is {**red, green_path, blue_path}, so it carries the red
        # exposure's plain hash. Half edits left on that file by an earlier half-frame
        # session must not make the assembled frame render as a diptych.
        ctrl = self._controller({"ha#1": WorkspaceConfig(), "ha#2": WorkspaceConfig()})
        triplet = {"path": "/p/a_r.cr3", "hash": "ha", "green_path": "/p/a_g.cr3", "blue_path": "/p/a_b.cr3"}
        assets = [triplet, {"path": "/p/b.tif", "hash": "ha"}]

        AppController._mark_diptychs(ctrl, assets)
        assert triplet["diptych"] is False
        assert assets[1]["diptych"] is True  # same hash, but a plain scan

        assert AppController.diptych_pair(ctrl, {"path": "/p/a_r.cr3", "hash": "ha", "green_path": "/p/a_g.cr3"}) is None
        assert AppController.diptych_pair(ctrl, {"path": "/p/a.tif", "hash": "ha", "stitch_paths": ["/p/x.tif"]}) is None
        assert AppController.diptych_pair(ctrl, {"path": "/p/a.tif", "hash": "ha", "hdr_paths": ["/p/x.tif"]}) is None

    def test_metering_a_half_does_not_create_its_edit(self):
        from negpy.desktop.controller import AppController

        # Looking at a half renders it, which meters it. That measurement must not file a
        # settings row of its own: the row is what later says the scan is a diptych, so a
        # Half Frame toggle on and straight off would leave the frame stuck as one.
        ctrl = self._controller({})
        ctrl.state = MagicMock()
        ctrl._measured_half_rows = set()

        ctrl.state.current_file_hash = "ha"  # whole scan: always persists
        assert AppController._may_persist_measured_bounds(ctrl) is True

        ctrl.session.repo.load_file_settings.return_value = None
        ctrl.state.current_file_hash = "ha#1"  # unedited half
        assert AppController._may_persist_measured_bounds(ctrl) is False

        ctrl.session.repo.load_file_settings.return_value = WorkspaceConfig()
        ctrl.state.current_file_hash = "ha#2"  # a half the user did edit
        assert AppController._may_persist_measured_bounds(ctrl) is True
        ctrl.session.repo.load_file_settings.reset_mock()
        assert AppController._may_persist_measured_bounds(ctrl) is True
        ctrl.session.repo.load_file_settings.assert_not_called()  # memoized

    def test_task_stamps_the_saved_split_geometry(self):
        from negpy.desktop.controller import AppController

        cfg = WorkspaceConfig()
        ctrl = self._controller({"ha#1": cfg, "ha#2": cfg})
        info, pair = AppController._diptych_task(ctrl, {"path": "/p/a.tif", "hash": "ha", "diptych": True})
        assert pair == (cfg, cfg)
        assert info["split_x"] == 0.4
        assert info["crop_rect"] == (0.1, 0.0, 0.9, 1.0)
        assert info["gutter_thickness"] == 0.05

    def test_task_prefers_the_files_own_override_over_the_profile(self):
        """A diptych's split geometry is whatever the two halves were actually cut
        with -- an override, when this file has one -- not blindly the roll's shared
        profile, which may since have been set to something else."""
        from negpy.desktop.controller import AppController

        cfg = WorkspaceConfig()
        ctrl = AppController.__new__(AppController)
        ctrl.session = MagicMock()
        ctrl.session.repo.load_file_settings_many.side_effect = lambda keys: {"ha#1": cfg, "ha#2": cfg}
        store = {
            SPLIT_SCANS_KEY: ["ha"],
            "half_frame_overrides": {"ha": {"crop_rect": [0.05, 0.0, 0.95, 1.0], "split_x": 0.6, "gutter_thickness": 0.03}},
            "half_frame_profile": {"crop_rect": [0.0, 0.0, 1.0, 1.0], "split_x": 0.5, "gutter_thickness": 0.0},
        }
        ctrl.session.repo.get_global_setting.side_effect = lambda key, default=None: store.get(key, default)
        ctrl._active_diptych_memo = ("", None)

        info, pair = AppController._diptych_task(ctrl, {"path": "/p/a.tif", "hash": "ha", "diptych": True})
        assert pair == (cfg, cfg)
        assert info["split_x"] == 0.6
        assert info["crop_rect"] == (0.05, 0.0, 0.95, 1.0)
        assert info["gutter_thickness"] == 0.03

    def test_a_flagged_negative_skips_the_lookup(self):
        from negpy.desktop.controller import AppController

        ctrl = self._controller({"ha#1": WorkspaceConfig()})
        info, pair = AppController._diptych_task(ctrl, {"hash": "ha", "diptych": False})
        assert pair is None and info["hash"] == "ha"
        ctrl.session.repo.load_file_settings_many.assert_not_called()

    def test_undiptych_forgets_the_split_and_deletes_both_halves(self):
        from negpy.desktop.controller import AppController

        ctrl = self._controller({"ha#1": WorkspaceConfig(), "ha#2": WorkspaceConfig()})
        asset = {"path": "/p/a.tif", "hash": "ha", "diptych": True}
        ctrl.state = MagicMock()
        ctrl.state.selected_file_idx = 0
        ctrl.state.uploaded_files = [asset]
        ctrl.state.current_file_hash = "ha"
        ctrl._measured_half_rows = {"ha#1"}
        ctrl.set_status = MagicMock()
        ctrl.load_file = MagicMock()

        AppController.request_undiptych(ctrl)

        ctrl.session.repo.save_global_setting.assert_called_once_with(SPLIT_SCANS_KEY, [])
        assert [c.args[0] for c in ctrl.session.repo.delete_file_settings.call_args_list] == ["ha#1", "ha#2"]
        assert asset["diptych"] is False
        assert ctrl._measured_half_rows == set()  # or a re-meter files a row again
        assert ctrl._active_diptych_memo == ("", None)
        ctrl.load_file.assert_called_once_with("/p/a.tif")

    def test_undiptych_leaves_a_plain_frame_alone(self):
        from negpy.desktop.controller import AppController

        ctrl = self._controller({})
        ctrl.state = MagicMock()
        ctrl.state.selected_file_idx = 0
        ctrl.state.uploaded_files = [{"path": "/p/a.tif", "hash": "ha"}]
        ctrl.set_status = MagicMock()

        AppController.request_undiptych(ctrl)

        ctrl.session.repo.delete_file_settings.assert_not_called()
        ctrl.session.repo.save_global_setting.assert_not_called()

    def test_composite_kind_and_summary(self):
        from negpy.desktop.session import composite_kind, composite_summary

        asset = {"hash": "ha", "diptych": True}
        assert composite_kind(asset) == "diptych"
        assert "Diptych" in composite_summary(asset)
        # A half still reads as a half while the mode is on.
        assert composite_kind({"hash": "ha#1", "half": 1, "diptych": True}) == "half"
