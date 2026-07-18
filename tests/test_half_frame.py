"""Half-frame mode: split detection, slicing, identities, and per-half plumbing."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from negpy.domain.models import ExportConfig, WorkspaceConfig
from negpy.services.assets.half_frame import (
    base_hash,
    detect_split_x,
    half_hash,
    half_name,
    slice_for_asset,
    slice_half,
)
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

    def test_half_name(self):
        assert half_name("IMG420.tif", 2) == "IMG420.tif [2]"

    def test_sidecar_path(self):
        assert sidecar_path_for("/a/roll.tif") == "/a/roll.negpy"
        assert sidecar_path_for("/a/roll.tif", 1) == "/a/roll.1.negpy"

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
