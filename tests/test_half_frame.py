"""Half-frame mode: split detection, slicing, identities, and per-half plumbing."""

import os

import numpy as np
import pytest
from unittest.mock import MagicMock

from negpy.domain.models import ExportConfig, WorkspaceConfig
from negpy.services.assets.half_frame import (
    base_hash,
    detect_split_x,
    SPLIT_SCANS_KEY,
    diptych_configs,
    gap_px,
    half_hash,
    half_name,
    join_halves,
    remember_split_scans,
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

    def test_a_flagged_negative_skips_the_lookup(self):
        from negpy.desktop.controller import AppController

        ctrl = self._controller({"ha#1": WorkspaceConfig()})
        info, pair = AppController._diptych_task(ctrl, {"hash": "ha", "diptych": False})
        assert pair is None and info["hash"] == "ha"
        ctrl.session.repo.load_file_settings_many.assert_not_called()

    def test_composite_kind_and_summary(self):
        from negpy.desktop.session import composite_kind, composite_summary

        asset = {"hash": "ha", "diptych": True}
        assert composite_kind(asset) == "diptych"
        assert "Diptych" in composite_summary(asset)
        # A half still reads as a half while the mode is on.
        assert composite_kind({"hash": "ha#1", "half": 1, "diptych": True}) == "half"
