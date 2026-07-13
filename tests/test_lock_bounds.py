import sys
import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QApplication

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState, DesktopSessionManager
from negpy.domain.models import WorkspaceConfig
from negpy.features.process.models import ProcessConfig, invalidate_local_bounds
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.rendering.preview_manager import PreviewManager

if not QApplication.instance():
    _app = QApplication(sys.argv)

_FLOORS = (0.1, 0.2, 0.3)
_CEILS = (0.8, 0.85, 0.9)


# ── Helper function ───────────────────────────────────────────────────────────


class TestInvalidateLocalBounds(unittest.TestCase):
    def test_unlocked_returns_zero_tuples(self):
        proc = ProcessConfig(local_floors=_FLOORS, local_ceils=_CEILS, lock_bounds=False)
        result = invalidate_local_bounds(proc)
        self.assertEqual(result, {"local_floors": (0.0, 0.0, 0.0), "local_ceils": (0.0, 0.0, 0.0)})

    def test_locked_returns_empty_dict(self):
        proc = ProcessConfig(local_floors=_FLOORS, local_ceils=_CEILS, lock_bounds=True)
        self.assertEqual(invalidate_local_bounds(proc), {})

    def test_default_config_is_unlocked(self):
        self.assertFalse(ProcessConfig().lock_bounds)

    def test_unlocked_replace_clears_bounds(self):
        proc = ProcessConfig(local_floors=_FLOORS, local_ceils=_CEILS, lock_bounds=False)
        result = replace(proc, **invalidate_local_bounds(proc))
        self.assertEqual(result.local_floors, (0.0, 0.0, 0.0))
        self.assertEqual(result.local_ceils, (0.0, 0.0, 0.0))

    def test_locked_replace_is_noop(self):
        proc = ProcessConfig(local_floors=_FLOORS, local_ceils=_CEILS, lock_bounds=True)
        result = replace(proc, **invalidate_local_bounds(proc))
        self.assertEqual(result.local_floors, _FLOORS)
        self.assertEqual(result.local_ceils, _CEILS)


# ── Session copy / paste ──────────────────────────────────────────────────────


class TestCopySettingsBounds(unittest.TestCase):
    def setUp(self):
        mock_repo = MagicMock(spec=StorageRepository)
        mock_repo.load_file_settings.return_value = None
        mock_repo.load_file_settings_by_path.return_value = None
        mock_repo.get_global_setting.return_value = None
        mock_repo.get_max_history_index.return_value = 0
        self.session = DesktopSessionManager(mock_repo)

        self.session.state.config = replace(
            WorkspaceConfig(),
            process=ProcessConfig(local_floors=_FLOORS, local_ceils=_CEILS, lock_bounds=True),
        )
        self.session.state.current_file_hash = "hash1"

    def test_copy_default_strips_local_bounds(self):
        self.session.copy_settings()
        proc = self.session.state.clipboard.process
        self.assertEqual(proc.local_floors, (0.0, 0.0, 0.0))
        self.assertEqual(proc.local_ceils, (0.0, 0.0, 0.0))

    def test_copy_default_strips_lock_flag(self):
        self.session.copy_settings()
        self.assertFalse(self.session.state.clipboard.process.lock_bounds)

    def test_copy_with_bounds_preserves_local_bounds(self):
        self.session.copy_settings_with_bounds()
        proc = self.session.state.clipboard.process
        self.assertEqual(proc.local_floors, _FLOORS)
        self.assertEqual(proc.local_ceils, _CEILS)

    def test_copy_with_bounds_preserves_lock_flag(self):
        self.session.copy_settings_with_bounds()
        self.assertTrue(self.session.state.clipboard.process.lock_bounds)

    def test_copy_default_preserves_other_process_fields(self):
        self.session.state.config = replace(
            self.session.state.config,
            process=replace(self.session.state.config.process, analysis_buffer=0.25, luma_range_clip=0.05),
        )
        self.session.copy_settings()
        proc = self.session.state.clipboard.process
        self.assertAlmostEqual(proc.analysis_buffer, 0.25)
        self.assertAlmostEqual(proc.luma_range_clip, 0.05)

    def test_copy_is_deep_copy(self):
        self.session.copy_settings_with_bounds()
        clipboard_proc = self.session.state.clipboard.process
        # Modifying source config should not affect clipboard
        self.session.state.config = replace(
            self.session.state.config,
            process=replace(self.session.state.config.process, analysis_buffer=0.99),
        )
        self.assertNotAlmostEqual(clipboard_proc.analysis_buffer, 0.99)


# ── Controller crop operations ────────────────────────────────────────────────


def _make_controller():
    mock_session = MagicMock(spec=DesktopSessionManager)
    mock_session.state = AppState()
    mock_session.repo = MagicMock()

    with (
        patch("negpy.desktop.controller.RenderWorker") as mock_rw,
        patch("negpy.desktop.controller.PreviewManager") as mock_pm,
    ):
        mock_rw.return_value = MagicMock()
        mock_pm.return_value = MagicMock(spec=PreviewManager)
        mock_pm.return_value.load_linear_preview.return_value = (None, (0, 0), {})
        ctrl = AppController(mock_session)

    ctrl.request_render = MagicMock()
    return ctrl


def _teardown_controller(ctrl):
    import gc

    for thread in [
        ctrl.render_thread,
        ctrl.export_thread,
        ctrl.thumb_thread,
        ctrl.norm_thread,
        ctrl.discovery_thread,
        ctrl.preview_load_thread,
        ctrl.scan_thread,
    ]:
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait()
    del ctrl
    gc.collect()


def _set_process(ctrl, **kwargs):
    ctrl.state.config = replace(
        ctrl.state.config,
        process=replace(ctrl.state.config.process, **kwargs),
    )


def _saved_process(ctrl):
    return ctrl.session.update_config.call_args.args[0].process


class TestCropClearsBoundsWhenUnlocked(unittest.TestCase):
    def setUp(self):
        self.ctrl = _make_controller()
        _set_process(self.ctrl, local_floors=_FLOORS, local_ceils=_CEILS, lock_bounds=False)

    def tearDown(self):
        _teardown_controller(self.ctrl)

    def test_apply_auto_crop_clears_bounds(self):
        self.ctrl.apply_auto_crop()
        proc = _saved_process(self.ctrl)
        self.assertEqual(proc.local_floors, (0.0, 0.0, 0.0))
        self.assertEqual(proc.local_ceils, (0.0, 0.0, 0.0))

    def test_reset_crop_clears_bounds(self):
        self.ctrl.reset_crop()
        proc = _saved_process(self.ctrl)
        self.assertEqual(proc.local_floors, (0.0, 0.0, 0.0))
        self.assertEqual(proc.local_ceils, (0.0, 0.0, 0.0))


class TestCropPreservesBoundsWhenLocked(unittest.TestCase):
    def setUp(self):
        self.ctrl = _make_controller()
        _set_process(self.ctrl, local_floors=_FLOORS, local_ceils=_CEILS, lock_bounds=True)

    def tearDown(self):
        _teardown_controller(self.ctrl)

    def test_apply_auto_crop_preserves_bounds(self):
        self.ctrl.apply_auto_crop()
        proc = _saved_process(self.ctrl)
        self.assertEqual(proc.local_floors, _FLOORS)
        self.assertEqual(proc.local_ceils, _CEILS)

    def test_reset_crop_preserves_bounds(self):
        self.ctrl.reset_crop()
        proc = _saved_process(self.ctrl)
        self.assertEqual(proc.local_floors, _FLOORS)
        self.assertEqual(proc.local_ceils, _CEILS)

    def test_handle_crop_rect_changed_preserves_bounds(self):
        from negpy.desktop.session import ToolMode

        self.ctrl.state.active_tool = ToolMode.CROP_MANUAL
        self.ctrl.handle_crop_rect_changed(0.1, 0.1, 0.9, 0.9, True)
        proc = _saved_process(self.ctrl)
        self.assertEqual(proc.local_floors, _FLOORS)
        self.assertEqual(proc.local_ceils, _CEILS)

    def test_detect_aspect_ratio_preserves_bounds(self):
        import numpy as np

        self.ctrl.state.preview_raw = np.zeros((300, 400, 3), dtype=np.uint8)
        with patch("negpy.desktop.controller.detect_closest_aspect_ratio", return_value="4:3"):
            geo = replace(self.ctrl.state.config.geometry, autocrop_ratio="3:2")
            self.ctrl.state.config = replace(self.ctrl.state.config, geometry=geo)
            self.ctrl.detect_aspect_ratio()
        proc = _saved_process(self.ctrl)
        self.assertEqual(proc.local_floors, _FLOORS)
        self.assertEqual(proc.local_ceils, _CEILS)


# ── Render write-back ─────────────────────────────────────────────────────────


class FakeBounds:
    def __init__(self, floors, ceils):
        self.floors = floors
        self.ceils = ceils


class TestRenderWritebackRespectsLock(unittest.TestCase):
    def setUp(self):
        self.ctrl = _make_controller()

    def tearDown(self):
        _teardown_controller(self.ctrl)

    def _call_metrics(self, floors, ceils, lock_bounds, use_luma_average=False, use_colour_average=False):
        _set_process(
            self.ctrl,
            local_floors=(0.0, 0.0, 0.0),
            local_ceils=(0.0, 0.0, 0.0),
            lock_bounds=lock_bounds,
            use_luma_average=use_luma_average,
            use_colour_average=use_colour_average,
        )
        self.ctrl._on_metrics_updated({"log_bounds": FakeBounds(floors, ceils)})

    def test_writeback_updates_bounds_when_unlocked(self):
        new_floors = (0.05, 0.06, 0.07)
        new_ceils = (0.91, 0.92, 0.93)
        self.ctrl._on_metrics_updated({"log_bounds": FakeBounds(new_floors, new_ceils)})
        proc = _saved_process(self.ctrl)
        self.assertEqual(proc.local_floors, new_floors)
        self.assertEqual(proc.local_ceils, new_ceils)

    def test_writeback_skips_when_locked(self):
        _set_process(self.ctrl, local_floors=(0.0, 0.0, 0.0), local_ceils=(0.0, 0.0, 0.0), lock_bounds=True)
        self.ctrl._on_metrics_updated({"log_bounds": FakeBounds((0.1, 0.1, 0.1), (0.9, 0.9, 0.9))})
        self.ctrl.session.update_config.assert_not_called()

    def test_writeback_skips_when_both_averages(self):
        _set_process(
            self.ctrl,
            local_floors=(0.0, 0.0, 0.0),
            local_ceils=(0.0, 0.0, 0.0),
            lock_bounds=False,
            use_luma_average=True,
            use_colour_average=True,
        )
        self.ctrl._on_metrics_updated({"log_bounds": FakeBounds((0.1, 0.1, 0.1), (0.9, 0.9, 0.9))})
        self.ctrl.session.update_config.assert_not_called()

    def test_writeback_runs_under_partial_roll(self):
        # Only one axis rides the roll — the per-frame component must still persist locally.
        _set_process(
            self.ctrl,
            local_floors=(0.0, 0.0, 0.0),
            local_ceils=(0.0, 0.0, 0.0),
            lock_bounds=False,
            use_luma_average=True,
            use_colour_average=False,
        )
        new_floors = (0.05, 0.06, 0.07)
        new_ceils = (0.91, 0.92, 0.93)
        self.ctrl._on_metrics_updated({"log_bounds": FakeBounds(new_floors, new_ceils)})
        proc = _saved_process(self.ctrl)
        self.assertEqual(proc.local_floors, new_floors)
        self.assertEqual(proc.local_ceils, new_ceils)

    def test_writeback_skips_when_bounds_unchanged(self):
        floors = (0.1, 0.1, 0.1)
        ceils = (0.9, 0.9, 0.9)
        _set_process(self.ctrl, local_floors=floors, local_ceils=ceils, lock_bounds=False)
        self.ctrl._on_metrics_updated({"log_bounds": FakeBounds(floors, ceils)})
        self.ctrl.session.update_config.assert_not_called()

    def test_writeback_skips_when_no_log_bounds_key(self):
        self.ctrl._on_metrics_updated({"histogram": [1, 2, 3]})
        self.ctrl.session.update_config.assert_not_called()

    def test_writeback_skips_when_ephemeral(self):
        # Splash (embedded-JPEG) first-paint render must not persist its bounds.
        _set_process(self.ctrl, local_floors=(0.0, 0.0, 0.0), local_ceils=(0.0, 0.0, 0.0), lock_bounds=False)
        self.ctrl._on_metrics_updated({"log_bounds": FakeBounds((0.1, 0.1, 0.1), (0.9, 0.9, 0.9)), "ephemeral": True})
        self.ctrl.session.update_config.assert_not_called()

    def test_writeback_skips_when_source_hash_mismatch(self):
        # Late metric from a different file (fast switch) must not poison the current file.
        _set_process(self.ctrl, local_floors=(0.0, 0.0, 0.0), local_ceils=(0.0, 0.0, 0.0), lock_bounds=False)
        self.ctrl._on_metrics_updated({"log_bounds": FakeBounds((0.1, 0.1, 0.1), (0.9, 0.9, 0.9)), "source_hash": "other_file"})
        self.ctrl.session.update_config.assert_not_called()

    def test_writeback_runs_when_source_hash_matches(self):
        _set_process(self.ctrl, local_floors=(0.0, 0.0, 0.0), local_ceils=(0.0, 0.0, 0.0), lock_bounds=False)
        new_floors = (0.05, 0.06, 0.07)
        new_ceils = (0.91, 0.92, 0.93)
        self.ctrl._on_metrics_updated({"log_bounds": FakeBounds(new_floors, new_ceils), "source_hash": self.ctrl.state.current_file_hash})
        proc = _saved_process(self.ctrl)
        self.assertEqual(proc.local_floors, new_floors)
        self.assertEqual(proc.local_ceils, new_ceils)

    def test_writeback_persists_base_not_mixed(self):
        # Persist the per-frame base, never the final mix — persisting the mix and
        # re-feeding it drifts (the colour-only-roll edit-stacking residual).
        _set_process(self.ctrl, local_floors=(0.0, 0.0, 0.0), local_ceils=(0.0, 0.0, 0.0), lock_bounds=False)
        base = ((0.05, 0.06, 0.07), (0.91, 0.92, 0.93))
        self.ctrl._on_metrics_updated(
            {
                "log_bounds": FakeBounds((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # mixed — must be ignored
                "log_bounds_base": FakeBounds(*base),
            }
        )
        proc = _saved_process(self.ctrl)
        self.assertEqual(proc.local_floors, base[0])
        self.assertEqual(proc.local_ceils, base[1])


if __name__ == "__main__":
    unittest.main()
