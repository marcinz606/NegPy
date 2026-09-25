import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import replace
from types import SimpleNamespace

import numpy as np


from PIL import Image
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from negpy.desktop.controller import AppController
from negpy.desktop.session import DesktopSessionManager, AppState, ToolMode
from negpy.desktop.settings_catalog import rows_for_fields
from negpy.desktop.workers.export import ExportTask, resolve_export_target_path
from negpy.features.geometry.logic import autocrop_detection_key
from negpy.domain.models import (
    ColorSpace,
    ExportConfig,
    ExportFormat,
    ExportPreset,
    ExportPresetOutputMode,
    ExportResolutionMode,
    WorkspaceConfig,
)
from negpy.infrastructure.scanners.params import ScanParams
from negpy.services.assets import rolls
from negpy.services.assets.thumbnails import asset_thumbnail_key
from negpy.services.rendering.preview_manager import PreviewManager

if not QApplication.instance():
    _app = QApplication(sys.argv)


def _slide_config(cfg):
    from negpy.features.process.models import ProcessMode

    return replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6))


def _positive_slide_config(cfg):
    """A Positive frame, which only Slide can be."""
    cfg = _slide_config(cfg)
    return replace(
        cfg,
        process=replace(cfg.process, positive_source=True),
        exposure=replace(cfg.exposure, auto_exposure=False, auto_normalize_contrast=False),
    )


class TestAppController(unittest.TestCase):
    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()

        # Patch GPU-touching classes before AppController.__init__ so no real GPU is created
        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)

    def tearDown(self):
        import gc

        # Stop all background threads before the controller is GC'd
        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_half_frame_profile_round_trip(self):
        self.controller.session.repo.get_global_setting.return_value = None
        self.assertIsNone(self.controller.half_frame_profile())

        self.controller.save_half_frame_profile([0.0, 0.0, 1.0, 1.0], 0.6, 0.02)
        args, _ = self.controller.session.repo.save_global_setting.call_args
        self.assertEqual(args[0], "half_frame_profile")
        self.assertEqual(args[1], {"crop_rect": [0.0, 0.0, 1.0, 1.0], "split_x": 0.6, "gutter_thickness": 0.02})

    def test_half_frame_override_round_trip(self):
        self.controller.session.repo.get_global_setting.return_value = None
        self.assertEqual(self.controller.half_frame_overrides(), {})
        self.assertIsNone(self.controller.half_frame_override("h1"))

        self.controller.save_half_frame_override("h1", [0.05, 0.0, 0.95, 1.0], 0.42, 0.01)
        args, _ = self.controller.session.repo.save_global_setting.call_args
        self.assertEqual(args[0], "half_frame_overrides")
        self.assertEqual(args[1], {"h1": {"crop_rect": [0.05, 0.0, 0.95, 1.0], "split_x": 0.42, "gutter_thickness": 0.01}})

    def test_clear_half_frame_override_only_writes_when_present(self):
        self.controller.session.repo.get_global_setting.return_value = {"h1": {"split_x": 0.4}}
        self.controller.clear_half_frame_override("h1")
        self.controller.session.repo.save_global_setting.assert_called_once_with("half_frame_overrides", {})

        self.controller.session.repo.save_global_setting.reset_mock()
        self.controller.session.repo.get_global_setting.return_value = {}
        self.controller.clear_half_frame_override("h2")
        self.controller.session.repo.save_global_setting.assert_not_called()

    def test_current_base_file_returns_the_base_hash_for_a_split_asset(self):
        """Both halves share one path, so matching by path alone would always return
        whichever comes first in the list -- never necessarily the active one -- and its
        own #1/#2 hash, which save_half_frame_override does not key by."""
        self.controller.state.uploaded_files = [
            {"path": "/tmp/scan.tif", "hash": "h1#1", "half": 1},
            {"path": "/tmp/scan.tif", "hash": "h1#2", "half": 2},
        ]
        self.controller.state.current_file_path = "/tmp/scan.tif"
        self.controller.state.current_file_hash = "h1#2"  # the active half, listed second

        self.assertEqual(self.controller.current_base_file(), ("/tmp/scan.tif", "h1"))

    def test_selected_base_hashes_dedupes_both_halves_and_drops_composites(self):
        self.controller.state.uploaded_files = [
            {"path": "/tmp/scan.tif", "hash": "h1#1", "half": 1},
            {"path": "/tmp/scan.tif", "hash": "h1#2", "half": 2},
            {"path": "/tmp/pano.tif", "hash": "hc", "stitch_paths": ("/tmp/a.tif",)},
        ]
        self.controller.state.selected_indices = [0, 1, 2]

        self.assertEqual(self.controller.selected_base_hashes(), ["h1"])

    def _patch_dialog(self, crop_rect=(0.1, 0.0, 0.9, 1.0), split_x=0.42, gutter=0.01, scope="current"):
        import numpy as np

        fake_img = np.zeros((4, 4, 3), dtype=np.uint8)
        decode_patch = patch("negpy.services.assets.thumbnails.decode_source_image", return_value=fake_img)
        decode_patch.start()
        self.addCleanup(decode_patch.stop)
        dialog_cls_patch = patch("negpy.desktop.view.widgets.half_frame_dialog.HalfFrameDialog")
        mock_dialog_cls = dialog_cls_patch.start()
        self.addCleanup(dialog_cls_patch.stop)
        mock_dialog = MagicMock()
        mock_dialog.exec.return_value = True
        mock_dialog.crop_rect.return_value = crop_rect
        mock_dialog.split_x.return_value = split_x
        mock_dialog.gutter_thickness.return_value = gutter
        mock_dialog.scope.return_value = scope
        mock_dialog_cls.return_value = mock_dialog
        return mock_dialog_cls

    def test_open_half_frame_dialog_current_scope_saves_an_override_not_the_profile(self):
        self._patch_dialog(scope="current")
        self.controller.session.repo.get_global_setting.return_value = None
        self.controller.session.repo.load_file_settings.return_value = None
        result = self.controller.open_half_frame_dialog("/p/a.tif", "ha")

        self.assertEqual(result, {"crop_rect": [0.1, 0.0, 0.9, 1.0], "split_x": 0.42, "gutter_thickness": 0.01})
        saved = {c.args[0]: c.args[1] for c in self.controller.session.repo.save_global_setting.call_args_list}
        self.assertEqual(
            saved["half_frame_overrides"], {"ha": {"crop_rect": [0.1, 0.0, 0.9, 1.0], "split_x": 0.42, "gutter_thickness": 0.01}}
        )
        # The chosen scope is remembered as next time's default.
        self.assertEqual(saved["half_frame_apply_scope"], "current")

    def test_open_half_frame_dialog_all_scope_saves_the_profile(self):
        self._patch_dialog(crop_rect=(0.0, 0.0, 1.0, 1.0), split_x=0.5, gutter=0.0, scope="all")
        self.controller.session.repo.get_global_setting.return_value = None
        self.controller.session.repo.load_file_settings.return_value = None
        result = self.controller.open_half_frame_dialog("/p/a.tif", "ha")

        self.assertEqual(result, {"crop_rect": [0.0, 0.0, 1.0, 1.0], "split_x": 0.5, "gutter_thickness": 0.0})
        saved = {c.args[0]: c.args[1] for c in self.controller.session.repo.save_global_setting.call_args_list}
        self.assertEqual(saved["half_frame_profile"], {"crop_rect": [0.0, 0.0, 1.0, 1.0], "split_x": 0.5, "gutter_thickness": 0.0})

    def test_open_half_frame_dialog_selected_scope_saves_an_override_on_each_hash(self):
        """Each save reads the settings store before writing, so a real repo (unlike
        a bare Mock) sees the prior hash's override still there for the next one."""
        self._patch_dialog(scope="selected")
        store: dict = {}
        self.controller.session.repo.get_global_setting.side_effect = lambda key, default=None: store.get(key, default)
        self.controller.session.repo.save_global_setting.side_effect = lambda key, value: store.__setitem__(key, value)
        self.controller.session.repo.load_file_settings.return_value = None
        self.controller.open_half_frame_dialog("/p/a.tif", "ha", selected_hashes=["ha", "hb"])

        overrides = store["half_frame_overrides"]
        self.assertEqual(set(overrides), {"ha", "hb"})
        for entry in overrides.values():
            self.assertEqual(entry, {"crop_rect": [0.1, 0.0, 0.9, 1.0], "split_x": 0.42, "gutter_thickness": 0.01})

    def test_open_half_frame_dialog_seeds_the_editor_from_the_remembered_scope(self):
        """No explicit initial_scope: the editor opens on whatever scope Apply last used."""
        mock_dialog_cls = self._patch_dialog()
        self.controller.session.repo.get_global_setting.side_effect = (
            lambda key, default=None: "all" if key == "half_frame_apply_scope" else None
        )
        self.controller.session.repo.load_file_settings.return_value = None
        self.controller.open_half_frame_dialog("/p/a.tif", "ha")
        self.assertEqual(mock_dialog_cls.call_args.kwargs["initial_scope"], "all")

    def test_open_half_frame_dialog_initial_scope_overrides_the_remembered_one(self):
        """The per-frame context menu always starts at 'current', whatever was last used."""
        mock_dialog_cls = self._patch_dialog()
        self.controller.session.repo.get_global_setting.side_effect = (
            lambda key, default=None: "all" if key == "half_frame_apply_scope" else None
        )
        self.controller.session.repo.load_file_settings.return_value = None
        self.controller.open_half_frame_dialog("/p/a.tif", "ha", initial_scope="current")
        self.assertEqual(mock_dialog_cls.call_args.kwargs["initial_scope"], "current")

    def test_open_half_frame_dialog_remaps_existing_manual_edits(self):
        """A frame with heal strokes already saved: moving the split re-anchors them
        instead of leaving them pointing at the old, now-wrong, position."""
        from negpy.domain.models import WorkspaceConfig
        from negpy.features.retouch.models import RetouchConfig

        self._patch_dialog(crop_rect=(0.0, 0.0, 1.0, 1.0), split_x=0.6, gutter=0.0, scope="current")
        old_profile = {"crop_rect": [0.0, 0.0, 1.0, 1.0], "split_x": 0.5, "gutter_thickness": 0.0}
        self.controller.session.repo.get_global_setting.side_effect = (
            lambda key, default=None: old_profile if key == "half_frame_profile" else None
        )

        half1 = WorkspaceConfig(retouch=RetouchConfig(manual_heal_strokes=[([[0.5, 0.5]], 10.0, 0.0, 0.0)]))
        self.controller.session.repo.load_file_settings.side_effect = lambda h: half1 if h == "ha#1" else None

        self.controller.open_half_frame_dialog("/p/a.tif", "ha")

        save_call = next(c for c in self.controller.session.repo.save_file_settings.call_args_list if c.args[0] == "ha#1")
        updated = save_call.args[1]
        # Old split 0.5, new split 0.6: half=1 local x=0.5 sat at the old gutter edge,
        # which the wider left half now places further along its own width.
        self.assertNotEqual(updated.retouch.manual_heal_strokes[0][0][0][0], 0.5)
        self.controller.session.push_external_history.assert_called_with("ha#1", half1, updated)

    def test_auto_detect_all_half_frame_splits_requests_one_path_per_file(self):
        """Off the GUI thread and deduped: a half-frame roll lists each file twice
        (one entry per half), a composite never wants its own split at all."""
        self.controller.session.state.uploaded_files = [
            {"path": "/p/a.tif", "hash": "ha#1"},
            {"path": "/p/a.tif", "hash": "ha#2"},
            {"path": "/p/b.tif", "hash": "hb"},
            {"path": "/p/c.tif", "hash": "hc", "green_path": "/p/g.tif", "blue_path": "/p/bl.tif"},
        ]
        requests = []
        self.controller.auto_detect_all_splits_requested.connect(lambda t: requests.append(t))
        self.controller.auto_detect_all_half_frame_splits()

        self.assertEqual(len(requests), 1)
        self.assertEqual(set(requests[0].paths), {"/p/a.tif", "/p/b.tif"})

    def test_auto_detect_all_half_frame_splits_no_op_with_nothing_loaded(self):
        self.controller.session.state.uploaded_files = []
        requests = []
        self.controller.auto_detect_all_splits_requested.connect(lambda t: requests.append(t))
        self.controller.auto_detect_all_half_frame_splits()
        self.assertEqual(requests, [])

    def test_on_splits_detected_saves_an_override_per_file_and_reloads(self):
        self.controller.session.state.uploaded_files = [
            {"path": "/p/a.tif", "hash": "ha#1"},
            {"path": "/p/a.tif", "hash": "ha#2"},
            {"path": "/p/b.tif", "hash": "hb#1"},
            {"path": "/p/b.tif", "hash": "hb#2"},
        ]
        store: dict = {}
        self.controller.session.repo.get_global_setting.side_effect = lambda key, default=None: store.get(key, default)
        self.controller.session.repo.save_global_setting.side_effect = lambda key, value: store.__setitem__(key, value)
        self.controller.session.repo.load_file_settings.return_value = None
        self.controller.request_asset_discovery = MagicMock()

        self.controller._on_splits_detected({"/p/a.tif": (0.4, 0.02, (0.05, 0.05, 0.95, 0.95)), "/p/b.tif": (0.6, 0.0, None)})

        overrides = store["half_frame_overrides"]
        self.assertEqual(overrides["ha"]["split_x"], 0.4)
        self.assertEqual(overrides["ha"]["gutter_thickness"], 0.02)
        self.assertEqual(overrides["ha"]["crop_rect"], [0.05, 0.05, 0.95, 0.95])
        self.assertEqual(overrides["hb"]["split_x"], 0.6)
        # No crop detected for this file: falls back to the full frame, same as before.
        self.assertEqual(overrides["hb"]["crop_rect"], [0.0, 0.0, 1.0, 1.0])
        self.controller.request_asset_discovery.assert_called_once()

    def test_on_splits_detected_keeps_the_existing_crop_when_none_is_detected(self):
        self.controller.session.state.uploaded_files = [{"path": "/p/a.tif", "hash": "ha#1"}]
        store = {"half_frame_overrides": {"ha": {"crop_rect": [0.1, 0.1, 0.9, 0.9], "split_x": 0.5, "gutter_thickness": 0.0}}}
        self.controller.session.repo.get_global_setting.side_effect = lambda key, default=None: store.get(key, default)
        self.controller.session.repo.save_global_setting.side_effect = lambda key, value: store.__setitem__(key, value)
        self.controller.session.repo.load_file_settings.return_value = None
        self.controller.request_asset_discovery = MagicMock()

        self.controller._on_splits_detected({"/p/a.tif": (0.4, 0.03, None)})

        self.assertEqual(store["half_frame_overrides"]["ha"]["crop_rect"], [0.1, 0.1, 0.9, 0.9])
        self.assertEqual(store["half_frame_overrides"]["ha"]["split_x"], 0.4)
        self.assertEqual(store["half_frame_overrides"]["ha"]["gutter_thickness"], 0.03)

    def test_on_splits_detected_no_op_when_nothing_matches(self):
        self.controller.session.state.uploaded_files = [{"path": "/p/a.tif", "hash": "ha#1"}]
        self.controller.request_asset_discovery = MagicMock()
        self.controller._on_splits_detected({"/p/other.tif": (0.4, 0.0, None)})
        self.controller.session.repo.save_global_setting.assert_not_called()
        self.controller.request_asset_discovery.assert_not_called()

    def _fake_settings_store(self) -> dict:
        store: dict = {}
        self.controller.session.repo.get_global_setting.side_effect = lambda key, default=None: store.get(key, default)
        self.controller.session.repo.save_global_setting.side_effect = lambda key, value: store.__setitem__(key, value)
        return store

    def test_half_frame_mode_for_roll_reads_that_rolls_own_entry(self):
        store = self._fake_settings_store()
        store["half_frame_mode_by_roll"] = {"r1": True, "r2": False}
        self.assertTrue(self.controller.half_frame_mode_for_roll("r1"))
        self.assertFalse(self.controller.half_frame_mode_for_roll("r2"))

    def test_half_frame_mode_for_roll_defaults_off_for_an_unseen_roll(self):
        self._fake_settings_store()
        self.assertFalse(self.controller.half_frame_mode_for_roll("new-roll"))

    def test_half_frame_mode_for_roll_falls_back_to_the_sticky_flag_with_no_roll(self):
        store = self._fake_settings_store()
        store["half_frame_mode"] = True
        store["half_frame_mode_by_roll"] = {"r1": False}
        self.assertTrue(self.controller.half_frame_mode_for_roll(None))

    def test_set_half_frame_mode_writes_the_active_rolls_own_entry(self):
        store = self._fake_settings_store()
        self.controller.state.active_roll_id = "r1"
        self.controller.session.state.uploaded_files = []
        self.controller.set_half_frame_mode(True)
        self.assertEqual(store["half_frame_mode_by_roll"], {"r1": True})
        self.assertNotIn("half_frame_mode", store)

    def test_set_half_frame_mode_writes_the_sticky_flag_with_no_active_roll(self):
        store = self._fake_settings_store()
        self.controller.state.active_roll_id = None
        self.controller.session.state.uploaded_files = []
        self.controller.set_half_frame_mode(True)
        self.assertEqual(store["half_frame_mode"], True)
        self.assertNotIn("half_frame_mode_by_roll", store)

    def test_open_roll_emits_that_rolls_own_half_frame_state(self):
        store = self._fake_settings_store()
        store["half_frame_mode_by_roll"] = {"r1": True}
        with patch("negpy.desktop.controller.rolls") as mock_rolls:
            mock_rolls.roll_for_id.return_value = {"kind": "folder", "folder_path": "/p", "extra_paths": []}
            self.controller.request_asset_discovery = MagicMock()
            seen = []
            self.controller.half_frame_mode_changed.connect(seen.append)
            self.controller.open_roll("r1")
        self.assertEqual(seen, [True])
        self.assertEqual(self.controller.state.active_roll_id, "r1")

    def test_create_roll_from_session_seeds_the_new_rolls_half_frame_state(self):
        """Saving the current ad hoc session as a roll must not silently reset its
        toggle to off the next time that roll is opened."""
        store = self._fake_settings_store()
        store["half_frame_mode"] = True
        self.controller.state.uploaded_files = [{"path": "/p/a.tif"}]
        with patch("negpy.desktop.controller.rolls") as mock_rolls:
            mock_rolls.create_virtual_roll.return_value = "new-roll"
            roll_id = self.controller.create_roll_from_session("My Roll")
        self.assertEqual(roll_id, "new-roll")
        self.assertEqual(store["half_frame_mode_by_roll"], {"new-roll": True})

    def test_busy_toast_is_taken_down_when_the_frame_lands(self):
        """A slow render step holds its toast open; the finished frame clears it, and a
        toast nobody claimed is left alone."""
        msgs = []
        self.controller.status_message_requested.connect(lambda text, *_: msgs.append(text))

        self.controller._on_render_busy("removing IR dust")
        self.assertEqual(msgs, ["removing IR dust"])
        self.controller._clear_busy_toast()
        self.assertEqual(msgs, ["removing IR dust", ""])

        self.controller._clear_busy_toast()
        self.assertEqual(len(msgs), 2, "nothing pending — an unrelated toast stays up")

    def test_export_failure_does_not_blank_the_canvas(self):
        """An export, thumbnail or search failure names its job and leaves the shown frame
        alone; only a failed load of that frame may clear the canvas."""
        msgs = []
        cleared = []
        self.controller.status_message_requested.connect(lambda text, _ms, kind: msgs.append((text, kind)))
        self.controller.load_failed.connect(lambda: cleared.append(True))

        self.controller._on_export_task_error("disk full")
        self.controller._on_library_search_error("permission denied")
        self.assertEqual(cleared, [])
        self.assertEqual(msgs[0], ("Export failed: disk full", "error"))
        self.assertEqual(msgs[1], ("Library search failed: permission denied", "error"))

        self.controller._on_render_error("decode boom")
        self.assertEqual(cleared, [True])
        self.assertEqual(msgs[-1], ("Failed to load file: decode boom", "error"))

    def test_load_file_emits_zoom_reset(self):
        """Test that loading a file normally resets the zoom."""
        mock_slot = MagicMock()
        self.controller.zoom_requested.connect(mock_slot)

        self.controller.load_file("dummy.dng")

        mock_slot.assert_called_once_with(1.0)
        self.assertFalse(self.controller.state.hq_preview)

    def test_load_file_tags_the_decode_with_the_current_generation(self):
        tasks = []
        self.controller.preview_load_requested.connect(tasks.append)

        self.controller.load_file("dummy.dng")

        self.assertEqual(tasks[-1].generation, self.controller._prefetch_gen)

    def test_lens_toggles_keep_the_displayed_texture_during_reload(self):
        from negpy.infrastructure.gpu.resources import GPUTexture
        from negpy.features.lens.models import LensCorrections
        from negpy.services.rendering.lens import lens_decode_token

        self.controller.preview_load_requested.disconnect(self.controller.preview_load_worker.process)
        state = self.controller.state
        state.current_file_path = "scan.arw"
        self.controller._requested_file_path = state.current_file_path
        texture = MagicMock(spec=GPUTexture)
        state.last_metrics["base_positive"] = texture
        loading, released, cleanup, decode, zoom = (MagicMock() for _ in range(5))
        self.controller.loading_started.connect(loading)
        self.controller.gpu_textures_released.connect(released)
        self.controller._render_cleanup_requested.connect(cleanup)
        self.controller.preview_load_requested.connect(decode)
        self.controller.zoom_requested.connect(zoom)

        previous = LensCorrections()
        for corrections in (LensCorrections(True, False), LensCorrections(True, True), LensCorrections(False, True), LensCorrections()):
            state.config = replace(
                state.config,
                geometry=replace(
                    state.config.geometry, lens_distortion_from_metadata=corrections.distortion, lens_ca_from_metadata=corrections.ca
                ),
            )
            state.preview_lens_token = lens_decode_token(previous, state.config.flatfield)
            self.controller.request_render()
            task = decode.call_args.args[0]
            self.assertEqual(task.lens_corrections, corrections)
            self.assertFalse(task.use_splash)
            self.assertIs(cleanup.call_args.args[0], texture)
            previous = corrections

        loading.assert_not_called()
        released.assert_not_called()
        zoom.assert_not_called()
        texture.destroy.assert_not_called()

    def test_cpu_reload_keeps_preview_but_navigation_shows_loading(self):
        import numpy as np

        self.controller.preview_load_requested.disconnect(self.controller.preview_load_worker.process)
        self.controller._requested_file_path = "scan.arw"
        self.controller.state.last_metrics["base_positive"] = np.ones((4, 4, 3), dtype=np.float32)
        loading, decode, repaint = (MagicMock() for _ in range(3))
        self.controller.loading_started.connect(loading)
        self.controller.preview_load_requested.connect(decode)
        self.controller.image_updated.connect(repaint)

        for _ in range(2):
            self.controller.load_file("scan.arw", preserve_zoom=True)
            self.assertFalse(decode.call_args.args[0].use_splash)
        loading.assert_not_called()
        repaint.assert_not_called()

        self.controller.load_file("next.arw", preserve_zoom=True)
        loading.assert_called_once()
        self.assertTrue(decode.call_args.args[0].use_splash)

    def test_slow_reload_arms_the_spinner_late(self):
        """keep_preview suppresses the spinner so a fast reload doesn't flicker, but a
        reload still in flight past the backstop delay must show one — otherwise a slow
        decode looks identical to nothing happening."""
        from negpy.desktop.controller import _KEEP_PREVIEW_SPINNER_DELAY_MS
        import numpy as np

        self.controller.preview_load_requested.disconnect(self.controller.preview_load_worker.process)
        self.controller._requested_file_path = "scan.arw"
        self.controller.state.last_metrics["base_positive"] = np.ones((4, 4, 3), dtype=np.float32)
        loading = MagicMock()
        self.controller.loading_started.connect(loading)

        self.controller.load_file("scan.arw", preserve_zoom=True)
        loading.assert_not_called()

        QTest.qWait(_KEEP_PREVIEW_SPINNER_DELAY_MS + 100)
        loading.assert_called_once()

    def test_reload_that_finishes_before_the_backstop_never_shows_the_spinner(self):
        from negpy.desktop.controller import _KEEP_PREVIEW_SPINNER_DELAY_MS
        import numpy as np

        self.controller.preview_load_requested.disconnect(self.controller.preview_load_worker.process)
        self.controller._requested_file_path = "scan.arw"
        self.controller.state.last_metrics["base_positive"] = np.ones((4, 4, 3), dtype=np.float32)
        loading = MagicMock()
        self.controller.loading_started.connect(loading)

        self.controller.load_file("scan.arw", preserve_zoom=True)
        self.controller._foreground_preview_generation = None  # decode already landed

        QTest.qWait(_KEEP_PREVIEW_SPINNER_DELAY_MS + 100)
        loading.assert_not_called()

    def test_lens_toggle_repaints_live_cached_texture_before_decode(self):
        from negpy.infrastructure.gpu.resources import GPUTexture

        self.controller.preview_load_requested.disconnect(self.controller.preview_load_worker.process)
        state = self.controller.state
        state.current_file_path = "scan.arw"
        state.current_file_hash = "scan"
        state.uploaded_files = [{"path": "scan.arw", "hash": "scan"}]
        self.controller._requested_file_path = state.current_file_path
        cached, outgoing = MagicMock(spec=GPUTexture), MagicMock(spec=GPUTexture)
        self.controller._render_memo.store("scan", "off", {"base_positive": cached})
        state.last_metrics["base_positive"] = outgoing
        self.controller._last_render_identity = ("scan", "on", None)
        events = []
        self.controller.image_updated.connect(lambda: events.append(("paint", state.last_metrics["base_positive"])))
        self.controller.preview_load_requested.connect(lambda task: events.append(("decode", task.file_path)))

        with patch.object(self.controller, "_render_memo_key", return_value="off"):
            self.controller.load_file(state.current_file_path, preserve_zoom=True)

        self.assertEqual(events, [("paint", cached), ("decode", "scan.arw")])
        cached.destroy.assert_not_called()
        outgoing.destroy.assert_not_called()
        self.assertIs(self.controller._render_memo.get("scan", "on")["base_positive"], outgoing)

        stale_metrics = {"source_hash": "scan", "memo_key": "on", "base_positive": outgoing}
        metrics_available = MagicMock()
        self.controller.metrics_available.connect(metrics_available)
        self.controller._on_render_finished(outgoing, stale_metrics)
        self.controller._on_metrics_updated(stale_metrics)

        self.assertEqual(events, [("paint", cached), ("decode", "scan.arw")])
        self.assertIs(state.last_metrics["base_positive"], cached)
        metrics_available.assert_not_called()

    def test_preview_load_defers_neighbor_prefetch_until_render_finishes(self):
        self.controller._requested_file_path = "/tmp/a.dng"
        self.controller.request_render = MagicMock()
        self.controller._schedule_prefetch_neighbors = MagicMock()

        self.controller._on_preview_loaded("/tmp/a.dng", object(), (10, 20), "", None, "")

        self.controller.request_render.assert_called_once_with()
        self.controller._schedule_prefetch_neighbors.assert_not_called()
        self.assertEqual(self.controller._neighbor_prefetch_generation, self.controller._prefetch_gen)

    def test_vram_capped_message_shown_when_the_setting_is_on(self):
        self.controller._requested_file_path = "scan.arw"
        self.controller.session.repo.get_global_setting.return_value = True
        self.controller.set_status = MagicMock()

        self.controller._on_hq_preview_vram_capped("scan.arw", 6144)

        self.controller.set_status.assert_called_once()

    def test_vram_capped_message_hidden_when_the_setting_is_off(self):
        """The status message is optional; the downsampling that triggered it (see
        preview_manager._load_from_open_raw) is not affected by this setting at all."""
        self.controller._requested_file_path = "scan.arw"
        self.controller.session.repo.get_global_setting.return_value = False
        self.controller.set_status = MagicMock()

        self.controller._on_hq_preview_vram_capped("scan.arw", 6144)

        self.controller.set_status.assert_not_called()

    def test_vram_capped_message_hidden_when_gpu_acceleration_is_off(self):
        """A CPU render never touches the capped GPU texture, so the message is moot."""
        self.controller._requested_file_path = "scan.arw"
        self.controller.state.gpu_enabled = False
        self.controller.session.repo.get_global_setting.return_value = True
        self.controller.set_status = MagicMock()

        self.controller._on_hq_preview_vram_capped("scan.arw", 6144)

        self.controller.set_status.assert_not_called()

    def test_foreground_render_queue_blocks_neighbor_prefetch(self):
        self.controller._foreground_preview_generation = None
        self.controller._neighbor_prefetch_generation = self.controller._prefetch_gen
        self.controller._is_rendering = True
        self.controller._pending_render_task = object()
        self.controller._schedule_prefetch_neighbors = MagicMock()

        self.controller._continue_background_work()

        self.controller._schedule_prefetch_neighbors.assert_not_called()

    def test_only_one_neighbor_prefetch_is_dispatched_at_a_time(self):
        first = MagicMock(generation=4)
        second = MagicMock(generation=4)
        controller = MagicMock()
        controller._foreground_work_active.return_value = False
        controller._prefetch_in_flight_generation = None
        controller._neighbor_prefetch_queue = [first, second]

        AppController._start_next_neighbor_prefetch(controller)
        AppController._start_next_neighbor_prefetch(controller)

        controller.preview_load_requested.emit.assert_called_once_with(first)
        self.assertEqual(controller._neighbor_prefetch_queue, [second])

    def test_neighbor_prefetch_protects_the_selected_frame_cache_entry(self):
        self.controller.session.repo.load_file_settings.return_value = None
        self.controller._half_slice_for_asset = MagicMock(return_value=None)
        asset = {"path": "/tmp/neighbor.dng", "hash": "neighbor"}

        task = self.controller._neighbor_prefetch_task(asset, generation=4, protected_file_hashes=("selected",))

        self.assertIsNotNone(task)
        self.assertEqual(task.protected_file_hashes, ("selected",))

    def test_second_neighbor_prefetch_protects_the_first_neighbor(self):
        state = self.controller.state
        state.uploaded_files = [
            {"path": "/tmp/previous.dng", "hash": "previous"},
            {"path": "/tmp/selected.dng", "hash": "selected"},
            {"path": "/tmp/next.dng", "hash": "next"},
        ]
        state.selected_file_idx = 1
        self.controller._prefetch_gen = 4
        self.controller.session.repo.load_file_settings.return_value = None
        self.controller.session.asset_model = MagicMock()
        self.controller.session.asset_model.visible_actual_indices_ordered.return_value = [0, 1, 2]
        self.controller._half_slice_for_asset = MagicMock(return_value=None)
        self.controller._start_next_neighbor_prefetch = MagicMock()

        with patch("negpy.desktop.controller.GPUDevice.get", return_value=SimpleNamespace(is_integrated=False)):
            self.controller._prepare_neighbor_prefetch(4)

        first, second = self.controller._neighbor_prefetch_queue
        self.assertEqual(first.protected_file_hashes, ("selected",))
        self.assertEqual(second.protected_file_hashes, ("selected", "previous"))

    def test_render_waits_for_running_neighbor_prefetch_to_stop(self):
        import numpy as np

        emitted = []
        self.controller.render_requested.connect(emitted.append)
        self.controller.state.preview_raw = np.zeros((4, 4, 3), dtype=np.float32)
        self.controller._prefetch_in_flight_generation = self.controller._prefetch_gen

        self.controller.request_render()

        self.assertEqual(emitted, [])
        self.assertIsNotNone(self.controller._pending_render_task)
        self.controller._on_neighbor_prefetch_finished(self.controller._prefetch_gen, "/neighbor.dng")
        self.assertEqual(len(emitted), 1)
        self.assertTrue(self.controller._is_rendering)

    def test_decode_failure_badges_file_and_success_clears_it(self):
        self.mock_session_manager.asset_model = MagicMock()
        state = self.mock_session_manager.state
        state.uploaded_files = [{"name": "a.dng", "path": "/tmp/a.dng", "hash": "h1"}]

        self.controller._on_preview_load_failed("/tmp/a.dng", "decode boom")
        self.assertEqual(state.uploaded_files[0]["decode_failed"], "decode boom")

        # A later successful load clears the badge even when the frame is no longer
        # the requested one (the handler prefix runs before the early return).
        self.controller._requested_file_path = "/tmp/other.dng"
        self.controller._on_preview_loaded("/tmp/a.dng", None, (0, 0), "", None, "")
        self.assertNotIn("decode_failed", state.uploaded_files[0])

    def test_normalization_finished_uses_hydrated_base_not_active_edit(self):
        """P0-5 Variant A: a frame with no saved edits must take its per-asset hydrated
        config as the write base, never the active frame's crop/heals/local sections."""
        from negpy.domain.models import GeometryConfig, RetouchConfig

        state = self.mock_session_manager.state
        state.uploaded_files = [
            {"name": "a.dng", "path": "/tmp/a.dng", "hash": "hash1"},
            {"name": "b.dng", "path": "/tmp/b.dng", "hash": "hash2"},
        ]
        state.current_file_hash = "hash1"
        state.config = replace(
            WorkspaceConfig(),
            geometry=GeometryConfig(crop_rect=(0.1, 0.1, 0.9, 0.9)),
            retouch=RetouchConfig(manual_dust_spots=[(0.5, 0.5, 3)]),
        )
        self.mock_session_manager.repo.load_file_settings.return_value = None
        self.mock_session_manager.config_for_asset.return_value = WorkspaceConfig()

        with patch.object(self.controller, "_end_batch"), patch.object(self.controller, "request_render"):
            self.controller._on_normalization_finished((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), [])

        saved = {c.args[0]: c.args[1] for c in self.mock_session_manager.repo.save_file_settings.call_args_list}
        self.assertIn("hash2", saved)
        self.assertIsNone(saved["hash2"].geometry.crop_rect)
        self.assertEqual(saved["hash2"].retouch.manual_dust_spots, [])
        # Baseline still broadcast onto the fresh frame.
        self.assertTrue(saved["hash2"].process.use_luma_average)
        self.mock_session_manager.config_for_asset.assert_any_call(state.uploaded_files[1])

    def test_write_edit_sidecars_uses_hydrated_base_not_active_edit(self):
        """A frame with no saved edit must get its own hydrated config written to its
        sidecar, never the active frame's edit (which load_or_promote would later promote
        into that file's persistent DB row)."""
        from negpy.domain.models import GeometryConfig

        state = self.mock_session_manager.state
        state.config = replace(state.config, geometry=GeometryConfig(crop_rect=(0.1, 0.1, 0.9, 0.9)))
        hydrated = WorkspaceConfig()
        self.mock_session_manager.config_for_asset.return_value = hydrated
        frame = {"name": "b.dng", "path": "/tmp/b.dng", "hash": "hash2"}

        with (
            patch("negpy.desktop.controller.load_or_promote", return_value=None),
            patch("negpy.desktop.controller.write_sidecar") as mock_write,
        ):
            written, failed = self.controller._write_edit_sidecars([frame])

        self.assertEqual((written, failed), (1, 0))
        self.mock_session_manager.config_for_asset.assert_called_once_with(frame)
        params = mock_write.call_args.args[1]
        self.assertIs(params, hydrated)
        self.assertIsNone(params.geometry.crop_rect)

    def _wire_repo_store(self) -> dict:
        """Backs the mocked repo's global settings with a real dict, so a roll write
        is readable back through rolls.py's own read/write helpers. Also makes
        update_config actually write state.config, like the real session does
        (session.py's own update_config sets it synchronously) -- needed by anything
        that reads state.config right back after applying an edit, such as
        _lock_roll_card's own divergence check."""
        store: dict = {}
        self.controller.session.repo.get_global_setting.side_effect = lambda key, default=None: store.get(key, default)
        self.controller.session.repo.save_global_setting.side_effect = lambda key, value: store.__setitem__(key, value)
        self.mock_session_manager.asset_model = MagicMock()
        state = self.mock_session_manager.state
        self.mock_session_manager.update_config.side_effect = lambda cfg, **kwargs: setattr(state, "config", cfg)
        return store

    def test_set_roll_default_with_no_active_roll_falls_back_to_a_per_frame_edit(self):
        state = self.mock_session_manager.state
        state.active_roll_id = None
        state.uploaded_files = [{"name": "a.dng", "path": "/a.dng", "hash": "h1"}]
        state.current_file_hash = "h1"

        self.controller.set_roll_default("sensor", hue_trim=2.5)

        cfg, kwargs = self.mock_session_manager.update_config.call_args
        self.assertEqual(cfg[0].process.hue_trim, 2.5)
        self.assertTrue(kwargs["persist"])

    def test_set_roll_default_locks_the_card_the_instant_it_changes(self):
        """Editing is a plain per-frame write now -- the roll's shared defaults are
        never touched here, only by apply_roll_cards_to_roll()."""
        from negpy.services.assets import rolls

        self._wire_repo_store()
        roll_id = rolls.create_virtual_roll(self.controller.session.repo, "Portra", [])
        state = self.mock_session_manager.state
        state.active_roll_id = roll_id
        state.uploaded_files = [{"name": "a.dng", "path": "/a.dng", "hash": "h1"}]
        state.current_file_hash = "h1"

        self.controller.set_roll_default("sensor", hue_trim=2.5)

        self.assertEqual(rolls.roll_defaults(self.controller.session.repo, roll_id), {})
        self.assertEqual(rolls.frame_override_cards(self.controller.session.repo, roll_id, "h1"), {"sensor"})
        cfg, kwargs = self.mock_session_manager.update_config.call_args
        self.assertEqual(cfg[0].process.hue_trim, 2.5)
        self.assertTrue(kwargs["persist"])

    def test_set_roll_default_writes_the_cards_own_config_section(self):
        """The Lens Correction card edits GeometryConfig, not ProcessConfig."""
        from negpy.services.assets import rolls

        self._wire_repo_store()
        roll_id = rolls.create_virtual_roll(self.controller.session.repo, "Portra", [])
        state = self.mock_session_manager.state
        state.active_roll_id = roll_id
        state.uploaded_files = [{"name": "a.dng", "path": "/a.dng", "hash": "h1"}]
        state.current_file_hash = "h1"

        self.controller.set_roll_default("lens", distortion_k1=0.02)

        self.assertEqual(rolls.frame_override_cards(self.controller.session.repo, roll_id, "h1"), {"lens"})
        cfg, _kwargs = self.mock_session_manager.update_config.call_args
        self.assertEqual(cfg[0].geometry.distortion_k1, 0.02)

    def test_setting_the_crop_ratio_locks_the_auto_crop_card(self):
        """Ratio keeps its own entry point, for the manual-crop reshape, so it has to
        lock the card by hand rather than through set_roll_default."""
        from negpy.services.assets import rolls

        self._wire_repo_store()
        roll_id = rolls.create_virtual_roll(self.controller.session.repo, "Portra", [])
        state = self.mock_session_manager.state
        state.active_roll_id = roll_id
        state.uploaded_files = [{"name": "a.dng", "path": "/a.dng", "hash": "h1"}]
        state.current_file_hash = "h1"

        self.controller.set_crop_ratio("5:4")

        self.assertEqual(self.mock_session_manager.update_config.call_args[0][0].geometry.autocrop_ratio, "5:4")
        self.assertEqual(rolls.frame_override_cards(self.controller.session.repo, roll_id, "h1"), {"autocrop"})

    def test_a_settled_auto_crop_edit_drops_the_frames_cached_bounds(self):
        """The crop feeds the meter, so a settled change to what auto crop looks for
        must re-meter; a mid-drag preview must not."""
        from negpy.services.assets import rolls

        self._wire_repo_store()
        roll_id = rolls.create_virtual_roll(self.controller.session.repo, "Portra", [])
        state = self.mock_session_manager.state
        state.active_roll_id = roll_id
        state.uploaded_files = [{"name": "a.dng", "path": "/a.dng", "hash": "h1"}]
        state.current_file_hash = "h1"
        state.config = replace(state.config, process=replace(state.config.process, local_floors=(0.1, 0.1, 0.1)))

        self.controller.set_roll_default("autocrop", persist=False, autocrop_offset=7)
        self.assertEqual(self.mock_session_manager.update_config.call_args[0][0].process.local_floors, (0.1, 0.1, 0.1))

        self.controller.set_roll_default("autocrop", autocrop_offset=9)
        self.assertEqual(self.mock_session_manager.update_config.call_args[0][0].process.local_floors, (0.0, 0.0, 0.0))

    def test_set_roll_default_mid_drag_never_locks(self):
        """persist=False (a slider mid-drag) previews on the active frame only -- the
        lock only follows the settled value."""
        from negpy.services.assets import rolls

        self._wire_repo_store()
        roll_id = rolls.create_virtual_roll(self.controller.session.repo, "Portra", [])
        state = self.mock_session_manager.state
        state.active_roll_id = roll_id
        state.uploaded_files = [{"name": "a.dng", "path": "/a.dng", "hash": "h1"}]
        state.current_file_hash = "h1"

        self.controller.set_roll_default("sensor", persist=False, hue_trim=2.5)

        self.assertEqual(rolls.frame_override_cards(self.controller.session.repo, roll_id, "h1"), set())
        cfg, kwargs = self.mock_session_manager.update_config.call_args
        self.assertEqual(cfg[0].process.hue_trim, 2.5)
        self.assertFalse(kwargs["persist"])

    def test_set_roll_default_does_not_relock_an_already_locked_card(self):
        from negpy.services.assets import rolls

        self._wire_repo_store()
        roll_id = rolls.create_virtual_roll(self.controller.session.repo, "Portra", [])
        rolls.set_frame_override(self.controller.session.repo, roll_id, "h1", "sensor", locked=True)
        state = self.mock_session_manager.state
        state.active_roll_id = roll_id
        state.uploaded_files = [{"name": "a.dng", "path": "/a.dng", "hash": "h1"}]
        state.current_file_hash = "h1"

        self.controller.set_roll_default("sensor", hue_trim=2.5)

        self.assertEqual(rolls.frame_override_cards(self.controller.session.repo, roll_id, "h1"), {"sensor"})
        cfg, kwargs = self.mock_session_manager.update_config.call_args
        self.assertEqual(cfg[0].process.hue_trim, 2.5)
        self.assertTrue(kwargs["persist"])

    def test_set_roll_default_unlocks_when_edited_back_to_the_rolls_own_value(self):
        """Editing a value away and then back to what the roll already says is not a
        divergence -- the card must not stay marked This Frame Only just because it
        was touched in between. "film" (2 fields) rather than "sensor" (9): every
        field in the card needs its own roll default before a match is possible, and
        this keeps the fixture to exactly the fields under test."""
        from negpy.features.process.models import ProcessMode
        from negpy.services.assets import rolls

        self._wire_repo_store()
        roll_id = rolls.create_virtual_roll(self.controller.session.repo, "Portra", [])
        rolls.set_roll_defaults(self.controller.session.repo, roll_id, process_mode=ProcessMode.C41, positive_source=False)
        rolls.set_frame_override(self.controller.session.repo, roll_id, "h1", "film", locked=True)
        state = self.mock_session_manager.state
        state.active_roll_id = roll_id
        state.uploaded_files = [{"name": "a.dng", "path": "/a.dng", "hash": "h1"}]
        state.current_file_hash = "h1"

        self.controller.set_roll_default("film", process_mode=ProcessMode.C41, positive_source=False)

        self.assertEqual(rolls.frame_override_cards(self.controller.session.repo, roll_id, "h1"), set())

    def test_set_roll_default_stays_locked_while_still_diverged(self):
        from negpy.services.assets import rolls

        self._wire_repo_store()
        roll_id = rolls.create_virtual_roll(self.controller.session.repo, "Portra", [])
        rolls.set_roll_defaults(self.controller.session.repo, roll_id, hue_trim=1.0)
        state = self.mock_session_manager.state
        state.active_roll_id = roll_id
        state.uploaded_files = [{"name": "a.dng", "path": "/a.dng", "hash": "h1"}]
        state.current_file_hash = "h1"

        self.controller.set_roll_default("sensor", hue_trim=2.5)

        self.assertEqual(rolls.frame_override_cards(self.controller.session.repo, roll_id, "h1"), {"sensor"})

    def test_set_process_mode_unlocks_when_switched_back_to_the_rolls_own_mode(self):
        from negpy.features.process.models import ProcessMode
        from negpy.services.assets import rolls

        self._wire_repo_store()
        roll_id = rolls.create_virtual_roll(self.controller.session.repo, "Portra", [])
        rolls.set_roll_defaults(self.controller.session.repo, roll_id, process_mode=ProcessMode.C41, positive_source=False)
        rolls.set_frame_override(self.controller.session.repo, roll_id, "h2", "film", locked=True)
        state = self.mock_session_manager.state
        state.active_roll_id = roll_id
        state.current_file_hash = "h2"

        self.controller.set_process_mode(ProcessMode.C41)

        self.assertEqual(rolls.frame_override_cards(self.controller.session.repo, roll_id, "h2"), set())

    def test_set_positive_source_unlocks_when_switched_back_to_the_rolls_own_value(self):
        from negpy.features.process.models import ProcessMode
        from negpy.services.assets import rolls

        self._wire_repo_store()
        roll_id = rolls.create_virtual_roll(self.controller.session.repo, "Portra", [])
        rolls.set_roll_defaults(self.controller.session.repo, roll_id, process_mode=ProcessMode.E6, positive_source=True)
        rolls.set_frame_override(self.controller.session.repo, roll_id, "h2", "film", locked=True)
        state = self.mock_session_manager.state
        state.active_roll_id = roll_id
        state.current_file_hash = "h2"
        state.config = _slide_config(self.mock_session_manager.state.config)

        self.controller.set_positive_source(True)

        self.assertEqual(rolls.frame_override_cards(self.controller.session.repo, roll_id, "h2"), set())

    def test_diverged_roll_cards_lists_locked_cards_in_a_fixed_order(self):
        from negpy.services.assets import rolls

        self._wire_repo_store()
        roll_id = rolls.create_virtual_roll(self.controller.session.repo, "Portra", [])
        rolls.set_frame_override(self.controller.session.repo, roll_id, "h1", "process", locked=True)
        rolls.set_frame_override(self.controller.session.repo, roll_id, "h1", "sensor", locked=True)
        state = self.mock_session_manager.state
        state.active_roll_id = roll_id
        state.current_file_hash = "h1"

        self.assertEqual(self.controller.diverged_roll_cards(), ["sensor", "process"])

    def test_diverged_roll_cards_empty_without_an_active_roll(self):
        self._wire_repo_store()
        self.mock_session_manager.state.active_roll_id = None
        self.assertEqual(self.controller.diverged_roll_cards(), [])

    def test_apply_roll_card_pushes_only_that_card(self):
        """The card's own Roll button, not the Roll tab's Apply: another diverged card
        stays marked."""
        from negpy.services.assets import rolls

        self._wire_repo_store()
        roll_id = rolls.create_virtual_roll(self.controller.session.repo, "Portra", [])
        for card in ("sensor", "autocrop"):
            rolls.set_frame_override(self.controller.session.repo, roll_id, "h1", card, locked=True)
        state = self.mock_session_manager.state
        state.active_roll_id = roll_id
        state.uploaded_files = [{"name": "a.dng", "path": "/a.dng", "hash": "h1"}]
        state.current_file_hash = "h1"
        state.config = replace(state.config, process=replace(state.config.process, hue_trim=2.5))

        touched = self.controller.apply_roll_card("sensor")

        self.assertEqual(touched, 1)
        self.assertEqual(rolls.roll_defaults(self.controller.session.repo, roll_id)["hue_trim"], 2.5)
        self.assertEqual(rolls.frame_override_cards(self.controller.session.repo, roll_id, "h1"), {"autocrop"})

    def test_apply_roll_card_refreshes_the_thumbnails_of_frames_that_follow_the_roll(self):
        """A frame locked on the pushed card keeps its own value, so it is neither
        flagged stale nor re-rendered."""
        from negpy.services.assets import rolls

        self._wire_repo_store()
        repo = self.controller.session.repo
        roll_id = rolls.create_virtual_roll(repo, "Portra", [])
        rolls.set_frame_override(repo, roll_id, "h1", "sensor", locked=True)
        rolls.set_frame_override(repo, roll_id, "h3", "sensor", locked=True)
        rolls.set_frame_override(repo, roll_id, "h4", "autocrop", locked=True)
        state = self.mock_session_manager.state
        state.active_roll_id = roll_id
        state.uploaded_files = [{"name": f"{h}.dng", "path": f"/{h}.dng", "hash": h} for h in ("h1", "h2", "h3", "h4")]
        state.current_file_hash = "h1"
        state.stale_thumbnails = set()
        state.config = replace(state.config, process=replace(state.config.process, hue_trim=2.5))

        self.controller.apply_roll_card("sensor")

        self.mock_session_manager.frames_edited_offscreen.emit.assert_called_once_with(["h2", "h4"])
        self.assertEqual(state.stale_thumbnails, {asset_thumbnail_key(f) for f in state.uploaded_files if f["hash"] in ("h2", "h4")})

    def test_apply_roll_card_on_a_metadata_card_leaves_the_thumbnails_alone(self):
        from negpy.services.assets import rolls

        self._wire_repo_store()
        repo = self.controller.session.repo
        roll_id = rolls.create_virtual_roll(repo, "Portra", [])
        rolls.set_frame_override(repo, roll_id, "h1", "metadata_gear", locked=True)
        state = self.mock_session_manager.state
        state.active_roll_id = roll_id
        state.uploaded_files = [{"name": f"{h}.dng", "path": f"/{h}.dng", "hash": h} for h in ("h1", "h2")]
        state.current_file_hash = "h1"
        state.stale_thumbnails = set()

        self.assertEqual(self.controller.apply_roll_card("metadata_gear"), 1)
        self.mock_session_manager.frames_edited_offscreen.emit.assert_not_called()
        self.assertEqual(state.stale_thumbnails, set())

    def test_apply_roll_card_on_a_card_that_follows_the_roll_is_a_noop(self):
        from negpy.services.assets import rolls

        self._wire_repo_store()
        roll_id = rolls.create_virtual_roll(self.controller.session.repo, "Portra", [])
        state = self.mock_session_manager.state
        state.active_roll_id = roll_id
        state.uploaded_files = [{"name": "a.dng", "path": "/a.dng", "hash": "h1"}]
        state.current_file_hash = "h1"

        self.assertEqual(self.controller.apply_roll_card("sensor"), 0)
        self.assertEqual(rolls.roll_defaults(self.controller.session.repo, roll_id), {})

    def test_a_frame_section_reads_frame_until_it_is_pushed(self):
        from negpy.services.assets import rolls

        self._wire_repo_store()
        roll_id = rolls.create_virtual_roll(self.controller.session.repo, "Portra", [])
        state = self.mock_session_manager.state
        state.active_roll_id = roll_id
        state.config = replace(state.config, exposure=replace(state.config.exposure, dye_separation=0.4))

        self.assertEqual(self.controller.frame_section_scope("tone"), "frame")

        self.controller.record_roll_apply(rows_for_fields(("dye_separation",)))
        self.assertEqual(self.controller.frame_section_scope("tone"), "roll")

    def test_a_frame_section_drops_back_to_frame_once_edited_again(self):
        """Nothing clears the record: the frame simply stops matching it."""
        from negpy.services.assets import rolls

        self._wire_repo_store()
        roll_id = rolls.create_virtual_roll(self.controller.session.repo, "Portra", [])
        state = self.mock_session_manager.state
        state.active_roll_id = roll_id
        state.config = replace(state.config, exposure=replace(state.config.exposure, dye_separation=0.4))
        self.controller.record_roll_apply(rows_for_fields(("dye_separation",)))

        state.config = replace(state.config, exposure=replace(state.config.exposure, dye_separation=0.9))

        self.assertEqual(self.controller.frame_section_scope("tone"), "frame")

    def test_frame_section_scopes_answers_every_card_off_one_roll_read(self):
        from negpy.services.assets import rolls

        self._wire_repo_store()
        repo = self.controller.session.repo
        roll_id = rolls.create_virtual_roll(repo, "Portra", [])
        state = self.mock_session_manager.state
        state.active_roll_id = roll_id
        state.config = replace(state.config, exposure=replace(state.config.exposure, dye_separation=0.4))
        self.controller.record_roll_apply(rows_for_fields(("dye_separation",)))
        repo.get_global_setting.reset_mock()

        scopes = self.controller.frame_section_scopes(("tone", "finish"))

        self.assertEqual(scopes, {"tone": "roll", "finish": "frame"})
        self.assertEqual(repo.get_global_setting.call_count, 1)

    def test_a_frame_section_reads_frame_with_no_roll_open(self):
        state = self.mock_session_manager.state
        state.active_roll_id = None
        self.assertEqual(self.controller.frame_section_scope("tone"), "frame")

    def test_recording_a_push_with_no_roll_open_is_a_noop(self):
        state = self.mock_session_manager.state
        state.active_roll_id = None
        self.controller.record_roll_apply(rows_for_fields(("dye_separation",)))
        self.assertEqual(self.controller.frame_section_scope("tone"), "frame")

    def _roll_with_frame(self) -> str:
        from negpy.services.assets import rolls

        self._wire_repo_store()
        roll_id = rolls.create_virtual_roll(self.controller.session.repo, "Portra", [])
        state = self.mock_session_manager.state
        state.active_roll_id = roll_id
        state.uploaded_files = [{"name": "a.dng", "path": "/a.dng", "hash": "h1"}]
        state.current_file_hash = "h1"
        state.selected_file_idx = 0
        return roll_id

    def test_a_whole_roll_apply_is_recorded_per_card(self):
        from negpy.desktop.settings_catalog import COLOR_FIELDS, GEOMETRY_FIELDS
        from negpy.services.assets import rolls

        roll_id = self._roll_with_frame()

        self.controller.record_roll_apply(rows_for_fields(GEOMETRY_FIELDS) + rows_for_fields(COLOR_FIELDS))

        pushes = rolls.roll_for_id(self.controller.session.repo, roll_id)["section_pushes"]
        self.assertEqual(set(pushes), {"geometry", "color"})

    def test_reset_to_roll_restores_only_what_the_roll_apply_carried(self):
        self._roll_with_frame()
        state = self.mock_session_manager.state
        state.config = replace(state.config, exposure=replace(state.config.exposure, density=1.2, dye_separation=0.4))
        self.controller.record_roll_apply(rows_for_fields(("density",)))
        state.config = replace(state.config, exposure=replace(state.config.exposure, density=0.5, dye_separation=0.9))
        self.assertEqual(self.controller.roll_revert_cards(("tone",)), {"tone"})

        self.assertEqual(self.controller.revert_to_roll(("tone",)), 1)

        self.assertEqual(state.config.exposure.density, 1.2)
        self.assertEqual(state.config.exposure.dye_separation, 0.9)
        self.assertEqual(self.controller.frame_section_scope("tone"), "roll")
        self.assertEqual(self.controller.roll_revert_cards(("tone",)), set())

    def test_a_pushed_crop_read_back_as_a_list_still_matches(self):
        """The roll store is JSON: a pushed crop comes back as a list, the frame holds a tuple."""
        from negpy.services.assets import rolls

        roll_id = self._roll_with_frame()
        state = self.mock_session_manager.state
        rolls.set_section_push(self.controller.session.repo, roll_id, "geometry", {"crop_rect": [0.1, 0.1, 0.9, 0.9]})
        state.config = replace(state.config, geometry=replace(state.config.geometry, crop_rect=(0.1, 0.1, 0.9, 0.9)))
        self.assertEqual(self.controller.frame_section_scope("geometry"), "roll")

        state.config = replace(state.config, geometry=replace(state.config.geometry, crop_rect=(0.2, 0.2, 0.8, 0.8)))
        self.controller.revert_to_roll(("geometry",))

        self.assertEqual(state.config.geometry.crop_rect, (0.1, 0.1, 0.9, 0.9))
        self.assertEqual(self.controller.frame_section_scope("geometry"), "roll")

    def test_reset_to_roll_unlocks_a_roll_card_and_takes_the_rolls_value(self):
        from negpy.services.assets import rolls

        roll_id = self._roll_with_frame()
        repo = self.controller.session.repo
        state = self.mock_session_manager.state
        rolls.set_roll_defaults(repo, roll_id, analysis_buffer=0.2)
        state.config = replace(state.config, process=replace(state.config.process, analysis_buffer=0.05))
        rolls.set_frame_override(repo, roll_id, "h1", "process", True)
        self.assertEqual(self.controller.roll_revert_cards(("process",)), {"process"})

        self.controller.revert_to_roll(("process",))

        self.assertEqual(state.config.process.analysis_buffer, 0.2)
        self.assertEqual(rolls.frame_override_cards(repo, roll_id, "h1"), set())
        self.assertEqual(self.controller.roll_revert_cards(("process",)), set())

    def test_reset_to_roll_on_film_mode_carries_the_modes_cast_removal(self):
        from negpy.features.process.models import ProcessMode, with_process_mode
        from negpy.services.assets import rolls

        roll_id = self._roll_with_frame()
        repo = self.controller.session.repo
        state = self.mock_session_manager.state
        rolls.set_roll_defaults(repo, roll_id, process_mode=ProcessMode.E6)
        rolls.set_frame_override(repo, roll_id, "h1", "film", True)
        before = state.config.exposure.cast_removal_strength
        expected = with_process_mode(state.config, ProcessMode.E6)

        self.controller.revert_to_roll(("film",))

        self.assertEqual(state.config.process.process_mode, ProcessMode.E6)
        self.assertEqual(state.config.exposure.cast_removal_strength, expected.exposure.cast_removal_strength)
        self.assertNotEqual(state.config.exposure.cast_removal_strength, before)

    def test_a_calibration_matrix_equal_to_the_rolls_does_not_lock(self):
        """A matrix read back from the roll store is nested lists; it still matches."""
        from negpy.services.assets import rolls

        roll_id = self._roll_with_frame()
        state = self.mock_session_manager.state
        matrix = ((1.0, 0.1, 0.0), (0.0, 1.0, 0.1), (0.1, 0.0, 1.0))
        state.config = replace(state.config, process=replace(state.config.process, crosstalk_matrix=matrix))
        stored = {name: getattr(state.config.process, name) for name in rolls.card_fields("sensor")}
        rolls.set_roll_defaults(self.controller.session.repo, roll_id, **{**stored, "crosstalk_matrix": [list(row) for row in matrix]})

        self.controller._lock_roll_card("sensor")

        self.assertNotIn("sensor", self.controller.locked_roll_cards())

    def test_resetting_several_cards_to_the_roll_is_one_edit(self):
        from negpy.services.assets import rolls

        roll_id = self._roll_with_frame()
        repo = self.controller.session.repo
        state = self.mock_session_manager.state
        rolls.set_roll_defaults(repo, roll_id, analysis_buffer=0.2)
        rolls.set_frame_override(repo, roll_id, "h1", "process", True)
        state.config = replace(state.config, exposure=replace(state.config.exposure, density=1.2))
        self.controller.record_roll_apply(rows_for_fields(("density",)))
        state.config = replace(state.config, exposure=replace(state.config.exposure, density=0.5))
        self.mock_session_manager.update_config.reset_mock()

        self.assertTrue(self.controller.can_revert_frame_to_roll())
        self.assertEqual(self.controller.revert_frame_to_roll(), 2)

        self.assertEqual(self.mock_session_manager.update_config.call_count, 1)
        self.assertEqual((state.config.process.analysis_buffer, state.config.exposure.density), (0.2, 1.2))
        self.assertFalse(self.controller.can_revert_frame_to_roll())

    def test_nothing_to_reset_to_the_roll(self):
        from negpy.services.assets import rolls

        roll_id = self._roll_with_frame()
        rolls.set_frame_override(self.controller.session.repo, roll_id, "h1", "process", True)
        self.mock_session_manager.update_config.reset_mock()

        # A locked card the roll holds no value for, and a frame card with no record.
        self.assertEqual(self.controller.roll_revert_cards(("process", "tone")), set())
        self.assertEqual(self.controller.revert_to_roll(("process", "tone")), 0)
        self.mock_session_manager.update_config.assert_not_called()

        self.mock_session_manager.state.active_roll_id = None
        self.assertEqual(self.controller.roll_revert_cards(("process", "tone")), set())

    def test_a_metadata_reset_to_the_roll_does_not_render(self):
        from negpy.services.assets import rolls

        roll_id = self._roll_with_frame()
        repo = self.controller.session.repo
        state = self.mock_session_manager.state
        rolls.set_roll_defaults(repo, roll_id, exposure_override="1/125 f/8")
        rolls.set_frame_override(repo, roll_id, "h1", "metadata_exposure", True)

        self.controller.revert_to_roll(("metadata_exposure",))

        self.assertEqual(state.config.metadata.exposure_override, "1/125 f/8")
        self.assertFalse(self.mock_session_manager.update_config.call_args.kwargs["render"])

    def test_locking_a_card_seeds_the_frames_own_row_then_sets_the_flag(self):
        from negpy.services.assets import rolls

        self._wire_repo_store()
        roll_id = rolls.create_virtual_roll(self.controller.session.repo, "Portra", [])
        rolls.set_roll_defaults(self.controller.session.repo, roll_id, hue_trim=2.5)
        state = self.mock_session_manager.state
        state.active_roll_id = roll_id
        state.uploaded_files = [{"name": "a.dng", "path": "/a.dng", "hash": "h1"}]
        state.selected_file_idx = 0
        state.current_file_hash = "h1"
        state.config = rolls.resolve_roll_config(self.controller.session.repo, roll_id, "h1", state.config)

        self.controller.set_roll_card_locked("sensor", locked=True)

        cfg, kwargs = self.mock_session_manager.update_config.call_args
        self.assertEqual(cfg[0].process.hue_trim, 2.5)
        self.assertTrue(kwargs["persist"])
        self.assertFalse(kwargs["render"])
        self.assertEqual(rolls.frame_override_cards(self.controller.session.repo, roll_id, "h1"), {"sensor"})

    def test_unlocking_a_card_reverts_to_the_rolls_current_value(self):
        from negpy.services.assets import rolls

        self._wire_repo_store()
        roll_id = rolls.create_virtual_roll(self.controller.session.repo, "Portra", [])
        rolls.set_roll_defaults(self.controller.session.repo, roll_id, hue_trim=9.0)
        rolls.set_frame_override(self.controller.session.repo, roll_id, "h1", "sensor", locked=True)
        state = self.mock_session_manager.state
        state.active_roll_id = roll_id
        asset = {"name": "a.dng", "path": "/a.dng", "hash": "h1"}
        state.uploaded_files = [asset]
        state.selected_file_idx = 0
        state.current_file_hash = "h1"
        self.mock_session_manager.config_for_asset.return_value = replace(state.config, process=replace(state.config.process, hue_trim=9.0))

        self.controller.set_roll_card_locked("sensor", locked=False)

        self.assertEqual(rolls.frame_override_cards(self.controller.session.repo, roll_id, "h1"), set())
        self.mock_session_manager.config_for_asset.assert_called_once_with(asset)
        cfg, kwargs = self.mock_session_manager.update_config.call_args
        self.assertEqual(cfg[0].process.hue_trim, 9.0)
        self.assertFalse(kwargs["persist"])

    def test_roll_lock_state_is_keyed_on_the_unforked_hash(self):
        """A locked card is about this physical frame's relationship to the roll, not
        its current edit identity -- it must read the same locked whether the frame is
        showing its forked edit or the shared one."""
        from negpy.services.assets import rolls

        self._wire_repo_store()
        roll_id = rolls.create_virtual_roll(self.controller.session.repo, "Portra", [])
        state = self.mock_session_manager.state
        state.active_roll_id = roll_id
        state.current_file_hash = rolls.roll_edit_hash("h1", roll_id)

        self.controller.set_roll_card_locked("sensor", locked=True)

        self.assertEqual(rolls.frame_override_cards(self.controller.session.repo, roll_id, "h1"), {"sensor"})
        self.assertTrue(self.controller.roll_card_locked("sensor"))

    def test_thumbnail_miss_does_not_mark_source_unreadable(self):
        from PIL import Image

        from negpy.services.assets.thumbnails import asset_thumbnail_key

        self.mock_session_manager.asset_model = MagicMock()
        state = self.mock_session_manager.state
        state.uploaded_files = [
            {"name": "bad.dng", "path": "/tmp/bad.dng", "hash": "h1"},
            {"name": "good.dng", "path": "/tmp/good.dng", "hash": "h2"},
        ]
        keys = [asset_thumbnail_key(f) for f in state.uploaded_files]
        self.controller._apply_thumbnails({keys[1]: Image.new("RGB", (4, 4))})

        self.assertNotIn("decode_failed", state.uploaded_files[0])
        self.assertNotIn("decode_failed", state.uploaded_files[1])

    def test_a_thumbnail_that_cannot_decode_does_not_badge_its_source(self):
        """A PIL image decodes lazily, so a truncated cache entry raises on the UI
        thread, inside a Qt slot, where an exception ends the process."""
        from PIL import Image

        from negpy.services.assets.thumbnails import asset_thumbnail_key

        self.mock_session_manager.asset_model = MagicMock()
        state = self.mock_session_manager.state
        state.uploaded_files = [{"name": "cut.dng", "path": "/tmp/cut.dng", "hash": "h1"}]
        key = asset_thumbnail_key(state.uploaded_files[0])
        broken = MagicMock(spec=Image.Image)
        broken.convert.side_effect = OSError("broken data stream when reading image file")
        self.controller._apply_thumbnails({key: broken})

        self.assertNotIn(key, state.thumbnails)
        self.assertNotIn("decode_failed", state.uploaded_files[0])

    def test_thumbnail_updates_do_not_badge_other_frames(self):
        from PIL import Image

        self.mock_session_manager.asset_model = MagicMock()
        state = self.mock_session_manager.state
        state.uploaded_files = [
            {"name": "a.dng", "path": "/tmp/a.dng", "hash": "h1"},
            {"name": "b.dng", "path": "/tmp/b.dng", "hash": "h2"},
        ]
        img = Image.new("RGB", (4, 4))
        self.controller._apply_thumbnails({"h1": img, "h2": img})
        self.assertNotIn("decode_failed", state.uploaded_files[0])

        # A second, narrower result must not badge frames absent from it.
        self.controller._apply_thumbnails({"h1": img})
        self.assertNotIn("decode_failed", state.uploaded_files[1])

    def test_batch_thumbnail_does_not_clobber_rendered(self):
        """A frame that already rendered on the canvas keeps its (correct, inverted)
        thumbnail even if the slower batch decode finishes afterward."""
        from PIL import Image

        self.mock_session_manager.asset_model = MagicMock()
        state = self.mock_session_manager.state
        state.uploaded_files = [{"name": "a.dng", "path": "/tmp/a.dng", "hash": "h1"}]
        rendered = Image.new("RGB", (4, 4), (255, 0, 0))
        placeholder = Image.new("RGB", (4, 4), (0, 255, 0))
        self.controller._on_rendered_thumbnail({"h1": rendered})
        self.controller._apply_thumbnails({"h1": placeholder})

        self.assertEqual(state.thumbnails["h1"].pixmap(4, 4).toImage().pixelColor(0, 0).red(), 255)

    def test_rendered_thumbnail_keys_by_asset_identity_not_filename(self):
        """Thumbnails are keyed by asset identity, so two same-named files in different
        folders can't overwrite each other, and an RGB-scan triplet's merged thumbnail
        lands under its own key rather than the red exposure's (issue #575)."""
        import numpy as np
        from negpy.features.rgbscan.models import RgbScanConfig
        from dataclasses import replace as dc_replace

        state = self.mock_session_manager.state
        state.uploaded_files = [
            {
                "name": "_DSC1316 (RGB)",
                "path": "/tmp/_DSC1316.NEF",
                "hash": "h1",
                "green_path": "/tmp/_DSC1317.NEF",
                "blue_path": "/tmp/_DSC1318.NEF",
            }
        ]
        state.selected_file_idx = 0
        state.current_file_path = "/tmp/_DSC1316.NEF"
        state.current_file_hash = "h1"
        state.config = dc_replace(
            state.config, rgbscan=RgbScanConfig(enabled=True, green_path="/tmp/_DSC1317.NEF", blue_path="/tmp/_DSC1318.NEF")
        )
        state.last_metrics = {"base_positive": np.zeros((2, 2, 3), dtype=np.float32), "source_hash": "h1"}

        captured = {}
        self.controller.thumbnail_update_requested.connect(lambda task: captured.setdefault("task", task))
        self.controller._update_thumbnail_from_state(persist=False)

        from negpy.services.assets.thumbnails import thumbnail_cache_key

        self.assertEqual(captured["task"].file_hash, thumbnail_cache_key("h1", is_triplet=True))
        self.assertNotEqual(captured["task"].file_hash, thumbnail_cache_key("h1", is_triplet=False))

    def test_capture_worker_cancelled_is_forwarded(self):
        cancelled = MagicMock()
        self.controller.capture_cancelled.connect(cancelled)

        self.controller.capture_worker.cancelled.emit()

        cancelled.assert_called_once_with()

    def test_scan_worker_cancelled_is_forwarded(self):
        cancelled = MagicMock()
        self.controller.scan_cancelled.connect(cancelled)

        self.controller.scan_worker.cancelled.emit()

        cancelled.assert_called_once_with()

    def test_start_scan_prepares_worker_before_emitting_signals(self):
        from negpy.desktop.workers.scan_worker import ScanRequest

        events: list[object] = []
        request = ScanRequest(
            device_id="coolscan3:test",
            params=ScanParams(dpi=4_000, depth=16, capture_ir=False),
            output_folder="/tmp",
            filename_pattern='scan-{{ "%03d" % seq }}',
            output_format="TIFF",
        )
        controller = SimpleNamespace(
            scan_worker=SimpleNamespace(prepare_scan=lambda: events.append("prepare")),
            scan_started=SimpleNamespace(emit=lambda: events.append("started")),
            scan_requested=SimpleNamespace(emit=lambda value: events.append(("request", value))),
        )

        AppController.start_scan(controller, request)

        self.assertEqual(events, ["prepare", "started", ("request", request)])

    def test_start_roll_preview_prepares_worker_and_emits_preview_only(self):
        from negpy.desktop.workers.scan_worker import RollPreviewRequest

        events: list[object] = []
        request = RollPreviewRequest(device=SimpleNamespace(id="coolscan3:test"), slots=(1, 2), dpi=500)
        controller = SimpleNamespace(
            scan_worker=SimpleNamespace(prepare_scan=lambda: events.append("prepare")),
            scan_started=SimpleNamespace(emit=lambda: events.append("started")),
            scan_roll_preview_requested=SimpleNamespace(emit=lambda value: events.append(("preview", value))),
        )

        AppController.start_roll_preview(controller, request)

        # No "started": a preview must not flip the main scan UI into scanning state.
        self.assertEqual(events, ["prepare", ("preview", request)])

    def test_thumbnail_refreshes_on_config_changed_settle(self):
        """Filmstrip thumbnail is re-captured on every settled render whose config
        differs from the last capture (covers in-place edits and reset), but not on a
        repeat settle with the same config object."""
        from negpy.domain.models import WorkspaceConfig

        self.controller._update_thumbnail_from_state = MagicMock()
        self.controller._pending_render_task = None
        self.controller._thumb_config = None

        cfg = WorkspaceConfig()
        self.controller.state.config = cfg
        self.controller._on_render_finished(None, {})
        self.assertEqual(self.controller._update_thumbnail_from_state.call_count, 1)
        self.controller._update_thumbnail_from_state.assert_called_with(persist=False)

        # Same config object -> no redundant refresh.
        self.controller._on_render_finished(None, {})
        self.assertEqual(self.controller._update_thumbnail_from_state.call_count, 1)

        self.controller.state.config = replace(cfg, exposure=replace(cfg.exposure, density=1.0))
        self.controller._on_render_finished(None, {})
        self.assertEqual(self.controller._update_thumbnail_from_state.call_count, 2)

    def test_thumbnail_not_refreshed_while_pending_or_ephemeral(self):
        """Don't capture a premature frame while a newer render is queued, nor the
        low-quality splash (ephemeral) render."""
        import numpy as np

        from negpy.domain.models import WorkspaceConfig
        from negpy.desktop.workers.render import RenderTask

        self.controller._update_thumbnail_from_state = MagicMock()
        self.controller._thumb_config = None
        self.controller.state.config = WorkspaceConfig()

        self.controller._pending_render_task = RenderTask(
            buffer=np.zeros((1, 1, 3), np.float32),
            config=WorkspaceConfig(),
            source_hash="x",
            preview_size=1.0,
        )
        self.controller._on_render_finished(None, {})
        self.controller._update_thumbnail_from_state.assert_not_called()

        self.controller._pending_render_task = None
        self.controller._on_render_finished(None, {"ephemeral": True})
        self.controller._update_thumbnail_from_state.assert_not_called()

    def test_render_of_a_frame_the_user_left_is_dropped(self):
        """Switching files mid-render leaves that render in flight. When it lands it must
        not repaint the canvas nor overwrite the new frame's metrics."""
        import numpy as np

        state = self.controller.state
        state.current_file_path = "/tmp/B.NEF"
        state.current_file_hash = "hB"
        current = np.full((2, 2, 3), 0.5, dtype=np.float32)
        state.last_metrics = {"base_positive": current, "source_hash": "hB", "log_bounds": (0.1, 0.9)}

        repaints = []
        self.controller.image_updated.connect(lambda: repaints.append(1))
        self.controller._update_thumbnail_from_state = MagicMock()

        stale = {"base_positive": np.zeros((2, 2, 3), dtype=np.float32), "source_hash": "hA", "log_bounds": (0.4, 0.4)}
        self.controller._on_render_finished(None, stale)
        self.controller._on_metrics_updated(stale)

        self.assertEqual(repaints, [])
        self.assertIs(state.last_metrics["base_positive"], current)
        self.assertEqual(state.last_metrics["log_bounds"], (0.1, 0.9))
        self.controller._update_thumbnail_from_state.assert_not_called()

        # The queue still drains — a stale frame must not wedge the render loop.
        self.assertFalse(self.controller._is_rendering)

    def test_render_of_the_current_frame_still_lands(self):
        """The guard keys on the frame, not on staleness in general: the selected frame's
        render — and an unhashed preview's, which carries the 'preview' placeholder —
        repaint as before."""
        import numpy as np

        state = self.controller.state
        state.current_file_path = "/tmp/B.NEF"
        state.current_file_hash = "hB"

        repaints = []
        self.controller.image_updated.connect(lambda: repaints.append(1))
        self.controller._update_thumbnail_from_state = MagicMock()

        self.controller._on_render_finished(None, {"base_positive": np.zeros((2, 2, 3), np.float32), "source_hash": "hB"})
        self.assertEqual(len(repaints), 1)

        state.current_file_hash = None
        self.controller._on_render_finished(None, {"base_positive": np.zeros((2, 2, 3), np.float32), "source_hash": "preview"})
        self.assertEqual(len(repaints), 2)

        # A render with no identity at all (older callers) is not treated as stale.
        self.controller._on_render_finished(None, {})
        self.assertEqual(len(repaints), 3)

    def test_navigate_back_paint_carries_its_own_frames_identity(self):
        """The memo fast path paints the incoming frame's last render. Leaving the
        outgoing frame's hash beside those pixels files them under it on the next
        thumbnail refresh."""
        import numpy as np

        state = self.controller.state
        state.uploaded_files = [
            {"name": "a.dng", "path": "/tmp/a.dng", "hash": "hA"},
            {"name": "b.dng", "path": "/tmp/b.dng", "hash": "hB"},
        ]
        state.current_file_path = "/tmp/a.dng"
        state.current_file_hash = "hA"
        state.last_metrics = {"source_hash": "hA"}

        cached = np.zeros((2, 2, 3), dtype=np.float32)
        self.controller._render_memo.store("hB", self.controller._render_memo_key(), {"base_positive": cached, "content_rect": None})

        self.controller.load_file("/tmp/b.dng")

        self.assertIs(state.last_metrics["base_positive"], cached)
        self.assertEqual(state.last_metrics["source_hash"], "hB")

    def test_pending_render_is_dispatched_after_a_dropped_frame(self):
        """A render queued behind the stale one must still be started."""
        import numpy as np

        from negpy.desktop.workers.render import RenderTask

        state = self.controller.state
        state.current_file_hash = "hB"
        queued = RenderTask(
            buffer=np.zeros((1, 1, 3), np.float32),
            config=WorkspaceConfig(),
            source_hash="hB",
            preview_size=1.0,
        )
        self.controller._pending_render_task = queued

        dispatched = []
        self.controller.render_requested.connect(dispatched.append)
        self.controller._on_render_finished(None, {"source_hash": "hA"})

        self.assertEqual(dispatched, [queued])
        self.assertIsNone(self.controller._pending_render_task)
        self.assertTrue(self.controller._is_rendering)

    def test_proof_active_gated_by_toggle(self):
        """proof_active() is False unless the soft-proof toggle is on, even with an
        export color space set (which always resolves an output profile)."""
        self.controller.state.soft_proof_enabled = False
        self.assertFalse(self.controller.proof_active())
        self.controller.state.soft_proof_enabled = True
        # An export color space resolves an effective output profile → proof active.
        self.assertTrue(self.controller.proof_active())

    def test_effective_input_icc(self):
        """Explicit Input ICC wins; Narrowband Scan supplies the bundled RGBScan
        profile when none is set; None when both are off."""
        state = self.controller.state
        self.assertIsNone(self.controller.effective_input_icc())

        state.config = replace(state.config, process=replace(state.config.process, narrowband_scan=True))
        path = self.controller.effective_input_icc()
        assert path is not None
        self.assertTrue(path.endswith(os.path.join("icc", "RGBScan.icc")))
        self.assertTrue(os.path.exists(path))

        state.icc_input_path = "/custom.icc"
        self.assertEqual(self.controller.effective_input_icc(), "/custom.icc")

    def test_narrowband_profile_suppressed_by_any_transparency(self):
        """The bundled profile describes narrowband capture of *negative* dyes, so it is
        refused for a slide whatever Positive says. Narrowband is a sticky setting, so
        this must hold without the user touching it."""
        from negpy.features.process.models import ProcessMode

        state = self.controller.state
        state.config = replace(state.config, process=replace(state.config.process, narrowband_scan=True))
        self.assertIsNotNone(self.controller.effective_input_icc())

        for positive in (True, False):
            state.config = replace(
                state.config,
                process=replace(state.config.process, process_mode=ProcessMode.E6, positive_source=positive),
            )
            self.assertIsNone(self.controller.effective_input_icc(), f"positive_source={positive}")
            # ...and the preview must not claim a proof it no longer applies.
            state.soft_proof_enabled = False
            self.assertFalse(self.controller.proof_active(), f"positive_source={positive}")

        # An explicit Input ICC is a deliberate choice about the source and still wins.
        state.icc_input_path = "/custom.icc"
        self.assertEqual(self.controller.effective_input_icc(), "/custom.icc")

    def test_effective_cam_xyz_stands_in_for_an_active_input_icc(self):
        """An active Input ICC supplies its own primaries rotation, so the camera's own
        must come out as identity — but the decode still needs the white-balance fold
        (#991), which nulling cam_xyz outright would also have dropped."""
        import numpy as np

        from negpy.features.process.capture_color import camera_to_working_matrix

        state = self.controller.state
        matrix = [[0.7, -0.1, -0.07], [-0.56, 1.34, 0.24], [-0.15, 0.22, 0.73]]
        wb = [1.9, 1.0, 1.6]
        state.preview_cam_xyz = matrix
        state.preview_camera_wb = wb

        cam_xyz, camera_wb = self.controller._effective_cam_xyz()
        self.assertEqual(cam_xyz, matrix)
        self.assertEqual(camera_wb, wb)

        state.icc_input_path = "/custom.icc"
        cam_xyz, camera_wb = self.controller._effective_cam_xyz()
        self.assertNotEqual(cam_xyz, matrix)
        self.assertEqual(camera_wb, wb)
        np.testing.assert_allclose(camera_to_working_matrix(cam_xyz, wb), np.diag(np.array(wb) / wb[1]), atol=1e-5)

    def _export_icc_input(self, **process):
        state = self.controller.state
        state.current_file_path = "/tmp/shot.dng"
        state.current_file_hash = "h1"
        state.flat_output = True
        state.config = replace(
            state.config,
            process=replace(state.config.process, **process),
            export=replace(state.config.export, output_mode=ExportPresetOutputMode.SAME_AS_SOURCE, export_path="/tmp"),
        )
        self.controller._run_export_tasks = MagicMock()
        self.controller.request_export()
        self.controller._run_export_tasks.assert_called_once()
        (tasks,), _ = self.controller._run_export_tasks.call_args
        return tasks[0].export_settings.icc_input_path

    def test_export_resolves_the_narrowband_profile_from_the_frames_own_process(self):
        """Regression: request_export resolved the implicit ICC from the pre-flatten view
        rather than the frame's settings, and silently dropped it from flat masters. The
        answer must not depend on which view asked."""
        path = self._export_icc_input(narrowband_scan=True)
        self.assertTrue(
            path is not None and path.endswith(os.path.join("icc", "RGBScan.icc")),
            "flat export dropped the implicit Narrowband profile",
        )

    def test_a_flat_export_of_a_slide_still_refuses_the_narrowband_profile(self):
        """The profile is refused for the dye set, not for the render, so flattening must not
        smuggle it back in."""
        from negpy.features.process.models import ProcessMode

        self.assertIsNone(self._export_icc_input(narrowband_scan=True, process_mode=ProcessMode.E6))

    def test_proof_active_with_narrowband_scan(self):
        """Narrowband Scan forces proofing on even with the soft-proof toggle off."""
        state = self.controller.state
        state.soft_proof_enabled = False
        self.assertFalse(self.controller.proof_active())
        state.config = replace(state.config, process=replace(state.config.process, narrowband_scan=True))
        self.assertTrue(self.controller.proof_active())

    def test_narrowband_profile_hidden_from_dropdown(self):
        from negpy.infrastructure.display.color_mgmt import ColorService

        profiles = ColorService.get_available_profiles()
        self.assertFalse(any(p.endswith("RGBScan.icc") for p in profiles))

    def test_load_file_preserve_zoom(self):
        """Test that load_file with preserve_zoom=True skips resetting zoom."""
        mock_slot = MagicMock()
        self.controller.zoom_requested.connect(mock_slot)

        self.controller.load_file("dummy.dng", preserve_zoom=True)

        mock_slot.assert_not_called()

    def test_file_selected_navigation_resets_zoom_by_default(self):
        """Switching images via session.file_selected resets zoom unless sticky_zoom is on."""
        self.controller.state.sticky_zoom = False
        mock_slot = MagicMock()
        self.controller.zoom_requested.connect(mock_slot)

        self.controller._on_file_selected_load("dummy.dng")

        mock_slot.assert_called_once_with(1.0)

    def test_file_selected_navigation_preserves_zoom_when_sticky(self):
        """With sticky_zoom on, switching images must not reset the zoom level."""
        self.controller.state.sticky_zoom = True
        mock_slot = MagicMock()
        self.controller.zoom_requested.connect(mock_slot)

        self.controller._on_file_selected_load("dummy.dng")

        mock_slot.assert_not_called()

    def test_toggle_hq_preview_preserves_zoom(self):
        """Test that toggling HQ mode persists via session and preserves zoom."""
        self.controller.state.current_file_path = "dummy.dng"

        mock_slot = MagicMock()
        self.controller.zoom_requested.connect(mock_slot)

        self.controller.toggle_hq_preview()

        # Persistence delegated to session
        self.mock_session_manager.set_hq_preview.assert_called_once_with(True)

        # Zoom should NOT be reset
        mock_slot.assert_not_called()

    def test_preview_loaded_updates_state_and_emits_signal(self):
        """Successful preview loads should publish dimensions before rendering starts."""
        mock_slot = MagicMock()
        self.controller.preview_loaded.connect(mock_slot)
        self.controller.request_render = MagicMock()
        self.controller._requested_file_path = "dummy.dng"

        raw = object()
        dims = (1234, 5678)

        self.controller._on_preview_loaded("dummy.dng", raw, dims, "", None, "")

        self.assertIs(self.controller.state.preview_raw, raw)
        self.assertEqual(self.controller.state.original_res, dims)
        self.assertEqual(self.controller.state.current_file_path, "dummy.dng")
        self.assertFalse(self.controller.state.has_ir)
        self.assertIsNone(self.controller.state.preview_ir)
        mock_slot.assert_called_once_with()
        self.controller.request_render.assert_called_once_with()

    def test_stale_preview_decode_is_dropped(self):
        """A decode that lands after the user switched files must not be applied —
        accepting it pairs the old buffer with the new file's hash and poisons the
        per-source analysis cache (green/red cast on the new file)."""
        self.controller.request_render = MagicMock()
        self.controller._requested_file_path = "current.dng"
        self.controller.state.preview_raw = None

        self.controller._on_preview_loaded("stale.dng", object(), (10, 20), "", None, "")

        self.assertIsNone(self.controller.state.preview_raw)
        self.controller.request_render.assert_not_called()

    def test_stale_lens_correction_decode_is_dropped(self):
        self.controller.request_render = MagicMock()
        self.controller._requested_file_path = "current.arw"
        state = self.controller.state
        state.config = replace(state.config, geometry=replace(state.config.geometry, lens_ca_from_metadata=True))
        current_raw = object()
        state.preview_raw = current_raw

        self.controller._on_preview_loaded("current.arw", object(), (10, 20), "", None, "", (None, None, None, ""))

        self.assertIs(state.preview_raw, current_raw)
        self.controller.request_render.assert_not_called()

    def test_apply_auto_crop_enables_auto_crop_and_clears_manual_rect(self):
        geometry = replace(self.controller.state.config.geometry, crop_rect=(0.1, 0.1, 0.9, 0.9), crop_from_auto=False)
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        self.controller.request_render = MagicMock()

        self.controller.apply_auto_crop()

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertTrue(saved_config.geometry.crop_from_auto)
        self.assertIsNone(saved_config.geometry.crop_rect)
        self.controller.request_render.assert_called_once_with()

    def test_reset_crop_disables_auto_crop_and_clears_manual_rect(self):
        geometry = replace(self.controller.state.config.geometry, crop_rect=(0.1, 0.1, 0.9, 0.9), crop_from_auto=True)
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        self.controller.request_render = MagicMock()

        self.controller.reset_crop()

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertFalse(saved_config.geometry.crop_from_auto)
        self.assertIsNone(saved_config.geometry.crop_rect)
        self.controller.request_render.assert_called_once_with()

    def test_set_crop_ratio_updates_config_when_no_manual_rect(self):
        self.controller.request_render = MagicMock()

        self.controller.set_crop_ratio("4:3")

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(saved_config.geometry.autocrop_ratio, "4:3")
        self.assertIsNone(saved_config.geometry.crop_rect)
        self.controller.request_render.assert_called_once_with()

    def test_set_crop_ratio_is_noop_when_unchanged(self):
        geometry = replace(self.controller.state.config.geometry, autocrop_ratio="3:2")
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        self.controller.request_render = MagicMock()

        self.controller.set_crop_ratio("3:2")

        self.mock_session_manager.update_config.assert_not_called()
        self.controller.request_render.assert_not_called()

    def test_set_crop_ratio_preserves_metering_bounds(self):
        """A ratio change is a pure reframe and must not re-meter. Clearing the
        per-file bounds makes the next render re-analyze over the new (smaller) ROI,
        which lands on different per-channel floors/ceils — a visible color cast
        shift on the canvas from an operation that only changed the frame."""
        import numpy as np

        self.controller.state.preview_raw = np.empty((800, 1200, 3), dtype=np.float32)
        floors, ceils = (-2.3, -2.4, -2.8), (-1.3, -1.2, -1.6)
        config = replace(
            self.controller.state.config, process=replace(self.controller.state.config.process, local_floors=floors, local_ceils=ceils)
        )
        config = replace(config, geometry=replace(config.geometry, crop_rect=(0.15, 0.15, 0.85, 0.85)))
        self.controller.state.config = config
        self.controller.request_render = MagicMock()

        self.controller.set_crop_ratio("4:3")

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(saved_config.process.local_floors, floors)
        self.assertEqual(saved_config.process.local_ceils, ceils)
        self.assertTrue(saved_config.process.is_local_initialized)

    def test_set_crop_ratio_reshape_never_grows_the_box(self):
        """The no-re-meter rule above is only safe because the reshape shrinks within
        the existing footprint — a box that could grow might pull film rebate into the
        metered region, which is exactly what the bounds invalidation elsewhere guards."""
        import numpy as np

        self.controller.state.preview_raw = np.empty((800, 1200, 3), dtype=np.float32)
        rect = (0.15, 0.15, 0.85, 0.85)
        self.controller.state.config = replace(
            self.controller.state.config,
            geometry=replace(self.controller.state.config.geometry, crop_rect=rect),
        )
        self.controller.request_render = MagicMock()

        for ratio in ("1:1", "4:3", "16:9", "65:24", "5:4"):
            self.mock_session_manager.reset_mock()
            self.controller.state.config = replace(
                self.controller.state.config,
                geometry=replace(self.controller.state.config.geometry, autocrop_ratio="Free", crop_rect=rect),
            )
            self.controller.set_crop_ratio(ratio)
            nx1, ny1, nx2, ny2 = self.mock_session_manager.update_config.call_args.args[0].geometry.crop_rect
            self.assertGreaterEqual(nx1, rect[0] - 1e-6, f"{ratio}: box grew left")
            self.assertGreaterEqual(ny1, rect[1] - 1e-6, f"{ratio}: box grew up")
            self.assertLessEqual(nx2, rect[2] + 1e-6, f"{ratio}: box grew right")
            self.assertLessEqual(ny2, rect[3] + 1e-6, f"{ratio}: box grew down")

    def test_set_crop_ratio_reshapes_manual_rect_centered_pixel_aware(self):
        """Reshaping must use real pixel dimensions, not normalized fractions —
        a non-square display image means "1:1" in normalized space isn't actually
        square on screen, so the controller (which has the image shape) must do
        this, not the sidebar."""
        import numpy as np

        self.controller.state.preview_raw = np.empty((800, 1200, 3), dtype=np.float32)  # h=800, w=1200
        geometry = replace(self.controller.state.config.geometry, crop_rect=(0.25, 0.25, 0.75, 0.75))
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        self.controller.request_render = MagicMock()

        self.controller.set_crop_ratio("1:1")

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(saved_config.geometry.autocrop_ratio, "1:1")
        nx1, ny1, nx2, ny2 = saved_config.geometry.crop_rect
        # Center unchanged.
        self.assertAlmostEqual((nx1 + nx2) / 2, 0.5, places=3)
        self.assertAlmostEqual((ny1 + ny2) / 2, 0.5, places=3)
        # True pixel square: (nx2-nx1)*1200 == (ny2-ny1)*800.
        px_w = (nx2 - nx1) * 1200
        px_h = (ny2 - ny1) * 800
        self.assertAlmostEqual(px_w, px_h, delta=1.0)
        self.controller.request_render.assert_called_once_with()

    def test_set_crop_ratio_accounts_for_90_degree_rotation(self):
        import numpy as np

        # Source is landscape (h=800, w=1200); a 90 rotation makes the display
        # portrait (h=1200, w=800) — the reshape must use the rotated dims.
        self.controller.state.preview_raw = np.empty((800, 1200, 3), dtype=np.float32)
        geometry = replace(
            self.controller.state.config.geometry,
            rotation=1,
            crop_rect=(0.25, 0.25, 0.75, 0.75),
        )
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        self.controller.request_render = MagicMock()

        self.controller.set_crop_ratio("1:1")

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        nx1, ny1, nx2, ny2 = saved_config.geometry.crop_rect
        # Display dims after a 90 rotation: h=1200, w=800.
        px_w = (nx2 - nx1) * 800
        px_h = (ny2 - ny1) * 1200
        self.assertAlmostEqual(px_w, px_h, delta=1.0)

    def _export_task(self, path, overwrite=False):
        preset = ExportPreset(
            name="t",
            output_mode=ExportPresetOutputMode.ABSOLUTE,
            output_path=os.path.dirname(path),
            overwrite=overwrite,
        )
        return ExportTask(
            file_info={"name": os.path.basename(path), "path": path, "hash": "h"},
            params=WorkspaceConfig(),
            export_settings=preset,
        )

    def _set_export_overwrite(self, value):
        cfg = self.controller.state.config
        self.controller.state.config = replace(cfg, export=replace(cfg.export, overwrite=value))

    def test_export_overwrite_pref_on_skips_prompt_and_overwrites(self):
        self._set_export_overwrite(True)
        with tempfile.TemporaryDirectory() as d:
            task = self._export_task(os.path.join(d, "A.RAF"))
            open(resolve_export_target_path(task), "wb").close()
            self.controller._prompt_overwrite_conflicts = MagicMock()
            out = self.controller._resolve_export_conflicts([task])
            self.assertTrue(out[0].export_settings.overwrite)
            self.controller._prompt_overwrite_conflicts.assert_not_called()

    def test_export_conflict_overwrite_sets_flag_true(self):
        self._set_export_overwrite(False)
        with tempfile.TemporaryDirectory() as d:
            task = self._export_task(os.path.join(d, "A.RAF"))
            open(resolve_export_target_path(task), "wb").close()
            self.controller._prompt_overwrite_conflicts = MagicMock(return_value=(True, False))
            out = self.controller._resolve_export_conflicts([task])
            self.assertEqual(len(out), 1)
            self.assertTrue(out[0].export_settings.overwrite)

    def test_export_conflict_rename_sets_flag_false(self):
        self._set_export_overwrite(False)
        with tempfile.TemporaryDirectory() as d:
            task = self._export_task(os.path.join(d, "A.RAF"))
            open(resolve_export_target_path(task), "wb").close()
            self.controller._prompt_overwrite_conflicts = MagicMock(return_value=(False, False))
            out = self.controller._resolve_export_conflicts([task])
            self.assertFalse(out[0].export_settings.overwrite)

    def test_export_conflict_cancel_returns_none(self):
        self._set_export_overwrite(False)
        with tempfile.TemporaryDirectory() as d:
            task = self._export_task(os.path.join(d, "A.RAF"))
            open(resolve_export_target_path(task), "wb").close()
            self.controller._prompt_overwrite_conflicts = MagicMock(return_value=(None, False))
            self.assertIsNone(self.controller._resolve_export_conflicts([task]))

    def test_export_conflict_remember_persists_preference(self):
        self._set_export_overwrite(False)
        with tempfile.TemporaryDirectory() as d:
            task = self._export_task(os.path.join(d, "A.RAF"))
            open(resolve_export_target_path(task), "wb").close()
            self.controller._prompt_overwrite_conflicts = MagicMock(return_value=(True, True))
            self.controller._set_overwrite_preference = MagicMock()
            self.controller._resolve_export_conflicts([task])
            self.controller._set_overwrite_preference.assert_called_once_with(True)

    def test_export_no_conflict_passes_through_without_prompt(self):
        self._set_export_overwrite(False)
        with tempfile.TemporaryDirectory() as d:
            task = self._export_task(os.path.join(d, "A.RAF"))  # target not created
            self.controller._prompt_overwrite_conflicts = MagicMock()
            out = self.controller._resolve_export_conflicts([task])
            self.assertEqual(out, [task])
            self.controller._prompt_overwrite_conflicts.assert_not_called()

    def _armed_auto_crop(self):
        geometry = replace(self.controller.state.config.geometry, crop_from_auto=True, crop_rect=None)
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        return autocrop_detection_key(geometry)

    def test_freeze_stores_the_crop_the_render_detected(self):
        key = self._armed_auto_crop()
        metrics = {"autocrop_resolved_rect": (0.05, 0.04, 0.99, 0.98), "autocrop_resolved_key": key}

        self.controller._freeze_resolved_auto_crop(metrics)

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(saved_config.geometry.crop_rect, (0.05, 0.04, 0.99, 0.98))
        self.assertEqual(saved_config.geometry.crop_detect_key, key)
        self.assertTrue(saved_config.geometry.crop_from_auto)

    def test_freeze_requests_no_render(self):
        """The rect is what was just painted, so re-rendering it would only cost a frame."""
        key = self._armed_auto_crop()
        self.controller._freeze_resolved_auto_crop({"autocrop_resolved_rect": (0.1, 0.1, 0.9, 0.9), "autocrop_resolved_key": key})
        self.assertFalse(self.mock_session_manager.update_config.call_args.kwargs["render"])

    def test_freeze_drops_a_result_the_user_has_moved_past(self):
        """Ratio changed while the render was in flight: a render under the new one is
        already queued, and storing this rect would file it under the wrong detection."""
        self._armed_auto_crop()
        metrics = {"autocrop_resolved_rect": (0.05, 0.04, 0.99, 0.98), "autocrop_resolved_key": "stale-key"}

        self.controller._freeze_resolved_auto_crop(metrics)

        self.mock_session_manager.update_config.assert_not_called()

    def test_freeze_ignores_a_render_of_a_manual_crop(self):
        geometry = replace(self.controller.state.config.geometry, crop_from_auto=False, crop_rect=(0.2, 0.2, 0.8, 0.8))
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)

        self.controller._freeze_resolved_auto_crop(
            {"autocrop_resolved_rect": (0.0, 0.0, 1.0, 1.0), "autocrop_resolved_key": autocrop_detection_key(geometry)}
        )

        self.mock_session_manager.update_config.assert_not_called()

    def test_apply_auto_crop_arms_without_a_rect(self):
        self.controller.request_render = MagicMock()
        self.controller.state.config = replace(
            self.controller.state.config,
            geometry=replace(self.controller.state.config.geometry, crop_rect=(0.2, 0.2, 0.8, 0.8)),
        )

        self.controller.apply_auto_crop()

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertTrue(saved_config.geometry.crop_from_auto)
        self.assertIsNone(saved_config.geometry.crop_rect)

    def test_crop_rect_changed_disables_auto_crop(self):
        geometry = replace(self.controller.state.config.geometry, crop_from_auto=True)
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        self.controller.state.active_tool = ToolMode.CROP_MANUAL
        self.controller.request_render = MagicMock()

        self.controller.handle_crop_rect_changed(0.2, 0.3, 0.8, 0.9, True)

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertFalse(saved_config.geometry.crop_from_auto)
        self.assertEqual(saved_config.geometry.crop_rect, (0.2, 0.3, 0.8, 0.9))
        self.controller.request_render.assert_called_once_with()

    def test_handle_crop_rect_changed_updates_rect(self):
        geometry = replace(self.controller.state.config.geometry, crop_rect=(0.2, 0.2, 0.6, 0.5))
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        self.controller.state.active_tool = ToolMode.CROP_MANUAL
        self.controller.request_render = MagicMock()

        self.controller.handle_crop_rect_changed(0.3, 0.25, 0.7, 0.55, True)

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(saved_config.geometry.crop_rect, (0.3, 0.25, 0.7, 0.55))
        self.controller.request_render.assert_called_once_with()

    def test_handle_crop_rect_changed_noop_when_tool_inactive(self):
        geometry = replace(self.controller.state.config.geometry, crop_rect=None)
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        self.controller.state.active_tool = ToolMode.NONE
        self.controller.request_render = MagicMock()

        self.controller.handle_crop_rect_changed(0.1, 0.1, 0.5, 0.5, True)

        self.mock_session_manager.update_config.assert_not_called()
        self.controller.request_render.assert_not_called()

    def test_handle_crop_rect_changed_does_not_deactivate_tool(self):
        geometry = replace(self.controller.state.config.geometry, crop_rect=(0.2, 0.2, 0.6, 0.5))
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        self.controller.state.active_tool = ToolMode.CROP_MANUAL
        self.controller.request_render = MagicMock()

        self.controller.handle_crop_rect_changed(0.3, 0.25, 0.7, 0.55, True)

        self.assertEqual(self.controller.state.active_tool, ToolMode.CROP_MANUAL)

    def test_handle_crop_rect_changed_live_drag_does_not_persist(self):
        geometry = replace(self.controller.state.config.geometry, crop_rect=(0.2, 0.2, 0.6, 0.5))
        self.controller.state.config = replace(self.controller.state.config, geometry=geometry)
        self.controller.state.active_tool = ToolMode.CROP_MANUAL
        self.controller.request_render = MagicMock()

        self.controller.handle_crop_rect_changed(0.3, 0.25, 0.7, 0.55, False)

        self.assertEqual(self.mock_session_manager.update_config.call_args.kwargs.get("persist"), False)
        self.controller.request_render.assert_not_called()

    def test_handle_crop_rect_changed_defers_bounds_invalidation(self):
        """During drag the auto-exposure bounds are left untouched (only flagged dirty),
        so the base cache survives and the frame doesn't re-normalize each step."""
        process = replace(self.controller.state.config.process, local_floors=(0.1, 0.2, 0.3), lock_bounds=False)
        self.controller.state.config = replace(self.controller.state.config, process=process)
        self.controller.state.active_tool = ToolMode.CROP_MANUAL
        self.controller.request_render = MagicMock()

        self.controller.handle_crop_rect_changed(0.2, 0.3, 0.8, 0.9, True)

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(saved_config.process.local_floors, (0.1, 0.2, 0.3))
        self.assertTrue(self.controller._crop_bounds_dirty)

    def test_leaving_crop_tool_invalidates_bounds_once(self):
        """Closing the crop tool with a pending change recomputes bounds a single time."""
        process = replace(
            self.controller.state.config.process,
            local_floors=(0.1, 0.2, 0.3),
            local_ceils=(0.4, 0.5, 0.6),
            lock_bounds=False,
        )
        self.controller.state.config = replace(self.controller.state.config, process=process)
        self.controller.state.active_tool = ToolMode.CROP_MANUAL
        self.controller._crop_bounds_dirty = True
        self.controller.request_render = MagicMock()

        self.controller.set_active_tool(ToolMode.NONE)

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(saved_config.process.local_floors, (0.0, 0.0, 0.0))
        self.assertEqual(saved_config.process.local_ceils, (0.0, 0.0, 0.0))
        self.assertFalse(self.controller._crop_bounds_dirty)
        self.controller.request_render.assert_called_once()

    def test_apply_auto_crop_exits_manual_crop_tool(self):
        """Enabling autocrop while the manual crop tool is active deactivates the tool."""
        self.controller.state.active_tool = ToolMode.CROP_MANUAL
        self.controller.request_render = MagicMock()

        self.controller.apply_auto_crop()

        self.assertEqual(self.controller.state.active_tool, ToolMode.NONE)

    def _seed_two_masks(self):
        from negpy.features.local.models import LocalAdjustmentsConfig, LocalMask

        verts = ((0.1, 0.1), (0.9, 0.1), (0.5, 0.9))
        masks = (
            LocalMask(vertices=verts, stops=-0.3, feather=0.02),
            LocalMask(vertices=verts, stops=0.3, feather=0.02),
        )
        self.controller.state.config = replace(self.controller.state.config, local=LocalAdjustmentsConfig(masks=masks))
        # Hidden-mask state is keyed by the open file's hash; give the tests one.
        self.controller.state.current_file_hash = "hashA"

    def test_set_local_mask_inverted_flips_only_that_mask(self):
        self._seed_two_masks()
        self.controller.request_render = MagicMock()

        self.controller.set_local_mask_inverted(1, True)

        saved = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual([m.invert for m in saved.local.masks], [False, True])
        self.controller.request_render.assert_called_once()

    def test_set_local_mask_visible_toggles_hidden_set(self):
        self._seed_two_masks()
        self.controller.canvas = None  # tolerate no registered canvas
        self.controller.set_local_mask_visible(1, False)
        self.assertEqual(self.controller.state.local_hidden_masks, {1})
        self.controller.set_local_mask_visible(1, True)
        self.assertEqual(self.controller.state.local_hidden_masks, set())

    def test_hidden_masks_persist_per_file_hash(self):
        self._seed_two_masks()
        self.controller.canvas = None
        self.controller.state.current_file_hash = "hashA"
        self.controller.set_local_mask_visible(1, False)
        self.assertEqual(self.controller.state.local_hidden_masks_by_hash["hashA"], {1})

        # Simulate switching away: another file's set is independent.
        self.controller.state.current_file_hash = "hashB"
        self.controller.state.local_hidden_masks = set()
        self.controller.set_local_mask_visible(0, False)
        self.assertEqual(self.controller.state.local_hidden_masks_by_hash["hashB"], {0})
        self.assertEqual(self.controller.state.local_hidden_masks_by_hash["hashA"], {1})

    def test_hidden_masks_cleared_hash_is_pruned(self):
        self._seed_two_masks()
        self.controller.canvas = None
        self.controller.state.current_file_hash = "hashA"
        self.controller.set_local_mask_visible(1, False)
        self.controller.set_local_mask_visible(1, True)
        self.assertNotIn("hashA", self.controller.state.local_hidden_masks_by_hash)

    def test_hidden_masks_clamped_when_mask_count_shrinks(self):
        from negpy.features.local.models import LocalAdjustmentsConfig, LocalMask

        self._seed_two_masks()  # 2 masks under hashA
        self.controller.canvas = None
        self.controller.set_local_mask_visible(1, False)
        self.assertEqual(self.controller.state.local_hidden_masks, {1})

        # Simulate an undo/redo/jump that swaps in a config with fewer masks: the stored
        # index 1 now points past the end and must be dropped from the returned set.
        verts = ((0.1, 0.1), (0.9, 0.1), (0.5, 0.9))
        one_mask = (LocalMask(vertices=verts, stops=-0.3, feather=0.02),)
        self.controller.state.config = replace(self.controller.state.config, local=LocalAdjustmentsConfig(masks=one_mask))
        self.assertEqual(self.controller.state.local_hidden_masks, set())

    def test_delete_local_mask_confirmed_remaps_view_indices(self):
        self._seed_two_masks()
        self.controller.request_render = MagicMock()
        self.controller.state.local_selected_mask = 1
        self.controller.state.local_hidden_masks = {1}

        with patch("negpy.desktop.view.confirm.confirm_delete_mask", return_value=True):
            self.controller.delete_local_mask(0)

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(len(saved_config.local.masks), 1)
        self.assertEqual(self.controller.state.local_selected_mask, 0)
        self.assertEqual(self.controller.state.local_hidden_masks, {0})

    def test_delete_local_mask_cancelled_is_noop(self):
        self._seed_two_masks()
        self.controller.request_render = MagicMock()
        self.mock_session_manager.update_config.reset_mock()

        with patch("negpy.desktop.view.confirm.confirm_delete_mask", return_value=False):
            self.controller.delete_local_mask(0)

        self.mock_session_manager.update_config.assert_not_called()

    def test_lasso_completion_adds_mask_and_exits_draw_mode(self):
        import numpy as np

        self.controller.state.active_tool = ToolMode.LOCAL_DRAW
        self.controller.state.last_metrics["uv_grid"] = np.zeros((2, 2, 2), dtype=np.float32)
        self.controller.request_render = MagicMock()

        self.controller.handle_local_mask_created("polygon", [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)])

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(len(saved_config.local.masks), 1)
        self.assertEqual(self.controller.state.active_tool, ToolMode.NONE)

    def test_a_card_edge_mask_needs_only_two_points(self):
        import numpy as np

        from negpy.features.local.models import MaskShape

        self.controller.state.active_tool = ToolMode.LOCAL_GRADIENT
        self.controller.state.last_metrics["uv_grid"] = np.zeros((2, 2, 2), dtype=np.float32)
        self.controller.request_render = MagicMock()

        self.controller.handle_local_mask_created("gradient", [(0.1, 0.1), (0.9, 0.9)])

        saved_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(len(saved_config.local.masks), 1)
        self.assertEqual(saved_config.local.masks[0].shape, MaskShape.GRADIENT)


class TestBatchExportFiltering(unittest.TestCase):
    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        self.mock_session_manager.repo.load_file_settings.return_value = None

        self.mock_session_manager.state.uploaded_files = [
            {"name": "IMG_0001.cr2", "path": "/tmp/IMG_0001.cr2", "hash": "h1"},
            {"name": "IMG_0002.cr2", "path": "/tmp/IMG_0002.cr2", "hash": "h2"},
            {"name": "scan.tif", "path": "/tmp/scan.tif", "hash": "h3"},
        ]

        self.visible_indices = [0, 1, 2]
        self.mock_session_manager.asset_model = MagicMock()
        self.mock_session_manager.asset_model.visible_actual_indices_ordered.side_effect = lambda: list(self.visible_indices)

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)

        self.controller._ensure_valid_export_path = MagicMock(return_value="/tmp/out")
        self.controller._run_export_tasks = MagicMock()
        self.controller._confirm_bulk_export = MagicMock(return_value=True)

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def _captured_tasks(self):
        self.controller._run_export_tasks.assert_called_once()
        return self.controller._run_export_tasks.call_args.args[0]

    def test_export_all_with_no_filter(self):
        self.visible_indices = [0, 1, 2]
        self.controller.request_batch_export()
        tasks = self._captured_tasks()
        self.assertEqual([t.file_info["name"] for t in tasks], ["IMG_0001.cr2", "IMG_0002.cr2", "scan.tif"])

    def test_export_all_respects_filter(self):
        self.visible_indices = [0, 1]  # only IMG_*
        self.controller.request_batch_export()
        tasks = self._captured_tasks()
        self.assertEqual([t.file_info["name"] for t in tasks], ["IMG_0001.cr2", "IMG_0002.cr2"])

    def test_export_all_zero_matches_does_not_dispatch(self):
        self.visible_indices = []
        self.controller.request_batch_export()
        self.controller._run_export_tasks.assert_not_called()

    def test_export_all_preserves_display_order(self):
        self.visible_indices = [2, 0]  # reversed visible order from sort+filter
        self.controller.request_batch_export()
        tasks = self._captured_tasks()
        self.assertEqual([t.file_info["name"] for t in tasks], ["scan.tif", "IMG_0001.cr2"])

    def test_export_all_applies_current_export_to_all(self):
        self.visible_indices = [0, 1]
        self.controller.state.config = replace(
            self.controller.state.config,
            export=replace(self.controller.state.config.export, export_path="/orig"),
        )
        self.controller.request_batch_export()
        tasks = self._captured_tasks()
        for t in tasks:
            self.assertEqual(t.params.export.export_path, "/tmp/out")

    def test_batch_export_ignores_stale_per_file_export_settings(self):
        """A frame's saved export block never overrides the panel (issue #750: batch
        exports came out at the stale 2000px target while the panel said Original)."""
        self.visible_indices = [0, 1]
        session_export = replace(
            self.controller.state.config.export,
            export_resolution_mode=ExportResolutionMode.ORIGINAL.value,
            jpeg_quality=90,
        )
        self.controller.state.config = replace(self.controller.state.config, export=session_export)

        stale_export = replace(
            session_export,
            output_mode=ExportPresetOutputMode.SAME_AS_SOURCE,
            export_path="/stale/default",
            output_subfolder="old_sub",
            export_fmt=ExportFormat.PNG,
            export_color_space=ColorSpace.ADOBE_RGB.value,
            export_resolution_mode=ExportResolutionMode.TARGET_PX.value,
            export_target_long_edge_px=2000,
            paper_aspect_ratio="1:1",
            export_print_size=10.0,
            export_dpi=150,
            jpeg_quality=50,
            filename_pattern="{{ original_name }}_stale",
        )
        stale_config = replace(self.controller.state.config, export=stale_export)
        self.mock_session_manager.repo.load_file_settings.return_value = stale_config
        self.controller.request_batch_export()
        tasks = self._captured_tasks()
        self.assertEqual(len(tasks), 2)
        for t in tasks:
            for field in (
                "output_mode",
                "output_subfolder",
                "export_fmt",
                "export_color_space",
                "export_resolution_mode",
                "export_target_long_edge_px",
                "paper_aspect_ratio",
                "export_print_size",
                "export_dpi",
                "jpeg_quality",
                "filename_pattern",
            ):
                self.assertEqual(getattr(t.params.export, field), getattr(session_export, field), field)
                self.assertEqual(getattr(t.export_settings, field), getattr(session_export, field), field)
            # export_path is validated by _ensure_valid_export_path (mocked to /tmp/out)
            self.assertEqual(t.params.export.export_path, "/tmp/out")

    def test_composite_frames_export_beside_a_half_frame(self):
        """An HDR or stitched composite hash ends in "#hdr"/"#stitch". The sibling
        lookup read that suffix as a half index, and int("hdr") stopped the batch
        before any frame was written."""
        from negpy.features.hdr.models import hdr_hash
        from negpy.features.stitch.models import stitch_hash

        self.mock_session_manager.state.uploaded_files = [
            {"name": "bracket", "path": "/tmp/_DSC1722.NEF", "hash": hdr_hash(["h1", "h2"])},
            {"name": "panorama", "path": "/tmp/_DSC1730.NEF", "hash": stitch_hash(["h3", "h4"])},
            {"name": "left half", "path": "/tmp/scan.tif", "hash": "h5#1"},
        ]
        self.visible_indices = [0, 1, 2]
        self.controller.request_batch_export()
        tasks = self._captured_tasks()
        self.assertEqual([t.file_info["name"] for t in tasks], ["bracket", "panorama", "left half"])


class TestLinearOutputExportCurrentFile(unittest.TestCase):
    """Regression: exporting Linear Output for the *current* file (files=None) must
    reuse its full asset dict, not a bare {path, name, hash} — otherwise
    resolve_asset_rgbscan sees no green_path/blue_path and silently strips the RGB-scan
    triplet, so only the primary (red) narrowband exposure gets exported."""

    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        self.mock_session_manager.repo.load_file_settings.return_value = None

        self.mock_session_manager.state.uploaded_files = [
            {
                "name": "IMG_0001_R.cr2",
                "path": "/tmp/IMG_0001_R.cr2",
                "hash": "h1",
                "green_path": "/tmp/IMG_0001_G.cr2",
                "blue_path": "/tmp/IMG_0001_B.cr2",
            }
        ]
        self.mock_session_manager.state.current_file_path = "/tmp/IMG_0001_R.cr2"
        self.mock_session_manager.state.current_file_hash = "h1"

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)

        self.controller.state.current_file_path = "/tmp/IMG_0001_R.cr2"
        self.controller.state.current_file_hash = "h1"
        self.controller._ensure_valid_export_path = MagicMock(return_value="/tmp/out")

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_current_file_triplet_survives_linear_export(self):
        # The write itself now runs in the export worker, so the triplet has to survive
        # into the dispatched task rather than into a direct call.
        files = [f for f in self.controller.state.uploaded_files]
        tasks = self.controller._linear_output_tasks(files, "/tmp/out")

        self.assertEqual(len(tasks), 1)
        rgbscan = tasks[0].options["rgbscan"]
        self.assertTrue(rgbscan.enabled)
        self.assertEqual(rgbscan.green_path, "/tmp/IMG_0001_G.cr2")
        self.assertEqual(rgbscan.blue_path, "/tmp/IMG_0001_B.cr2")


class TestLinearOutputDestination(unittest.TestCase):
    """Regression (#859): Linear Output built its own destination — always the absolute
    export path, always `<stem>_linear` — instead of the Export panel's destination rules.
    Selecting "Same as source" or a filename template did nothing under the Linear intent."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.source = os.path.join(self.tmp.name, "src", "IMG_0001.dng")
        os.makedirs(os.path.dirname(self.source))
        open(self.source, "w").close()

        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        self.mock_session_manager.repo.load_file_settings.return_value = None
        self.file_info = {"name": "IMG_0001.dng", "path": self.source, "hash": "h1"}
        self.mock_session_manager.state.uploaded_files = [self.file_info]

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()
        self.tmp.cleanup()

    def _set_export(self, **kwargs):
        state = self.controller.state
        state.config = replace(state.config, export=replace(state.config.export, **kwargs))

    def _out_path(self, export_path="/abs/out"):
        tasks = self.controller._linear_output_tasks([self.file_info], export_path)
        self.assertEqual(len(tasks), 1)
        return tasks[0].out_path

    def test_same_as_source_writes_beside_the_source(self):
        self._set_export(output_mode=ExportPresetOutputMode.SAME_AS_SOURCE)
        self.assertEqual(self._out_path(), os.path.join(os.path.dirname(self.source), "IMG_0001_linear.tiff"))

    def test_subfolder_of_source_writes_into_the_subfolder(self):
        self._set_export(output_mode=ExportPresetOutputMode.SUBFOLDER_OF_SOURCE, output_subfolder="linear")
        expected = os.path.join(os.path.dirname(self.source), "linear", "IMG_0001_linear.tiff")
        self.assertEqual(self._out_path(), expected)

    def test_absolute_writes_to_the_export_path(self):
        self._set_export(output_mode=ExportPresetOutputMode.ABSOLUTE)
        self.assertEqual(self._out_path(), os.path.join("/abs/out", "IMG_0001_linear.tiff"))

    def test_filename_template_is_honoured_and_keeps_the_linear_suffix(self):
        # The suffix is not cosmetic: without it, "same as source" plus the default pattern
        # writes the dump over the source file it was decoded from.
        self._set_export(
            output_mode=ExportPresetOutputMode.ABSOLUTE,
            filename_pattern="{{ original_name }}_{{ format }}",
        )
        self.assertEqual(self._out_path(), os.path.join("/abs/out", "IMG_0001_TIFF_linear.tiff"))

    def test_format_variable_names_the_linear_format_not_the_print_one(self):
        self.controller.state.linear_format = "jxl"
        self._set_export(
            output_mode=ExportPresetOutputMode.ABSOLUTE,
            export_fmt=ExportFormat.JPEG,
            filename_pattern="{{ original_name }}_{{ format }}",
        )
        self.assertEqual(self._out_path(), os.path.join("/abs/out", "IMG_0001_JXL_linear.jxl"))

    def test_existing_file_is_renamed_unless_overwrite_is_set(self):
        out_dir = os.path.dirname(self.source)
        open(os.path.join(out_dir, "IMG_0001_linear.tiff"), "w").close()

        self._set_export(output_mode=ExportPresetOutputMode.SAME_AS_SOURCE, overwrite=False)
        self.assertEqual(self._out_path(), os.path.join(out_dir, "IMG_0001_linear_2.tiff"))

        self._set_export(overwrite=True)
        self.assertEqual(self._out_path(), os.path.join(out_dir, "IMG_0001_linear.tiff"))

    def test_unset_export_path_does_not_cancel_a_source_relative_export(self):
        # `not export_path` used to abort here: the source-relative modes never read the
        # path, so an empty one made Export do nothing at all, without a message.
        self._set_export(output_mode=ExportPresetOutputMode.SAME_AS_SOURCE, export_path="")
        self.assertEqual(self.controller._ensure_valid_export_path(), "")


class TestPresetExportCurrentFileTriplet(unittest.TestCase):
    """Regression: request_preset_export() (the "Export Presets" button's current-file
    scope) built a bare {path, name, hash} dict for every call, unconditionally — never
    looking up uploaded_files at all. Same failure mode as the Linear Output current-file
    bug: resolve_asset_rgbscan/resolve_asset_stitch see no green_path/blue_path and reset
    the triplet, so a preset export of the current file silently used only the primary
    (red) narrowband exposure instead of the merged RGB."""

    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        self.mock_session_manager.repo.load_file_settings.return_value = None

        self.mock_session_manager.state.uploaded_files = [
            {
                "name": "IMG_0001_R.cr2",
                "path": "/tmp/IMG_0001_R.cr2",
                "hash": "h1",
                "green_path": "/tmp/IMG_0001_G.cr2",
                "blue_path": "/tmp/IMG_0001_B.cr2",
            }
        ]
        self.mock_session_manager.state.current_file_path = "/tmp/IMG_0001_R.cr2"
        self.mock_session_manager.state.current_file_hash = "h1"
        self.mock_session_manager.state.export_presets = [
            ExportPreset(name="JPEG", enabled=True, export_fmt=ExportFormat.JPEG),
        ]

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)

        self.controller.state.current_file_path = "/tmp/IMG_0001_R.cr2"
        self.controller.state.current_file_hash = "h1"
        self.controller._validate_preset_paths = MagicMock(return_value=True)
        self.controller._run_export_tasks = MagicMock()

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_current_file_triplet_survives_preset_export(self):
        self.controller.request_preset_export()

        self.controller._run_export_tasks.assert_called_once()
        tasks = self.controller._run_export_tasks.call_args.args[0]
        self.assertEqual(len(tasks), 1)
        rgbscan = tasks[0].params.rgbscan
        self.assertTrue(rgbscan.enabled)
        self.assertEqual(rgbscan.green_path, "/tmp/IMG_0001_G.cr2")
        self.assertEqual(rgbscan.blue_path, "/tmp/IMG_0001_B.cr2")


class TestPresetBatchExport(unittest.TestCase):
    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        self.mock_session_manager.repo.load_file_settings.return_value = None

        self.mock_session_manager.state.uploaded_files = [
            {"name": "IMG_0001.cr2", "path": "/tmp/IMG_0001.cr2", "hash": "h1"},
            {"name": "IMG_0002.cr2", "path": "/tmp/IMG_0002.cr2", "hash": "h2"},
            {"name": "scan.tif", "path": "/tmp/scan.tif", "hash": "h3"},
        ]
        self.mock_session_manager.state.export_presets = [
            ExportPreset(name="JPEG", enabled=True, export_fmt=ExportFormat.JPEG),
            ExportPreset(name="TIFF", enabled=True, export_fmt=ExportFormat.TIFF),
        ]

        self.visible_indices = [0, 1, 2]
        self.mock_session_manager.asset_model = MagicMock()
        self.mock_session_manager.asset_model.visible_actual_indices_ordered.side_effect = lambda: list(self.visible_indices)

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)

        self.controller._validate_preset_paths = MagicMock(return_value=True)
        self.controller._run_export_tasks = MagicMock()

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def _captured_tasks(self):
        self.controller._run_export_tasks.assert_called_once()
        return self.controller._run_export_tasks.call_args.args[0]

    @patch("negpy.desktop.controller.QMessageBox.question")
    def test_preset_batch_export_respects_filter(self, mock_question):
        from PyQt6.QtWidgets import QMessageBox

        mock_question.return_value = QMessageBox.StandardButton.Yes
        self.visible_indices = [0, 1]
        self.controller.request_preset_batch_export()
        tasks = self._captured_tasks()
        self.assertEqual(len(tasks), 4)
        self.assertEqual([t.file_info["name"] for t in tasks], ["IMG_0001.cr2"] * 2 + ["IMG_0002.cr2"] * 2)

    @patch("negpy.desktop.controller.QMessageBox.question")
    def test_preset_batch_export_zero_visible_does_not_dispatch(self, mock_question):
        self.visible_indices = []
        self.controller.request_preset_batch_export()
        self.controller._run_export_tasks.assert_not_called()
        mock_question.assert_not_called()

    @patch("negpy.desktop.controller.QMessageBox.question")
    def test_preset_batch_export_cancel_does_not_dispatch(self, mock_question):
        from PyQt6.QtWidgets import QMessageBox

        mock_question.return_value = QMessageBox.StandardButton.Cancel
        self.controller.request_preset_batch_export()
        self.controller._run_export_tasks.assert_not_called()

    @patch("negpy.desktop.controller.QMessageBox.question")
    def test_preset_batch_export_confirmation_message(self, mock_question):
        from PyQt6.QtWidgets import QMessageBox

        mock_question.return_value = QMessageBox.StandardButton.Yes
        self.visible_indices = [0, 1, 2]
        self.controller.request_preset_batch_export()
        mock_question.assert_called_once()
        message = mock_question.call_args.args[2]
        self.assertIn("3 frames", message)
        self.assertIn("2 presets", message)
        self.assertIn("6 files", message)


class TestPresetExportSelected(unittest.TestCase):
    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        self.mock_session_manager.repo.load_file_settings.return_value = None
        self.mock_session_manager.state.current_file_path = "/tmp/IMG_0002.cr2"
        self.mock_session_manager.state.current_file_hash = "h2"

        self.mock_session_manager.state.uploaded_files = [
            {"name": "IMG_0001.cr2", "path": "/tmp/IMG_0001.cr2", "hash": "h1"},
            {"name": "IMG_0002.cr2", "path": "/tmp/IMG_0002.cr2", "hash": "h2"},
            {"name": "scan.tif", "path": "/tmp/scan.tif", "hash": "h3"},
        ]
        self.mock_session_manager.state.export_presets = [
            ExportPreset(name="JPEG", enabled=True, export_fmt=ExportFormat.JPEG),
            ExportPreset(name="TIFF", enabled=True, export_fmt=ExportFormat.TIFF),
        ]
        self.mock_session_manager.state.selected_indices = [2, 0]

        self.visible_indices = [0, 1, 2]
        self.mock_session_manager.asset_model = MagicMock()
        self.mock_session_manager.asset_model.visible_actual_indices_ordered.side_effect = lambda: list(self.visible_indices)

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)

        self.controller._validate_preset_paths = MagicMock(return_value=True)
        self.controller._run_export_tasks = MagicMock()

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_preset_export_selected_confirms_and_uses_display_order(self):
        from PyQt6.QtWidgets import QMessageBox

        self.mock_session_manager.state.selected_indices = [2, 0]
        with patch("negpy.desktop.controller.QMessageBox.question") as mock_question:
            mock_question.return_value = QMessageBox.StandardButton.Yes
            self.controller.request_preset_export_selected()
            mock_question.assert_called_once()

        tasks = self.controller._run_export_tasks.call_args.args[0]
        self.assertEqual(len(tasks), 4)
        self.assertEqual([t.file_info["name"] for t in tasks], ["IMG_0001.cr2"] * 2 + ["scan.tif"] * 2)

    def test_preset_export_single_selection_uses_preview_frame(self):
        self.mock_session_manager.state.selected_indices = [0]
        self.mock_session_manager.state.selected_file_idx = 2
        self.mock_session_manager.state.current_file_path = "/tmp/scan.tif"
        self.controller.request_preset_export_selected()
        tasks = self.controller._run_export_tasks.call_args.args[0]
        self.assertEqual(len(tasks), 2)
        self.assertEqual({t.file_info["name"] for t in tasks}, {"scan.tif"})

    def test_preset_export_selected_skips_excluded(self):
        self.mock_session_manager.state.uploaded_files[0]["excluded"] = True
        with patch("negpy.desktop.controller.QMessageBox.question") as mock_question:
            self.controller.request_preset_export_selected()
            mock_question.assert_not_called()

        tasks = self.controller._run_export_tasks.call_args.args[0]
        self.assertEqual([t.file_info["name"] for t in tasks], ["scan.tif"] * 2)

    def test_preset_export_current_frame_menu_unchanged(self):
        self.controller.request_preset_export()
        tasks = self.controller._run_export_tasks.call_args.args[0]
        self.assertEqual(len(tasks), 2)
        self.assertEqual({t.file_info["name"] for t in tasks}, {"IMG_0002.cr2"})

    def test_batch_export_default_skips_rejected(self):
        self.mock_session_manager.state.uploaded_files[1]["excluded"] = True
        self.controller._ensure_valid_export_path = MagicMock(return_value="/tmp")
        self.controller._confirm_bulk_export = MagicMock(return_value=True)

        self.controller.request_batch_export()

        tasks = self.controller._run_export_tasks.call_args.args[0]
        names = [t.file_info["name"] for t in tasks]
        self.assertEqual(names, ["IMG_0001.cr2", "scan.tif"])

    def test_export_selected_skips_rejected(self):
        self.mock_session_manager.state.uploaded_files[0]["excluded"] = True
        self.controller._ensure_valid_export_path = MagicMock(return_value="/tmp")
        self.controller._confirm_bulk_export = MagicMock(return_value=True)

        self.controller.request_export_selected()

        tasks = self.controller._run_export_tasks.call_args.args[0]
        self.assertEqual([t.file_info["name"] for t in tasks], ["scan.tif"])

    def test_batch_normalization_records_history_for_other_files(self):
        self.mock_session_manager.repo.load_file_settings.return_value = None
        self.mock_session_manager.config_for_asset.return_value = WorkspaceConfig()
        self.controller._on_normalization_finished((0.1, 0.1, 0.1), (0.9, 0.9, 0.9), [])

        pushed = {c.args[0] for c in self.mock_session_manager.push_external_history.call_args_list}
        # The active file (h2) records its step via update_config(persist=True) instead.
        self.assertEqual(pushed, {"h1", "h3"})
        self.mock_session_manager.update_config.assert_called()

    def test_locked_frame_keeps_its_own_exposure_after_batch_analysis(self):
        locked_cfg = replace(WorkspaceConfig(), process=replace(WorkspaceConfig().process, lock_bounds=True))
        self.mock_session_manager.repo.load_file_settings.side_effect = lambda h: locked_cfg if h == "h1" else None
        self.mock_session_manager.config_for_asset.return_value = WorkspaceConfig()

        self.controller._on_normalization_finished((0.1, 0.1, 0.1), (0.9, 0.9, 0.9), [])

        saved = {c.args[0]: c.args[1] for c in self.mock_session_manager.repo.save_file_settings.call_args_list}
        self.assertNotIn("h1", saved)  # locked frame is never rewritten
        self.assertIn("h3", saved)
        self.assertTrue(saved["h3"].process.use_luma_average)
        pushed = {c.args[0] for c in self.mock_session_manager.push_external_history.call_args_list}
        self.assertNotIn("h1", pushed)  # not even entered into history

    def test_locked_active_frame_is_not_overwritten_in_memory(self):
        locked_cfg = replace(WorkspaceConfig(), process=replace(WorkspaceConfig().process, lock_bounds=True))
        self.mock_session_manager.repo.load_file_settings.return_value = None
        self.mock_session_manager.config_for_asset.return_value = WorkspaceConfig()
        self.mock_session_manager.state.config = locked_cfg  # h2, the active frame

        self.controller._on_normalization_finished((0.1, 0.1, 0.1), (0.9, 0.9, 0.9), [])

        self.mock_session_manager.update_config.assert_not_called()

    def test_status_message_reports_locked_and_outlier_frames(self):
        locked_cfg = replace(WorkspaceConfig(), process=replace(WorkspaceConfig().process, lock_bounds=True))
        self.mock_session_manager.repo.load_file_settings.side_effect = lambda h: locked_cfg if h == "h1" else None
        self.mock_session_manager.config_for_asset.return_value = WorkspaceConfig()
        msgs = []
        self.controller.status_message_requested.connect(lambda text, *_: msgs.append(text))

        self.controller._on_normalization_finished((0.1, 0.1, 0.1), (0.9, 0.9, 0.9), ["h3"])

        message = next(m for m in msgs if "far from the roll color keeps its own exposure and color" in m)
        self.assertIn("locked frame", message)
        self.assertIn("scan.tif", message)

    def test_batch_normalization_records_the_rolls_own_baseline_when_a_roll_is_active(self):
        self.mock_session_manager.state.active_roll_id = "roll-1"
        self.mock_session_manager.repo.load_file_settings.return_value = None
        self.mock_session_manager.config_for_asset.return_value = WorkspaceConfig()

        with patch.object(rolls, "set_roll_normalization") as mock_set:
            self.controller._on_normalization_finished((0.1, 0.1, 0.1), (0.9, 0.9, 0.9), [])

        mock_set.assert_called_once_with(self.mock_session_manager.repo, "roll-1", (0.1, 0.1, 0.1), (0.9, 0.9, 0.9), outliers=(), axis=None)

    def test_outlier_keeps_its_own_bounds(self):
        self.mock_session_manager.state.active_roll_id = "roll-1"
        self.mock_session_manager.repo.load_file_settings.return_value = None
        self.mock_session_manager.config_for_asset.return_value = WorkspaceConfig()

        with patch.object(rolls, "set_roll_normalization") as mock_set:
            self.controller._on_normalization_finished((0.1, 0.1, 0.1), (0.9, 0.9, 0.9), ["h1", "h2"])

        saved = {c.args[0]: c.args[1].process for c in self.mock_session_manager.repo.save_file_settings.call_args_list}
        self.assertFalse(saved["h1"].use_luma_average)
        self.assertFalse(saved["h1"].use_color_average)
        self.assertTrue(saved["h3"].use_luma_average)
        self.assertTrue(saved["h3"].use_color_average)
        active = self.mock_session_manager.update_config.call_args.args[0].process  # h2
        self.assertFalse(active.use_luma_average)
        self.assertFalse(active.use_color_average)
        self.assertEqual(mock_set.call_args.kwargs["outliers"], ("h1", "h2"))

    def test_pooled_axis_rides_on_inliers_and_is_recorded_with_the_baseline(self):
        self.mock_session_manager.state.active_roll_id = "roll-1"
        self.mock_session_manager.repo.load_file_settings.return_value = None
        self.mock_session_manager.config_for_asset.return_value = WorkspaceConfig()
        axis = ((-1.0, -1.1, -1.2), (-0.4, -0.5, -0.6), None, 0.8)

        with patch.object(rolls, "set_roll_normalization") as mock_set:
            self.controller._on_normalization_finished((0.1, 0.1, 0.1), (0.9, 0.9, 0.9), ["h1"], axis)

        saved = {c.args[0]: c.args[1].process for c in self.mock_session_manager.repo.save_file_settings.call_args_list}
        self.assertFalse(saved["h1"].use_cast_average)
        self.assertTrue(saved["h3"].use_cast_average)
        self.assertEqual(saved["h3"].locked_neutral_axis, axis)
        self.assertEqual(mock_set.call_args.kwargs["axis"], axis)

    def test_no_pooled_axis_leaves_cast_average_off(self):
        self.mock_session_manager.repo.load_file_settings.return_value = None
        self.mock_session_manager.config_for_asset.return_value = WorkspaceConfig()

        self.controller._on_normalization_finished((0.1, 0.1, 0.1), (0.9, 0.9, 0.9), [], None)

        saved = {c.args[0]: c.args[1].process for c in self.mock_session_manager.repo.save_file_settings.call_args_list}
        self.assertFalse(saved["h3"].use_cast_average)
        self.assertTrue(saved["h3"].use_color_average)

    def test_apply_normalization_roll_keeps_a_recorded_outlier_on_its_own_bounds(self):
        self.mock_session_manager.repo.load_file_settings.return_value = None
        self.mock_session_manager.config_for_asset.return_value = WorkspaceConfig()
        data = {"floors": (0.1, 0.1, 0.1), "ceils": (0.9, 0.9, 0.9), "cast": (0.0, 0.0, 0.0), "outliers": ("h1",)}

        with (
            patch.object(rolls, "roll_normalization", return_value=data),
            patch.object(rolls, "roll_for_id", return_value={"name": "Tri-X"}),
        ):
            self.controller.apply_normalization_roll("roll-1")

        saved = {c.args[0]: c.args[1].process for c in self.mock_session_manager.repo.save_file_settings.call_args_list}
        self.assertFalse(saved["h1"].use_color_average)
        self.assertFalse(saved["h1"].use_luma_average)
        self.assertTrue(saved["h3"].use_color_average)

    def test_batch_normalization_does_not_touch_the_roll_store_without_an_active_roll(self):
        self.mock_session_manager.state.active_roll_id = None
        self.mock_session_manager.repo.load_file_settings.return_value = None
        self.mock_session_manager.config_for_asset.return_value = WorkspaceConfig()

        with patch.object(rolls, "set_roll_normalization") as mock_set:
            self.controller._on_normalization_finished((0.1, 0.1, 0.1), (0.9, 0.9, 0.9), [])

        mock_set.assert_not_called()

    def _scene_setup(self):
        self.mock_session_manager.state.active_roll_id = "roll-1"
        self.mock_session_manager.repo.load_file_settings.return_value = None
        self.mock_session_manager.config_for_asset.return_value = WorkspaceConfig()
        return patch.object(rolls, "scene_by_hash", return_value={"h1": (1, "s1", "Beach"), "h3": (1, "s1", "Beach")})

    def test_roll_analysis_skips_scene_members(self):
        with self._scene_setup(), patch.object(rolls, "set_roll_normalization") as mock_set:
            self.controller._on_normalization_finished((0.1, 0.1, 0.1), (0.9, 0.9, 0.9), [])

        saved = {c.args[0]: c.args[1] for c in self.mock_session_manager.repo.save_file_settings.call_args_list}
        self.assertEqual(set(saved), {"h2"})
        mock_set.assert_called_once()
        new_cfg = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(new_cfg.process.baseline_source, "roll:roll-1")

    def test_scene_analysis_writes_members_and_stores_on_the_scene(self):
        self.controller._normalization_scope = "s1"
        with (
            self._scene_setup(),
            patch.object(rolls, "set_roll_normalization") as mock_roll,
            patch.object(rolls, "set_scene_normalization") as mock_scene,
        ):
            self.controller._on_normalization_finished((0.1, 0.1, 0.1), (0.9, 0.9, 0.9), [])

        saved = {c.args[0]: c.args[1] for c in self.mock_session_manager.repo.save_file_settings.call_args_list}
        self.assertEqual(set(saved), {"h1", "h3"})
        self.assertEqual(saved["h1"].process.baseline_source, "scene:s1")
        mock_roll.assert_not_called()
        mock_scene.assert_called_once_with(
            self.mock_session_manager.repo, "roll-1", "s1", (0.1, 0.1, 0.1), (0.9, 0.9, 0.9), outliers=(), axis=None
        )
        # h2, the active frame, is outside the scene, so its in-memory config is untouched.
        self.mock_session_manager.update_config.assert_not_called()

    def test_apply_normalization_roll_skips_scene_members(self):
        data = {"floors": (0.1, 0.1, 0.1), "ceils": (0.9, 0.9, 0.9), "cast": (0.0, 0.0, 0.0)}
        with (
            self._scene_setup(),
            patch.object(rolls, "roll_normalization", return_value=data),
            patch.object(rolls, "roll_for_id", return_value={"name": "Tri-X"}),
        ):
            self.controller.apply_normalization_roll("roll-1")

        saved = {c.args[0] for c in self.mock_session_manager.repo.save_file_settings.call_args_list}
        self.assertEqual(saved, {"h2"})

    def test_analyze_all_scenes_runs_each_scene_in_turn(self):
        by_hash = {"h1": (1, "s1", "Beach"), "h3": (2, "s2", "Night")}
        scenes = [("s1", {"name": "Beach"}), ("s2", {"name": "Night"})]
        emitted = []
        self.controller.normalization_requested.connect(emitted.append)
        with (
            patch.object(rolls, "scene_by_hash", return_value=by_hash),
            patch.object(rolls, "roll_scenes", return_value=scenes),
            patch.object(rolls, "set_scene_normalization") as mock_scene,
            patch.object(self.controller, "_confirm_normalization", return_value=True),
        ):
            self.mock_session_manager.state.active_roll_id = "roll-1"
            self.mock_session_manager.repo.load_file_settings.return_value = None
            self.mock_session_manager.config_for_asset.return_value = WorkspaceConfig()
            self.controller.request_analyze_all_scenes()
            self.assertEqual([f.file_info["hash"] for f in emitted[0].frames], ["h1"])

            self.controller._on_normalization_finished((0.1, 0.1, 0.1), (0.9, 0.9, 0.9), [])
            self.assertEqual([f.file_info["hash"] for f in emitted[1].frames], ["h3"])
            self.assertEqual(self.controller._active_batch, "normalization")

            self.controller._on_normalization_finished((0.2, 0.2, 0.2), (0.8, 0.8, 0.8), [])
            self.assertIsNone(self.controller._active_batch)

        self.assertEqual([c.args[2] for c in mock_scene.call_args_list], ["s1", "s2"])

    def test_cancel_clears_the_scene_queue(self):
        self.controller._scene_queue = ["s2"]
        self.controller._on_normalization_cancelled()
        self.assertEqual(self.controller._scene_queue, [])

    def test_group_as_scene_needs_a_roll(self):
        self.mock_session_manager.state.active_roll_id = None
        self.mock_session_manager.state.selected_indices = [0, 1]
        with patch.object(rolls, "create_scene") as mock_create:
            self.assertIsNone(self.controller.request_group_as_scene("Beach"))
        mock_create.assert_not_called()

    def test_group_as_scene_uses_the_selection(self):
        self.mock_session_manager.state.active_roll_id = "roll-1"
        self.mock_session_manager.state.selected_indices = [0, 2]
        with patch.object(rolls, "create_scene", return_value="s1") as mock_create:
            self.assertEqual(self.controller.request_group_as_scene("Beach"), "s1")
        mock_create.assert_called_once_with(self.mock_session_manager.repo, "roll-1", "Beach", ["h1", "h3"])
        self.mock_session_manager.refresh_scene_marks.assert_called_once()

    def test_only_the_rolls_first_scene_announces_itself(self):
        self.mock_session_manager.state.active_roll_id = "roll-1"
        self.mock_session_manager.state.selected_indices = [0]
        fired = []
        self.controller.first_scene_created.connect(lambda: fired.append(True))

        with patch.object(rolls, "roll_scenes", return_value=[]), patch.object(rolls, "create_scene", return_value="s1"):
            self.controller.request_group_as_scene("Beach")
        with patch.object(rolls, "roll_scenes", return_value=[("s1", {})]), patch.object(rolls, "create_scene", return_value="s2"):
            self.controller.request_group_as_scene("Night")

        self.assertEqual(fired, [True])

    def test_apply_normalization_roll_is_a_noop_for_an_unanalyzed_roll(self):
        with patch.object(rolls, "roll_normalization", return_value=None):
            self.controller.apply_normalization_roll("roll-1")

        self.mock_session_manager.repo.save_file_settings.assert_not_called()
        self.mock_session_manager.update_config.assert_not_called()

    def test_apply_normalization_roll_loads_the_saved_baseline_onto_every_file(self):
        self.mock_session_manager.repo.load_file_settings.return_value = None
        self.mock_session_manager.config_for_asset.return_value = WorkspaceConfig()
        data = {"floors": (0.1, 0.1, 0.1), "ceils": (0.9, 0.9, 0.9), "cast": (0.0, 0.0, 0.0)}

        with (
            patch.object(rolls, "roll_normalization", return_value=data),
            patch.object(rolls, "roll_for_id", return_value={"name": "Tri-X"}),
        ):
            self.controller.apply_normalization_roll("roll-1")

        saved = {c.args[0]: c.args[1] for c in self.mock_session_manager.repo.save_file_settings.call_args_list}
        self.assertEqual(set(saved), {"h1", "h2", "h3"})
        self.assertTrue(saved["h1"].process.use_luma_average)
        self.assertTrue(saved["h1"].process.use_color_average)
        self.assertEqual(saved["h1"].process.locked_floors, (0.1, 0.1, 0.1))
        self.assertEqual(saved["h1"].process.roll_name, "Tri-X")
        # h2 is the active frame -- its in-memory state also updates via update_config.
        pushed = {c.args[0] for c in self.mock_session_manager.push_external_history.call_args_list}
        self.assertEqual(pushed, {"h1", "h3"})
        self.mock_session_manager.update_config.assert_called_once()
        new_cfg = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(new_cfg.process.roll_name, "Tri-X")

    def test_apply_normalization_roll_skips_locked_frames(self):
        locked_cfg = replace(WorkspaceConfig(), process=replace(WorkspaceConfig().process, lock_bounds=True))
        self.mock_session_manager.repo.load_file_settings.side_effect = lambda h: locked_cfg if h == "h1" else None
        self.mock_session_manager.config_for_asset.return_value = WorkspaceConfig()
        data = {"floors": (0.1, 0.1, 0.1), "ceils": (0.9, 0.9, 0.9), "cast": (0.0, 0.0, 0.0)}

        with (
            patch.object(rolls, "roll_normalization", return_value=data),
            patch.object(rolls, "roll_for_id", return_value={"name": "Tri-X"}),
        ):
            self.controller.apply_normalization_roll("roll-1")

        saved = {c.args[0]: c.args[1] for c in self.mock_session_manager.repo.save_file_settings.call_args_list}
        self.assertNotIn("h1", saved)  # locked frame keeps its own exposure
        self.assertIn("h3", saved)

    def test_set_roll_baseline_from_frame_saves_the_frames_own_bounds(self):
        cfg = WorkspaceConfig()
        self.mock_session_manager.state.config = replace(
            cfg, process=replace(cfg.process, local_floors=(0.1, 0.2, 0.3), local_ceils=(0.9, 0.8, 0.7))
        )

        with (
            patch.object(rolls, "set_roll_normalization") as mock_set,
            patch.object(self.controller, "apply_normalization_roll") as mock_apply,
        ):
            self.controller.set_roll_baseline_from_frame("roll-1")

        mock_set.assert_called_once_with(self.mock_session_manager.repo, "roll-1", (0.1, 0.2, 0.3), (0.9, 0.8, 0.7))
        mock_apply.assert_called_once_with("roll-1")

    def test_set_roll_baseline_from_frame_needs_a_metered_frame(self):
        """An unrendered frame has all-zero bounds; writing those would blank the roll."""
        self.mock_session_manager.state.config = WorkspaceConfig()

        with (
            patch.object(rolls, "set_roll_normalization") as mock_set,
            patch.object(self.controller, "apply_normalization_roll") as mock_apply,
        ):
            self.controller.set_roll_baseline_from_frame("roll-1")

        mock_set.assert_not_called()
        mock_apply.assert_not_called()

    def test_set_process_mode_locks_the_film_card_when_a_roll_is_active(self):
        """Editing is a plain per-frame write now, same as any other Roll-tab card
        (set_roll_default) -- Apply to Whole Roll is the only thing that pushes it out."""
        from negpy.features.process.models import ProcessMode

        self.mock_session_manager.state.active_roll_id = "roll-1"

        with (
            patch.object(rolls, "set_frame_override") as mock_lock,
            patch.object(rolls, "frame_override_cards", return_value=set()),
        ):
            self.controller.set_process_mode(ProcessMode.BW)

        mock_lock.assert_called_once_with(self.mock_session_manager.repo, "roll-1", "h2", "film", True)

    def test_set_process_mode_does_not_touch_the_roll_without_an_active_roll(self):
        from negpy.features.process.models import ProcessMode

        self.mock_session_manager.state.active_roll_id = None

        with patch.object(rolls, "set_frame_override") as mock_lock:
            self.controller.set_process_mode(ProcessMode.BW)

        mock_lock.assert_not_called()

    def test_set_process_mode_does_not_relock_an_already_locked_film_card(self):
        from negpy.features.process.models import ProcessMode

        self.mock_session_manager.state.active_roll_id = "roll-1"

        with (
            patch.object(rolls, "set_frame_override") as mock_lock,
            patch.object(rolls, "frame_override_cards", return_value={"film"}),
        ):
            self.controller.set_process_mode(ProcessMode.BW)

        mock_lock.assert_not_called()

    def test_set_positive_source_locks_the_film_card_when_a_roll_is_active(self):
        self.mock_session_manager.state.active_roll_id = "roll-1"
        self.mock_session_manager.state.config = _slide_config(self.mock_session_manager.state.config)

        with (
            patch.object(rolls, "set_frame_override") as mock_lock,
            patch.object(rolls, "frame_override_cards", return_value=set()),
        ):
            self.controller.set_positive_source(True)

        mock_lock.assert_called_once_with(self.mock_session_manager.repo, "roll-1", "h2", "film", True)

    def test_set_positive_source_does_not_touch_the_roll_without_an_active_roll(self):
        self.mock_session_manager.state.active_roll_id = None
        self.mock_session_manager.state.config = _slide_config(self.mock_session_manager.state.config)

        with patch.object(rolls, "set_frame_override") as mock_lock:
            self.controller.set_positive_source(True)

        mock_lock.assert_not_called()

    def test_set_positive_source_leaves_auto_density_grade_on(self):
        """Auto Density/Auto Grade meter a raw slide and a Positive frame alike, so the
        toggle carries them over as they are."""
        self.mock_session_manager.state.active_roll_id = None
        cfg = _slide_config(self.mock_session_manager.state.config)
        self.mock_session_manager.state.config = replace(
            cfg, exposure=replace(cfg.exposure, auto_exposure=True, auto_normalize_contrast=True)
        )

        self.controller.set_positive_source(True)

        passed = self.mock_session_manager.update_config.call_args.args[0]
        self.assertTrue(passed.exposure.auto_exposure)
        self.assertTrue(passed.exposure.auto_normalize_contrast)

    def test_set_positive_source_leaves_a_deliberate_auto_choice_alone(self):
        self.mock_session_manager.state.active_roll_id = None
        cfg = _slide_config(self.mock_session_manager.state.config)
        self.mock_session_manager.state.config = replace(
            cfg, exposure=replace(cfg.exposure, auto_exposure=False, auto_normalize_contrast=False)
        )

        self.controller.set_positive_source(True)

        passed = self.mock_session_manager.update_config.call_args.args[0]
        self.assertFalse(passed.exposure.auto_exposure)
        self.assertFalse(passed.exposure.auto_normalize_contrast)

    def test_set_positive_source_off_leaves_auto_density_grade_off(self):
        self.mock_session_manager.state.active_roll_id = None
        self.mock_session_manager.state.config = _positive_slide_config(self.mock_session_manager.state.config)

        self.controller.set_positive_source(False)

        passed = self.mock_session_manager.update_config.call_args.args[0]
        self.assertFalse(passed.exposure.auto_exposure)
        self.assertFalse(passed.exposure.auto_normalize_contrast)

    def test_leaving_slide_drops_positive_and_restores_the_autos(self):
        """Positive is Slide-only, so a mode switch away from Slide clears it, and Auto
        Density/Auto Grade go back to the negative's default (auto_meter_for_mode)."""
        from negpy.features.process.models import ProcessMode

        self.mock_session_manager.state.active_roll_id = None
        self.mock_session_manager.state.config = _positive_slide_config(self.mock_session_manager.state.config)

        self.controller.set_process_mode(ProcessMode.C41)

        passed = self.mock_session_manager.update_config.call_args.args[0]
        self.assertFalse(passed.process.positive_source)
        self.assertTrue(passed.exposure.auto_exposure)
        self.assertTrue(passed.exposure.auto_normalize_contrast)

    def test_request_reset_roll_resets_every_visible_frame(self):
        self.controller.request_reset_roll()

        visible = [self.mock_session_manager.state.uploaded_files[i] for i in self.visible_indices]
        self.mock_session_manager.reset_roll.assert_called_once_with(visible)

    def test_request_reset_roll_with_nothing_visible_does_nothing(self):
        self.visible_indices = []

        self.controller.request_reset_roll()

        self.mock_session_manager.reset_roll.assert_not_called()

    def test_batch_normalization_offers_other_files_for_a_thumbnail_refresh(self):
        self.mock_session_manager.repo.load_file_settings.return_value = None
        self.mock_session_manager.config_for_asset.return_value = WorkspaceConfig()

        self.controller._on_normalization_finished((0.1, 0.1, 0.1), (0.9, 0.9, 0.9), [])

        self.mock_session_manager.frames_edited_offscreen.emit.assert_called_once_with(["h1", "h3"])

    def test_apply_normalization_roll_offers_other_files_for_a_thumbnail_refresh(self):
        self.mock_session_manager.repo.load_file_settings.return_value = None
        self.mock_session_manager.config_for_asset.return_value = WorkspaceConfig()
        data = {"floors": (0.1, 0.1, 0.1), "ceils": (0.9, 0.9, 0.9), "cast": (0.0, 0.0, 0.0)}

        with (
            patch.object(rolls, "roll_normalization", return_value=data),
            patch.object(rolls, "roll_for_id", return_value={"name": "Roll A"}),
        ):
            self.controller.apply_normalization_roll("roll-1")

        self.mock_session_manager.frames_edited_offscreen.emit.assert_called_once_with(["h1", "h3"])


class TestSessionRestore(unittest.TestCase):
    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            self.controller = AppController(self.mock_session_manager)
        self.controller.request_asset_discovery = MagicMock()

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def _mock_settings(self, files, active):
        def get(key, default=None):
            return {"session_files": files, "session_active_path": active}.get(key, default)

        self.mock_session_manager.repo.get_global_setting.side_effect = get

    def test_saved_session_paths_filters_missing(self):
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".dng") as tf:
            self._mock_settings([tf.name, "/does/not/exist.dng"], tf.name)
            self.assertEqual(self.controller.saved_session_paths(), [tf.name])
            self.assertFalse(os.path.exists("/does/not/exist.dng"))

    def test_restore_session_selects_active_and_discovers(self):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".dng") as a, tempfile.NamedTemporaryFile(suffix=".dng") as b:
            self._mock_settings([a.name, b.name], b.name)
            self.controller.restore_session()
            self.assertEqual(self.controller._pending_scanned_file, b.name)
            self.controller.request_asset_discovery.assert_called_once_with([a.name, b.name], auto_open=True, restore_triplets={})

    def test_restore_session_no_saved_files_is_noop(self):
        self._mock_settings([], None)
        self.controller.restore_session()
        self.controller.request_asset_discovery.assert_not_called()

    def _mock_settings_with_rolls(self, files, active, rolls_store):
        from negpy.services.assets import rolls

        def get(key, default=None):
            return {"session_files": files, "session_active_path": active, rolls.ROLLS_KEY: rolls_store}.get(key, default)

        self.mock_session_manager.repo.get_global_setting.side_effect = get

    def test_restore_session_recognizes_the_roll_all_restored_paths_belong_to(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            a, b = os.path.join(folder, "a.dng"), os.path.join(folder, "b.dng")
            open(a, "w").close()
            open(b, "w").close()
            self._mock_settings_with_rolls([a, b], b, {"roll1": {"kind": "folder", "folder_path": folder, "extra_paths": []}})
            self.controller.restore_session()
            self.assertEqual(self.controller.state.active_roll_id, "roll1")

    def test_restore_session_leaves_active_roll_id_none_when_paths_disagree(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as folder_a, tempfile.TemporaryDirectory() as folder_b:
            a, b = os.path.join(folder_a, "a.dng"), os.path.join(folder_b, "b.dng")
            open(a, "w").close()
            open(b, "w").close()
            self._mock_settings_with_rolls([a, b], a, {"roll1": {"kind": "folder", "folder_path": folder_a, "extra_paths": []}})
            self.controller.restore_session()
            self.assertIsNone(self.controller.state.active_roll_id)


class TestRgbScanModeReload(unittest.TestCase):
    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            self.controller = AppController(self.mock_session_manager)
        self.controller.request_asset_discovery = MagicMock()

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_toggle_with_no_files_only_saves_flag(self):
        self.controller.set_rgb_scan_mode(True)
        self.mock_session_manager.repo.save_global_setting.assert_any_call("rgbscan_mode", True)
        self.controller.request_asset_discovery.assert_not_called()

    def test_enabling_sets_sticky_narrowband_default(self):
        self.controller.set_rgb_scan_mode(True)
        self.mock_session_manager.repo.save_global_setting.assert_any_call("last_narrowband_scan", True)

    def test_disabling_does_not_touch_narrowband(self):
        self.controller.set_rgb_scan_mode(False)
        calls = [c.args for c in self.mock_session_manager.repo.save_global_setting.call_args_list]
        self.assertNotIn(("last_narrowband_scan", True), calls)

    def test_enabling_forces_narrowband_on_active_config(self):
        state = self.mock_session_manager.state
        state.uploaded_files = [{"name": "a", "path": "/a.dng", "hash": "h1"}]
        state.current_file_path = "/a.dng"
        self.assertFalse(state.config.process.narrowband_scan)

        self.controller.set_rgb_scan_mode(True)

        updated_config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertTrue(updated_config.process.narrowband_scan)

    def test_toggle_with_loaded_files_rediscovers_all_exposures(self):
        state = self.mock_session_manager.state
        state.uploaded_files = [
            {"name": "a (RGB)", "path": "/r1.dng", "hash": "h1", "green_path": "/g1.dng", "blue_path": "/b1.dng"},
            {"name": "c", "path": "/c.dng", "hash": "h2"},
        ]
        state.current_file_path = "/r1.dng"
        self.controller.set_rgb_scan_mode(False)
        self.controller.request_asset_discovery.assert_called_once_with(
            ["/r1.dng", "/g1.dng", "/b1.dng", "/c.dng"], replace_existing=True, reselect_path="/r1.dng", announce_rgb=False
        )

    def test_turning_the_mode_on_allows_the_empty_report(self):
        """Turning RGB Scan on is the user asking for triplets, so a folder that yields
        none is worth a dialog. Turning it off is not, and neither is anything else."""
        state = self.mock_session_manager.state
        state.uploaded_files = [{"name": "a", "path": "/a.dng", "hash": "h1"}]

        self.controller.set_rgb_scan_mode(True)
        self.assertTrue(self.controller.request_asset_discovery.call_args.kwargs["announce_rgb"])

        self.controller.request_asset_discovery.reset_mock()
        self.controller.set_rgb_scan_mode(False)
        self.assertFalse(self.controller.request_asset_discovery.call_args.kwargs["announce_rgb"])

    def test_discovery_finished_replace_rebuilds_and_reselects(self):
        state = self.mock_session_manager.state
        state.uploaded_files = [
            {"name": "r", "path": "/r.dng", "hash": "h1"},
            {"name": "g", "path": "/g.dng", "hash": "h2"},
            {"name": "b", "path": "/b.dng", "hash": "h3"},
        ]

        def add_files(_paths, validated_info=None):
            state.uploaded_files.extend(validated_info or [])

        self.mock_session_manager.add_files.side_effect = add_files
        self.controller.generate_missing_thumbnails = MagicMock()
        self.controller._replace_after_discovery = True
        self.controller._reselect_after_discovery = "/g.dng"  # was viewing the green exposure

        merged = [{"name": "r (RGB)", "path": "/r.dng", "hash": "h1", "green_path": "/g.dng", "blue_path": "/b.dng"}]
        self.controller._on_discovery_finished(merged)

        self.assertEqual(state.uploaded_files, merged)
        self.mock_session_manager.select_file.assert_called_once_with(0)

    def test_discovery_finished_indexes_files_whose_thumbnails_are_already_cached(self):
        """A whole-library search's own matches already have their thumbnails cached
        (that's how CLIP embedded them to begin with), so generate_missing_thumbnails
        claims no batch and _on_thumbnails_finished -- the only other caller of
        generate_missing_embeddings -- never arrives to refresh the model. Discovery
        must run it directly whenever no thumbnail batch was claimed, not only release
        the hot-folder flag."""
        state = self.mock_session_manager.state
        state.uploaded_files = []
        state.semantic_search_enabled = True

        def add_files(_paths, validated_info=None):
            state.uploaded_files.extend(validated_info or [])

        self.mock_session_manager.add_files.side_effect = add_files
        self.mock_session_manager.asset_model = MagicMock()
        self.mock_session_manager.repo.load_embeddings_for.return_value = {"h1": object()}
        # Nothing missing to thumbnail -- generate_missing_thumbnails claims no batch,
        # exactly as it would when every match's thumbnail is already cached.
        self.controller.generate_missing_thumbnails = MagicMock()
        self.controller._replace_after_discovery = True
        self.controller._reselect_after_discovery = None

        discovered = [{"name": "cat", "path": "/cat.tif", "hash": "h1"}]
        with patch("negpy.desktop.controller.semantic_model.clip_model_ready", return_value=True):
            self.controller._on_discovery_finished(discovered)

        self.mock_session_manager.asset_model.refresh.assert_called()

    def test_replace_fresh_open_selects_first_in_sorted_order(self):
        # Library double-click loads via replace_existing=True with no frame to reselect;
        # the fallback must land on the sorted-first frame, not discovery index 0.
        state = self.mock_session_manager.state
        state.uploaded_files = []

        def add_files(_paths, validated_info=None):
            state.uploaded_files.extend(validated_info or [])

        self.mock_session_manager.add_files.side_effect = add_files
        self.mock_session_manager.asset_model = MagicMock()
        self.mock_session_manager.asset_model.visible_actual_indices_ordered.return_value = [1, 2, 0]
        self.controller.generate_missing_thumbnails = MagicMock()
        self.controller._replace_after_discovery = True
        self.controller._reselect_after_discovery = None

        discovered = [
            {"name": "c", "path": "/c.dng", "hash": "h3"},
            {"name": "a", "path": "/a.dng", "hash": "h1"},
            {"name": "b", "path": "/b.dng", "hash": "h2"},
        ]
        self.controller._on_discovery_finished(discovered)

        self.mock_session_manager.select_file.assert_called_once_with(1)

    def test_auto_open_selects_first_in_sorted_order_not_discovery_order(self):
        state = self.mock_session_manager.state
        state.uploaded_files = []
        state.current_file_path = None

        def add_files(_paths, validated_info=None):
            state.uploaded_files.extend(validated_info or [])

        self.mock_session_manager.add_files.side_effect = add_files
        self.mock_session_manager.asset_model = MagicMock()
        self.controller.generate_missing_thumbnails = MagicMock()
        self.controller._auto_open_after_discovery = True

        # Discovery order c, a, b (indices 0,1,2); filmstrip sorts by name to a, b, c.
        discovered = [
            {"name": "c", "path": "/c.dng", "hash": "h3"},
            {"name": "a", "path": "/a.dng", "hash": "h1"},
            {"name": "b", "path": "/b.dng", "hash": "h2"},
        ]
        self.mock_session_manager.asset_model.visible_actual_indices_ordered.return_value = [1, 2, 0]

        self.controller._on_discovery_finished(discovered)

        # First in sorted order is "a" at actual index 1, not the first-discovered "c" at 0.
        self.mock_session_manager.select_file.assert_called_once_with(1)


class TestDiscoveryProgressPopup(unittest.TestCase):
    """Folder-load hashing drives the shared batch progress popup."""

    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        self.mock_session_manager.repo.get_global_setting.return_value = False
        self.mock_session_manager.asset_model = MagicMock()
        self.mock_session_manager.asset_model.visible_actual_indices_ordered.return_value = []

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            self.controller = AppController(self.mock_session_manager)

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_request_discovery_opens_popup(self):
        started = []
        self.controller.batch_started.connect(lambda title, ab: started.append((title, ab)))
        self.controller.request_asset_discovery(["/a.dng"])
        self.assertEqual(started, [("Hashing files", False)])

    def _captured_task(self, **discovery_kwargs):
        self.controller.asset_discovery_requested.disconnect(self.controller.discovery_worker.process)
        tasks = []
        self.controller.asset_discovery_requested.connect(tasks.append)
        self.controller.request_asset_discovery(["/a.dng"], **discovery_kwargs)
        return tasks[0]

    def test_no_active_roll_never_splits_half_frames(self):
        """A batch with no single shared roll (a library-wide search's mixed results)
        has no roll-wide toggle to apply, and the "confirmed diptych" set is recorded
        by whatever roll's toggle happened to be on at discovery time -- not a per-file
        fact -- so it is not a safe signal here either. Nothing splits."""
        self.mock_session_manager.state.active_roll_id = None
        self.mock_session_manager.repo.get_global_setting.side_effect = lambda key, default=None: (
            ["ha", "hb"] if key == "half_frame_scans" else True if key == "half_frame_mode" else default
        )

        task = self._captured_task()

        self.assertFalse(task.half_frame)

    def test_an_active_roll_uses_its_own_toggle(self):
        self.mock_session_manager.state.active_roll_id = "r1"
        by_roll = {"r1": True}
        self.mock_session_manager.repo.get_global_setting.side_effect = lambda key, default=None: (
            by_roll if key == "half_frame_mode_by_roll" else ["ha"] if key == "half_frame_scans" else default
        )

        task = self._captured_task()

        self.assertTrue(task.half_frame)

    def test_progress_feeds_popup(self):
        progress = []
        self.controller.batch_progress.connect(lambda c, t, n: progress.append((c, t, n)))
        self.controller._on_discovery_progress(2, 5, "x")
        self.assertEqual(progress, [(2, 5, "x")])

    def test_finished_closes_popup_before_thumbnails(self):
        order = []
        self.controller.batch_finished.connect(lambda: order.append("finished"))
        self.controller.generate_missing_thumbnails = MagicMock(side_effect=lambda: order.append("thumbs"))
        self.controller._replace_after_discovery = True
        self.controller._reselect_after_discovery = "/r.dng"
        self.mock_session_manager.add_files.side_effect = lambda _p, validated_info=None: None
        self.mock_session_manager.state.uploaded_files = [{"name": "r", "path": "/r.dng", "hash": "h1"}]

        self.controller._on_discovery_finished([{"name": "r", "path": "/r.dng", "hash": "h1"}])

        self.assertEqual(order, ["finished", "thumbs"])

    def test_thumbnail_queue_does_not_delay_a_new_folder_discovery(self):
        state = self.mock_session_manager.state
        state.uploaded_files = [{"name": "old.dng", "path": "/old.dng", "hash": "old"}]
        self.controller.generate_missing_thumbnails()
        self.assertIsNone(self.controller._active_batch)

        self.controller.asset_discovery_requested.disconnect(self.controller.discovery_worker.process)
        tasks = []
        self.controller.asset_discovery_requested.connect(tasks.append)
        self.controller.request_asset_discovery(["/new-folder"], auto_open=True, replace_existing=True)

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].paths, ["/new-folder"])

    def test_back_to_back_capture_completions_are_discovered_in_order(self):
        self.controller.asset_discovery_requested.disconnect(self.controller.discovery_worker.process)
        tasks = []
        self.controller.asset_discovery_requested.connect(tasks.append)
        self.controller.generate_missing_thumbnails = MagicMock()
        state = self.mock_session_manager.state
        self.mock_session_manager.add_files.side_effect = lambda _paths, validated_info=None: state.uploaded_files.extend(
            validated_info or []
        )
        req = MagicMock()
        req.white_mode = False
        req.rgb_mode = True
        self.controller._last_capture_req = req

        first_paths = ["/roll/frame1_R.dng", "/roll/frame1_G.dng", "/roll/frame1_B.dng"]
        second_paths = ["/roll/frame2_R.dng", "/roll/frame2_G.dng", "/roll/frame2_B.dng"]
        self.controller._on_capture_finished(first_paths)
        self.controller._on_capture_finished(second_paths)

        self.assertEqual([task.paths for task in tasks], [first_paths])

        self.controller._on_discovery_finished([{"name": "frame1", "path": first_paths[0], "hash": "h1"}])
        self.assertEqual([task.paths for task in tasks], [first_paths, second_paths])
        self.assertIn(os.path.normcase(os.path.abspath(first_paths[0])), self.controller._pending_capture_imports)
        self.assertIn(os.path.normcase(os.path.abspath(second_paths[0])), self.controller._pending_capture_imports)

        self.controller._on_discovery_finished([{"name": "frame2", "path": second_paths[0], "hash": "h2"}])
        self.assertEqual([f["path"] for f in state.uploaded_files], [first_paths[0], second_paths[0]])
        self.mock_session_manager.select_file.assert_called_with(1)
        self.assertIn(os.path.normcase(os.path.abspath(first_paths[0])), self.controller._pending_capture_imports)
        self.assertNotIn(os.path.normcase(os.path.abspath(second_paths[0])), self.controller._pending_capture_imports)


class TestHotFolderSequenceState(unittest.TestCase):
    """`hot_folder_sequence_active` belongs only to hot-folder discovery."""

    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        self.mock_session_manager.repo.get_global_setting.return_value = False
        self.mock_session_manager.asset_model = MagicMock()
        self.mock_session_manager.asset_model.visible_actual_indices_ordered.return_value = []
        self.mock_session_manager.add_files.side_effect = lambda _paths, validated_info=None: (
            self.mock_session_manager.state.uploaded_files.extend(validated_info or [])
        )

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            self.controller = AppController(self.mock_session_manager)

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_manual_discovery_never_claims_the_sequence(self):
        self.assertFalse(self.controller.hot_folder_sequence_active)
        self.controller.request_asset_discovery(["/manual.dng"])
        self.assertFalse(self.controller.hot_folder_sequence_active, "a manual request must not claim the hot-folder sequence")

    def test_hot_folder_sequence_ends_when_discovery_finishes(self):
        self.assertFalse(self.controller.hot_folder_sequence_active)

        self.controller.request_asset_discovery(["/hot.dng"], hot_folder=True)
        self.assertTrue(self.controller.hot_folder_sequence_active, "a hot-folder discovery must claim the sequence")

        self.controller.generate_missing_thumbnails = MagicMock()
        self.controller._on_discovery_finished([{"name": "hot", "path": "/hot.dng", "hash": "h1"}])
        self.assertFalse(self.controller.hot_folder_sequence_active)

    def test_hot_folder_sequence_clears_on_discovery_error(self):
        self.controller.request_asset_discovery(["/hot.dng"], hot_folder=True)
        self.assertTrue(self.controller.hot_folder_sequence_active)
        self.controller._on_discovery_batch_error("boom")
        self.assertFalse(self.controller.hot_folder_sequence_active)

    def test_hot_folder_sequence_clears_when_no_assets_found(self):
        self.controller.request_asset_discovery(["/hot.dng"], hot_folder=True)
        self.assertTrue(self.controller.hot_folder_sequence_active)
        self.controller._on_discovery_finished([])
        self.assertFalse(self.controller.hot_folder_sequence_active)

    def test_hot_folder_sequence_clears_when_thumbnails_already_cached(self):
        asset = {"name": "hot", "path": "/hot.dng", "hash": "h1"}
        self.mock_session_manager.state.thumbnails[asset_thumbnail_key(asset)] = object()

        self.controller.request_asset_discovery(["/hot.dng"], hot_folder=True)
        self.assertTrue(self.controller.hot_folder_sequence_active)
        self.controller._on_discovery_finished([asset])
        self.assertFalse(self.controller.hot_folder_sequence_active, "nothing to thumbnail is itself the end of the sequence")


class TestBatchAnalysisFiltering(unittest.TestCase):
    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        self.mock_session_manager.repo.load_file_settings.return_value = None

        self.mock_session_manager.state.uploaded_files = [
            {"name": "IMG_0001.cr2", "path": "/tmp/IMG_0001.cr2", "hash": "h1"},
            {"name": "IMG_0002.cr2", "path": "/tmp/IMG_0002.cr2", "hash": "h2"},
            {"name": "scan.tif", "path": "/tmp/scan.tif", "hash": "h3"},
        ]

        self.visible_indices = [0, 1, 2]
        self.mock_session_manager.asset_model = MagicMock()
        self.mock_session_manager.asset_model.visible_actual_indices_ordered.side_effect = lambda: list(self.visible_indices)

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)

        self.emitted = []
        self.controller.normalization_requested.connect(self.emitted.append)

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_analysis_respects_filter(self):
        self.visible_indices = [0, 1]  # only IMG_*
        with patch("negpy.desktop.controller.QMessageBox") as mock_box:
            mock_box.StandardButton.Yes = 1
            mock_box.question.return_value = 1
            self.controller.request_batch_normalization()
        self.assertEqual(len(self.emitted), 1)
        self.assertEqual([f.file_info["name"] for f in self.emitted[0].frames], ["IMG_0001.cr2", "IMG_0002.cr2"])

    def test_analysis_zero_matches_does_not_dispatch(self):
        self.visible_indices = []
        with patch("negpy.desktop.controller.QMessageBox") as mock_box:
            mock_box.StandardButton.Yes = 1
            mock_box.question.return_value = 1
            self.controller.request_batch_normalization()
        self.assertEqual(self.emitted, [])


class TestContactSheetOutputDir(unittest.TestCase):
    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        self.mock_session_manager.asset_model = MagicMock()

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)

        self.visible_files = [
            {"name": "a.cr2", "path": "/rolls/frame/a.cr2", "hash": "h1"},
            {"name": "b.cr2", "path": "/rolls/frame/b.cr2", "hash": "h2"},
        ]

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_custom_path_wins_over_export_destination(self):
        export = ExportConfig(
            contact_sheet_output_path="/custom/contact",
            output_mode=ExportPresetOutputMode.SAME_AS_SOURCE,
        )
        self.controller.state.config = replace(self.controller.state.config, export=export)
        out = self.controller._contact_sheet_output_dir(self.visible_files)
        self.assertEqual(out, "/custom/contact")

    def test_empty_path_uses_source_folder_when_same_as_source(self):
        export = ExportConfig(
            contact_sheet_output_path="",
            output_mode=ExportPresetOutputMode.SAME_AS_SOURCE,
        )
        self.controller.state.config = replace(self.controller.state.config, export=export)
        out = self.controller._contact_sheet_output_dir(self.visible_files)
        self.assertEqual(out, "/rolls/frame")

    def test_empty_path_uses_export_path_when_absolute(self):
        export = ExportConfig(
            contact_sheet_output_path="",
            output_mode=ExportPresetOutputMode.ABSOLUTE,
            export_path="/home/user/NegPy/export",
        )
        self.controller.state.config = replace(self.controller.state.config, export=export)
        out = self.controller._contact_sheet_output_dir(self.visible_files)
        self.assertEqual(out, "/home/user/NegPy/export")

    def test_whitespace_only_path_falls_back_to_export_rules(self):
        export = ExportConfig(
            contact_sheet_output_path="   ",
            output_mode=ExportPresetOutputMode.SAME_AS_SOURCE,
        )
        self.controller.state.config = replace(self.controller.state.config, export=export)
        out = self.controller._contact_sheet_output_dir(self.visible_files)
        self.assertEqual(out, "/rolls/frame")

    def _dict_repo(self) -> None:
        store: dict = {}
        self.controller.session.repo.get_global_setting.side_effect = lambda key, default=None: store.get(key, default)
        self.controller.session.repo.save_global_setting.side_effect = lambda key, value: store.__setitem__(key, value)

    def test_subfolder_of_source_redirects_a_virtual_roll_with_no_folder(self):
        """A virtual roll's files share no folder to build a subfolder under, so
        Subfolder of Source gathers them under the data folder instead."""
        from negpy.kernel.system.paths import get_default_user_dir
        from negpy.services.assets.rolls import create_virtual_roll

        self._dict_repo()
        roll_id = create_virtual_roll(self.controller.session.repo, "Portra 400", ["/a.nef", "/b.nef"])
        self.controller.state.active_roll_id = roll_id
        export = ExportConfig(output_mode=ExportPresetOutputMode.SUBFOLDER_OF_SOURCE, output_subfolder="export")
        self.controller.state.config = replace(self.controller.state.config, export=export)

        out = self.controller._contact_sheet_output_dir(self.visible_files)

        self.assertEqual(out, os.path.join(get_default_user_dir(), "Portra 400", "export"))

    def test_subfolder_of_source_keeps_per_file_resolution_for_a_folder_roll(self):
        """A folder roll's own folder is already the roll's folder, so it resolves
        exactly as it would with no roll at all."""
        from negpy.services.assets.rolls import recognize_folder

        self._dict_repo()
        roll_id = recognize_folder(self.controller.session.repo, "/rolls/frame")
        self.controller.state.active_roll_id = roll_id
        export = ExportConfig(output_mode=ExportPresetOutputMode.SUBFOLDER_OF_SOURCE, output_subfolder="export")
        self.controller.state.config = replace(self.controller.state.config, export=export)

        out = self.controller._contact_sheet_output_dir(self.visible_files)

        self.assertEqual(out, os.path.join("/rolls/frame", "export"))

    def test_subfolder_of_source_with_no_active_roll_resolves_per_file(self):
        self._dict_repo()
        export = ExportConfig(output_mode=ExportPresetOutputMode.SUBFOLDER_OF_SOURCE, output_subfolder="export")
        self.controller.state.config = replace(self.controller.state.config, export=export)

        out = self.controller._contact_sheet_output_dir(self.visible_files)

        self.assertEqual(out, os.path.join("/rolls/frame", "export"))

    def test_virtual_roll_redirect_warns_once(self):
        from negpy.services.assets.rolls import create_virtual_roll

        self._dict_repo()
        roll_id = create_virtual_roll(self.controller.session.repo, "Portra 400", ["/a.nef", "/b.nef"])
        self.controller.state.active_roll_id = roll_id
        export = ExportConfig(output_mode=ExportPresetOutputMode.SUBFOLDER_OF_SOURCE, output_subfolder="export")
        self.controller.state.config = replace(self.controller.state.config, export=export)

        msgs = []
        self.controller.status_message_requested.connect(lambda text, _ms, kind: msgs.append((text, kind)))
        self.controller._contact_sheet_output_dir(self.visible_files)

        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0][1], "warning")
        self.assertIn("Portra 400", msgs[0][0])

    def test_folder_roll_does_not_warn(self):
        from negpy.services.assets.rolls import recognize_folder

        self._dict_repo()
        roll_id = recognize_folder(self.controller.session.repo, "/rolls/frame")
        self.controller.state.active_roll_id = roll_id
        export = ExportConfig(output_mode=ExportPresetOutputMode.SUBFOLDER_OF_SOURCE, output_subfolder="export")
        self.controller.state.config = replace(self.controller.state.config, export=export)

        msgs = []
        self.controller.status_message_requested.connect(lambda text, _ms, kind: msgs.append((text, kind)))
        self.controller._contact_sheet_output_dir(self.visible_files)

        self.assertEqual(msgs, [])


class TestRetouchPersistence(unittest.TestCase):
    """Regression: heal/scratch edits must persist=True like every other discrete
    canvas action (e.g. _handle_wb_pick) — otherwise select_file's "save before
    switching" guard (gated on the dirty flag persist=True sets) skips them, and
    switching files silently discards heals that were never written to disk."""

    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)
        self.controller.request_render = MagicMock()

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def _stroke(self):
        return ([[0.1, 0.1]], 5.0, 0.01, -0.01)

    def test_commit_heal_stroke_via_dust_pick_persists(self):
        self.controller.state.active_tool = ToolMode.DUST_PICK
        self.controller.state.last_metrics["uv_grid"] = MagicMock()
        with patch("negpy.desktop.controller.CoordinateMapping") as mock_map:
            mock_map.map_click_to_raw.return_value = (0.5, 0.5)
            self.controller.handle_canvas_clicked(0.5, 0.5)
        self.mock_session_manager.update_config.assert_called_once()
        self.assertTrue(self.mock_session_manager.update_config.call_args.kwargs.get("persist"))
        saved = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(len(saved.retouch.manual_heal_strokes), 1)

    def test_handle_heal_stroke_completed_persists(self):
        self.controller.state.last_metrics["uv_grid"] = MagicMock()
        with patch("negpy.desktop.controller.CoordinateMapping") as mock_map:
            mock_map.map_click_to_raw.return_value = (0.5, 0.5)
            self.controller.handle_heal_stroke_completed([(0.4, 0.4), (0.6, 0.6)])
        self.assertTrue(self.mock_session_manager.update_config.call_args.kwargs.get("persist"))

    def test_undo_last_retouch_persists(self):
        retouch = replace(self.controller.state.config.retouch, manual_heal_strokes=[self._stroke()])
        self.controller.state.config = replace(self.controller.state.config, retouch=retouch)

        self.controller.undo_last_retouch()

        self.assertTrue(self.mock_session_manager.update_config.call_args.kwargs.get("persist"))
        saved = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(saved.retouch.manual_heal_strokes, [])

    def test_delete_heal_persists(self):
        retouch = replace(self.controller.state.config.retouch, manual_heal_strokes=[self._stroke(), self._stroke()])
        self.controller.state.config = replace(self.controller.state.config, retouch=retouch)

        self.controller.delete_heal("stroke", 0)

        self.assertTrue(self.mock_session_manager.update_config.call_args.kwargs.get("persist"))
        saved = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(len(saved.retouch.manual_heal_strokes), 1)

    def test_clear_retouch_persists(self):
        retouch = replace(self.controller.state.config.retouch, manual_heal_strokes=[self._stroke()])
        self.controller.state.config = replace(self.controller.state.config, retouch=retouch)

        with patch("negpy.desktop.view.confirm.confirm_clear_heals", return_value=True):
            self.controller.clear_retouch()

        self.assertTrue(self.mock_session_manager.update_config.call_args.kwargs.get("persist"))
        saved = self.mock_session_manager.update_config.call_args.args[0]
        self.assertEqual(saved.retouch.manual_heal_strokes, [])

    def test_cycle_dust_overlay_with_ir(self):
        self.controller.state.has_ir = True
        self.controller.state.dust_overlay_mode = "off"
        seq = []
        for _ in range(5):
            self.controller.cycle_dust_overlay()
            seq.append(self.controller.state.dust_overlay_mode)
        self.assertEqual(seq, ["marked", "ir", "off", "marked", "ir"])

    def test_cycle_dust_overlay_skips_ir_without_ir(self):
        self.controller.state.has_ir = False
        self.controller.state.dust_overlay_mode = "off"
        seq = []
        for _ in range(4):
            self.controller.cycle_dust_overlay()
            seq.append(self.controller.state.dust_overlay_mode)
        self.assertEqual(seq, ["marked", "off", "marked", "off"])

    def test_cycle_dust_overlay_from_ir_when_ir_lost(self):
        # Mode was "ir" but the new frame has none: cycling treats it as off.
        self.controller.state.has_ir = False
        self.controller.state.dust_overlay_mode = "ir"
        self.controller.cycle_dust_overlay()
        self.assertEqual(self.controller.state.dust_overlay_mode, "marked")


if __name__ == "__main__":
    unittest.main()


class TestDisplayTransformParams(unittest.TestCase):
    """The canvas and the filmstrip thumbnail must derive their display transform
    from the same place. When a soft proof is active the render worker has already
    baked source->output->monitor into the buffer, so the transform has to be a
    no-op; treating that buffer as working-space re-applies ProPhoto->sRGB and the
    thumbnail comes out visibly oversaturated next to the canvas."""

    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)
        self.controller.state.monitor_icc_bytes = b"fake-monitor-profile"

    def tearDown(self):
        import gc

        # Same teardown as TestAppController: the controller owns live QThreads and
        # letting it be collected while they run crashes the interpreter.
        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_proof_active_still_reports_the_working_space(self):
        """The proof is not baked into the buffer — it rides the display LUT.

        Reporting sRGB here (as the baked-buffer era did) would skip the
        working→display conversion the render still needs.
        """
        self.controller.state.soft_proof_enabled = True
        cs, monitor, proof = self.controller.display_transform_params()
        self.assertEqual(cs, self.controller.state.workspace_color_space)
        self.assertEqual(monitor, b"fake-monitor-profile")
        self.assertIsNotNone(proof)

    def test_proof_inactive_converts_from_the_working_space(self):
        self.controller.proof_profiles = lambda process=None: None
        cs, monitor, proof = self.controller.display_transform_params()
        self.assertEqual(cs, self.controller.state.workspace_color_space)
        self.assertEqual(monitor, b"fake-monitor-profile")
        self.assertIsNone(proof)

    def test_splash_buffer_is_treated_as_srgb(self):
        self.controller.proof_profiles = lambda process=None: None
        cs, monitor, proof = self.controller.display_transform_params(splash=True)
        self.assertEqual(cs, ColorSpace.SRGB.value)
        self.assertEqual(monitor, b"fake-monitor-profile")
        self.assertIsNone(proof)

    def test_proof_profiles_follows_the_soft_proof_toggle(self):
        self.controller.state.soft_proof_enabled = False
        self.controller.state.config = replace(
            self.controller.state.config,
            process=replace(self.controller.state.config.process, narrowband_scan=False),
        )
        self.assertIsNone(self.controller.proof_profiles())
        self.controller.state.soft_proof_enabled = True
        self.assertIsNotNone(self.controller.proof_profiles())

    def test_narrowband_supplies_an_input_profile_with_the_toggle_off(self):
        """Narrowband Scan proofs through its own input profile regardless."""
        self.controller.state.soft_proof_enabled = False
        self.controller.state.config = replace(
            self.controller.state.config,
            process=replace(self.controller.state.config.process, narrowband_scan=True),
        )
        proof = self.controller.proof_profiles()
        self.assertIsNotNone(proof)
        self.assertTrue(proof[0], "narrowband must supply the input profile")
        self.assertIsNone(proof[1], "the output profile stays gated on the toggle")

    def test_thumbnail_task_carries_the_same_params_as_the_canvas(self):
        """The actual regression: the thumbnail used to hardcode the working space."""
        import numpy as np

        self.controller.state.soft_proof_enabled = True
        state = self.controller.state
        state.uploaded_files = [{"name": "frame.cr2", "path": "/tmp/frame.cr2", "hash": "hash-1"}]
        state.selected_file_idx = 0
        state.current_file_path = "/tmp/frame.cr2"
        state.current_file_hash = "hash-1"
        state.last_metrics = {"base_positive": np.zeros((4, 4, 3), dtype=np.float32), "source_hash": "hash-1"}

        emitted = []
        # Drop the real worker connection first: emitting would otherwise hand the
        # buffer to the thumbnail QThread, which then races this test's teardown.
        try:
            self.controller.thumbnail_update_requested.disconnect()
        except TypeError:
            pass
        self.controller.thumbnail_update_requested.connect(emitted.append)
        self.controller._update_thumbnail_from_state()

        self.assertEqual(len(emitted), 1)
        task = emitted[0]
        self.assertEqual(
            (task.color_space, task.monitor_icc_bytes, task.proof),
            self.controller.display_transform_params(),
        )
        # The proof must reach the filmstrip too, or it shows an unproofed frame
        # next to a proofed canvas.
        self.assertIsNotNone(task.proof)

    def test_unproofed_working_space_buffer_keeps_the_conversion(self):
        """The negative peek wants the working->display conversion without the paper
        simulation. Reporting sRGB to get rid of the proof would lose both."""
        self.controller.state.soft_proof_enabled = True
        cs, monitor, proof = self.controller.display_transform_params(proofed=False)
        self.assertEqual(cs, self.controller.state.workspace_color_space)
        self.assertEqual(monitor, b"fake-monitor-profile")
        self.assertIsNone(proof)


class TestNegativePeekColor(unittest.TestCase):
    """The peek paints camera-native pixels, so it owns the camera matrix itself.

    Without it the buffer goes to the canvas as though it were already display RGB,
    which drains the film base: a C-41 mask reads far weaker than the file's own.
    """

    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    # A real decoder matrix (Nikon D3300), so the test fails on a plausible transform
    # rather than only on an artificial one.
    D3300 = [
        [0.6988000273704529, -0.13840000331401825, -0.0714000016450882],
        [-0.5630999803543091, 1.340999960899353, 0.24469999969005585],
        [-0.148499995470047, 0.22040000557899475, 0.7318000197410583],
    ]

    @staticmethod
    def _source():
        import numpy as np

        # An orange-mask film base: red passes, blue is held back. The top rows are the
        # bare light around the rebate, which is what the peek references itself to.
        img = np.full((8, 8, 3), 0.1, dtype=np.float32) * np.array([1.0, 0.45, 0.2], dtype=np.float32)
        img[:2, :, :] = 0.6
        return img

    def _paint(self, cam_xyz, camera_wb=None):
        state = self.controller.state
        state.preview_raw = self._source()
        state.original_res = (8, 8)
        state.preview_cam_xyz = cam_xyz
        state.preview_camera_wb = camera_wb
        self.controller._paint_negative_peek()
        return state.last_metrics

    def test_the_peek_applies_the_camera_matrix(self):
        import numpy as np

        from negpy.features.process.capture_color import apply_camera_matrix, camera_to_working_matrix, lightbox_level
        from negpy.kernel.image.logic import working_oetf_encode

        metrics = self._paint(self.D3300)
        painted = metrics["base_positive"]

        source = self.controller.state.preview_raw
        matrix = camera_to_working_matrix(self.D3300, None)
        expected = working_oetf_encode(apply_camera_matrix(source, matrix) * lightbox_level(source, matrix))
        np.testing.assert_allclose(painted, expected, atol=1e-5)
        # And it is not the un-matrixed buffer, which is what shipped the weak mask.
        self.assertFalse(np.allclose(painted, working_oetf_encode(source * lightbox_level(source, None)), atol=1e-3))

    def test_the_peek_is_color_managed_but_never_proofed(self):
        metrics = self._paint(self.D3300)
        self.assertFalse(metrics["splash"], "a camera-matrixed buffer is in the working space")
        self.assertFalse(metrics["proof"], "the peek shows the scan, not a print")

    def test_a_source_with_no_matrix_takes_the_level_alone(self):
        """Scanner TIFF and JPEG carry no camera matrix; they are already profiled, so the
        display level is the only thing between the buffer and the canvas."""
        import numpy as np

        from negpy.features.process.capture_color import lightbox_level
        from negpy.kernel.image.logic import working_oetf_encode

        metrics = self._paint(None)
        source = self.controller.state.preview_raw
        np.testing.assert_allclose(metrics["base_positive"], working_oetf_encode(source * lightbox_level(source, None)), atol=1e-6)

    def test_linear_raw_folds_the_multipliers_back_in(self):
        """The decode's white balance must not change what the mask looks like, or
        Linear RAW silently restyles the negative instead of leaving it alone."""
        import numpy as np

        wb = [1.891, 1.0, 1.578]
        with_wb = np.array(self._paint(self.D3300, camera_wb=wb)["base_positive"])

        state = self.controller.state
        state.config = replace(state.config, process=replace(state.config.process, linear_raw=True))
        # The Linear RAW decode skips the multipliers, so its buffer is the unbalanced one.
        state.preview_raw = (self._source() / np.array(wb, dtype=np.float32)).astype(np.float32)
        state.original_res = (8, 8)
        state.preview_cam_xyz = self.D3300
        state.preview_camera_wb = wb
        self.controller._paint_negative_peek()
        without_wb = np.array(state.last_metrics["base_positive"])

        np.testing.assert_allclose(with_wb, without_wb, atol=1e-5)

    def test_a_narrowband_capture_folds_the_multipliers_the_render_path_refuses(self):
        """should_fold_camera_wb refuses them on narrowband, where no scene white balance
        exists to reconstruct. The peek folds them anyway: it only has to show the film the
        way every raw viewer does, and unbalanced sensor RGB renders an orange mask green."""
        import numpy as np

        from negpy.features.process.capture_color import apply_camera_matrix, camera_to_working_matrix, lightbox_level
        from negpy.kernel.image.logic import working_oetf_encode

        state = self.controller.state
        state.config = replace(state.config, process=replace(state.config.process, linear_raw=True, narrowband_scan=True))
        painted = self._paint(self.D3300, camera_wb=[1.891, 1.0, 1.578])["base_positive"]

        source = state.preview_raw
        matrix = camera_to_working_matrix(self.D3300, [1.891, 1.0, 1.578])
        expected = working_oetf_encode(apply_camera_matrix(source, matrix) * lightbox_level(source, matrix))
        np.testing.assert_allclose(painted, expected, atol=1e-5)

    def test_the_peek_clears_a_stale_interactive_flag(self):
        """Flat Peek renders with readback_metrics=False, which tags its metrics
        interactive; switching straight to Negative Peek must not inherit that flag,
        or right_panel's analysis-chart refresh mistakes the settled peek frame for a
        mid-gesture one and never re-syncs the histogram."""
        self.controller.state.last_metrics["interactive"] = True
        metrics = self._paint(self.D3300)
        self.assertFalse(metrics["interactive"])


class TestEmbeddedPeek(unittest.TestCase):
    """The reference view: the camera's own JPEG of the capture, not NegPy's decode."""

    def setUp(self):
        import numpy as np

        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)
        self.controller.state.preview_raw = np.empty((8, 8, 3), dtype=np.float32)
        self.controller.state.current_file_path = "/scans/frame.nef"

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    @staticmethod
    def _preview():
        import numpy as np

        return np.linspace(0.0, 1.0, 6 * 8 * 3, dtype=np.float32).reshape(6, 8, 3)

    def test_it_paints_the_embedded_preview_as_the_srgb_it_is(self):
        """No working OETF and `splash` set: the camera already encoded this buffer, and the
        curve it put there is what the view exists to show."""
        import numpy as np

        preview = self._preview()
        with patch("negpy.desktop.controller.PreviewManager.try_splash_preview", return_value=(preview, (8, 6))):
            self.controller.toggle_embedded_peek(force=True)

        self.assertTrue(self.controller.state.embedded_peek)
        metrics = self.controller.state.last_metrics
        np.testing.assert_allclose(metrics["base_positive"], preview)
        self.assertTrue(metrics["splash"])
        self.assertFalse(metrics["proof"])

    def test_it_is_read_once_and_kept_for_the_frame(self):
        with patch("negpy.desktop.controller.PreviewManager.try_splash_preview", return_value=(self._preview(), (8, 6))) as read:
            self.controller.toggle_embedded_peek(force=True)
            self.controller.toggle_embedded_peek(force=False)
            self.controller.toggle_embedded_peek(force=True)
        read.assert_called_once()

    def test_a_source_with_no_preview_says_so_and_stays_off(self):
        seen: list = []
        self.controller.embedded_peek_changed.connect(seen.append)
        with patch("negpy.desktop.controller.PreviewManager.try_splash_preview", return_value=None):
            self.controller.toggle_embedded_peek(force=True)
        self.assertFalse(self.controller.state.embedded_peek)
        self.assertEqual(seen, [False], "the menu item must not stay checked on a file with no preview")

    def test_the_peeks_are_mutually_exclusive(self):
        with patch("negpy.desktop.controller.PreviewManager.try_splash_preview", return_value=(self._preview(), (8, 6))):
            self.controller.state.negative_peek = True
            self.controller.toggle_embedded_peek(force=True)
            self.assertFalse(self.controller.state.negative_peek)

            self.controller.toggle_negative_peek(force=True)
            self.assertFalse(self.controller.state.embedded_peek)

    def test_leaving_it_re_renders_the_edit(self):
        self.controller.state.embedded_peek = True
        with patch.object(self.controller, "request_render") as rr:
            self.controller.toggle_embedded_peek(force=False)
        self.assertFalse(self.controller.state.embedded_peek)
        rr.assert_called_once()

    def test_it_needs_a_loaded_source(self):
        self.controller.state.preview_raw = None
        self.controller.toggle_embedded_peek(force=True)
        self.assertFalse(self.controller.state.embedded_peek)


class TestCompareFlatPeekInteraction(unittest.TestCase):
    """Before/After and flat-peek are mutually exclusive overlays; a geometry op must
    keep whichever one is active instead of dropping the user back to the plain edit."""

    def setUp(self):
        import numpy as np

        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)
        # toggle_compare / rerender_active_view early-return without a preview buffer.
        self.controller.state.preview_raw = np.empty((8, 8, 3), dtype=np.float32)

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_enabling_compare_clears_an_active_flat_peek(self):
        """Regression: turning on Before/After while flat-peek was on left flat-peek's
        toggle lit though the compare baseline was what actually rendered."""
        self.controller.state.flat_peek = True
        seen: list = []
        self.controller.flat_peek_changed.connect(seen.append)
        with patch.object(self.controller, "request_render"):
            self.controller.toggle_compare()
        self.assertTrue(self.controller.state.compare_mode)
        self.assertFalse(self.controller.state.flat_peek)
        self.assertIn(False, seen)

    def test_rerender_active_view_keeps_the_compare_split_on_a_plain_render(self):
        self.controller.state.compare_mode = True
        with patch.object(self.controller, "request_render") as rr:
            self.controller.rerender_active_view()
        _, kwargs = rr.call_args
        # The split renders the edit as usual; the baseline half re-captures on its own
        # once the geometry key moves.
        self.assertIsNone(kwargs.get("config_override"))
        self.assertTrue(self.controller.state.compare_mode)

    def test_rerender_active_view_re_renders_the_flat_master(self):
        from negpy.domain.models import flat_master_config

        self.controller.state.flat_peek = True
        with patch.object(self.controller, "request_render") as rr:
            self.controller.rerender_active_view()
        _, kwargs = rr.call_args
        self.assertEqual(kwargs.get("config_override"), flat_master_config(self.controller.state.config))

    def test_rerender_active_view_is_a_plain_render_when_no_overlay(self):
        with patch.object(self.controller, "request_render") as rr:
            self.controller.rerender_active_view()
        _, kwargs = rr.call_args
        self.assertIsNone(kwargs.get("config_override"))

    def test_negative_peek_paints_the_source_with_only_the_level_and_the_oetf(self):
        import numpy as np

        from negpy.features.process.capture_color import lightbox_level
        from negpy.kernel.image.logic import working_oetf_encode

        source = np.linspace(0.0, 1.0, 8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)
        self.controller.state.preview_raw = source
        painted: list = []
        self.controller.image_updated.connect(lambda: painted.append(True))

        self.controller.toggle_negative_peek(force=True)

        self.assertTrue(self.controller.state.negative_peek)
        self.assertTrue(painted)
        metrics = self.controller.state.last_metrics
        # No camera matrix on this source, so the display level and the encode are
        # all that separate it from the buffer the loader read. See TestNegativePeekColor
        # for the camera-native path.
        np.testing.assert_allclose(metrics["base_positive"], working_oetf_encode(source * lightbox_level(source, None)))
        # Working space, so the display conversion runs; the proof does not.
        self.assertFalse(metrics["splash"])
        self.assertFalse(metrics["proof"])

    def test_leaving_the_negative_peek_re_renders_the_edit(self):
        self.controller.state.negative_peek = True
        with patch.object(self.controller, "request_render") as rr:
            self.controller.toggle_negative_peek(force=False)
        self.assertFalse(self.controller.state.negative_peek)
        rr.assert_called_once()

    def test_negative_peek_needs_a_loaded_source(self):
        self.controller.state.preview_raw = None
        seen: list = []
        self.controller.negative_peek_changed.connect(seen.append)
        self.controller.toggle_negative_peek(force=True)
        self.assertFalse(self.controller.state.negative_peek)
        self.assertEqual(seen, [])

    def test_the_two_peeks_are_mutually_exclusive(self):
        self.controller.state.flat_peek = True
        self.controller.toggle_negative_peek(force=True)
        self.assertTrue(self.controller.state.negative_peek)
        self.assertFalse(self.controller.state.flat_peek)

        with patch.object(self.controller, "request_render"):
            self.controller.toggle_flat_peek(force=True)
        self.assertTrue(self.controller.state.flat_peek)
        self.assertFalse(self.controller.state.negative_peek)

    def test_enabling_compare_clears_an_active_negative_peek(self):
        self.controller.state.negative_peek = True
        seen: list = []
        self.controller.negative_peek_changed.connect(seen.append)
        with patch.object(self.controller, "request_render"):
            self.controller.toggle_compare()
        self.assertTrue(self.controller.state.compare_mode)
        self.assertFalse(self.controller.state.negative_peek)
        self.assertIn(False, seen)

    def test_the_negative_peek_takes_the_geometry(self):
        import numpy as np
        from dataclasses import replace

        # Landscape, so a quarter turn is visible in the shape alone.
        source = np.random.default_rng(0).random((6, 10, 3), dtype=np.float32)
        self.controller.state.preview_raw = source
        geo = replace(self.controller.state.config.geometry, rotation=1, crop_rect=(0.0, 0.0, 0.5, 1.0))
        self.controller.state.config = replace(self.controller.state.config, geometry=geo)

        self.controller.toggle_negative_peek(force=True)

        painted = self.controller.state.last_metrics["base_positive"]
        # Quarter turn swaps the axes, then the crop keeps the left half of the width.
        self.assertEqual(painted.shape, (10, 3, 3))
        # No border stage ran, so nothing may claim the frame is inset.
        self.assertIsNone(self.controller.state.last_metrics["content_rect"])

    def test_the_crop_tool_peeks_the_uncropped_frame(self):
        import numpy as np
        from dataclasses import replace

        from negpy.desktop.session import ToolMode

        source = np.zeros((6, 10, 3), dtype=np.float32)
        self.controller.state.preview_raw = source
        geo = replace(self.controller.state.config.geometry, crop_rect=(0.0, 0.0, 0.5, 1.0))
        self.controller.state.config = replace(self.controller.state.config, geometry=geo)
        self.controller.state.active_tool = ToolMode.CROP_MANUAL

        self.controller.toggle_negative_peek(force=True)

        # Framing a crop against a pre-cropped frame would be impossible.
        self.assertEqual(self.controller.state.last_metrics["base_positive"].shape, (6, 10, 3))

    def test_any_plain_render_leaves_the_negative_peek(self):
        self.controller.state.negative_peek = True
        seen: list = []
        self.controller.negative_peek_changed.connect(seen.append)
        with patch.object(self.controller, "_dispatch_pending_render"):
            self.controller.request_render()
        self.assertFalse(self.controller.state.negative_peek)
        self.assertIn(False, seen)

    def test_rerender_active_view_keeps_the_negative_peek(self):
        import numpy as np

        self.controller.state.preview_raw = np.zeros((8, 8, 3), dtype=np.float32)
        self.controller.state.negative_peek = True
        with patch.object(self.controller, "request_render") as rr:
            self.controller.rerender_active_view()
        # A geometry op must not drop the peek, and the peek is not a render.
        rr.assert_not_called()
        self.assertTrue(self.controller.state.negative_peek)
        self.assertIn("base_positive", self.controller.state.last_metrics)


class TestClearThumbnailCache(unittest.TestCase):
    """'Clear Thumbnails' has to drop the disk cache and the in-memory icons, then refill."""

    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        self.mock_session_manager.asset_model = MagicMock()

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            self.controller = AppController(self.mock_session_manager)
        self.controller.asset_store = MagicMock()

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_clear_wipes_disk_and_memory(self):
        state = self.mock_session_manager.state
        state.thumbnails["a"] = object()
        state.rendered_thumbnails.add("a")
        self.controller.generate_missing_thumbnails = MagicMock()

        self.controller.clear_thumbnail_cache()

        self.controller.asset_store.clear_thumbnails.assert_called_once_with()
        self.assertEqual(state.thumbnails, {})
        self.assertEqual(state.rendered_thumbnails, set())
        self.mock_session_manager.asset_model.refresh.assert_called_once_with()
        self.controller.generate_missing_thumbnails.assert_called_once_with()

    def test_regeneration_runs_against_an_emptied_cache(self):
        # generate_missing_thumbnails only enqueues names absent from state.thumbnails,
        # so clearing has to happen first or nothing comes back.
        state = self.mock_session_manager.state
        state.uploaded_files = [{"name": "a", "path": "/a.dng", "hash": "h1"}]
        state.thumbnails["a"] = object()
        seen = []
        self.controller.generate_missing_thumbnails = MagicMock(side_effect=lambda: seen.append(dict(state.thumbnails)))

        self.controller.clear_thumbnail_cache()

        self.assertEqual(seen, [{}])


class TestSemanticIndexing(unittest.TestCase):
    """generate_missing_embeddings: the CLIP indexing pass that follows a thumbnail
    batch, and the handlers that apply its results."""

    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        self.mock_session_manager.asset_model = MagicMock()

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            self.controller = AppController(self.mock_session_manager)

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_embed_search_query_returns_none_without_a_downloaded_model(self):
        with patch("negpy.desktop.controller.semantic_model.clip_model_ready", return_value=False):
            self.assertIsNone(self.controller.embed_search_query("cats"))

    def test_embed_search_query_returns_none_for_empty_text(self):
        self.assertIsNone(self.controller.embed_search_query(""))

    def test_embed_search_query_reuses_one_model_instance(self):
        vector = object()
        with (
            patch("negpy.desktop.controller.semantic_model.clip_model_ready", return_value=True),
            patch("negpy.desktop.controller.semantic_model.ClipModel") as model_cls,
        ):
            model_cls.return_value.embed_text.return_value = vector
            first = self.controller.embed_search_query("cats")
            second = self.controller.embed_search_query("dogs")

        self.assertIs(first, vector)
        self.assertIs(second, vector)
        model_cls.assert_called_once_with()

    def test_noop_when_the_feature_is_off(self):
        state = self.mock_session_manager.state
        state.uploaded_files = [{"name": "a", "path": "/a.dng", "hash": "h1"}]
        state.semantic_search_enabled = False

        self.controller.generate_missing_embeddings()

        self.mock_session_manager.repo.load_embeddings_for.assert_not_called()

    def test_noop_when_the_model_is_not_downloaded(self):
        state = self.mock_session_manager.state
        state.uploaded_files = [{"name": "a", "path": "/a.dng", "hash": "h1"}]
        state.semantic_search_enabled = True

        with patch("negpy.desktop.controller.semantic_model.clip_model_ready", return_value=False):
            self.controller.generate_missing_embeddings()

        self.mock_session_manager.repo.load_embeddings_for.assert_not_called()

    def test_only_files_missing_from_the_db_are_requested(self):
        state = self.mock_session_manager.state
        state.semantic_search_enabled = True
        state.uploaded_files = [
            {"name": "a", "path": "/a.dng", "hash": "h1"},
            {"name": "b", "path": "/b.dng", "hash": "h2"},
        ]
        self.mock_session_manager.repo.load_embeddings_for.return_value = {"h1": object()}
        self.controller.embedding_requested = MagicMock()

        with patch("negpy.desktop.controller.semantic_model.clip_model_ready", return_value=True):
            self.controller.generate_missing_embeddings()

        self.assertIn("h1", state.embeddings)
        (requested,), _ = self.controller.embedding_requested.emit.call_args
        self.assertEqual([f["hash"] for f in requested], ["h2"])

    def test_nothing_missing_is_a_noop_after_the_db_check(self):
        state = self.mock_session_manager.state
        state.semantic_search_enabled = True
        state.uploaded_files = [{"name": "a", "path": "/a.dng", "hash": "h1"}]
        self.mock_session_manager.repo.load_embeddings_for.return_value = {"h1": object()}
        self.controller.embedding_requested = MagicMock()

        with patch("negpy.desktop.controller.semantic_model.clip_model_ready", return_value=True):
            self.controller.generate_missing_embeddings()

        self.controller.embedding_requested.emit.assert_not_called()

    def test_refreshes_the_model_even_when_nothing_was_missing(self):
        """A whole-library search hands off files that are, by construction, already
        indexed -- so an in-session semantic filter left active from before the hand-off
        would otherwise never see these newly cached embeddings and would exclude every
        one of them (absent-from-embeddings means excluded, not zero-scored)."""
        state = self.mock_session_manager.state
        state.semantic_search_enabled = True
        state.uploaded_files = [{"name": "a", "path": "/a.dng", "hash": "h1"}]
        self.mock_session_manager.repo.load_embeddings_for.return_value = {"h1": object()}

        with patch("negpy.desktop.controller.semantic_model.clip_model_ready", return_value=True):
            self.controller.generate_missing_embeddings()

        self.mock_session_manager.asset_model.refresh.assert_called_once_with()

    def test_the_thumbnail_queue_going_idle_triggers_indexing(self):
        """Indexing follows the thumbnails so each file's preview is already on disk;
        the queue's own idle transition is what says they are all in."""
        state = self.mock_session_manager.state
        state.semantic_search_enabled = True
        state.uploaded_files = [{"name": "a", "path": "/a.dng", "hash": "h1"}]
        self.controller.generate_missing_embeddings = MagicMock()
        self.controller._thumbnail_queue_active = True

        self.controller._on_thumbnail_activity("")

        self.controller.generate_missing_embeddings.assert_called_once_with()

    def test_an_already_idle_thumbnail_queue_does_not_trigger_indexing(self):
        self.controller.generate_missing_embeddings = MagicMock()
        self.controller._thumbnail_queue_active = False

        self.controller._on_thumbnail_activity("")

        self.controller.generate_missing_embeddings.assert_not_called()

    def test_apply_embeddings_updates_state_and_refreshes_the_model(self):
        vector = object()
        self.controller.state.uploaded_files = [{"name": "a", "path": "/a.dng", "hash": "h1"}]

        self.controller._apply_embeddings({"h1": vector})

        self.assertEqual(self.controller.state.embeddings["h1"], vector)
        self.mock_session_manager.asset_model.refresh.assert_called_once_with()

    def test_apply_embeddings_ignores_a_file_not_in_the_live_session(self):
        """A library-wide indexing pass's results belong in the DB, already saved per
        file as they land -- not in the current session's in-memory cache."""
        self.controller._apply_embeddings({"not_loaded": object()})

        self.assertNotIn("not_loaded", self.controller.state.embeddings)

    def test_finished_releases_the_batch_lane(self):
        self.controller._begin_batch("embeddings", "Indexing for search by meaning", abortable=False)

        self.controller._on_embeddings_finished({"h1": object()})

        self.assertIsNone(self.controller._active_batch)

    def test_batch_error_releases_the_lane_without_raising(self):
        self.controller._begin_batch("embeddings", "Indexing for search by meaning", abortable=False)

        self.controller._on_embedding_batch_error("boom")

        self.assertIsNone(self.controller._active_batch)


class TestRotateThumbnails(unittest.TestCase):
    """A batch rotate turns other selected frames' saved geometry without opening
    them, so their filmstrip thumbnail has to turn too, in place, disk cache
    included — generate_missing_thumbnails would re-derive from the source instead
    and never see the rotation."""

    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        self.mock_session_manager.asset_model = MagicMock()

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            self.controller = AppController(self.mock_session_manager)
        self.controller.asset_store = MagicMock()

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    @staticmethod
    def _corner_icon(w=4, h=2, corner=(255, 0, 0)):
        """A QIcon whose (w-1, 0) pixel is `corner`, everything else black."""
        from PyQt6.QtGui import QColor, QImage

        img = QImage(w, h, QImage.Format.Format_RGB32)
        img.fill(0)
        img.setPixelColor(w - 1, 0, QColor(*corner))
        return QIcon(QPixmap.fromImage(img))

    @staticmethod
    def _icon_pixel(icon, x, y):
        img = icon.pixmap(icon.availableSizes()[0]).toImage()
        c = img.pixelColor(x, y)
        return (c.red(), c.green(), c.blue())

    def test_rotates_memory_icon_and_disk_cache_independently(self):
        # Different markers in each store: memory (canvas-rendered) can be ahead of
        # disk (persisted lazily), so deriving one from the other would be wrong.
        self.mock_session_manager.state.thumbnails["hash1-v3"] = self._corner_icon(corner=(255, 0, 0))
        disk_cached = Image.new("RGB", (4, 2), (0, 0, 0))
        disk_cached.putpixel((3, 0), (0, 255, 0))
        self.controller.asset_store.get_thumbnail.return_value = disk_cached

        self.controller.rotate_thumbnails(["hash1-v3"], 1)

        # A quarter-turn CCW swaps the axes: the top-right marker lands top-left,
        # the same corner np.rot90(k=1) puts it at.
        self.assertEqual(self._icon_pixel(self.mock_session_manager.state.thumbnails["hash1-v3"], 0, 0), (255, 0, 0))
        saved_key, saved_img = self.controller.asset_store.save_thumbnail.call_args.args
        self.assertEqual(saved_key, "hash1-v3")
        self.assertEqual(saved_img.size, (2, 4))
        self.assertEqual(saved_img.getpixel((0, 0)), (0, 255, 0))
        self.mock_session_manager.asset_model.refresh.assert_called_once_with()

    def test_rotates_disk_only_when_no_memory_icon_yet(self):
        disk_cached = Image.new("RGB", (4, 2), (0, 0, 0))
        disk_cached.putpixel((3, 0), (0, 255, 0))
        self.controller.asset_store.get_thumbnail.return_value = disk_cached

        self.controller.rotate_thumbnails(["hash1-v3"], 1)

        self.controller.asset_store.save_thumbnail.assert_called_once()
        # No memory icon existed, so none is fabricated from the disk copy.
        self.assertNotIn("hash1-v3", self.mock_session_manager.state.thumbnails)
        self.mock_session_manager.asset_model.refresh.assert_not_called()

    def test_rotates_memory_only_when_no_disk_cache_yet(self):
        self.mock_session_manager.state.thumbnails["hash1-v3"] = self._corner_icon(corner=(255, 0, 0))
        self.controller.asset_store.get_thumbnail.return_value = None

        self.controller.rotate_thumbnails(["hash1-v3"], 1)

        self.assertEqual(self._icon_pixel(self.mock_session_manager.state.thumbnails["hash1-v3"], 0, 0), (255, 0, 0))
        self.controller.asset_store.save_thumbnail.assert_not_called()
        self.mock_session_manager.asset_model.refresh.assert_called_once_with()

    def test_skips_a_key_with_no_cached_thumbnail_anywhere_yet(self):
        self.controller.asset_store.get_thumbnail.return_value = None

        self.controller.rotate_thumbnails(["hash1-v3"], 1)

        self.controller.asset_store.save_thumbnail.assert_not_called()
        self.mock_session_manager.asset_model.refresh.assert_not_called()

    def test_zero_quarter_turns_is_a_noop(self):
        self.controller.rotate_thumbnails(["hash1-v3"], 0)

        self.controller.asset_store.get_thumbnail.assert_not_called()

    def test_rotate_clears_the_stale_flag_the_bulk_write_set(self):
        """rotate_selected_frames flags a batch-rotated frame stale (a DB write with no
        render); the turn above brings the cached bitmap into agreement with it, so the
        flag must not survive to show a spurious dot on an already-correct thumbnail."""
        self.mock_session_manager.state.stale_thumbnails.add("hash1-v3")

        self.controller.rotate_thumbnails(["hash1-v3"], 1)

        self.assertNotIn("hash1-v3", self.mock_session_manager.state.stale_thumbnails)

    def test_flip_mirrors_memory_icon_and_disk_cache_independently(self):
        self.mock_session_manager.state.thumbnails["hash1-v3"] = self._corner_icon(corner=(255, 0, 0))
        disk_cached = Image.new("RGB", (4, 2), (0, 0, 0))
        disk_cached.putpixel((3, 0), (0, 255, 0))
        self.controller.asset_store.get_thumbnail.return_value = disk_cached

        self.controller.flip_thumbnails(["hash1-v3"], True)

        # A horizontal flip moves the top-right marker to top-left.
        self.assertEqual(self._icon_pixel(self.mock_session_manager.state.thumbnails["hash1-v3"], 0, 0), (255, 0, 0))
        saved_key, saved_img = self.controller.asset_store.save_thumbnail.call_args.args
        self.assertEqual(saved_key, "hash1-v3")
        self.assertEqual(saved_img.getpixel((0, 0)), (0, 255, 0))
        self.mock_session_manager.asset_model.refresh.assert_called_once_with()

    def test_flip_skips_a_key_with_no_cached_thumbnail_anywhere_yet(self):
        self.controller.asset_store.get_thumbnail.return_value = None

        self.controller.flip_thumbnails(["hash1-v3"], True)

        self.controller.asset_store.save_thumbnail.assert_not_called()
        self.mock_session_manager.asset_model.refresh.assert_not_called()

    def test_flip_with_no_keys_is_a_noop(self):
        self.controller.flip_thumbnails([], True)

        self.controller.asset_store.get_thumbnail.assert_not_called()

    def test_flip_clears_the_stale_flag_the_bulk_write_set(self):
        self.mock_session_manager.state.stale_thumbnails.add("hash1-v3")

        self.controller.flip_thumbnails(["hash1-v3"], True)

        self.assertNotIn("hash1-v3", self.mock_session_manager.state.stale_thumbnails)

    def test_rotate_before_decode_finishes_corrects_the_stale_delivery(self):
        """A frame with no cached thumbnail yet is still being decoded by
        generate_missing_thumbnails; that decode's own (orientation-blind) result,
        landing after the rotate, must come out corrected rather than stale."""
        asset = {"name": "f1.dng", "path": "/roll/f1.dng", "hash": "h1"}
        key = asset_thumbnail_key(asset)
        self.mock_session_manager.state.uploaded_files = [asset]
        self.controller.asset_store.get_thumbnail.return_value = None  # still mid-decode

        self.controller.rotate_thumbnails([key], 1)

        self.controller.asset_store.save_thumbnail.assert_not_called()
        self.assertNotIn(key, self.mock_session_manager.state.thumbnails)

        # The in-flight decode's own result lands afterward, in the old orientation.
        stale = Image.new("RGB", (4, 2), (0, 0, 0))
        stale.putpixel((3, 0), (0, 255, 0))
        self.controller._apply_thumbnails({key: stale})

        saved_key, saved_img = self.controller.asset_store.save_thumbnail.call_args.args
        self.assertEqual(saved_key, key)
        self.assertEqual(saved_img.size, (2, 4))
        self.assertEqual(saved_img.getpixel((0, 0)), (0, 255, 0))
        self.assertEqual(self._icon_pixel(self.mock_session_manager.state.thumbnails[key], 0, 0), (0, 255, 0))

    def test_rotate_correction_is_not_replayed_a_second_time(self):
        asset = {"name": "f1.dng", "path": "/roll/f1.dng", "hash": "h1"}
        key = asset_thumbnail_key(asset)
        self.mock_session_manager.state.uploaded_files = [asset]
        self.controller.asset_store.get_thumbnail.return_value = None

        self.controller.rotate_thumbnails([key], 1)
        self.controller._apply_thumbnails({key: Image.new("RGB", (4, 2), (0, 0, 0))})
        self.controller.asset_store.save_thumbnail.reset_mock()

        # A later, unrelated re-delivery for the same key (e.g. clear_thumbnail_cache)
        # must not be turned again — the correction was a one-time catch-up.
        self.controller._apply_thumbnails({key: Image.new("RGB", (2, 4), (0, 0, 0))})

        self.controller.asset_store.save_thumbnail.assert_not_called()

    def test_rotate_correction_is_dropped_once_the_frame_leaves_the_roll(self):
        asset = {"name": "f1.dng", "path": "/roll/f1.dng", "hash": "h1"}
        key = asset_thumbnail_key(asset)
        self.mock_session_manager.state.uploaded_files = [asset]
        self.controller.asset_store.get_thumbnail.return_value = None

        self.controller.rotate_thumbnails([key], 1)
        self.mock_session_manager.state.uploaded_files = []  # removed before the decode lands

        self.controller._apply_thumbnails({key: Image.new("RGB", (4, 2), (0, 0, 0))})

        self.controller.asset_store.save_thumbnail.assert_not_called()
        self.assertNotIn(key, self.controller._thumbnail_pending_correction)


class TestLibraryIndexing(unittest.TestCase):
    """index_library / _on_library_index_scanned / request_library_semantic_search --
    the whole-library counterpart to the in-session embedding pass and search."""

    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.state.semantic_search_enabled = True
        self.mock_session_manager.repo = MagicMock()
        self.mock_session_manager.asset_model = MagicMock()
        self.mock_session_manager.asset_model.semantic_query_active = False

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            self.controller = AppController(self.mock_session_manager)

        self.scan_requests = []
        self.controller.library_index_scan_requested.connect(self.scan_requests.append)
        self.controller.embedding_requested = MagicMock()
        self._ready_patch = patch("negpy.desktop.controller.semantic_model.clip_model_ready", return_value=True)
        self._ready_patch.start()

    def tearDown(self):
        import gc

        self._ready_patch.stop()
        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def _set_roots(self, roots):
        self.mock_session_manager.repo.get_global_setting.side_effect = lambda key, default=None: (
            roots if key == "library_roots" else default
        )

    def test_noop_when_the_feature_is_off(self):
        self.mock_session_manager.state.semantic_search_enabled = False
        self._set_roots(["/photos"])

        self.controller.index_library()

        self.assertEqual(self.scan_requests, [])
        self.assertIsNone(self.controller._active_batch)

    def test_noop_without_library_roots(self):
        self._set_roots([])

        self.controller.index_library()

        self.assertEqual(self.scan_requests, [])
        self.assertIsNone(self.controller._active_batch)

    def test_claims_the_batch_lane_and_requests_a_scan(self):
        self._set_roots(["/photos"])

        self.controller.index_library()

        self.assertEqual(self.controller._active_batch, "library_index")
        self.assertEqual(self.scan_requests, [["/photos"]])
        self.assertEqual(self.controller._embedding_batch_owner, "library_index")

    def test_scanned_files_missing_from_the_db_are_requested(self):
        self._set_roots(["/photos"])
        self.controller.index_library()
        self.mock_session_manager.repo.load_embeddings_for.return_value = {"h1": object()}

        self.controller._on_library_index_scanned(
            [{"name": "a", "path": "/photos/a.nef", "hash": "h1"}, {"name": "b", "path": "/photos/b.nef", "hash": "h2"}]
        )

        (requested,), _ = self.controller.embedding_requested.emit.call_args
        self.assertEqual([f["hash"] for f in requested], ["h2"])
        self.assertEqual(self.controller._active_batch, "library_index")  # released only once embedding finishes

    def test_nothing_missing_ends_the_batch_without_embedding(self):
        self._set_roots(["/photos"])
        self.controller.index_library()
        self.mock_session_manager.repo.load_embeddings_for.return_value = {"h1": object()}

        self.controller._on_library_index_scanned([{"name": "a", "path": "/photos/a.nef", "hash": "h1"}])

        self.controller.embedding_requested.emit.assert_not_called()
        self.assertIsNone(self.controller._active_batch)

    def test_a_stray_scan_result_with_no_active_batch_is_ignored(self):
        """Defends against the scan signal firing when nothing claimed the lane --
        it shouldn't be reachable, but must not crash or start an embedding batch."""
        self.controller._on_library_index_scanned([{"name": "a", "path": "/a.nef", "hash": "h1"}])
        self.controller.embedding_requested.emit.assert_not_called()

    def test_cancelling_during_the_scan_skips_the_embedding_batch(self):
        self._set_roots(["/photos"])
        self.controller.index_library()

        self.controller.abort_active_batch()
        self.controller._on_library_index_scanned([{"name": "a", "path": "/photos/a.nef", "hash": "h1"}])

        self.controller.embedding_requested.emit.assert_not_called()
        self.assertIsNone(self.controller._active_batch)

    def test_abort_cancels_the_shared_embedding_worker(self):
        self._set_roots(["/photos"])
        self.controller.index_library()
        self.controller.embedding_worker.cancel = MagicMock()

        self.controller.abort_active_batch()

        self.controller.embedding_worker.cancel.assert_called_once_with()

    def test_walk_progress_is_labeled_as_scanning_while_indexing(self):
        self._set_roots(["/photos"])
        self.controller.index_library()

        with patch.object(self.controller, "set_status") as status:
            self.controller._on_library_walk_progress(5)

        status.assert_called_once_with("Scanning library… 5 files")

    def test_walk_progress_is_labeled_as_searching_otherwise(self):
        with patch.object(self.controller, "set_status") as status:
            self.controller._on_library_walk_progress(5)

        status.assert_called_once_with("Searching library… 5 files")

    def test_a_scan_error_ends_the_batch_and_is_labeled_as_indexing(self):
        self._set_roots(["/photos"])
        self.controller.index_library()

        with patch.object(self.controller, "_report_worker_error") as report:
            self.controller._on_library_search_error("disk unplugged")

        report.assert_called_once_with("Library indexing", "disk unplugged")
        self.assertIsNone(self.controller._active_batch)

    def test_a_plain_search_error_is_still_labeled_as_search(self):
        with patch.object(self.controller, "_report_worker_error") as report:
            self.controller._on_library_search_error("disk unplugged")

        report.assert_called_once_with("Library search", "disk unplugged")

    def test_semantic_search_ranks_and_opens_matches(self):
        """The threshold is relative to the whole candidate pool's own score spread, so
        it needs a real background of unrelated scores to separate a match from -- one
        match against one lone unrelated file can't stand out from each other the way a
        real search's rare match stands out from its library's own baseline."""
        query = np.array([0.0, 1.0], dtype=np.float32)
        background = {
            f"far{i}": (f"/photos/far{i}.nef", np.array([np.sqrt(1.0 - c**2), c], dtype=np.float32))
            for i, c in enumerate(np.linspace(0.05, 0.15, 20))
        }
        self.mock_session_manager.repo.load_all_embeddings.return_value = {
            **background,
            "h1": ("/photos/close.nef", np.array([0.1, 0.9], dtype=np.float32) / np.linalg.norm([0.1, 0.9])),
        }
        with (
            patch.object(self.controller, "embed_search_query", return_value=query),
            patch.object(self.controller, "request_asset_discovery") as discovery,
        ):
            self.controller.request_library_semantic_search("a sunset")

        discovery.assert_called_once_with(["/photos/close.nef"], auto_open=True, replace_existing=True)

    def test_semantic_search_clears_both_filters_before_the_hand_off(self):
        """A search box left in search-by-meaning mode from before this run already
        applied its own in-session query (set_semantic_query, via the sidebar's live
        filter) to whatever was loaded previously. Left active, it would re-run the same
        outlier check against just the hand-off's own already-selected matches -- a small,
        mutually similar set with no background left to stand out from -- and could
        exclude the lot. A stale plain-text filter from an earlier, unrelated search is
        just as capable of zeroing this batch (its filenames are never going to contain
        the query text) once clearing the semantic query stops masking it -- clear_filters
        clears both in one rebuild rather than just the one that happened to be active.
        The hand-off's own ranking already is the filtered result."""
        query = np.array([0.0, 1.0], dtype=np.float32)
        background = {
            f"far{i}": (f"/photos/far{i}.nef", np.array([np.sqrt(1.0 - c**2), c], dtype=np.float32))
            for i, c in enumerate(np.linspace(0.05, 0.15, 20))
        }
        self.mock_session_manager.repo.load_all_embeddings.return_value = {
            **background,
            "h1": ("/photos/close.nef", np.array([0.1, 0.9], dtype=np.float32) / np.linalg.norm([0.1, 0.9])),
        }
        with (
            patch.object(self.controller, "embed_search_query", return_value=query),
            patch.object(self.controller, "request_asset_discovery"),
        ):
            self.controller.request_library_semantic_search("a sunset")

        self.mock_session_manager.asset_model.clear_filters.assert_called_once_with()

    def test_semantic_search_excludes_a_row_with_no_path(self):
        """A vector saved before the file_path column existed can't be opened from a
        library-wide match -- excluded rather than crashing on an empty path."""
        query = np.array([0.0, 1.0], dtype=np.float32)
        self.mock_session_manager.repo.load_all_embeddings.return_value = {"h1": ("", query)}
        with (
            patch.object(self.controller, "embed_search_query", return_value=query),
            patch.object(self.controller, "request_asset_discovery") as discovery,
        ):
            self.controller.request_library_semantic_search("a sunset")

        discovery.assert_not_called()

    def test_semantic_search_excludes_a_whole_scan_embedding_superseded_by_its_own_halves(self):
        """The whole-library indexer walks every physical file with no half-frame
        awareness, so a file keeps its whole-scan embedding once its two halves get
        their own -- unrotated, both subjects at once, a worse candidate than either
        half and never the one actually shown once split. Real half-hash embeddings
        existing for it (not merely split_scans() saying so, which is written from
        a roll-wide toggle and is not a per-file fact) is what proves it superseded."""
        query = np.array([0.0, 1.0], dtype=np.float32)
        background = {
            f"far{i}": (f"/photos/far{i}.nef", np.array([np.sqrt(1.0 - c**2), c], dtype=np.float32))
            for i, c in enumerate(np.linspace(0.05, 0.15, 20))
        }
        close = np.array([0.1, 0.9], dtype=np.float32) / np.linalg.norm([0.1, 0.9])
        self.mock_session_manager.repo.load_all_embeddings.return_value = {
            **background,
            "h1": ("/photos/whole.nef", close),
            "h1#1": ("/photos/whole.nef", close),
        }
        with (
            patch.object(self.controller, "embed_search_query", return_value=query),
            patch.object(self.controller, "request_asset_discovery") as discovery,
        ):
            self.controller.request_library_semantic_search("a sunset")

        discovery.assert_called_once_with(["/photos/whole.nef"], auto_open=True, replace_existing=True)

    def test_semantic_search_with_no_matches_does_not_open_anything(self):
        with (
            patch.object(self.controller, "embed_search_query", return_value=np.zeros(2, dtype=np.float32)),
            patch.object(self.controller, "request_asset_discovery") as discovery,
        ):
            self.mock_session_manager.repo.load_all_embeddings.return_value = {}
            self.controller.request_library_semantic_search("a sunset")

        discovery.assert_not_called()

    def test_semantic_search_with_no_embedding_is_a_noop(self):
        with (
            patch.object(self.controller, "embed_search_query", return_value=None),
            patch.object(self.controller, "request_asset_discovery") as discovery,
        ):
            self.controller.request_library_semantic_search("a sunset")

        discovery.assert_not_called()
        self.mock_session_manager.repo.load_all_embeddings.assert_not_called()

    def test_semantic_search_with_no_model_reports_not_ready(self):
        """Non-empty text but still no embedding means the model isn't ready --
        distinct from an empty box, which asks for a search instead."""
        with (
            patch.object(self.controller, "embed_search_query", return_value=None),
            patch.object(self.controller, "set_status") as status,
        ):
            self.controller.request_library_semantic_search("a sunset")

        self.assertEqual(status.call_args[0][0], "Search by meaning is not ready yet")

    def test_semantic_search_with_empty_text_asks_for_a_query(self):
        with (
            patch.object(self.controller, "embed_search_query", return_value=None),
            patch.object(self.controller, "set_status") as status,
        ):
            self.controller.request_library_semantic_search("   ")

        self.assertEqual(status.call_args[0][0], "Type a search first")


class TestLibrarySearch(unittest.TestCase):
    """The library search runs the film-strip query against folders on disk and opens
    what it finds. It must never hash: identity stays the loader's job."""

    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()
        self.mock_session_manager.repo.load_settings_by_path.return_value = {}
        self.mock_session_manager.repo.load_file_marks_by_path.return_value = {}
        self.mock_session_manager.asset_model = MagicMock()

        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            self.controller = AppController(self.mock_session_manager)

        self.tasks = []
        self.controller.library_search_requested.connect(self.tasks.append)

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def _set_roots(self, roots):
        self.mock_session_manager.repo.get_global_setting.side_effect = lambda key, default=None: (
            roots if key == "library_roots" else default
        )

    def test_empty_query_does_not_search(self):
        self._set_roots(["/photos"])
        self.controller.request_library_search("   ")
        self.assertEqual(self.tasks, [])

    def test_search_without_roots_asks_for_a_folder_first(self):
        self._set_roots([])
        self.controller.request_library_search("film:portra")
        self.assertEqual(self.tasks, [])

    def test_search_carries_roots_edits_and_marks(self):
        self._set_roots(["/photos"])
        self.mock_session_manager.repo.load_settings_by_path.return_value = {"/photos/a.nef": WorkspaceConfig()}
        self.mock_session_manager.repo.load_file_marks_by_path.return_value = {"/photos/a.nef": "keeper"}

        self.controller.request_library_search("film:portra")

        self.assertEqual(len(self.tasks), 1)
        task = self.tasks[0]
        self.assertEqual(task.roots, ["/photos"])
        self.assertEqual(task.query, "film:portra")
        self.assertEqual(set(task.configs_by_path), {"/photos/a.nef"})
        self.assertEqual(task.marks_by_path, {"/photos/a.nef": "keeper"})

    def test_results_replace_the_session(self):
        with patch.object(self.controller, "request_asset_discovery") as discovery:
            self.controller._on_library_search_finished(["/photos/a.nef", "/photos/b.nef"])

        discovery.assert_called_once_with(["/photos/a.nef", "/photos/b.nef"], auto_open=True, replace_existing=True)

    def test_results_clear_a_stale_semantic_query_before_the_hand_off(self):
        """A semantic query left over from an earlier, unrelated search-by-meaning run
        would otherwise re-rank this plain keyword search's own matches by an embedding
        that has nothing to do with them, dropping every file with no cached vector yet
        -- the mirror image of the semantic hand-off's own stale-filter problem. The
        hand-off's own match list already is the filtered result."""
        with patch.object(self.controller, "request_asset_discovery"):
            self.controller._on_library_search_finished(["/photos/a.nef", "/photos/b.nef"])

        self.mock_session_manager.asset_model.clear_filters.assert_called_once_with()

    def test_no_results_leaves_the_session_alone(self):
        with patch.object(self.controller, "request_asset_discovery") as discovery:
            self.controller._on_library_search_finished([])

        discovery.assert_not_called()

    def test_open_library_folder_replaces_or_appends(self):
        with patch("negpy.desktop.controller.os.path.isdir", return_value=True):
            with patch.object(self.controller, "request_asset_discovery") as discovery:
                self.controller.open_library_folder("/photos/roll_a")
                self.assertTrue(discovery.call_args.kwargs["replace_existing"])

                self.controller.open_library_folder("/photos/roll_a", add_to_session=True)
                self.assertFalse(discovery.call_args.kwargs["replace_existing"])

    def test_missing_folder_is_reported_not_opened(self):
        with patch("negpy.desktop.controller.os.path.isdir", return_value=False):
            with patch.object(self.controller, "request_asset_discovery") as discovery:
                self.controller.open_library_folder("/photos/gone")
        discovery.assert_not_called()

    def _dict_repo(self) -> None:
        """Swaps self.controller.session.repo's global settings for a real dict, so a
        roll written in one call is readable back in the next -- the class-wide fixture's
        bare MagicMock does not round-trip."""
        store: dict = {}
        self.controller.session.repo.get_global_setting.side_effect = lambda key, default=None: store.get(key, default)
        self.controller.session.repo.save_global_setting.side_effect = lambda key, value: store.__setitem__(key, value)

    def test_opening_one_folder_recognizes_and_activates_it(self):
        self._dict_repo()
        with patch("negpy.desktop.controller.os.path.isdir", return_value=True):
            with patch.object(self.controller, "request_asset_discovery"):
                self.controller.open_library_folder("/photos/roll_a")

        from negpy.services.assets.rolls import folder_roll_id_for_path

        roll_id = folder_roll_id_for_path(self.controller.session.repo, "/photos/roll_a")
        self.assertIsNotNone(roll_id)
        self.assertEqual(self.controller.state.active_roll_id, roll_id)

    def test_opening_several_folders_leaves_no_single_active_roll(self):
        self._dict_repo()
        with patch("negpy.desktop.controller.os.path.isdir", return_value=True):
            with patch.object(self.controller, "request_asset_discovery"):
                self.controller.open_library_folders(["/photos/a", "/photos/b"])

        self.assertIsNone(self.controller.state.active_roll_id)

    def test_adding_to_session_does_not_recognize_a_folder(self):
        self._dict_repo()
        with patch("negpy.desktop.controller.os.path.isdir", return_value=True):
            with patch.object(self.controller, "request_asset_discovery"):
                self.controller.open_library_folder("/photos/roll_a", add_to_session=True)

        from negpy.services.assets.rolls import saved_rolls

        self.assertEqual(saved_rolls(self.controller.session.repo), {})

    def test_open_roll_loads_a_folder_rolls_own_and_extra_paths(self):
        self._dict_repo()
        from negpy.services.assets.rolls import add_extra_members, recognize_folder

        roll_id = recognize_folder(self.controller.session.repo, "/photos/roll_a")
        add_extra_members(self.controller.session.repo, roll_id, ["/elsewhere/c.nef"])

        with patch.object(self.controller, "request_asset_discovery") as discovery:
            self.controller.open_roll(roll_id)

        discovery.assert_called_once_with(["/photos/roll_a", "/elsewhere/c.nef"], auto_open=True, replace_existing=True)
        self.assertEqual(self.controller.state.active_roll_id, roll_id)

    def test_open_roll_loads_a_virtual_rolls_member_paths(self):
        self._dict_repo()
        from negpy.services.assets.rolls import create_virtual_roll

        roll_id = create_virtual_roll(self.controller.session.repo, "Portra", ["/a.nef", "/b.nef"])

        with patch.object(self.controller, "request_asset_discovery") as discovery:
            self.controller.open_roll(roll_id)

        discovery.assert_called_once_with(["/a.nef", "/b.nef"], auto_open=True, replace_existing=True)

    def test_open_roll_reports_an_unknown_id(self):
        self._dict_repo()
        with patch.object(self.controller, "request_asset_discovery") as discovery:
            self.controller.open_roll("not-a-real-id")
        discovery.assert_not_called()

    def test_create_roll_from_session_saves_the_loaded_paths(self):
        self._dict_repo()
        self.controller.state.uploaded_files = [{"path": "/a.nef"}, {"path": "/b.nef"}]

        roll_id = self.controller.create_roll_from_session("Portra")

        from negpy.services.assets.rolls import roll_for_id

        entry = roll_for_id(self.controller.session.repo, roll_id)
        self.assertEqual(entry["kind"], "virtual")
        self.assertEqual(entry["name"], "Portra")
        self.assertEqual(entry["member_paths"], ["/a.nef", "/b.nef"])
        self.assertEqual(self.controller.state.active_roll_id, roll_id)

    def test_request_rename_roll_display_name_only(self):
        self._dict_repo()
        from negpy.services.assets.rolls import create_virtual_roll, roll_for_id

        roll_id = create_virtual_roll(self.controller.session.repo, "Portra", [])

        result = self.controller.request_rename_roll(roll_id, "Portra 400", False)

        self.assertTrue(result)
        self.assertEqual(roll_for_id(self.controller.session.repo, roll_id)["name"], "Portra 400")
        self.controller.session.rehome_folder_paths.assert_not_called()

    def test_request_rename_roll_also_renames_the_folder_when_not_active(self):
        self._dict_repo()
        from negpy.services.assets.rolls import recognize_folder, roll_for_id

        with tempfile.TemporaryDirectory() as d:
            old_path = os.path.join(d, "roll_a")
            os.mkdir(old_path)
            roll_id = recognize_folder(self.controller.session.repo, old_path)
            self.controller.state.active_roll_id = None

            result = self.controller.request_rename_roll(roll_id, "roll_b", True)

            self.assertTrue(result)
            new_path = os.path.join(d, "roll_b")
            self.assertTrue(os.path.isdir(new_path))
            self.assertEqual(roll_for_id(self.controller.session.repo, roll_id)["folder_path"], new_path)
            self.assertEqual(roll_for_id(self.controller.session.repo, roll_id)["name"], "roll_b")
            self.controller.session.rehome_folder_paths.assert_not_called()

    def test_request_rename_roll_rehomes_the_active_rolls_paths(self):
        self._dict_repo()
        from negpy.services.assets.rolls import recognize_folder

        with tempfile.TemporaryDirectory() as d:
            old_path = os.path.join(d, "roll_a")
            os.mkdir(old_path)
            roll_id = recognize_folder(self.controller.session.repo, old_path)
            self.controller.state.active_roll_id = roll_id

            result = self.controller.request_rename_roll(roll_id, "roll_b", True)

            self.assertTrue(result)
            new_path = os.path.join(d, "roll_b")
            self.controller.session.rehome_folder_paths.assert_called_once_with(old_path, new_path)

    def test_request_rename_roll_disk_failure_leaves_the_display_name_alone(self):
        self._dict_repo()
        from negpy.services.assets.rolls import recognize_folder, roll_for_id

        with tempfile.TemporaryDirectory() as d:
            old_path = os.path.join(d, "roll_a")
            os.mkdir(old_path)
            os.mkdir(os.path.join(d, "roll_b"))  # collides with the requested new name
            roll_id = recognize_folder(self.controller.session.repo, old_path)

            result = self.controller.request_rename_roll(roll_id, "roll_b", True)

            self.assertFalse(result)
            entry = roll_for_id(self.controller.session.repo, roll_id)
            self.assertEqual(entry["name"], "roll_a")
            self.assertEqual(entry["folder_path"], old_path)
            self.controller.session.rehome_folder_paths.assert_not_called()

    def test_create_roll_from_session_with_nothing_loaded(self):
        self._dict_repo()
        self.controller.state.uploaded_files = []
        self.assertIsNone(self.controller.create_roll_from_session("Portra"))

    def test_appending_while_a_roll_is_active_extends_its_membership(self):
        self._dict_repo()
        from negpy.services.assets.rolls import create_virtual_roll, roll_for_id

        roll_id = create_virtual_roll(self.controller.session.repo, "Portra", ["/a.nef"])
        self.controller.state.active_roll_id = roll_id

        self.controller._replace_after_discovery = False
        self.controller._on_discovery_finished([{"path": "/b.nef", "hash": "hb", "name": "b.nef"}])

        self.assertEqual(roll_for_id(self.controller.session.repo, roll_id)["member_paths"], ["/a.nef", "/b.nef"])

    def test_replacing_does_not_extend_the_previously_active_roll(self):
        self._dict_repo()
        from negpy.services.assets.rolls import create_virtual_roll, roll_for_id

        roll_id = create_virtual_roll(self.controller.session.repo, "Portra", ["/a.nef"])
        self.controller.state.active_roll_id = roll_id

        # The replace branch reaches further into session/asset_model than this fixture's
        # bare mock supports; only what this test cares about (roll membership) needs it.
        self.mock_session_manager.asset_model = MagicMock(visible_actual_indices_ordered=MagicMock(return_value=[]))
        self.controller._replace_after_discovery = True
        self.controller._on_discovery_finished([{"path": "/b.nef", "hash": "hb", "name": "b.nef"}])

        self.assertEqual(roll_for_id(self.controller.session.repo, roll_id)["member_paths"], ["/a.nef"])

    def test_library_search_results_clear_the_active_roll(self):
        self._dict_repo()
        from negpy.services.assets.rolls import create_virtual_roll

        roll_id = create_virtual_roll(self.controller.session.repo, "Portra", ["/a.nef"])
        self.controller.state.active_roll_id = roll_id

        with patch.object(self.controller, "request_asset_discovery"):
            self.controller._on_library_search_finished(["/c.nef"])

        self.assertIsNone(self.controller.state.active_roll_id)

    def test_has_rolls_reflects_the_store(self):
        self._dict_repo()
        self.assertFalse(self.controller.has_rolls())

        from negpy.services.assets.rolls import create_virtual_roll

        create_virtual_roll(self.controller.session.repo, "Portra", [])
        self.assertTrue(self.controller.has_rolls())

    def test_opening_a_folder_registers_it_as_a_search_root(self):
        self._dict_repo()
        with patch("negpy.desktop.controller.os.path.isdir", return_value=True):
            with patch.object(self.controller, "request_asset_discovery"):
                self.controller.open_library_folder("/photos/roll_a")

        self.assertIn("/photos/roll_a", self.controller.library_roots())

    def test_adding_to_session_does_not_register_a_search_root(self):
        self._dict_repo()
        with patch("negpy.desktop.controller.os.path.isdir", return_value=True):
            with patch.object(self.controller, "request_asset_discovery"):
                self.controller.open_library_folder("/photos/roll_a", add_to_session=True)

        self.assertEqual(self.controller.library_roots(), [])

    def test_import_subfolders_as_rolls_recognizes_each_one_and_registers_the_parent(self):
        self._dict_repo()
        found = ["/scans/roll_a", "/scans/2024/roll_b"]
        with patch("negpy.services.assets.rolls.discover_roll_folders", return_value=found):
            roll_ids = self.controller.import_subfolders_as_rolls("/scans")

        self.assertEqual(len(roll_ids), 2)
        self.assertIn("/scans", self.controller.library_roots())

    def test_import_subfolders_as_rolls_with_none_found_registers_nothing(self):
        self._dict_repo()
        with patch("negpy.services.assets.rolls.discover_roll_folders", return_value=[]):
            roll_ids = self.controller.import_subfolders_as_rolls("/scans")

        self.assertEqual(roll_ids, [])
        self.assertEqual(self.controller.library_roots(), [])

    def test_rediscover_rolls_walks_every_import_source_again(self):
        self._dict_repo()
        found = {"/scans": ["/scans/roll_a"]}
        with patch("negpy.services.assets.rolls.discover_roll_folders", side_effect=lambda p, _filters: found[p]):
            self.controller.import_subfolders_as_rolls("/scans")
            found["/scans"] = ["/scans/roll_a", "/scans/roll_b"]
            new, dropped = self.controller.rediscover_rolls()

        self.assertEqual((new, dropped), (1, 0))
        self.assertTrue(self.controller.has_rolls())


class TestSplashPreviewRaceGuard(unittest.TestCase):
    """A backlogged splash-decode worker can land after the real render for the same
    file already has -- e.g. a prefetched neighbour whose full pipeline finishes
    before its own splash request reaches the front of the queue. Painting a late
    splash then would stomp the correct positive with the raw, un-inverted embedded
    thumbnail (glaringly wrong on a negative)."""

    def _panel(self, *, requested_path="a.dng", hash_for_path="h1"):
        panel = MagicMock()
        panel._requested_file_path = requested_path
        panel._file_hash_for_path.return_value = hash_for_path
        panel._split_active_half.return_value = ("RAW", (100, 100))
        panel.state = AppState()
        return panel

    def test_splash_skipped_once_the_real_render_for_this_file_already_landed(self):
        panel = self._panel(hash_for_path="h1")
        panel.state.last_metrics["splash"] = False
        panel.state.last_metrics["source_hash"] = "h1"

        AppController._on_splash_preview(panel, "a.dng", "RAW", (100, 100))

        self.assertNotIn("base_positive", panel.state.last_metrics)
        panel.image_updated.emit.assert_not_called()

    def test_splash_paints_normally_before_any_real_render_has_landed(self):
        panel = self._panel(hash_for_path="h1")

        AppController._on_splash_preview(panel, "a.dng", "RAW", (100, 100))

        self.assertEqual(panel.state.last_metrics["base_positive"], "RAW")
        self.assertTrue(panel.state.last_metrics["splash"])
        panel.image_updated.emit.assert_called_once()

    def test_splash_not_suppressed_by_a_different_files_render(self):
        """The guard must key off the file splash is arriving for, not just whether
        *any* render recently landed -- else a legitimate splash for a fresh frame
        would wrongly be swallowed by the frame just left."""
        panel = self._panel(requested_path="b.dng", hash_for_path="h2")
        panel.state.last_metrics["splash"] = False
        panel.state.last_metrics["source_hash"] = "h1"  # a different file's render

        AppController._on_splash_preview(panel, "b.dng", "RAW", (100, 100))

        self.assertEqual(panel.state.last_metrics["base_positive"], "RAW")

    def test_splash_ignored_for_a_stale_navigation_request(self):
        panel = self._panel(requested_path="b.dng")

        AppController._on_splash_preview(panel, "a.dng", "RAW", (100, 100))

        self.assertNotIn("base_positive", panel.state.last_metrics)
        panel.image_updated.emit.assert_not_called()
