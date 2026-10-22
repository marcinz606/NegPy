import sys
import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QApplication

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState, DesktopSessionManager
from negpy.desktop.view.sidebar.scan_setup import ScanSetupSidebar, resolved_rig
from negpy.domain.models import WorkspaceConfig
from negpy.features.process.models import ProcessMode, scan_setup_rig, scan_setup_values, with_process_mode
from negpy.services.rendering.preview_manager import PreviewManager

if not QApplication.instance():
    _app = QApplication(sys.argv)


class TestScanSetupValues(unittest.TestCase):
    def test_camera_white_light_is_the_only_as_shot_wb_decode(self):
        self.assertEqual(scan_setup_values("camera", "white"), (False, "neutral"))

    def test_camera_narrowband(self):
        self.assertEqual(scan_setup_values("camera", "narrowband"), (True, "pakon"))

    def test_scanner_white_light(self):
        self.assertEqual(scan_setup_values("scanner", "white"), (True, "neutral"))

    def test_scanner_narrowband(self):
        self.assertEqual(scan_setup_values("scanner", "narrowband"), (True, "pakon"))


class TestScanSetupRig(unittest.TestCase):
    def test_auto_decodes_as_a_camera(self):
        self.assertEqual(scan_setup_values("auto", "white"), (False, "neutral"))
        self.assertEqual(scan_setup_values("auto", "narrowband"), (True, "pakon"))

    def test_nothing_saved_reads_auto_under_white_light(self):
        self.assertEqual(scan_setup_rig(None), ("auto", "white"))
        self.assertEqual(scan_setup_rig({"capture": "drum", "light": 3}), ("auto", "white"))
        self.assertEqual(scan_setup_rig({"capture": "scanner", "light": "narrowband"}), ("scanner", "narrowband"))


def _setup_controller(setup=None, **state):
    controller = MagicMock()
    controller.state = AppState()
    for key, value in state.items():
        setattr(controller.state, key, value)
    controller.session.repo.get_global_setting.side_effect = lambda key, default=None: setup if key == "scan_setup" else default
    controller.active_diptych.return_value = None
    controller.rgb_scan_mode_for_roll.return_value = False
    return controller


class TestScanSetupCard(unittest.TestCase):
    def test_shows_the_saved_setup_and_applies_a_change(self):
        controller = _setup_controller({"capture": "scanner", "light": "white"})
        card = ScanSetupSidebar(controller)
        card.sync_ui()
        self.assertEqual((card.capture_btn.currentData(), card.light_btn.currentData()), ("scanner", "white"))

        card.light_btn.setCurrentIndex(card.light_btn.findData("narrowband"))
        controller.apply_scan_setup.assert_called_once_with("scanner", "narrowband")

    def test_the_capture_is_read_from_the_file_until_picked(self):
        controller = _setup_controller(None, current_file_hash="h", preview_cam_xyz=None)
        card = ScanSetupSidebar(controller)
        card.sync_ui()
        self.assertEqual(card.capture_btn.count(), 2)
        self.assertEqual(card.capture_btn.currentData(), "scanner")

        card.light_btn.setCurrentIndex(card.light_btn.findData("narrowband"))
        controller.apply_scan_setup.assert_called_with("auto", "narrowband")
        card.capture_btn.setCurrentIndex(card.capture_btn.findData("camera"))
        controller.apply_scan_setup.assert_called_with("camera", "narrowband")

    def test_bw_hides_the_light_and_keeps_it(self):
        controller = _setup_controller({"capture": "scanner", "light": "narrowband"})
        config = controller.state.config
        controller.state.config = replace(config, process=replace(config.process, process_mode=ProcessMode.BW))
        card = ScanSetupSidebar(controller)
        card.sync_ui()
        self.assertTrue(card.light_btn.isHidden())
        self.assertEqual(card.light_btn.currentData(), "narrowband")
        self.assertTrue(card.hint.isHidden())

        controller.state.config = config
        card.sync_ui()
        self.assertFalse(card.light_btn.isHidden())

    def test_auto_reads_the_open_frame(self):
        camera = _setup_controller(None, current_file_hash="h", preview_cam_xyz=((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        scanner = _setup_controller(None, current_file_hash="h", preview_cam_xyz=None)
        self.assertEqual(resolved_rig(camera.session.repo, camera.state), ("camera", "white"))
        self.assertEqual(resolved_rig(scanner.session.repo, scanner.state), ("scanner", "white"))
        self.assertEqual(resolved_rig(_setup_controller(None).session.repo, AppState()), ("camera", "white"))


class TestRigCards(unittest.TestCase):
    def _panel(self, setup, **state):
        from negpy.desktop.view.sidebar.controls_panel import ControlsPanel

        controller = _setup_controller(setup, **state)
        panel = ControlsPanel(controller)
        panel._sync_all_sidebars()
        return panel

    def test_a_film_scanner_hides_the_camera_cards(self):
        panel = self._panel({"capture": "scanner", "light": "narrowband"})
        self.assertTrue(panel.demosaic_section.isHidden())
        self.assertTrue(panel.optics_section.isHidden())
        self.assertTrue(panel.trichrome_sidebar.isHidden())
        self.assertTrue(panel.sensor_sidebar.light_fold.content.isHidden())
        self.assertTrue(panel.sensor_sidebar.narrowband_fold.content.isHidden())
        self.assertFalse(panel.sensor_section.isHidden(), "crosstalk stays")

    def test_a_narrowband_camera_shows_everything(self):
        panel = self._panel({"capture": "camera", "light": "narrowband"})
        for widget in (panel.demosaic_section, panel.optics_section, panel.trichrome_sidebar):
            self.assertFalse(widget.isHidden())
        self.assertFalse(panel.sensor_sidebar.light_fold.content.isHidden())
        self.assertFalse(panel.sensor_sidebar.narrowband_fold.content.isHidden())

    def test_a_white_light_camera_hides_the_narrowband_rows(self):
        panel = self._panel({"capture": "camera", "light": "white"})
        self.assertFalse(panel.demosaic_section.isHidden())
        self.assertTrue(panel.trichrome_sidebar.isHidden())
        self.assertTrue(panel.sensor_sidebar.light_fold.content.isHidden())

    def test_a_setting_keeps_its_control_in_view(self):
        config = WorkspaceConfig()
        config = replace(config, process=replace(config.process, hue_trim=4.0))
        panel = self._panel({"capture": "scanner", "light": "white"}, config=config)
        self.assertFalse(panel.sensor_sidebar.light_fold.content.isHidden())

    def test_bw_on_a_white_light_scanner_leaves_calibration_empty(self):
        config = with_process_mode(WorkspaceConfig(), ProcessMode.BW)
        panel = self._panel({"capture": "scanner", "light": "white"}, config=config)
        self.assertTrue(panel.sensor_section.isHidden())


class TestApplyScanSetup(unittest.TestCase):
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
        self.controller.load_file = MagicMock()

        state = self.mock_session_manager.state
        state.current_file_hash = "h1"
        state.current_file_path = "/a.dng"
        state.uploaded_files = [
            {"name": "a", "path": "/a.dng", "hash": "h1"},
            {"name": "b", "path": "/b.dng", "hash": "h2"},
            {"name": "c", "path": "/c.dng", "hash": "h3"},
        ]
        self.repo = self.mock_session_manager.repo
        self.repo.load_file_settings.side_effect = lambda h: WorkspaceConfig() if h == "h2" else None

    def tearDown(self):
        import gc

        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
            self.controller.prefetch_load_thread,
            self.controller.scan_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_writes_the_gate_and_the_sticky_defaults(self):
        self.controller.apply_scan_setup("scanner", "narrowband")
        saved = self.repo.save_global_settings.call_args.args[0]
        self.assertEqual(saved["scan_setup"], {"capture": "scanner", "light": "narrowband"})
        self.assertTrue(saved["last_linear_raw"])
        self.assertNotIn("last_narrowband_scan", saved)

    def test_updates_the_open_frame(self):
        state = self.mock_session_manager.state
        state.config = replace(state.config, process=replace(state.config.process, linear_raw=True))
        self.controller.apply_scan_setup("camera", "white")
        config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertFalse(config.process.linear_raw)

    def _open_frame_paper(self, light, paper="neutral", mode=ProcessMode.C41):
        state = self.mock_session_manager.state
        state.config = with_process_mode(replace(state.config, exposure=replace(state.config.exposure, paper_profile=paper)), mode)
        self.controller.apply_scan_setup("camera", light)
        return self.mock_session_manager.update_config.call_args.args[0].exposure.paper_profile

    def test_narrowband_prints_the_open_frame_on_pakon_and_white_light_on_digital(self):
        self.assertEqual(self._open_frame_paper("narrowband"), "pakon")
        self.assertEqual(self._open_frame_paper("white", paper="pakon"), "neutral")

    def test_another_stock_or_film_mode_keeps_its_paper(self):
        self.assertEqual(self._open_frame_paper("narrowband", paper="kodak_supra_endura"), "kodak_supra_endura")
        self.assertEqual(self._open_frame_paper("narrowband", mode=ProcessMode.BW), "neutral")

    def test_new_files_default_to_the_lights_stock(self):
        self.repo.get_global_setting.side_effect = (
            lambda key, default=None: {"paper_profile": "neutral", "toe": 0.2} if key == "sticky_config" else default
        )
        self.controller.apply_scan_setup("camera", "narrowband")
        self.repo.save_global_setting.assert_any_call("sticky_config", {"paper_profile": "pakon", "toe": 0.2})

    def test_new_files_keep_a_carried_stock_the_wizard_does_not_own(self):
        self.repo.get_global_setting.side_effect = (
            lambda key, default=None: {"paper_profile": "fuji_crystal"} if key == "sticky_config" else default
        )
        self.controller.apply_scan_setup("camera", "narrowband")
        self.assertNotIn("sticky_config", [c.args[0] for c in self.repo.save_global_setting.call_args_list])

    def test_linear_raw_flip_reloads_the_open_frame(self):
        self.controller.apply_scan_setup("scanner", "white")
        self.controller.load_file.assert_called_once_with("/a.dng")
        self.assertFalse(self.mock_session_manager.update_config.call_args.kwargs["render"])

    def test_unchanged_linear_raw_renders_without_reloading(self):
        self._open_frame_paper("white", paper="pakon")
        self.controller.load_file.assert_not_called()
        self.assertTrue(self.mock_session_manager.update_config.call_args.kwargs["render"])

    def test_a_new_stock_alone_keeps_the_frames_bounds(self):
        state = self.mock_session_manager.state
        state.config = replace(state.config, process=replace(state.config.process, linear_raw=True, local_floors=(0.1, 0.1, 0.1)))
        self._open_frame_paper("narrowband")
        self.assertEqual(self.mock_session_manager.update_config.call_args.args[0].process.local_floors, (0.1, 0.1, 0.1))

    def test_frames_the_setup_does_not_change_are_left_alone(self):
        self.controller.apply_scan_setup("camera", "white")
        self.mock_session_manager.update_config.assert_not_called()
        self.repo.save_file_settings.assert_not_called()
        self.mock_session_manager.push_external_history.assert_not_called()
        self.mock_session_manager.frames_edited_offscreen.emit.assert_not_called()

    def test_rewrites_other_frames_that_have_saved_edits(self):
        self.controller.apply_scan_setup("scanner", "narrowband")
        self.repo.save_file_settings.assert_called_once()
        args, kwargs = self.repo.save_file_settings.call_args
        self.assertEqual(args[0], "h2")
        self.assertTrue(args[1].process.linear_raw)
        self.assertEqual(args[1].exposure.paper_profile, "pakon")
        self.assertEqual(kwargs["file_path"], "/b.dng")

    def test_untouched_frames_are_left_to_the_sticky_defaults(self):
        self.controller.apply_scan_setup("scanner", "narrowband")
        saved_hashes = [c.args[0] for c in self.repo.save_file_settings.call_args_list]
        self.assertNotIn("h3", saved_hashes)

    def test_rewritten_frames_are_offered_for_a_background_thumbnail_refresh(self):
        self.controller.apply_scan_setup("scanner", "narrowband")
        self.mock_session_manager.frames_edited_offscreen.emit.assert_called_once_with(["h2"])

    def test_no_rewrites_emits_nothing(self):
        self.repo.load_file_settings.side_effect = lambda h: None
        self.controller.apply_scan_setup("scanner", "narrowband")
        self.mock_session_manager.frames_edited_offscreen.emit.assert_not_called()

    def test_bulk_rewrite_is_undoable_per_frame(self):
        self.controller.apply_scan_setup("scanner", "narrowband")
        self.mock_session_manager.push_external_history.assert_called_once()
        self.assertEqual(self.mock_session_manager.push_external_history.call_args.args[0], "h2")

    def test_no_open_frame_still_sets_the_defaults(self):
        state = self.mock_session_manager.state
        state.current_file_hash = ""
        state.current_file_path = ""
        state.uploaded_files = []
        self.controller.apply_scan_setup("camera", "narrowband")
        self.repo.save_global_settings.assert_called_once()
        self.mock_session_manager.update_config.assert_not_called()
        self.controller.load_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
