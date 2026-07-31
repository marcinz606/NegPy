import sys
import unittest
from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QApplication, QDialog

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState, DesktopSessionManager
from negpy.desktop.view.widgets.scan_setup_dialog import ScanSetupDialog
from negpy.domain.models import WorkspaceConfig
from negpy.features.process.models import scan_setup_values
from negpy.services.rendering.preview_manager import PreviewManager

if not QApplication.instance():
    _app = QApplication(sys.argv)


class TestScanSetupValues(unittest.TestCase):
    def test_camera_white_light_is_the_only_as_shot_wb_decode(self):
        self.assertEqual(scan_setup_values("camera", "white"), (False, False))

    def test_camera_narrowband(self):
        self.assertEqual(scan_setup_values("camera", "narrowband"), (True, True))

    def test_scanner_white_light(self):
        self.assertEqual(scan_setup_values("scanner", "white"), (True, False))

    def test_scanner_narrowband(self):
        self.assertEqual(scan_setup_values("scanner", "narrowband"), (True, True))


class TestScanSetupDialog(unittest.TestCase):
    def test_starts_on_capture_page(self):
        dlg = ScanSetupDialog(None)
        self.assertEqual(dlg.pages.currentIndex(), 0)
        self.assertFalse(dlg.back_btn.isVisibleTo(dlg))
        self.assertEqual(dlg.next_btn.text(), "Next →")
        self.assertEqual(dlg.choice(), {"capture": "camera", "light": "white"})

    def test_next_advances_and_back_returns(self):
        dlg = ScanSetupDialog(None)
        dlg.next_btn.click()
        self.assertEqual(dlg.pages.currentIndex(), 1)
        self.assertEqual(dlg.next_btn.text(), "Apply")
        self.assertTrue(dlg.back_btn.isVisibleTo(dlg))
        dlg.back_btn.click()
        self.assertEqual(dlg.pages.currentIndex(), 0)

    def test_light_page_names_the_chosen_hardware_class(self):
        dlg = ScanSetupDialog(None)
        dlg.capture_group.button(1).setChecked(True)  # film scanner
        self.assertIn("Plustek", dlg.light_hints[0].text())
        self.assertIn("Coolscan", dlg.light_hints[1].text())

        dlg.capture_group.button(0).setChecked(True)  # digital camera
        self.assertIn("lightbox", dlg.light_hints[0].text())
        self.assertIn("Scanlight", dlg.light_hints[1].text())

    def test_choice_round_trips_through_the_pages(self):
        dlg = ScanSetupDialog(None)
        dlg.capture_group.button(1).setChecked(True)
        dlg.next_btn.click()
        dlg.light_buttons[1].setChecked(True)
        self.assertEqual(dlg.choice(), {"capture": "scanner", "light": "narrowband"})

    def test_apply_accepts(self):
        dlg = ScanSetupDialog(None)
        dlg.next_btn.click()
        dlg.next_btn.click()
        self.assertEqual(dlg.result(), QDialog.DialogCode.Accepted)

    def test_preselects_the_saved_choice(self):
        dlg = ScanSetupDialog(None, {"capture": "scanner", "light": "narrowband"})
        self.assertTrue(dlg.capture_group.button(1).isChecked())
        self.assertTrue(dlg.light_buttons[1].isChecked())
        self.assertEqual(dlg.choice(), {"capture": "scanner", "light": "narrowband"})

    def test_summary_reports_what_apply_will_set(self):
        dlg = ScanSetupDialog(None)
        dlg.next_btn.click()
        self.assertIn("Linear RAW off", dlg.summary.text())
        dlg.light_buttons[1].setChecked(True)
        self.assertIn("Linear RAW on", dlg.summary.text())
        self.assertIn("Narrowband on", dlg.summary.text())


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
        self.assertTrue(saved["last_narrowband_scan"])

    def test_updates_the_open_frame(self):
        self.controller.apply_scan_setup("camera", "white")
        config = self.mock_session_manager.update_config.call_args.args[0]
        self.assertFalse(config.process.linear_raw)
        self.assertFalse(config.process.narrowband_scan)

    def test_linear_raw_flip_reloads_the_open_frame(self):
        self.controller.apply_scan_setup("scanner", "white")
        self.controller.load_file.assert_called_once_with("/a.dng")
        self.assertFalse(self.mock_session_manager.update_config.call_args.kwargs["render"])

    def test_unchanged_linear_raw_renders_without_reloading(self):
        self.controller.apply_scan_setup("camera", "white")
        self.controller.load_file.assert_not_called()
        self.assertTrue(self.mock_session_manager.update_config.call_args.kwargs["render"])

    def test_rewrites_other_frames_that_have_saved_edits(self):
        self.controller.apply_scan_setup("scanner", "narrowband")
        self.repo.save_file_settings.assert_called_once()
        args, kwargs = self.repo.save_file_settings.call_args
        self.assertEqual(args[0], "h2")
        self.assertTrue(args[1].process.narrowband_scan)
        self.assertEqual(kwargs["file_path"], "/b.dng")

    def test_untouched_frames_are_left_to_the_sticky_defaults(self):
        self.controller.apply_scan_setup("scanner", "narrowband")
        saved_hashes = [c.args[0] for c in self.repo.save_file_settings.call_args_list]
        self.assertNotIn("h3", saved_hashes)

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
