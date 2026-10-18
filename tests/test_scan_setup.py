import sys
import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QApplication, QDialog

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState, DesktopSessionManager
from negpy.desktop.view.widgets.scan_setup_dialog import ScanSetupDialog
from negpy.domain.models import WorkspaceConfig
from negpy.features.process.models import ProcessMode, scan_setup_values, with_process_mode
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

    def test_a_scanner_is_asked_its_light_with_scanner_examples(self):
        dlg = ScanSetupDialog(None)
        dlg.capture_btn.setCurrentIndex(1)  # film scanner
        dlg.next_btn.click()
        self.assertEqual(dlg.pages.currentIndex(), 1)
        dlg.light_btn.setCurrentIndex(1)
        self.assertIn("Coolscan", dlg.light_hint.text())
        self.assertEqual(dlg.choice(), {"capture": "scanner", "light": "narrowband"})
        self.assertEqual("Sets Linear RAW · Print stock Digital (Pakon Minilab)", dlg.summary.text())

    def test_choice_round_trips_through_the_pages(self):
        dlg = ScanSetupDialog(None)
        dlg.next_btn.click()
        dlg.light_btn.setCurrentIndex(1)
        self.assertEqual(dlg.choice(), {"capture": "camera", "light": "narrowband"})

    def test_apply_accepts(self):
        dlg = ScanSetupDialog(None)
        dlg.next_btn.click()
        dlg.next_btn.click()
        self.assertEqual(dlg.result(), QDialog.DialogCode.Accepted)

    def test_preselects_the_saved_choice(self):
        dlg = ScanSetupDialog(None, {"capture": "camera", "light": "narrowband"})
        self.assertEqual(dlg.capture_btn.currentIndex(), 0)
        self.assertEqual(dlg.light_btn.currentIndex(), 1)
        self.assertEqual(dlg.choice(), {"capture": "camera", "light": "narrowband"})
        self.assertEqual(ScanSetupDialog(None, {"capture": "scanner"}).capture_btn.currentIndex(), 1)

    def test_summary_reports_what_apply_will_set(self):
        dlg = ScanSetupDialog(None)
        dlg.next_btn.click()
        self.assertEqual("Sets Camera WB · Print stock Digital (Adobe RGB)", dlg.summary.text())
        dlg.light_btn.setCurrentIndex(1)
        self.assertEqual("Sets Linear RAW · Print stock Digital (Pakon Minilab)", dlg.summary.text())

    def test_the_light_page_suggests_other_print_stocks(self):
        dlg = ScanSetupDialog(None)
        self.assertFalse(dlg.stock_hint.isVisibleTo(dlg))
        dlg.next_btn.click()
        self.assertTrue(dlg.stock_hint.isVisibleTo(dlg))
        self.assertIn("Print stock", dlg.stock_hint.text())
        dlg.light_btn.setCurrentIndex(1)
        self.assertIn("Kodak Supra Endura", dlg.stock_hint.text())


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
        self.controller.apply_scan_setup("camera", "white")
        self.controller.load_file.assert_not_called()
        self.assertTrue(self.mock_session_manager.update_config.call_args.kwargs["render"])

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
