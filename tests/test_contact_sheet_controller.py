"""AppController's Contact Sheet flow: read the frames' facts, open the dialog, queue the job."""

import gc
import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QDialog

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState, DesktopSessionManager
from negpy.domain.models import WorkspaceConfig
from negpy.services.export.contact_sheet_layout import ContactSheetSettings, SheetFormat
from negpy.services.export.contact_sheet_roll import FrameFacts, SheetLook
from negpy.services.rendering.preview_manager import PreviewManager


class TestContactSheetRequest(unittest.TestCase):
    def setUp(self):
        session = MagicMock(spec=DesktopSessionManager)
        session.state = AppState()
        session.repo = MagicMock()
        session.repo.get_global_setting.return_value = None
        session.asset_model = MagicMock()
        with (
            patch("negpy.desktop.controller.RenderWorker") as render_worker,
            patch("negpy.desktop.controller.PreviewManager") as preview_manager,
        ):
            render_worker.return_value = MagicMock()
            preview_manager.return_value = MagicMock(spec=PreviewManager)
            preview_manager.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(session)
        self.session = session
        self.controller.contact_sheet_preview = MagicMock()
        self.controller.contact_sheet_preview.prepare.return_value = 7
        self.controller.contact_sheet_requested.disconnect()
        self.jobs = []
        self.controller.contact_sheet_requested.connect(self.jobs.append)
        self.controller.load_gear_library = MagicMock(return_value=None)
        self.controller._contact_sheet_output_dir = MagicMock(return_value="/out")

        self.files = [
            {"name": "b.tif", "path": "/roll/b.tif", "hash": "hb"},
            {"name": "a.tif", "path": "/roll/a.tif", "hash": "ha"},
        ]
        self.controller.state.uploaded_files = self.files
        session.asset_model.visible_actual_indices_ordered.return_value = [0, 1]
        self.own = {
            "ha": replace(WorkspaceConfig(), geometry=replace(WorkspaceConfig().geometry, rotation=0)),
            "hb": replace(WorkspaceConfig(), geometry=replace(WorkspaceConfig().geometry, rotation=0)),
        }
        session.config_for_asset.side_effect = lambda asset: self.own[asset["hash"]]
        # The active frame is neither of them and is turned: nothing of it may leak.
        self.controller.state.current_file_hash = "other"
        self.controller.state.config = replace(WorkspaceConfig(), geometry=replace(WorkspaceConfig().geometry, rotation=1))

    def tearDown(self):
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

    def _dialog(self, accepted=True):
        dialog_cls = patch("negpy.desktop.view.widgets.contact_sheet_dialog.ContactSheetDialog")
        mock_cls = dialog_cls.start()
        self.addCleanup(dialog_cls.stop)
        dialog = MagicMock()
        dialog.exec.return_value = QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected
        dialog.settings.return_value = ContactSheetSettings(dpi=600)
        dialog.film.return_value = (SheetFormat.FULL_FRAME, "6×6")
        dialog.look.return_value = SheetLook()
        dialog.kept_frames.side_effect = lambda: tuple(mock_cls.call_args.args[0])
        dialog.numbers.side_effect = lambda: tuple(range(len(mock_cls.call_args.args[0])))
        dialog.breaks.return_value = ()
        mock_cls.return_value = dialog
        return mock_cls

    def _prepared(self, birth_a=10.0, birth_b=20.0):
        facts = [FrameFacts(birth_time=birth_b), FrameFacts(birth_time=birth_a)]
        self.controller._on_contact_sheet_prepared(7, facts)

    def test_a_running_batch_blocks_before_anything_is_read(self):
        self.controller._active_batch = "export"
        self.controller._active_batch_title = "Exporting"
        self.controller.request_contact_sheet()
        self.controller.contact_sheet_preview.prepare.assert_not_called()

    def test_no_output_folder_ends_the_request(self):
        self.controller._contact_sheet_output_dir.return_value = None
        self.controller.request_contact_sheet()
        self.controller.contact_sheet_preview.prepare.assert_not_called()

    def test_accepted_dialog_queues_the_frames_in_creation_order_with_their_own_edits(self):
        mock_cls = self._dialog()
        self.controller.request_contact_sheet()
        self.controller.contact_sheet_preview.prepare.assert_called_once()
        self._prepared(birth_a=10.0, birth_b=20.0)

        frames = mock_cls.call_args.args[0]
        self.assertEqual([f.name for f in frames], ["a.tif", "b.tif"])
        self.assertEqual(len(self.jobs), 1)
        job = self.jobs[0]
        self.assertEqual([f.asset["hash"] for f in job.frames], ["ha", "hb"])
        self.assertTrue(all(f.config.geometry.rotation == 0 for f in job.frames))
        self.assertEqual(job.out_dir, "/out")
        self.assertEqual(job.settings.dpi, 600)
        self.assertEqual(self.controller._active_batch, "contact_sheet")
        saved = [c.args for c in self.session.repo.save_global_setting.call_args_list if c.args[0] == "contact_sheet_settings"]
        self.assertEqual(saved[0][1]["dpi"], 600)

    def test_frames_left_out_keep_the_others_roll_numbers(self):
        mock_cls = self._dialog()
        dialog = mock_cls.return_value
        dialog.kept_frames.side_effect = lambda: tuple(mock_cls.call_args.args[0][1:])
        dialog.numbers.side_effect = lambda: (1,)
        self.controller.request_contact_sheet()
        self._prepared()
        job = self.jobs[0]
        self.assertEqual([f.name for f in job.frames], ["b.tif"])
        self.assertEqual(job.numbers, (1,))

    def test_the_dialog_offers_a_straight_proof_from_the_roll_analysis(self):
        mock_cls = self._dialog()
        self.controller.state.active_roll_id = "roll-1"
        baseline = {"floors": (-2.0, -2.0, -2.0), "ceils": (-0.5, -0.5, -0.5), "cast": (0, 0, 0), "axis": None, "outliers": ()}
        with (
            patch("negpy.desktop.controller.rolls.roll_normalization", return_value=baseline),
            patch.object(self.controller, "half_frame_mode_for_roll", return_value=False),
            patch("negpy.desktop.controller.rolls.roll_for_id", return_value={"name": "Roll 1"}),
        ):
            self.controller.request_contact_sheet()
            self._prepared()
        proof = mock_cls.call_args.kwargs["proof"]
        self.assertTrue(proof.available)
        self.assertEqual([f.config.process.locked_floors for f in proof.frames], [(-2.0, -2.0, -2.0)] * 2)

    def test_scene_metering_reaches_the_scene_proof_and_breaks_reach_the_job(self):
        mock_cls = self._dialog()
        self.controller.state.active_roll_id = "roll-1"
        self.files[0]["scene"] = (1, "s1", "Scene 1")
        scene_baseline = {"floors": (-3.0, -3.0, -3.0), "ceils": (-1.0, -1.0, -1.0), "axis": None, "outliers": ()}
        mock_cls.return_value.breaks.return_value = (1,)
        with (
            patch("negpy.desktop.controller.rolls.roll_normalization", return_value=None),
            patch("negpy.desktop.controller.rolls.roll_scenes", return_value=[("s1", {"name": "Scene 1"})]),
            patch("negpy.desktop.controller.rolls.scene_normalization", return_value=scene_baseline),
            patch("negpy.desktop.controller.rolls.roll_for_id", return_value={"name": "Roll 1"}),
            patch.object(self.controller, "half_frame_mode_for_roll", return_value=False),
        ):
            self.controller.request_contact_sheet()
            self._prepared()
        kwargs = mock_cls.call_args.kwargs
        self.assertFalse(kwargs["proof"].available)
        # b.tif (in scene 1) meters on its own; a.tif is in no scene and the roll was never analyzed.
        self.assertFalse(kwargs["scene_proof"].available)
        self.assertEqual(self.jobs[0].breaks, (1,))

    def test_without_roll_analysis_the_proof_says_why(self):
        mock_cls = self._dialog()
        self.controller.request_contact_sheet()
        self._prepared()
        proof = mock_cls.call_args.kwargs["proof"]
        self.assertFalse(proof.available)
        self.assertIn("Roll Analysis", proof.reason)

    def test_rejected_dialog_queues_nothing(self):
        self._dialog(accepted=False)
        self.controller.request_contact_sheet()
        self._prepared()
        self.assertEqual(self.jobs, [])
        self.assertIsNone(self.controller._active_batch)

    def test_a_stale_generation_is_ignored(self):
        mock_cls = self._dialog()
        self.controller.request_contact_sheet()
        self.controller._on_contact_sheet_prepared(3, [FrameFacts(), FrameFacts()])
        mock_cls.assert_not_called()

    def test_a_batch_started_while_the_dialog_is_open_is_named(self):
        self.assertEqual(self.controller._contact_sheet_lane_message(), "")
        self.controller._active_batch = "discovery"
        self.controller._active_batch_title = "Importing"
        self.assertIn("Importing", self.controller._contact_sheet_lane_message())

    def test_finishing_reports_the_folder(self):
        statuses = []
        self.controller.status_message_requested.connect(lambda text, *_: statuses.append(text))
        self.controller._active_batch = "contact_sheet"
        self.controller._on_contact_sheet_written("/out")
        self.controller._on_export_finished()
        self.assertIn("Contact sheet saved to /out", statuses)
