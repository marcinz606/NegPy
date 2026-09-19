"""The filmstrip must not guess a frame's film process when the frame already knows it.

A slide reached the filmstrip inverted while the canvas rendered it as a positive: the
batch path re-detected the mode from an 8-bit preview, and a warm slide reads as an orange
mask. The stored mode is the answer the canvas uses, so the filmstrip uses it too.
"""

import unittest
from unittest.mock import MagicMock, patch

from negpy.desktop.controller import AppController
from negpy.desktop.session import DesktopSessionManager

SLIDE = {"name": "a.nef", "path": "/f/a.nef", "hash": "hash-a"}
FRESH = {"name": "b.nef", "path": "/f/b.nef", "hash": "hash-b"}


def _controller(stored: dict):
    c = MagicMock()
    c.state.uploaded_files = [dict(SLIDE), dict(FRESH)]
    c.state.thumbnails = {}
    c.session.stored_process_mode = lambda asset: stored.get(asset["hash"], "")
    return c


def _emitted(controller) -> list:
    controller.thumbnail_requested.emit.assert_called_once()
    return controller.thumbnail_requested.emit.call_args.args[0]


class BatchRequest(unittest.TestCase):
    def test_stored_mode_rides_along(self):
        controller = _controller({"hash-a": "Transparency"})

        AppController.generate_missing_thumbnails(controller)

        modes = {f["hash"]: f["process_mode"] for f in _emitted(controller)}
        self.assertEqual(modes, {"hash-a": "Transparency", "hash-b": ""})

    def test_the_session_assets_are_not_touched(self):
        """The dicts cross to a worker thread, and a mode written back here would outlive
        the frame's own settings."""
        controller = _controller({"hash-a": "Transparency"})

        AppController.generate_missing_thumbnails(controller)

        self.assertFalse(any("process_mode" in f for f in controller.state.uploaded_files))
        self.assertIsNot(_emitted(controller)[0], controller.state.uploaded_files[0])

    def test_thumbnail_queue_does_not_claim_the_batch_lane(self):
        controller = _controller({})

        AppController.generate_missing_thumbnails(controller)

        controller._begin_batch.assert_not_called()

    def test_foreground_load_cancels_a_running_thumbnail_queue(self):
        controller = _controller({})
        controller.thumb_worker = MagicMock()
        controller.thumbnail_cancel_requested = MagicMock()
        controller._thumbnail_queue_active = True
        controller._thumbnails_paused_for_foreground = False

        AppController._pause_background_thumbnails(controller)

        self.assertTrue(controller._thumbnails_paused_for_foreground)
        controller.thumb_worker.cancel_pending.assert_called_once()
        controller.thumbnail_cancel_requested.emit.assert_called_once()

    def test_completed_thumbnail_queue_is_not_retried_on_file_load(self):
        controller = _controller({})
        controller.thumb_worker = MagicMock()
        controller.thumbnail_cancel_requested = MagicMock()
        controller._thumbnail_queue_active = False
        controller._thumbnails_paused_for_foreground = False

        AppController._pause_background_thumbnails(controller)

        self.assertFalse(controller._thumbnails_paused_for_foreground)
        controller.thumb_worker.cancel_pending.assert_not_called()
        controller.thumbnail_cancel_requested.emit.assert_not_called()

    def test_empty_thumbnail_activity_marks_the_queue_idle(self):
        controller = MagicMock()
        controller._thumbnail_queue_active = True

        AppController._on_thumbnail_activity(controller, "")

        self.assertFalse(controller._thumbnail_queue_active)
        controller.thumbnail_activity_changed.emit.assert_called_once_with("")

    def test_rendered_active_thumbnail_resumes_the_background_queue(self):
        controller = MagicMock()
        controller._thumbnails_paused_for_foreground = True
        controller._foreground_preview_generation = None
        controller._is_rendering = False
        controller._pending_render_task = None

        AppController._resume_background_thumbnails(controller)

        self.assertFalse(controller._thumbnails_paused_for_foreground)
        controller.generate_missing_thumbnails.assert_called_once()

    def test_background_queue_waits_until_foreground_render_is_idle(self):
        controller = MagicMock()
        controller._thumbnails_paused_for_foreground = True
        controller._foreground_preview_generation = None
        controller._is_rendering = True
        controller._pending_render_task = None

        AppController._resume_background_thumbnails(controller)

        self.assertTrue(controller._thumbnails_paused_for_foreground)
        controller.generate_missing_thumbnails.assert_not_called()

    def test_idle_continuation_starts_deferred_prefetch_before_thumbnails(self):
        controller = MagicMock()
        controller._thumbnails_paused_for_foreground = True
        controller._foreground_preview_generation = None
        controller._is_rendering = False
        controller._pending_render_task = None
        controller._active_batch = None
        controller._neighbor_prefetch_generation = 3
        controller._prefetch_gen = 3
        controller._prefetch_in_flight_generation = None
        controller._neighbor_prefetch_queue = []

        AppController._continue_background_work(controller)

        controller._schedule_prefetch_neighbors.assert_called_once_with()
        controller.generate_missing_thumbnails.assert_not_called()

    def test_preview_load_error_does_not_clear_an_unrelated_render(self):
        controller = MagicMock()
        controller._foreground_preview_generation = 2
        controller._neighbor_prefetch_generation = 2
        controller._neighbor_prefetch_queue = [object()]
        controller._is_rendering = True

        AppController._on_preview_load_error(controller, "decode failed")

        self.assertTrue(controller._is_rendering)
        self.assertIsNone(controller._foreground_preview_generation)
        self.assertIsNone(controller._neighbor_prefetch_generation)
        self.assertEqual(controller._neighbor_prefetch_queue, [])
        controller.generate_missing_thumbnails.assert_not_called()


class StoredMode(unittest.TestCase):
    def test_a_composite_uses_the_mode_it_inherited(self):
        session = MagicMock()
        asset = {**SLIDE, "process_mode": "Transparency", "hdr_paths": ("/f/x.nef",)}

        with patch("negpy.desktop.session.load_or_promote") as load:
            self.assertEqual(DesktopSessionManager.stored_process_mode(session, asset), "Transparency")
            load.assert_not_called()

    def test_a_saved_edit_answers(self):
        session = MagicMock()
        saved = MagicMock()
        saved.process.process_mode = "Transparency"

        with patch("negpy.desktop.session.load_or_promote", return_value=saved):
            self.assertEqual(DesktopSessionManager.stored_process_mode(session, dict(SLIDE)), "Transparency")

    def test_an_undecided_frame_answers_nothing(self):
        """Not the sticky global: that is a guess about the next file, and the caller wants
        to know whether an answer exists at all."""
        session = MagicMock()

        with patch("negpy.desktop.session.load_or_promote", return_value=None):
            self.assertEqual(DesktopSessionManager.stored_process_mode(session, dict(FRESH)), "")


if __name__ == "__main__":
    unittest.main()
