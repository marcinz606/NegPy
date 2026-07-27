"""The BEFORE badge must track the frame on screen, not the toggle.

Compare flips state.compare_mode and *then* requests a render; on a slow render (HQ,
large scan) the badge would appear over the still-displayed edit, labelling the after
image as the before one. The flag travels with the render task and comes back in
metrics, so the badge only lights once the baseline pixels land.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from negpy.domain.models import WorkspaceConfig


class TestCompareFlagRoundTrip(unittest.TestCase):
    def test_worker_echoes_the_compare_flag_into_metrics(self):
        with patch("negpy.desktop.workers.render.ImageProcessor") as MockIP:
            from negpy.desktop.workers.render import RenderTask, RenderWorker

            # Fresh metrics dict per call — a shared one would be mutated by the second render.
            MockIP.return_value.run_pipeline.side_effect = lambda *a, **k: (np.zeros((2, 2, 3), np.float32), {})
            worker = RenderWorker()
            seen: list = []
            worker.finished.connect(lambda _r, m: seen.append(m))

            common = dict(buffer=np.zeros((2, 2, 3), np.float32), config=WorkspaceConfig(), preview_size=512.0)
            worker.process(RenderTask(source_hash="f1", compare=True, **common))
            worker.process(RenderTask(source_hash="f1", compare=False, **common))

        self.assertTrue(seen[0]["compare"])
        self.assertFalse(seen[1]["compare"])


class TestOverlayBadgeCondition(unittest.TestCase):
    """The badge is keyed off the painted frame's metrics, not state.compare_mode."""

    def _badge_drawn(self, *, compare_mode: bool, painted_compare: bool) -> bool:
        from PyQt6.QtCore import QRectF
        from PyQt6.QtGui import QPainter, QPixmap

        from negpy.desktop.session import AppState
        from negpy.desktop.view.canvas.overlay import CanvasOverlay

        state = AppState()
        state.compare_mode = compare_mode
        state.last_metrics["compare"] = painted_compare
        overlay = CanvasOverlay(state)
        overlay._view_rect = QRectF(0, 0, 200, 160)

        pixmap = QPixmap(200, 160)
        painter = QPainter(pixmap)
        with patch.object(overlay, "_draw_compare_badge") as badge:
            overlay._draw_ui(painter)
        painter.end()
        return badge.called

    def test_toggle_alone_does_not_badge_the_still_displayed_edit(self):
        self.assertFalse(self._badge_drawn(compare_mode=True, painted_compare=False))

    def test_baseline_frame_badges_even_after_the_toggle_flips_back(self):
        self.assertTrue(self._badge_drawn(compare_mode=False, painted_compare=True))


class TestCompareBadgeFollowsPaintedFrame(unittest.TestCase):
    """Overlay reads last_metrics['compare']; assert what that dict holds over a toggle."""

    def setUp(self):
        from negpy.desktop.controller import AppController
        from negpy.desktop.session import AppState, DesktopSessionManager
        from negpy.services.rendering.preview_manager import PreviewManager

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
        self.tasks: list = []
        self.controller.render_requested.connect(self.tasks.append)

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

    def _finish(self, task) -> None:
        """Stand in for the worker completing `task`."""
        self.controller._on_render_finished(None, {"base_positive": np.zeros((2, 2, 3), np.float32), "compare": task.compare})

    def test_badge_waits_for_the_baseline_render(self):
        self.controller.toggle_compare()
        self.assertTrue(self.controller.state.compare_mode)
        # Render still in flight: the edit is on screen, so no badge yet.
        self.assertFalse(self.controller.state.last_metrics.get("compare", False))

        self._finish(self.tasks[-1])
        self.assertTrue(self.controller.state.last_metrics["compare"])

    def test_badge_persists_until_the_edit_is_repainted(self):
        self.controller.toggle_compare()
        self._finish(self.tasks[-1])

        self.controller._is_rendering = False
        self.controller.toggle_compare()
        self.assertFalse(self.controller.state.compare_mode)
        # Baseline pixels are still displayed — badge stays up until the edit lands.
        self.assertTrue(self.controller.state.last_metrics["compare"])

        self._finish(self.tasks[-1])
        self.assertFalse(self.controller.state.last_metrics["compare"])

    def test_splash_and_memo_repaints_clear_a_stale_badge(self):
        self.controller.state.last_metrics["compare"] = True
        self.controller._on_splash_preview(self.controller._requested_file_path, np.zeros((2, 2, 3), np.float32), (2, 2))
        self.assertFalse(self.controller.state.last_metrics["compare"])


if __name__ == "__main__":
    unittest.main()
