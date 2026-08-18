"""The before/after split: what gets painted, what the divider does, and what the
controller keeps.

The baseline half is rendered once and stashed — it is never displayed as the frame — so
the split only appears when those pixels exist, and the edit beside it keeps rendering
normally while the mode is on.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QMouseEvent, QPainter, QPixmap

from negpy.domain.models import WorkspaceConfig

W = H = 200


def _overlay(*, compare_mode: bool = True, before: bool = True):
    from negpy.desktop.session import AppState
    from negpy.desktop.view.canvas.overlay import CanvasOverlay

    state = AppState()
    state.compare_mode = compare_mode
    if before:
        state.compare_before = np.zeros((H, W, 3), dtype=np.float32)
    overlay = CanvasOverlay(state)
    overlay._view_rect = QRectF(0, 0, W, H)
    overlay._current_size = (W, H)
    return overlay


def _press(x: float, y: float) -> QMouseEvent:
    return QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(x, y),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def _move(x: float, y: float) -> QMouseEvent:
    return QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(x, y),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


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


class TestOverlaySplitCondition(unittest.TestCase):
    """The split needs both the mode and the stashed baseline pixels."""

    def _split_drawn(self, **kwargs) -> bool:
        overlay = _overlay(**kwargs)
        pixmap = QPixmap(W, H)
        painter = QPainter(pixmap)
        with patch.object(overlay, "_draw_compare_split") as spy:
            overlay._draw_ui(painter)
        painter.end()
        return spy.called

    def test_toggle_alone_does_not_split_the_still_displayed_edit(self):
        self.assertFalse(self._split_drawn(compare_mode=True, before=False))

    def test_leaving_compare_drops_the_split_even_with_pixels_in_hand(self):
        self.assertFalse(self._split_drawn(compare_mode=False, before=True))

    def test_mode_plus_baseline_pixels_splits(self):
        self.assertTrue(self._split_drawn(compare_mode=True, before=True))


class TestDividerDrag(unittest.TestCase):
    def test_the_divider_is_grabbed_only_near_it(self):
        overlay = _overlay()
        self.assertTrue(overlay._hit_compare_split(QPointF(W * 0.5, H * 0.5)))
        self.assertFalse(overlay._hit_compare_split(QPointF(W * 0.5 + 40, H * 0.5)))
        # Outside the frame vertically is not the divider.
        self.assertFalse(overlay._hit_compare_split(QPointF(W * 0.5, H + 30)))

    def test_a_grab_beats_the_pan_that_would_otherwise_claim_the_click(self):
        overlay = _overlay()
        overlay.zoom_level = 2.0  # left-drag pans at this zoom
        overlay.mousePressEvent(_press(W * 0.5, H * 0.5))
        self.assertTrue(overlay._split_dragging)

    def test_dragging_moves_the_divider_and_clamps_at_the_edges(self):
        overlay = _overlay()
        overlay._split_dragging = True

        overlay.mouseMoveEvent(_move(W * 0.25, H * 0.5))
        self.assertAlmostEqual(overlay.state.compare_split, 0.25, places=3)

        overlay.mouseMoveEvent(_move(-500, H * 0.5))
        self.assertEqual(overlay.state.compare_split, 0.0)
        overlay.mouseMoveEvent(_move(W + 500, H * 0.5))
        self.assertEqual(overlay.state.compare_split, 1.0)

    def test_a_tool_owns_its_clicks(self):
        from negpy.desktop.session import ToolMode

        overlay = _overlay()
        overlay.set_tool_mode(ToolMode.DUST_PICK)
        self.assertFalse(overlay._hit_compare_split(QPointF(W * 0.5, H * 0.5)))


class TestControllerKeepsTheSplit(unittest.TestCase):
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

    def _finish(self, task, **extra) -> None:
        """Stand in for the worker completing `task`."""
        self.controller._is_rendering = False
        metrics = {"base_positive": np.zeros((2, 2, 3), np.float32), "compare": task.compare}
        metrics.update(extra)
        self.controller._on_render_finished(None, metrics)

    def test_the_baseline_is_stashed_not_displayed(self):
        self.controller.toggle_compare()
        task = self.tasks[-1]
        self.assertTrue(task.compare)
        # Full resolution, or one half of the split would be softer than the other.
        self.assertFalse(task.interactive)

        painted: list = []
        self.controller.image_updated.connect(lambda: painted.append(True))
        self._finish(task, content_rect=(0, 0, 2, 2))

        self.assertIsInstance(self.controller.state.compare_before, np.ndarray)
        self.assertEqual(self.controller.state.compare_before_rect, (0, 0, 2, 2))
        self.assertEqual(painted, [])
        self.assertNotIn("base_positive", self.controller.state.last_metrics)

    def test_the_edit_is_printed_again_after_the_baseline_render(self):
        """The engine pool hands every render the same output texture, so the baseline
        render overwrites the edit the canvas samples — both halves would show the
        baseline without a repaint."""
        self.controller.toggle_compare()
        self._finish(self.tasks[-1])

        self.assertFalse(self.tasks[-1].compare)
        self.assertIs(self.tasks[-1].config, self.controller.state.config)

    def test_an_edit_keeps_the_split_and_reuses_the_stashed_baseline(self):
        self.controller.toggle_compare()
        self._finish(self.tasks[-1])
        before = self.controller.state.compare_before

        self.controller.request_render()
        self._finish(self.tasks[-1])

        self.assertTrue(self.controller.state.compare_mode)
        self.assertIs(self.controller.state.compare_before, before)

    def test_a_geometry_change_re_captures_the_baseline(self):
        from dataclasses import replace

        self.controller.toggle_compare()
        self._finish(self.tasks[-1])
        self._finish(self.tasks[-1])  # the repaint of the edit

        self.controller.state.config = replace(
            self.controller.state.config,
            geometry=replace(self.controller.state.config.geometry, rotation=90),
        )
        self.controller.request_render()
        self._finish(self.tasks[-1])

        self.assertTrue(self.tasks[-1].compare)

    def test_toggling_off_drops_the_stashed_frame(self):
        self.controller.toggle_compare()
        self._finish(self.tasks[-1])

        self.controller.toggle_compare()
        self.assertFalse(self.controller.state.compare_mode)
        self.assertIsNone(self.controller.state.compare_before)

    def test_leaving_the_frame_drops_the_split(self):
        self.controller.toggle_compare()
        self._finish(self.tasks[-1])

        with patch.object(self.controller, "_file_hash_for_path", return_value=None):
            self.controller.load_file("/nowhere/frame.tif")
        self.assertFalse(self.controller.state.compare_mode)
        self.assertIsNone(self.controller.state.compare_before)


if __name__ == "__main__":
    unittest.main()
