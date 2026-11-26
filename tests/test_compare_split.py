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

    def test_compare_hold_disables_divider_hit_test(self):
        overlay = _overlay()
        overlay.state.compare_hold = True
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
            self.controller.prefetch_load_thread,
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
        seen_on_paint: list[object] = []
        self.controller.image_updated.connect(lambda: seen_on_paint.append(self.controller.state.compare_before))
        self.controller.request_render()
        self._finish(self.tasks[-1])

        # Stale unrotated baseline is dropped BEFORE painting the rotated edit.
        self.assertEqual(seen_on_paint, [None])
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

    def _step_config(self) -> WorkspaceConfig:
        from dataclasses import replace

        config = self.controller.state.config
        return replace(config, exposure=replace(config.exposure, density=config.exposure.density + 0.3))

    def test_a_history_step_is_the_before_half(self):
        step = self._step_config()
        self.controller.state.current_file_hash = "abc"
        self.controller.state.undo_index = 3
        self.controller.session.repo.load_history_step.return_value = step

        self.controller.compare_with_history_step(1)

        self.controller.session.repo.load_history_step.assert_called_once_with("abc", 1)
        self.assertTrue(self.controller.state.compare_mode)
        self.assertTrue(self.tasks[-1].compare)
        self.assertEqual(self.tasks[-1].config.exposure, step.exposure)
        self.assertEqual(self.controller.state.compare_reference_label, "STEP 1")

    def test_a_new_reference_re_captures_the_open_split(self):
        self.controller.state.current_file_hash = "abc"
        self.controller.toggle_compare()
        self._finish(self.tasks[-1])
        self._finish(self.tasks[-1])  # the repaint of the edit

        step = self._step_config()
        self.controller.session.repo.load_work_print.return_value = step
        self.controller.compare_with_work_print("Warm")

        self.assertTrue(self.controller.state.compare_mode)
        self.assertIsNone(self.controller.state.compare_before)
        self.assertTrue(self.tasks[-1].compare)
        self.assertEqual(self.tasks[-1].config.exposure, step.exposure)
        self.assertEqual(self.controller.state.compare_reference_label, "WARM")

    def test_leaving_the_split_forgets_the_reference(self):
        self.controller.state.current_file_hash = "abc"
        self.controller.session.repo.load_work_print.return_value = self._step_config()
        self.controller.compare_with_work_print("Warm")

        self.controller.toggle_compare()
        self.assertIsNone(self.controller.state.compare_reference)
        self.assertEqual(self.controller.state.compare_reference_label, "")
        from negpy.desktop.controller import baseline_compare_config

        self.assertEqual(self.controller._baseline_compare_config(), baseline_compare_config(self.controller.state.config))

    def test_gpu_baseline_clones_in_vram_without_host_readback(self):
        class _FakeGPUTex:
            def __init__(self):
                self.width, self.height = 4096, 3276
                self.view = object()
                self.readbacks = 0
                self.clones = 0
                self.destroyed = False

            def clone_gpu(self):
                self.clones += 1
                return _FakeGPUTex()

            def readback(self):
                self.readbacks += 1
                return np.zeros((2, 2, 4), dtype=np.float32)

            def destroy(self):
                self.destroyed = True

        self.controller.state.gpu_enabled = True
        self.controller.state.gpu_viewport_failed = False
        staged_events: list[bool] = []
        ready_events: list[bool] = []
        self.controller.compare_before_staged.connect(lambda dirty: staged_events.append(dirty))
        self.controller.compare_frame_ready.connect(lambda: ready_events.append(True))

        self.controller.toggle_compare()
        task = self.tasks[-1]

        pool_tex = _FakeGPUTex()
        with patch("negpy.desktop.controller.GPUTexture", _FakeGPUTex):
            self.controller._is_rendering = False
            self.controller._on_render_finished(
                None, {"base_positive": pool_tex, "compare": task.compare, "content_rect": (0, 0, 4096, 3276)}
            )
            stashed = self.controller.state.compare_before
            self.assertIsInstance(stashed, _FakeGPUTex)
            self.assertIsNot(stashed, pool_tex)
            self.assertEqual(pool_tex.clones, 1)
            self.assertEqual(pool_tex.readbacks, 0)
            # Split stages without prematurely redrawing while pool_tex holds BEFORE.
            self.assertEqual(len(staged_events), 1)
            self.assertEqual(len(ready_events), 0)

            # Signal detaches canvas view BEFORE hardware destroy() runs.
            detached_before_destroy: list[bool] = []
            self.controller.compare_changed.connect(
                lambda _on: detached_before_destroy.append(self.controller.state.compare_before is None and not stashed.destroyed)
            )
            self.controller.exit_compare()
            self.assertEqual(detached_before_destroy, [True])
            self.assertTrue(stashed.destroyed)
            self.assertIsNone(self.controller.state.compare_before)

    def test_gpu_hold_lifecycle_and_stale_invalidation(self):
        from dataclasses import replace

        class _FakeGPUTex:
            def __init__(self):
                self.width, self.height = 1000, 800
                self.view = object()
                self.destroyed = False

            def clone_gpu(self):
                return _FakeGPUTex()

            def destroy(self):
                self.destroyed = True

        self.controller.state.gpu_enabled = True
        self.controller.state.gpu_viewport_failed = False
        with patch("negpy.desktop.controller.GPUTexture", _FakeGPUTex):
            self.controller.begin_compare_hold()
            task = self.tasks[-1]
            self.controller._is_rendering = False
            self.controller._on_render_finished(None, {"base_positive": _FakeGPUTex(), "compare": task.compare})
            first_tex = self.controller.state.compare_before
            self.assertIsInstance(first_tex, _FakeGPUTex)
            self.controller.end_compare_hold()
            self.assertFalse(first_tex.destroyed)

            # Rotating makes first_tex stale; next hold drops it before compare_hold_changed(True).
            self.controller.state.config = replace(
                self.controller.state.config,
                geometry=replace(self.controller.state.config.geometry, rotation=90),
            )
            seen_during_hold: list[object] = []
            self.controller.compare_hold_changed.connect(lambda on: seen_during_hold.append((on, self.controller.state.compare_before)))
            self.controller.begin_compare_hold()
            self.assertTrue(first_tex.destroyed)
            self.assertEqual(seen_during_hold[-1], (True, None))

            # Exiting compare while hold is active clears compare_hold and detaches before destroy.
            second_tex = _FakeGPUTex()
            self.controller.state.compare_before = second_tex
            order_ok: list[bool] = []
            self.controller.compare_hold_changed.connect(
                lambda on: order_ok.append((not on) and self.controller.state.compare_before is None and not second_tex.destroyed)
            )
            self.controller.exit_compare()
            self.assertFalse(self.controller.state.compare_hold)
            self.assertIn(True, order_ok)
            self.assertTrue(second_tex.destroyed)

    def test_disabling_gpu_drops_vram_baseline(self):
        class _FakeGPUTex:
            def __init__(self):
                self.destroyed = False

            def destroy(self):
                self.destroyed = True

        tex = _FakeGPUTex()
        with patch("negpy.desktop.controller.GPUTexture", _FakeGPUTex):
            self.controller.state.gpu_enabled = False
            self.controller.state.compare_before = tex
            self.controller._render_if_state_moved()
        self.assertTrue(tex.destroyed)
        self.assertIsNone(self.controller.state.compare_before)

    def test_session_emptied_exits_compare_and_destroys_vram_baseline(self):
        self.mock_session_manager.session_emptied.connect.assert_any_call(self.controller.exit_compare)
        tex = MagicMock()
        self.controller.state.compare_mode = True
        self.controller.state.compare_before = tex
        self.controller.exit_compare()
        self.assertFalse(self.controller.state.compare_mode)
        self.assertIsNone(self.controller.state.compare_before)
        tex.destroy.assert_called_once()

    def test_toggle_compare_reuses_fresh_hold_baseline_and_drops_mismatched_one(self):
        class _FakeGPUTex:
            def __init__(self):
                self.width, self.height = 1000, 800
                self.view = object()
                self.destroyed = False

            def clone_gpu(self):
                return _FakeGPUTex()

            def destroy(self):
                self.destroyed = True

        self.controller.state.gpu_enabled = True
        self.controller.state.gpu_viewport_failed = False
        with patch("negpy.desktop.controller.GPUTexture", _FakeGPUTex):
            self.controller.begin_compare_hold()
            self._finish(self.tasks[-1], base_positive=_FakeGPUTex())
            self._finish(self.tasks[-1], base_positive=_FakeGPUTex())  # edit reprint
            self.controller.end_compare_hold()
            retained = self.controller.state.compare_before
            self.assertIsInstance(retained, _FakeGPUTex)

            # Matching baseline key: toggle_compare reuses `retained` with zero extra renders.
            before_count = len(self.tasks)
            self.controller.toggle_compare()
            self.assertEqual(len(self.tasks), before_count)
            self.assertIs(self.controller.state.compare_before, retained)

            # Reset to non-split with retained baseline, then open work-print compare:
            # mismatched baseline is dropped BEFORE compare_changed(True) fires.
            self.controller.state.compare_mode = False
            self.controller.state.current_file_hash = "abc"
            self.controller.session.repo.load_work_print.return_value = self._step_config()
            seen_on_open: list[object] = []
            self.controller.compare_changed.connect(lambda on: seen_on_open.append((on, self.controller.state.compare_before)))
            self.controller.compare_with_work_print("Warm")
            self.assertTrue(retained.destroyed)
            self.assertEqual(seen_on_open[-1], (True, None))
            self.assertTrue(self.tasks[-1].compare)


class TestGPUOverlayZOrderParity(unittest.TestCase):
    def test_awaiting_edit_latches_hold_and_blocks_pool_readback_until_edit_lands(self):
        from negpy.desktop.session import AppState
        from negpy.desktop.view.canvas.widget import ImageCanvas

        class _FakeTex:
            width, height = W, H
            view = object()

            def readback(self):
                raise AssertionError("overwritten pool texture must not be read back while awaiting edit")

            def readback_region(self, *_a):
                raise AssertionError("overwritten pool texture must not be probed while awaiting edit")

        state = AppState()
        state.gpu_enabled = True
        state.gpu_viewport_failed = ""
        with patch("negpy.desktop.view.canvas.widget.GPUDevice") as mock_dev:
            mock_dev.get.return_value.is_available = False
            canvas = ImageCanvas(state)

        calls: list[tuple[int, bool]] = []
        canvas.gpu_widget.set_compare_state = lambda _b, _br, _ar, _s, mode, *, redraw=True: calls.append((mode, redraw))

        with (
            patch("negpy.desktop.view.canvas.widget.GPUTexture", _FakeTex),
            patch("negpy.desktop.view.canvas.overlay.GPUTexture", _FakeTex),
        ):
            pool_tex = _FakeTex()
            canvas._last_buffer = pool_tex
            canvas.overlay._gpu_texture = pool_tex
            state.compare_mode = True
            state.compare_before = _FakeTex()
            canvas.stage_compare_before(True)
            self.assertTrue(canvas._compare_awaiting_edit)
            self.assertEqual(calls[-1], (0, False))
            self.assertIsNone(canvas.get_pixel_rgb(0.5, 0.5))
            self.assertIsNone(canvas.overlay._host_buffer())

            # Exiting compare before edit lands keeps _compare_awaiting_edit True (redraw=False)
            # so the dirty pool_tex is neither drawn nor read back.
            state.compare_mode = False
            state.compare_before = None
            canvas.refresh_compare()
            self.assertTrue(canvas._compare_awaiting_edit)
            self.assertEqual(calls[-1], (0, False))
            self.assertIsNone(canvas.get_pixel_rgb(0.5, 0.5))

            state.compare_mode = True
            state.compare_before = _FakeTex()
            canvas.stage_compare_before(True)

            # Tapping and releasing hold while edit render is still in flight latches mode 2
            # (since BEFORE was drawn to the GPU surface) until update_buffer restores `tex`.
            state.compare_hold = True
            canvas.refresh_compare()
            self.assertTrue(canvas._compare_awaiting_edit)
            self.assertEqual(calls[-1], (2, True))

            state.compare_hold = False
            canvas.refresh_compare()
            self.assertTrue(canvas._compare_awaiting_edit)
            self.assertTrue(canvas.hold_latched())
            self.assertEqual(calls[-1], (2, True))
            self.assertFalse(canvas.overlay._compare_split_active())

            # Once the edit render lands via update_buffer, hold latch clears and split (mode 1) activates.
            canvas.update_buffer(_FakeTex(), "sRGB", (0, 0, W, H))
            self.assertFalse(canvas._compare_awaiting_edit)
            self.assertFalse(canvas.hold_latched())
            self.assertEqual(calls[-1], (1, True))

    def test_gpu_split_clips_edit_marks_off_before_half_and_hold_suppresses_them(self):
        from PyQt6.QtWidgets import QWidget

        class _FakeGPU(QWidget):
            def isVisible(self):
                return True

        class _FakeParent(QWidget):
            def __init__(self):
                super().__init__()
                self.gpu_widget = _FakeGPU(self)
                self._compare_awaiting_edit = False

        class _FakeTex:
            width, height = W, H

        parent = _FakeParent()
        overlay = _overlay(compare_mode=True, before=False)
        overlay.setParent(parent)
        overlay.resize(W, H)
        overlay._rotation_grid_visible = True

        with patch("negpy.desktop.view.canvas.overlay.GPUTexture", _FakeTex):
            overlay.state.compare_before = _FakeTex()
            overlay.state.compare_split = 0.5
            pm_split = QPixmap(W, H)
            pm_split.fill(Qt.GlobalColor.transparent)
            p = QPainter(pm_split)
            overlay._draw_ui(p)
            p.end()
            img_split = pm_split.toImage()
            # Lower-left quadrant (below BEFORE badge, left of split x=100) has no rotation grid lines.
            left_marks = sum(1 for y in range(120, 190) for x in range(10, 85) if img_split.pixelColor(x, y).alpha() > 0)
            right_marks = sum(1 for y in range(120, 190) for x in range(115, 190) if img_split.pixelColor(x, y).alpha() > 0)
            self.assertEqual(left_marks, 0)
            self.assertGreater(right_marks, 0)

            # During compare_hold, edit marks are suppressed across BOTH halves.
            overlay.state.compare_mode = False
            overlay.state.compare_hold = True
            pm_hold = QPixmap(W, H)
            pm_hold.fill(Qt.GlobalColor.transparent)
            p2 = QPainter(pm_hold)
            overlay._draw_ui(p2)
            p2.end()
            img_hold = pm_hold.toImage()
            lower_marks = sum(1 for y in range(120, 190) for x in range(10, 190) if img_hold.pixelColor(x, y).alpha() > 0)
            self.assertEqual(lower_marks, 0)


class TestPeekBadge(unittest.TestCase):
    """A peek replaces the print with something that is not one. The badge is what says so
    on the canvas itself; the toolbar toggle that opened it is off at the edge of the view."""

    @staticmethod
    def _painted(**flags) -> QPixmap:
        overlay = _overlay(compare_mode=False, before=False)
        for name, value in flags.items():
            setattr(overlay.state, name, value)
        pixmap = QPixmap(W, H)
        pixmap.fill()
        painter = QPainter(pixmap)
        overlay._draw_ui(painter)
        painter.end()
        return pixmap

    @staticmethod
    def _badge_pixels(pixmap: QPixmap) -> int:
        """Non-white pixels in the badge's corner of an otherwise blank canvas."""
        image = pixmap.toImage()
        return sum(1 for y in range(12, 34) for x in range(12, 120) if image.pixelColor(x, y).lightness() < 200)

    def test_each_peek_paints_a_badge(self):
        self.assertGreater(self._badge_pixels(self._painted(negative_peek=True)), 100)
        self.assertGreater(self._badge_pixels(self._painted(embedded_peek=True)), 100)
        self.assertGreater(self._badge_pixels(self._painted(flat_peek=True)), 100)
        self.assertGreater(self._badge_pixels(self._painted(flatfield_peek=True)), 100)

    def test_the_plain_edit_paints_none(self):
        self.assertEqual(self._badge_pixels(self._painted()), 0)


if __name__ == "__main__":
    unittest.main()
