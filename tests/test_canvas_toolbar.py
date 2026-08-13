import sys
import unittest
from unittest.mock import MagicMock

from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.canvas.toolbar import ActionToolbar
from negpy.domain.models import WorkspaceConfig

if not QApplication.instance():
    _app = QApplication(sys.argv)


def _make_toolbar() -> ActionToolbar:
    controller = MagicMock()
    controller.session = MagicMock()
    controller.session.state = MagicMock()
    controller.session.state.gpu_enabled = False
    controller.session.state.hq_preview = False
    controller.session.state.compare_mode = False
    controller.session.state.flat_peek = False
    controller.session.state.zones_overlay = False
    controller.session.state.grain_focuser = False
    controller.session.state.selected_file_idx = 0
    controller.session.state.undo_index = 0
    controller.session.state.max_history_index = 0
    controller.session.state.clipboard = None
    controller.session.state.config.geometry.flip_horizontal = False
    controller.session.state.config.geometry.flip_vertical = False
    controller.session.state.canvas_bg_index = 0
    controller.session.asset_model.actual_to_display.return_value = 0
    controller.session.asset_model.rowCount.return_value = 1
    controller.session.repo.get_global_setting.return_value = 1.0
    controller.canvas = None
    controller.render_worker.processor.backend_name = "CPU"
    return ActionToolbar(controller)


def _visible_group_count(tb: ActionToolbar) -> int:
    return sum(1 for group in tb._collapse_groups if any(w.isVisible() for w in group))


class TestCanvasToolbarResponsive(unittest.TestCase):
    def test_pill_width_never_exceeds_budget(self):
        tb = _make_toolbar()
        tb.show()
        QApplication.processEvents()

        for canvas_w in (480, 640, 800, 1200):
            tb.set_available_width(canvas_w)
            QApplication.processEvents()
            self.assertLessEqual(tb._pill_width(), tb._toolbar_width_budget(canvas_w))

    def test_wider_canvas_shows_more_controls(self):
        tb = _make_toolbar()
        tb.show()
        QApplication.processEvents()

        counts: list[int] = []
        widths: list[int] = []
        for canvas_w in (480, 640, 800, 1600):
            tb.set_available_width(canvas_w)
            QApplication.processEvents()
            counts.append(_visible_group_count(tb))
            widths.append(tb._pill_width())

        for prev, nxt in zip(counts[:-1], counts[1:], strict=True):
            self.assertGreaterEqual(nxt, prev)
        for prev, nxt in zip(widths[:-1], widths[1:], strict=True):
            self.assertGreaterEqual(nxt, prev)

    def test_core_controls_always_visible(self):
        tb = _make_toolbar()
        tb.show()
        QApplication.processEvents()

        for canvas_w in (480, 800, 1200):
            tb.set_available_width(canvas_w)
            QApplication.processEvents()
            self.assertTrue(tb.btn_prev.isVisible())
            self.assertTrue(tb.btn_next.isVisible())
            self.assertTrue(tb.btn_overflow.isVisible())

    def test_full_width_shows_all_optional_groups(self):
        tb = _make_toolbar()
        tb.show()
        QApplication.processEvents()

        tb.set_available_width(2000)
        QApplication.processEvents()

        self.assertEqual(_visible_group_count(tb), len(tb._collapse_groups))
        self.assertTrue(tb.btn_compare.isVisible())
        self.assertTrue(tb.btn_undo.isVisible())
        self.assertTrue(tb.btn_zoom_fit.isVisible())

    def test_every_visible_control_shares_one_row_height(self):
        """btn_zoom_fit/btn_zoom_original once skipped the standard sizing loop and fell
        back to style defaults (29px and 25px against everyone else's 32), leaving their
        hover rectangles visibly short of their neighbours'."""
        tb = _make_toolbar()
        tb.set_available_width(1600)
        tb.show()
        QApplication.processEvents()

        row = [
            tb.btn_toggle_left,
            tb.btn_prev,
            tb.btn_next,
            tb.zoom_label,
            tb.btn_zoom_fit,
            tb.btn_zoom_original,
            tb.btn_hq,
            tb.btn_rot_l,
            tb.btn_rot_r,
            tb.btn_flip_h,
            tb.btn_flip_v,
            tb.btn_undo,
            tb.btn_redo,
            tb.btn_compare,
            tb.btn_zones,
            tb.btn_loupe,
            tb.btn_overflow,
            tb.btn_toggle_right,
        ]
        edges = {(w.geometry().y(), w.geometry().bottom()) for w in row if w.isVisible()}
        self.assertEqual(len(edges), 1, f"toolbar controls disagree on top/bottom edges: {sorted(edges)}")

    def _all_overflow_actions(self, tb: ActionToolbar) -> list:
        return [
            tb._ov_hq_action,
            tb._ov_gpu_action,
            *tb._ov_color_actions,
            tb._ov_fit_action,
            tb._ov_original_action,
            tb._ov_compare_action,
            tb._ov_flat_peek_action,
            tb._ov_zones_action,
            tb._ov_loupe_action,
            tb._ov_undo_action,
            tb._ov_redo_action,
            tb._ov_rot_l_action,
            tb._ov_rot_r_action,
            tb._ov_flip_h_action,
            tb._ov_flip_v_action,
        ]

    def test_grain_focuser_button_drives_the_controller(self):
        tb = _make_toolbar()
        tb.btn_loupe.click()
        tb.controller.toggle_grain_focuser.assert_called_once_with(force=True)

        # A programmatic state sync must not echo back as another toggle.
        tb._on_grain_focuser_changed(False)
        self.assertFalse(tb.btn_loupe.isChecked())
        tb.controller.toggle_grain_focuser.assert_called_once_with(force=True)

    def test_ring_around_is_not_in_the_toolbar(self):
        tb = _make_toolbar()
        labels = [a.text() for a in tb.btn_overflow.menu().actions()]
        self.assertNotIn("Color Ring-Around", labels)
        self.assertFalse(hasattr(tb, "_ov_ring_action"))

    def test_overflow_menu_always_shows_full_action_set(self):
        """Regression: the overflow menu previously mirrored only whatever the row's
        responsive collapse hid, so a control moving into the row (e.g. a side panel
        toggle freeing up width) made it vanish from the menu too. The menu must stay
        complete regardless of how much of the row is currently collapsed."""
        tb = _make_toolbar()
        tb.show()
        QApplication.processEvents()

        for canvas_w in (320, 480, 640, 800, 1200, 2000):
            tb.set_available_width(canvas_w)
            QApplication.processEvents()
            for action in self._all_overflow_actions(tb):
                self.assertTrue(action.isVisible(), f"{action.text()!r} hidden from overflow at width {canvas_w}")

    def test_narrow_canvas_still_shows_row_controls_via_overflow(self):
        tb = _make_toolbar()
        tb.show()
        QApplication.processEvents()

        tb.set_available_width(480)
        QApplication.processEvents()
        narrow_count = _visible_group_count(tb)

        tb.set_available_width(2000)
        QApplication.processEvents()
        wide_count = _visible_group_count(tb)

        self.assertLess(narrow_count, wide_count)
        # Whatever the row hides at the narrow width, the overflow copy still works.
        self.assertTrue(tb._ov_compare_action.isVisible())
        self.assertTrue(tb._ov_undo_action.isVisible())


class TestRotateRouting(unittest.TestCase):
    """The 90° rotate buttons are shared: a proof on the canvas takes the turn instead of
    the image, so the ladder can be brought onto a different part of the frame."""

    def test_a_proof_on_the_canvas_takes_the_rotation(self):
        tb = _make_toolbar()
        tb.controller.rotate_test_strip.return_value = True

        tb.rotate(1)

        tb.controller.rotate_test_strip.assert_called_once_with(1)
        tb.session.update_config.assert_not_called()

    def test_with_no_proof_up_the_image_still_rotates(self):
        tb = _make_toolbar()
        tb.controller.rotate_test_strip.return_value = False
        tb.session.state.config = WorkspaceConfig()

        tb.rotate(1)

        tb.session.update_config.assert_called_once()
        self.assertEqual(tb.session.update_config.call_args.args[0].geometry.rotation, 1)
        tb.controller.rerender_active_view.assert_called_once()


if __name__ == "__main__":
    unittest.main()
