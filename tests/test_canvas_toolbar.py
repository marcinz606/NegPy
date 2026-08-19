import sys
import unittest
from unittest.mock import MagicMock

from PyQt6.QtWidgets import QApplication, QDialog

from negpy.desktop.view.canvas.toolbar import DEFAULT_TOOLBAR_IDS, TOOLBAR_ITEM_BY_ID, ActionToolbar, load_toolbar_items
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


def _visible_item_count(tb: ActionToolbar) -> int:
    return sum(1 for item_id in tb._item_ids if not tb._row_widgets[item_id].isHidden())


def _row_order(tb: ActionToolbar) -> list[str]:
    """Ids in layout order, so a reorder is checked against the real row, not the stored list."""
    by_widget = {widget: item_id for item_id, widget in tb._row_widgets.items()}
    layout = tb._row_layout
    return [by_widget[w] for i in range(layout.count()) if (w := layout.itemAt(i).widget()) in by_widget]


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
            counts.append(_visible_item_count(tb))
            widths.append(tb._pill_width())

        for prev, nxt in zip(counts[:-1], counts[1:], strict=True):
            self.assertGreaterEqual(nxt, prev)
        for prev, nxt in zip(widths[:-1], widths[1:], strict=True):
            self.assertGreaterEqual(nxt, prev)

    def test_anchors_and_core_controls_always_visible(self):
        tb = _make_toolbar()
        tb.show()
        QApplication.processEvents()

        for canvas_w in (480, 800, 1200):
            tb.set_available_width(canvas_w)
            QApplication.processEvents()
            self.assertTrue(tb.btn_prev.isVisible())
            self.assertTrue(tb.btn_next.isVisible())
            self.assertTrue(tb.btn_overflow.isVisible())
            self.assertTrue(tb.btn_toggle_left.isVisible())
            self.assertTrue(tb.btn_toggle_right.isVisible())

    def test_full_width_shows_every_chosen_control(self):
        tb = _make_toolbar()
        tb.show()
        QApplication.processEvents()

        tb.set_available_width(2000)
        QApplication.processEvents()

        self.assertEqual(_visible_item_count(tb), len(tb._item_ids))
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
        narrow_count = _visible_item_count(tb)

        tb.set_available_width(2000)
        QApplication.processEvents()
        wide_count = _visible_item_count(tb)

        self.assertLess(narrow_count, wide_count)
        # Whatever the row hides at the narrow width, the overflow copy still works.
        self.assertTrue(tb._ov_compare_action.isVisible())
        self.assertTrue(tb._ov_undo_action.isVisible())


class TestToolbarCustomization(unittest.TestCase):
    def test_default_row_is_used_when_nothing_is_stored(self):
        tb = _make_toolbar()
        self.assertEqual(tb._item_ids, list(DEFAULT_TOOLBAR_IDS))
        self.assertEqual(_row_order(tb), list(DEFAULT_TOOLBAR_IDS))

    def test_load_drops_unknown_ids(self):
        repo = MagicMock()
        repo.get_global_setting.return_value = ["loupe", "retired_button", "prev"]
        self.assertEqual(load_toolbar_items(repo), ["loupe", "prev"])

    def test_load_tolerates_a_malformed_setting(self):
        repo = MagicMock()
        repo.get_global_setting.return_value = "prev"
        self.assertEqual(load_toolbar_items(repo), list(DEFAULT_TOOLBAR_IDS))

    def test_stored_order_drives_the_row(self):
        tb = _make_toolbar()
        tb.controller.session.repo.get_global_setting.return_value = ["loupe", "prev", "undo"]
        tb._item_ids = load_toolbar_items(tb.session.repo)
        tb._rebuild_row()

        self.assertEqual(_row_order(tb), ["loupe", "prev", "undo"])
        self.assertTrue(tb.btn_next.isHidden())

    def test_a_separator_sits_at_every_visible_category_boundary(self):
        tb = _make_toolbar()
        tb.show()
        tb.set_available_width(2000)
        QApplication.processEvents()

        categories = [TOOLBAR_ITEM_BY_ID[i].category for i in tb._item_ids]
        boundaries = sum(1 for a, b in zip(categories[:-1], categories[1:], strict=True) if a != b)
        visible_separators = sum(1 for widget, is_sep in tb._row_sequence if is_sep and not widget.isHidden())
        self.assertEqual(visible_separators, boundaries)

    def test_no_dangling_separator_once_the_row_collapses(self):
        tb = _make_toolbar()
        tb.show()
        QApplication.processEvents()

        for canvas_w in (320, 480, 640, 900):
            tb.set_available_width(canvas_w)
            QApplication.processEvents()
            visible = [(w, is_sep) for w, is_sep in tb._row_sequence if not w.isHidden()]
            if not visible:
                continue
            self.assertFalse(visible[0][1], f"row starts with a separator at width {canvas_w}")
            self.assertFalse(visible[-1][1], f"row ends with a separator at width {canvas_w}")

    def test_collapse_eats_from_the_end_of_the_users_order(self):
        tb = _make_toolbar()
        tb.show()
        QApplication.processEvents()

        tb.set_available_width(480)
        QApplication.processEvents()
        visible = [i for i in tb._item_ids if not tb._row_widgets[i].isHidden()]
        self.assertLess(len(visible), len(tb._item_ids))
        self.assertEqual(visible, tb._item_ids[: len(visible)])

    def test_flat_peek_can_be_put_on_the_row_and_drives_the_controller(self):
        tb = _make_toolbar()
        tb._item_ids = ["flat_peek"]
        tb._rebuild_row()

        self.assertEqual(_row_order(tb), ["flat_peek"])
        tb.btn_flat_peek.click()
        tb.controller.toggle_flat_peek.assert_called_once_with(force=True)

    def test_editing_the_toolbar_saves_and_rebuilds(self):
        tb = _make_toolbar()
        with unittest.mock.patch("negpy.desktop.view.widgets.favourites_dialog.FavouritesDialog") as dialog_cls:
            dialog = dialog_cls.return_value
            dialog.exec.return_value = QDialog.DialogCode.Accepted
            dialog.selected_ids.return_value = ["next", "prev"]
            tb.open_toolbar_editor()

        tb.session.repo.save_global_setting.assert_called_once_with("toolbar_items", ["next", "prev"])
        self.assertEqual(_row_order(tb), ["next", "prev"])

    def test_a_cancelled_edit_changes_nothing(self):
        tb = _make_toolbar()
        with unittest.mock.patch("negpy.desktop.view.widgets.favourites_dialog.FavouritesDialog") as dialog_cls:
            dialog_cls.return_value.exec.return_value = QDialog.DialogCode.Rejected
            tb.open_toolbar_editor()

        tb.session.repo.save_global_setting.assert_not_called()
        self.assertEqual(_row_order(tb), list(DEFAULT_TOOLBAR_IDS))

    def test_the_overflow_menu_stays_complete_with_an_empty_row(self):
        tb = _make_toolbar()
        tb._item_ids = []
        tb._rebuild_row()
        tb.show()
        tb.set_available_width(480)
        QApplication.processEvents()

        for action in TestCanvasToolbarResponsive()._all_overflow_actions(tb):
            self.assertTrue(action.isVisible(), f"{action.text()!r} hidden from overflow")
        self.assertTrue(tb.btn_overflow.isVisible())


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


class TestOneToOneButtonState(unittest.TestCase):
    """The 1:1 button reads out the current zoom, the way HQ reads out its mode."""

    def _toolbar_at(self, percent: int) -> ActionToolbar:
        tb = _make_toolbar()
        tb.controller.canvas = MagicMock()
        tb.controller.canvas.current_zoom_percent.return_value = percent
        return tb

    def test_lit_at_one_to_one(self):
        tb = self._toolbar_at(100)
        tb._on_zoom_changed(1.0)
        self.assertTrue(tb.btn_zoom_original.isChecked())
        self.assertTrue(tb._ov_original_action.isChecked())

    def test_dark_at_any_other_zoom(self):
        tb = self._toolbar_at(65)
        tb._on_zoom_changed(1.0)
        self.assertFalse(tb.btn_zoom_original.isChecked())
        self.assertFalse(tb._ov_original_action.isChecked())

    def test_a_click_leaves_the_state_to_the_zoom_it_causes(self):
        """The click's own toggle is overwritten by the resulting zoom_changed."""
        tb = self._toolbar_at(100)
        tb.btn_zoom_original.click()
        tb.controller.canvas.zoom_to_original.assert_called_once()
        tb._on_zoom_changed(1.0)
        self.assertTrue(tb.btn_zoom_original.isChecked())

    def test_a_click_with_no_canvas_does_not_stick(self):
        tb = _make_toolbar()  # controller.canvas is None
        tb.btn_zoom_original.click()
        self.assertFalse(tb.btn_zoom_original.isChecked())
