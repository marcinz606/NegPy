import sys
import unittest
from unittest.mock import MagicMock

from PyQt6.QtWidgets import QApplication, QDialog

from negpy.desktop.view.canvas.toolbar import DEFAULT_TOOLBAR_IDS, TOOLBAR_ITEM_BY_ID, TOOLBAR_ITEMS, ActionToolbar, load_toolbar_items
from negpy.desktop.session import ToolMode
from negpy.desktop.view.styles.theme import THEME
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
    controller.session.state.negative_peek = False
    controller.session.state.embedded_peek = False
    controller.session.state.flatfield_peek = False
    controller.session.state.flat_output = False
    controller.session.state.config.flatfield.profile_id = ""
    controller.session.state.zones_overlay = False
    controller.session.state.test_strip = False
    controller.session.state.test_strip_kind = "tone"
    controller.session.state.test_strip_expanded = None
    controller.session.state.clipping_overlay = False
    controller.session.state.printing_notes = False
    controller.session.state.grain_focuser = False
    controller.session.state.cursor_probe = False
    controller.session.state.canvas_info_level = 1
    controller.session.state.zones_style = "false_color"
    controller.session.state.color_vision = "standard"
    controller.session.state.active_tool = ToolMode.NONE
    controller.session.state.clone_picking = False
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


def _repo(stored, added=("tools",)) -> MagicMock:
    """A stored row whose new items were already offered, unless *added* says otherwise."""
    repo = MagicMock()
    settings = {"toolbar_items": stored, "toolbar_added_items": list(added)}
    repo.get_global_setting.side_effect = lambda key, default=None: settings.get(key, default)
    return repo


def _visible_item_count(tb: ActionToolbar) -> int:
    return sum(1 for item_id in tb._item_ids if not tb._row_widgets[item_id].isHidden())


def _row_order(tb: ActionToolbar) -> list[str]:
    """Ids in layout order, so a reorder is checked against the real row, not the stored list."""
    by_widget = {widget: item_id for item_id, widget in tb._row_widgets.items()}
    layout = tb._row_layout
    return [by_widget[w] for i in range(layout.count()) if (w := layout.itemAt(i).widget()) in by_widget]


def test_construction_does_not_show_orphan_controls(top_level_show_spy):
    _make_toolbar()

    assert top_level_show_spy.events == []


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
        self.assertTrue(tb.source_choice.isVisible())
        self.assertTrue(tb.btn_undo.isVisible())
        self.assertTrue(tb.btn_zoom.isVisible())

    def test_every_visible_control_shares_one_row_height(self):
        tb = _make_toolbar()
        tb.set_available_width(1600)
        tb.show()
        QApplication.processEvents()

        row = [
            tb.btn_toggle_left,
            tb.btn_prev,
            tb.position_label,
            tb.btn_next,
            tb.btn_zoom,
            tb.btn_rot_l,
            tb.btn_rot_r,
            tb.btn_flip_h,
            tb.btn_flip_v,
            tb.btn_tools,
            tb.btn_undo,
            tb.btn_redo,
            tb.source_choice,
            tb.overlays_menu,
            tb.btn_overflow,
            tb.btn_toggle_right,
        ]
        edges = {(w.geometry().y(), w.geometry().bottom()) for w in row if w.isVisible()}
        self.assertEqual(len(edges), 1, f"toolbar controls disagree on top/bottom edges: {sorted(edges)}")

    def _all_overflow_actions(self, tb: ActionToolbar) -> list:
        return [
            tb._ov_hq_action,
            tb._ov_fit_action,
            tb._ov_original_action,
            tb._ov_compare_action,
            tb._ov_flat_peek_action,
            tb._ov_negative_peek_action,
            tb._ov_embedded_peek_action,
            tb._ov_zones_action,
            tb._ov_loupe_action,
            tb._ov_clipping_action,
            tb._ov_notes_action,
            tb._ov_undo_action,
            tb._ov_redo_action,
            tb._ov_rot_l_action,
            tb._ov_rot_r_action,
            tb._ov_flip_h_action,
            tb._ov_flip_v_action,
        ]

    def test_ring_around_is_not_in_the_toolbar(self):
        tb = _make_toolbar()
        labels = [a.text() for a in tb.btn_overflow.menu().actions()]
        self.assertNotIn("Color Ring-Around", labels)
        self.assertFalse(hasattr(tb, "_ov_ring_action"))

    def test_app_settings_moved_out_of_the_overflow_menu(self):
        tb = _make_toolbar()
        labels = [a.text() for a in tb.btn_overflow.menu().actions()]
        for gone in (
            "GPU Acceleration",
            "Multi-core CPU Rendering",
            "Canvas: Black",
            "UI Scale",
            "Immersive Canvas",
            "Sticky Zoom",
            "Reverse Scroll Zoom",
            "Show Slider Values",
            "Reset Panel Layout",
            "Edit Toolbar…",
            "Manage Database…",
        ):
            self.assertNotIn(gone, labels)
        self.assertTrue(any("Preferences" in label for label in labels))

    def test_a_checkable_overflow_item_carries_no_icon(self):
        """Under the app stylesheet Qt draws a menu icon in the check column, so an icon on
        a checkable item costs the checkmark — and with it the only sign in the menu that
        the view is already on."""
        tb = _make_toolbar()
        for action in self._all_overflow_actions(tb):
            if action.isCheckable():
                self.assertTrue(action.icon().isNull(), f"{action.text()!r} would lose its checkmark to its icon")

    def test_panel_toggles_are_not_checkable(self):
        """An open panel is the resting state; the accent fill means armed."""
        tb = _make_toolbar()
        self.assertFalse(tb.btn_toggle_left.isCheckable())
        self.assertFalse(tb.btn_toggle_right.isCheckable())

    def test_every_keyed_overflow_item_shows_its_key(self):
        tb = _make_toolbar()
        self.assertTrue(tb._ov_fit_action.text().endswith("0"))
        self.assertTrue(tb._ov_rot_r_action.text().endswith("]"))
        self.assertTrue(tb._ov_negative_peek_action.text().endswith("N"))

    def test_right_click_offers_edit_toolbar(self):
        tb = _make_toolbar()
        seen: list[str] = []
        with unittest.mock.patch(
            "negpy.desktop.view.canvas.toolbar.QMenu.exec", lambda menu, *_a: seen.extend(a.text() for a in menu.actions())
        ):
            tb.contextMenuEvent(MagicMock())
        self.assertEqual(seen, ["Edit Toolbar…"])

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
        repo = _repo(["overlays", "retired_button", "prev"])
        self.assertEqual(load_toolbar_items(repo), ["overlays", "prev"])

    def test_folded_buttons_become_their_menu_and_the_row_is_saved_once(self):
        repo = _repo(
            [
                "prev",
                "next",
                "zoom_label",
                "zoom_fit",
                "hq",
                "undo",
                "compare",
                "negative_peek",
                "zones",
                "loupe",
            ]
        )
        migrated = ["prev", "position", "next", "zoom", "undo", "source", "overlays"]
        self.assertEqual(load_toolbar_items(repo), migrated)
        repo.save_global_setting.assert_called_once_with("toolbar_items", migrated)

    def test_a_removed_frame_counter_stays_removed(self):
        repo = _repo(["prev", "next", "zoom"])
        self.assertEqual(load_toolbar_items(repo), ["prev", "next", "zoom"])
        repo.save_global_setting.assert_not_called()

    def test_a_stored_row_gets_tools_once_before_undo(self):
        repo = _repo(["prev", "undo", "redo"], added=())
        self.assertEqual(load_toolbar_items(repo), ["prev", "tools", "undo", "redo"])
        repo.save_global_settings.assert_called_once_with(
            {"toolbar_items": ["prev", "tools", "undo", "redo"], "toolbar_added_items": ["tools"]}
        )

    def test_a_stored_row_without_undo_gets_tools_last(self):
        self.assertEqual(load_toolbar_items(_repo(["prev", "next"], added=())), ["prev", "next", "tools"])

    def test_a_removed_tools_button_stays_removed(self):
        repo = _repo(["prev", "undo"])
        self.assertEqual(load_toolbar_items(repo), ["prev", "undo"])
        repo.save_global_settings.assert_not_called()

    def test_load_tolerates_a_malformed_setting(self):
        repo = MagicMock()
        repo.get_global_setting.return_value = "prev"
        self.assertEqual(load_toolbar_items(repo), list(DEFAULT_TOOLBAR_IDS))

    def test_stored_order_drives_the_row(self):
        tb = _make_toolbar()
        tb._item_ids = load_toolbar_items(_repo(["overlays", "prev", "undo"]))
        tb._rebuild_row()

        self.assertEqual(_row_order(tb), ["overlays", "prev", "undo"])
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

    def test_collapse_keeps_navigation_zoom_and_view_longest(self):
        tb = _make_toolbar()
        tb.show()
        QApplication.processEvents()

        tb.set_available_width(560)
        QApplication.processEvents()
        visible = [i for i in tb._item_ids if not tb._row_widgets[i].isHidden()]
        self.assertLess(len(visible), len(tb._item_ids))
        self.assertIn("prev", visible)
        self.assertIn("zoom", visible)
        self.assertNotIn("flip_v", visible)

    def test_menu_actions_on_the_row_hide_before_the_stock_controls(self):
        tb = _make_toolbar()
        tb._item_ids = ["copy", *DEFAULT_TOOLBAR_IDS]
        tb._rebuild_row()
        tb.show()
        QApplication.processEvents()

        tb.set_available_width(4000)
        QApplication.processEvents()
        full = tb._pill_width()
        tb.set_available_width(full + 2 * THEME.space_xl - 1)
        QApplication.processEvents()
        self.assertTrue(tb.btn_copy.isHidden())
        self.assertTrue(all(not tb._row_widgets[i].isHidden() for i in DEFAULT_TOOLBAR_IDS))

    def test_every_item_can_be_put_on_the_row(self):
        tb = _make_toolbar()
        tb._item_ids = [item.id for item in TOOLBAR_ITEMS]
        tb._rebuild_row()

        self.assertEqual(_row_order(tb), tb._item_ids)

    def test_menu_only_actions_on_the_row_drive_the_controller(self):
        tb = _make_toolbar()
        tb._item_ids = ["copy", "reset_to_roll"]
        tb._rebuild_row()

        tb.btn_copy.click()
        tb.session.copy_settings.assert_called_once()
        tb.btn_reset_to_roll.click()
        tb.controller.revert_frame_to_roll.assert_called_once()

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
        tb.session.state.selected_indices = [0]

        tb.rotate(1)

        tb.session.update_config.assert_called_once()
        self.assertEqual(tb.session.update_config.call_args.args[0].geometry.rotation, 1)
        tb.session.rotate_selected_frames.assert_called_once_with(1, active_included=True)
        tb.controller.rerender_active_view.assert_called_once()

    def test_a_selection_that_excludes_the_active_frame_leaves_it_alone(self):
        tb = _make_toolbar()
        tb.controller.rotate_test_strip.return_value = False
        tb.session.state.selected_file_idx = 0
        tb.session.state.selected_indices = [1, 2]

        tb.rotate(1)

        tb.session.update_config.assert_not_called()
        tb.session.rotate_selected_frames.assert_called_once_with(1, active_included=False)
        tb.controller.rerender_active_view.assert_not_called()

    def test_a_selection_that_includes_the_active_frame_still_rotates_it(self):
        tb = _make_toolbar()
        tb.controller.rotate_test_strip.return_value = False
        tb.session.state.config = WorkspaceConfig()
        tb.session.state.selected_file_idx = 0
        tb.session.state.selected_indices = [0, 1, 2]

        tb.rotate(1)

        tb.session.update_config.assert_called_once()
        tb.session.rotate_selected_frames.assert_called_once_with(1, active_included=True)
        tb.controller.rerender_active_view.assert_called_once()


class TestFlipRouting(unittest.TestCase):
    """See TestRotateRouting: flip shares the same active/selection targeting rule."""

    def test_a_selection_that_excludes_the_active_frame_leaves_it_alone(self):
        tb = _make_toolbar()
        tb.session.state.selected_file_idx = 0
        tb.session.state.selected_indices = [1, 2]

        tb.flip("horizontal")

        tb.session.update_config.assert_not_called()
        tb.session.flip_selected_frames.assert_called_once_with(True, active_included=False)
        tb.controller.rerender_active_view.assert_not_called()

    def test_a_selection_that_includes_the_active_frame_still_flips_it(self):
        tb = _make_toolbar()
        tb.session.state.config = WorkspaceConfig()
        tb.session.state.selected_file_idx = 0
        tb.session.state.selected_indices = [0, 1, 2]

        tb.flip("horizontal")

        tb.session.update_config.assert_called_once()
        tb.session.flip_selected_frames.assert_called_once_with(True, active_included=True)
        tb.controller.rerender_active_view.assert_called_once()


if __name__ == "__main__":
    unittest.main()


class TestZoomButton(unittest.TestCase):
    def _toolbar_at(self, percent: int) -> ActionToolbar:
        tb = _make_toolbar()
        tb.controller.canvas = MagicMock()
        tb.controller.canvas.current_zoom_percent.return_value = percent
        return tb

    def test_reads_out_the_true_zoom(self):
        tb = self._toolbar_at(65)
        tb._on_zoom_changed(1.0)
        self.assertEqual(tb.btn_zoom.text(), "65%")
        self.assertFalse(tb._ov_original_action.isChecked())

    def test_says_when_hq_is_on(self):
        tb = self._toolbar_at(100)
        tb.session.state.hq_preview = True
        tb._on_zoom_changed(1.0)
        self.assertEqual(tb.btn_zoom.text(), "100% HQ")
        self.assertTrue(tb._ov_original_action.isChecked())

    def test_a_readout_longer_than_its_slot_relays_the_pill(self):
        from PyQt6.QtWidgets import QWidget

        class _Canvas(QWidget):
            relayouts = 0

            def relayout_floating_widgets(self):
                _Canvas.relayouts += 1

        canvas = _Canvas()
        tb = self._toolbar_at(1600)
        tb.setParent(canvas)
        canvas.show()
        tb.session.state.hq_preview = True
        tb._on_zoom_changed(1.0)
        self.assertGreater(_Canvas.relayouts, 0)

    def test_a_preset_sets_that_zoom(self):
        tb = self._toolbar_at(65)
        preset = next(a for a in tb.btn_zoom.choice_menu.actions() if a.text() == "200%")
        preset.trigger()
        tb.controller.canvas.zoom_to_percent.assert_called_once_with(200)

    def test_hq_item_drives_the_controller(self):
        tb = _make_toolbar()
        tb._zoom_hq_action.trigger()
        tb.controller.toggle_hq_preview.assert_called_once()


class TestFrameCounter(unittest.TestCase):
    def test_shows_the_place_in_the_strip(self):
        tb = _make_toolbar()
        tb.session.asset_model.actual_to_display.return_value = 1
        tb.session.asset_model.rowCount.return_value = 5
        tb._update_ui_state()
        self.assertEqual(tb.position_label.text(), "2 / 5")


class TestViewChooser(unittest.TestCase):
    def _choose(self, tb: ActionToolbar, label: str) -> None:
        index = next(i for i in range(tb.source_choice.count()) if tb.source_choice.choice_action(i).text().startswith(label))
        tb.source_choice.setCurrentIndex(index)

    def test_a_peek_drives_the_controller(self):
        tb = _make_toolbar()
        self._choose(tb, "Negative")
        tb.controller.toggle_negative_peek.assert_called_once_with(force=True)

    def test_before_after_turns_compare_on(self):
        tb = _make_toolbar()
        self._choose(tb, "Before / After")
        tb.controller.toggle_compare.assert_called_once()

    def test_print_leaves_whatever_has_the_canvas(self):
        tb = _make_toolbar()
        tb.session.state.compare_mode = True
        tb._sync_view_state()
        self._choose(tb, "Print")
        tb.controller.exit_compare.assert_called_once()

        tb.session.state.compare_mode = False
        tb.session.state.negative_peek = True
        tb._sync_view_state()
        self._choose(tb, "Print")
        tb.controller.toggle_negative_peek.assert_called_once_with(force=False)

    def test_the_label_follows_a_peek_started_by_its_key(self):
        tb = _make_toolbar()
        tb.session.state.embedded_peek = True
        tb._sync_view_state()
        self.assertEqual(tb.source_choice.text().strip(), "Embedded")
        tb.controller.toggle_embedded_peek.assert_not_called()

    def test_a_refused_peek_falls_back_to_print(self):
        """A file with no embedded preview leaves the state off; the chooser must not claim it is on."""
        tb = _make_toolbar()
        self._choose(tb, "Embedded")
        self.assertEqual(tb.source_choice.text().strip(), "Print")

    def test_flat_scan_needs_no_flat_output(self):
        """The peek renders the flat master from any frame, like its key."""
        tb = _make_toolbar()
        tb.session.state.flat_output = False
        tb._sync_view_state()
        flat = next(i for i in range(tb.source_choice.count()) if tb.source_choice.choice_action(i).text().startswith("Flat Scan"))
        self.assertTrue(tb.source_choice.is_choice_enabled(flat))
        self.assertTrue(tb._ov_flat_peek_action.isEnabled())


class TestOverlaysMenu(unittest.TestCase):
    def test_an_overlay_drives_the_controller(self):
        tb = _make_toolbar()
        tb._overlay_actions["clipping_overlay"].trigger()
        tb.controller.toggle_clipping_overlay.assert_called_once_with(force=True)

    def test_state_sync_checks_the_item_and_lights_the_button_without_echo(self):
        tb = _make_toolbar()
        tb.session.state.grain_focuser = True
        tb._sync_view_state()
        self.assertTrue(tb._overlay_actions["grain_focuser"].isChecked())
        self.assertTrue(tb.overlays_menu.isChecked())
        tb.controller.toggle_grain_focuser.assert_not_called()

        tb.session.state.grain_focuser = False
        tb._sync_view_state()
        self.assertFalse(tb.overlays_menu.isChecked())


class TestToolRow(unittest.TestCase):
    def _armed(self, mode: ToolMode) -> ActionToolbar:
        tb = _make_toolbar()
        tb.session.state.active_tool = mode
        tb._sync_tool_row()
        return tb

    def test_arming_a_tool_swaps_the_rows(self):
        tb = self._armed(ToolMode.DUST_PICK)
        self.assertTrue(tb._view_row.isHidden())
        self.assertFalse(tb._tool_row.isHidden())
        self.assertEqual(tb.tool_name.text(), "Heal")
        self.assertIn("dust", tb.tool_hint.text())

    def test_leaving_the_tool_brings_the_view_row_back(self):
        tb = self._armed(ToolMode.STRAIGHTEN)
        tb.session.state.active_tool = ToolMode.NONE
        tb._sync_tool_row()
        self.assertFalse(tb._view_row.isHidden())
        self.assertTrue(tb._tool_row.isHidden())

    def test_every_tool_has_a_row(self):
        from negpy.desktop.view.canvas.toolbar import _TOOL_ROWS

        self.assertEqual(set(_TOOL_ROWS), set(ToolMode) - {ToolMode.NONE})

    def test_clone_asks_for_a_source_first(self):
        tb = _make_toolbar()
        tb.session.state.active_tool = ToolMode.CLONE
        tb.session.state.clone_picking = True
        tb._sync_tool_row()
        self.assertIn("pick the area to copy from", tb.tool_hint.text())

    def test_done_finishes_the_draft_then_leaves(self):
        from PyQt6.QtWidgets import QWidget

        canvas = QWidget()
        canvas.overlay = MagicMock()
        tb = self._armed(ToolMode.SCRATCH_PICK)
        tb.setParent(canvas)
        tb.btn_tool_done.click()
        canvas.overlay.finish_in_progress.assert_called_once()
        tb.controller.cancel_active_tool.assert_called_once()

    def test_the_slider_mirrors_the_sidebar_both_ways(self):
        from negpy.desktop.view.widgets.sliders import CompactSlider

        source = CompactSlider("Brush Size", 1, 50, 10, step=1)
        source.setValue(10)
        tb = _make_toolbar()
        with unittest.mock.patch.object(tb, "_find_widget", side_effect=lambda path: source if path.endswith("slider") else None):
            tb.session.state.active_tool = ToolMode.DUST_PICK
            tb._sync_tool_row()
        mirror = tb._tool_slider
        self.assertEqual(mirror.value(), 10)

        source.mirror_value(20, commit=False)
        self.assertEqual(mirror.value(), 20)
        mirror.valueChanged.emit(30.0)
        self.assertEqual(source.value(), 30)

        tb.session.state.active_tool = ToolMode.NONE
        tb._sync_tool_row()
        self.assertIsNone(tb._tool_slider)
        source.mirror_value(25, commit=False)  # a dropped mirror is no longer connected

    def test_a_narrow_canvas_drops_the_hint_before_done(self):
        tb = self._armed(ToolMode.KEYSTONE_LINES)
        tb.show()
        QApplication.processEvents()
        tb.set_available_width(320)
        QApplication.processEvents()
        self.assertTrue(tb.tool_hint.isHidden())
        self.assertFalse(tb.btn_tool_done.isHidden())


class TestToolsMenu(unittest.TestCase):
    def _items(self, menu):
        from PyQt6.QtWidgets import QWidgetAction

        return [a for a in menu.actions() if not a.isSeparator() and not isinstance(a, QWidgetAction)]

    def test_every_menu_tool_has_a_row_and_a_registry_action(self):
        from negpy.desktop.view.canvas.toolbar import _CANVAS_TOOLS, _TOOL_ROWS
        from negpy.desktop.view.shortcut_registry import REGISTRY

        tools = [spec for spec in _CANVAS_TOOLS if not isinstance(spec, str)]
        self.assertEqual({mode for mode, _a, _p in tools}, set(_TOOL_ROWS) - {ToolMode.ZONE_PLACE})
        for _mode, action_id, _path in tools:
            self.assertIn(action_id, REGISTRY)
        self.assertIn("open_tools_menu", REGISTRY)

    def test_the_menu_is_grouped_under_panel_section_titles(self):
        from PyQt6.QtWidgets import QWidgetAction

        tb = _make_toolbar()
        headers = [a.defaultWidget().text() for a in tb.btn_tools.choice_menu.actions() if isinstance(a, QWidgetAction)]
        self.assertEqual(headers, ["GEOMETRY", "METERING", "FILTRATION", "DODGE & BURN", "LIGHT LEAK", "RETOUCH"])

    def test_tools_is_on_the_stock_row_and_collapses(self):
        from negpy.desktop.view.canvas.toolbar import _COLLAPSE_ORDER

        self.assertIn("tools", DEFAULT_TOOLBAR_IDS)
        self.assertIn("tools", _COLLAPSE_ORDER)

    def test_an_item_runs_the_tools_shortcut_action(self):
        tb = _make_toolbar()
        with unittest.mock.patch.object(tb, "_arm_tool") as arm:
            heal = next(a for a in self._items(tb.btn_tools.choice_menu) if a.text().startswith("Heal"))
            heal.trigger()
        arm.assert_called_once_with("pick_dust")

    def test_a_gated_tool_and_the_armed_tool_are_grayed_out(self):
        from PyQt6.QtWidgets import QPushButton

        tb = _make_toolbar()
        disabled = QPushButton()
        disabled.setEnabled(False)
        tb.session.state.active_tool = ToolMode.CROP_MANUAL
        with unittest.mock.patch.object(
            tb, "_find_widget", side_effect=lambda path: disabled if path.endswith("clone_btn") else QPushButton()
        ):
            tb.tool_name.choice_menu.aboutToShow.emit()
        enabled = {a.text().split()[0]: a.isEnabled() for a in self._items(tb.tool_name.choice_menu)}
        self.assertFalse(enabled["Clone"])
        self.assertFalse(enabled["Crop"])
        self.assertTrue(enabled["Heal"])

    def test_the_overflow_menu_carries_the_tools(self):
        tb = _make_toolbar()
        submenu = next(a.menu() for a in tb.btn_overflow.menu().actions() if a.menu() is not None and a.text() == "Tools")
        self.assertEqual(len(self._items(submenu)), len(self._items(tb.btn_tools.choice_menu)))


class TestZoneRow(unittest.TestCase):
    def _rows(self, tb: ActionToolbar) -> tuple[bool, bool, bool]:
        return (not tb._view_row.isHidden(), not tb._tool_row.isHidden(), not tb._zone_row.isHidden())

    def test_zone_overlay_swaps_in_its_row_and_an_armed_tool_outranks_it(self):
        tb = _make_toolbar()
        state = tb.session.state
        state.zones_overlay = True
        tb._sync_tool_row()
        self.assertEqual(self._rows(tb), (False, False, True))
        self.assertEqual(tb.zone_style.currentData(), "false_color")

        state.active_tool = ToolMode.CROP_MANUAL
        tb._sync_tool_row()
        self.assertEqual(self._rows(tb), (False, True, False))

        state.active_tool = ToolMode.NONE
        state.zones_overlay = False
        tb._sync_tool_row()
        self.assertEqual(self._rows(tb), (True, False, False))

    def test_zone_row_closes_and_switches_the_look(self):
        tb = _make_toolbar()
        tb.btn_zone_close.click()
        tb.controller.toggle_zones_overlay.assert_called_with(force=False)
        tb.zone_style.setCurrentIndex(1)
        tb.controller.set_zones_style.assert_called_with("zone_map")
        tb.zone_style.setCurrentIndex(2)
        tb.controller.set_zones_style.assert_called_with("grid")

    def test_legend_hover_highlights_whatever_the_strip_setting(self):
        tb = _make_toolbar()
        tb.zone_legend.zone_hovered.emit(6)
        tb.controller.highlight_zone.assert_called_with(6)


class TestPaperStripRow(unittest.TestCase):
    KEYS = ["neutral", "pakon", "kodak_endura", "fuji_crystal"]

    def _rows(self, tb: ActionToolbar) -> tuple[bool, bool, bool, bool]:
        rows = (tb._view_row, tb._tool_row, tb._paper_row, tb._zone_row)
        return tuple(not row.isHidden() for row in rows)

    def _paper_up(self, expanded=None) -> ActionToolbar:
        tb = _make_toolbar()
        state = tb.session.state
        state.test_strip = True
        state.test_strip_kind = "paper"
        state.test_strip_expanded = expanded
        state.test_strip_layout = "bands"
        tb.controller.paper_strip_keys.return_value = self.KEYS
        tb._sync_tool_row()
        return tb

    def test_the_paper_strip_swaps_in_its_row_and_an_armed_tool_outranks_it(self):
        tb = self._paper_up()
        self.assertEqual(self._rows(tb), (False, False, True, False))
        self.assertFalse(tb.paper_strip_hint.isHidden())
        self.assertTrue(tb.paper_strip_combo.isHidden())
        self.assertTrue(tb.btn_paper_keep.isHidden())

        tb.session.state.zones_overlay = True
        tb._sync_tool_row()
        self.assertEqual(self._rows(tb), (False, False, True, False))

        tb.session.state.active_tool = ToolMode.CROP_MANUAL
        tb._sync_tool_row()
        self.assertEqual(self._rows(tb), (False, True, False, False))

    def test_a_tone_strip_keeps_the_view_row(self):
        tb = self._paper_up()
        tb.session.state.test_strip_kind = "tone"
        tb._sync_tool_row()
        self.assertEqual(self._rows(tb), (True, False, False, False))

    def test_a_paper_shown_full_frame_is_named_and_can_be_kept(self):
        tb = self._paper_up(expanded=2)
        self.assertTrue(tb.paper_strip_hint.isHidden())
        self.assertFalse(tb.paper_strip_combo.isHidden())
        self.assertEqual(tb.paper_strip_combo.currentText(), "Kodak Endura Premier")
        self.assertFalse(tb.btn_paper_keep.isHidden())
        tb.controller.show_paper_strip.assert_not_called()

    def test_the_row_drives_the_paper_strip(self):
        tb = self._paper_up(expanded=2)
        tb.paper_strip_combo.setCurrentIndex(3)
        tb.controller.show_paper_strip.assert_called_once_with(3)
        tb.btn_paper_back.click()
        tb.controller.show_paper_strip.assert_called_with(None)
        tb.btn_paper_keep.click()
        tb.controller.keep_paper_strip.assert_called_once_with()
        tb.btn_paper_close.click()
        tb.controller.toggle_paper_strip.assert_called_once_with(force=False)

    def test_the_strip_or_grid_choice_lays_the_papers_out(self):
        tb = self._paper_up()
        self.assertFalse(tb.paper_layout.isHidden())
        self.assertEqual(tb.paper_layout.currentData(), "bands")
        tb.paper_layout.setCurrentIndex(1)
        tb.controller.set_paper_strip_layout.assert_called_once_with("grid")

        tb.session.state.test_strip_layout = "grid"
        tb.session.state.test_strip_expanded = 1
        tb._sync_tool_row()
        self.assertTrue(tb.paper_layout.isHidden())
        self.assertEqual(tb.btn_paper_back.text().strip(), "Grid")
