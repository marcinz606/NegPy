import unittest
from unittest.mock import patch

from PyQt6.QtWidgets import QDialog

from negpy.desktop.view.widgets.preferences_dialog import NUMBER_ROWS, PreferencesDialog, default_for
from tests.conftest import FakeController, FakeRepo


def _dlg(pinned=None, **settings) -> PreferencesDialog:
    controller = FakeController(FakeRepo(**settings))
    return PreferencesDialog(controller, None, pinned_keys=pinned or set())


class TestPreferencesDialog(unittest.TestCase):
    def test_every_performance_row_gets_a_spin_box(self):
        dlg = _dlg()
        self.assertEqual(set(dlg._spins), {row.key for row in NUMBER_ROWS})

    def test_ui_scale_is_saved_as_a_fraction(self):
        dlg = _dlg(ui_scale=1.0)
        dlg.scale_combo.setCurrentIndex(0)  # 80%
        self.assertAlmostEqual(dlg.repo.data["ui_scale"], 0.8)

    def test_the_scale_combo_opens_on_the_saved_value(self):
        self.assertEqual(_dlg(ui_scale=1.2).scale_combo.currentText(), "120%")

    def test_canvas_background_pills_cover_every_colour(self):
        from negpy.desktop.view.canvas.toolbar import CANVAS_COLORS

        dlg = _dlg()
        self.assertEqual(len(dlg.canvas_pills), len(CANVAS_COLORS))
        self.assertTrue(dlg.canvas_pills[0].isChecked())
        self.assertEqual([p.toolTip() for p in dlg.canvas_pills], [label for _, _, label in CANVAS_COLORS])

    def test_clicking_a_pill_sets_that_background(self):
        dlg = _dlg()
        dlg.canvas_pills[2].click()
        dlg.session.set_canvas_bg.assert_called_once_with(2)
        self.assertTrue(dlg.canvas_pills[2].isChecked())
        self.assertFalse(dlg.canvas_pills[0].isChecked())

    def test_view_toggles_go_through_the_session(self):
        dlg = _dlg()
        dlg.immersive_box.setChecked(not dlg.immersive_box.isChecked())
        dlg.sticky_zoom_box.setChecked(not dlg.sticky_zoom_box.isChecked())
        dlg.session.set_immersive_canvas.assert_called_once()
        dlg.session.set_sticky_zoom.assert_called_once()

    def test_sticky_settings_box_reflects_state_and_toggles_through_the_session(self):
        dlg = _dlg()
        self.assertTrue(dlg.sticky_settings_box.isChecked())
        self.assertTrue(dlg._persistent_settings_button.isEnabled())

        dlg.sticky_settings_box.setChecked(False)
        dlg.session.set_sticky_settings_enabled.assert_called_once_with(False)
        self.assertFalse(dlg._persistent_settings_button.isEnabled())

    def test_sticky_settings_box_opens_unchecked_when_disabled(self):
        controller = FakeController(FakeRepo())
        controller.session.state.sticky_settings_enabled = False
        dlg = PreferencesDialog(controller, None)
        self.assertFalse(dlg.sticky_settings_box.isChecked())
        self.assertFalse(dlg._persistent_settings_button.isEnabled())

    def test_the_cache_limit_is_shown_in_mb_and_stored_in_bytes(self):
        dlg = _dlg()
        dlg._spins["preview_cache_max_bytes"].setValue(256)
        self.assertEqual(dlg.repo.data["preview_cache_max_bytes"], 256 * 1024 * 1024)

    def test_a_plain_number_row_is_stored_as_it_reads(self):
        dlg = _dlg()
        dlg._spins["render_memo_max_entries"].setValue(12)
        self.assertEqual(dlg.repo.data["render_memo_max_entries"], 12)

    def test_semantic_search_box_reflects_state(self):
        controller = FakeController(FakeRepo())
        controller.session.state.semantic_search_enabled = True
        with patch("negpy.desktop.view.widgets.preferences_dialog.semantic_model.clip_model_ready", return_value=True):
            dlg = PreferencesDialog(controller, None)
        self.assertTrue(dlg.semantic_box.isChecked())

    def test_turning_it_on_with_the_model_ready_skips_the_download_dialog(self):
        with patch("negpy.desktop.view.widgets.preferences_dialog.semantic_model.clip_model_ready", return_value=True):
            dlg = _dlg()
            with patch("negpy.desktop.view.widgets.preferences_dialog.ClipDownloadDialog") as dialog_cls:
                dlg.semantic_box.setChecked(True)
        dialog_cls.assert_not_called()
        dlg.session.set_semantic_search_enabled.assert_called_once_with(True)

    def test_turning_it_on_without_the_model_prompts_a_download(self):
        with patch("negpy.desktop.view.widgets.preferences_dialog.semantic_model.clip_model_ready", return_value=False):
            dlg = _dlg()
            with patch("negpy.desktop.view.widgets.preferences_dialog.ClipDownloadDialog") as dialog_cls:
                dialog_cls.return_value.exec.return_value = QDialog.DialogCode.Accepted
                dlg.semantic_box.setChecked(True)
        dialog_cls.return_value.exec.assert_called_once_with()
        dlg.session.set_semantic_search_enabled.assert_called_once_with(True)

    def test_cancelling_the_download_leaves_the_box_unchecked_and_the_preference_off(self):
        with patch("negpy.desktop.view.widgets.preferences_dialog.semantic_model.clip_model_ready", return_value=False):
            dlg = _dlg()
            with patch("negpy.desktop.view.widgets.preferences_dialog.ClipDownloadDialog") as dialog_cls:
                dialog_cls.return_value.exec.return_value = QDialog.DialogCode.Rejected
                dlg.semantic_box.setChecked(True)
        self.assertFalse(dlg.semantic_box.isChecked())
        dlg.session.set_semantic_search_enabled.assert_not_called()

    def test_the_restart_hint_waits_for_a_startup_change(self):
        dlg = _dlg()
        self.assertTrue(dlg._restart_hint.isHidden())
        dlg.immersive_box.setChecked(not dlg.immersive_box.isChecked())
        self.assertTrue(dlg._restart_hint.isHidden())
        dlg._spins["preview_render_size"].setValue(2048)
        self.assertFalse(dlg._restart_hint.isHidden())

    def test_a_row_override_toml_pins_stands_down(self):
        dlg = _dlg(pinned={"preview_render_size"})
        pinned = dlg._spins["preview_render_size"]
        self.assertFalse(pinned.isEnabled())
        self.assertTrue(dlg._spins["render_memo_max_entries"].isEnabled())
        pinned.setValue(4096)
        self.assertNotIn("preview_render_size", dlg.repo.data)

    def test_texture_cap_defaults_to_hardware_choice(self):
        self.assertEqual(default_for("max_texture_size"), 0)

    def test_every_number_row_has_a_default_inside_its_range(self):
        for row in NUMBER_ROWS:
            with self.subTest(row.key):
                self.assertGreaterEqual(default_for(row.key) // row.scale, row.minimum)
                self.assertLessEqual(default_for(row.key) // row.scale, row.maximum)


if __name__ == "__main__":
    unittest.main()
