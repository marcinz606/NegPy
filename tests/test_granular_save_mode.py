"""GranularSettingsDialog ask_name / exclude_sections modes (preset save flow)."""

from dataclasses import replace

from negpy.desktop.view.widgets.granular_settings_dialog import GranularSettingsDialog
from negpy.domain.models import WorkspaceConfig


def _edited_cfg() -> WorkspaceConfig:
    c = WorkspaceConfig()
    return replace(
        c,
        exposure=replace(c.exposure, density=1.5),
        geometry=replace(c.geometry, manual_crop_rect=(0.1, 0.1, 0.9, 0.9)),
    )


def test_save_mode_button_title_and_gating(qapp):
    dlg = GranularSettingsDialog(None, _edited_cfg(), "current settings", ask_name=True)
    assert dlg.windowTitle() == "Save Preset"
    assert dlg.apply_btn.text() == "Save"
    assert not dlg.apply_btn.isEnabled()

    dlg._name_edit.setText("  Portra  ")
    assert dlg.apply_btn.isEnabled()
    assert dlg.name() == "Portra"

    dlg._set_all_checked(False)
    assert not dlg.apply_btn.isEnabled()


def test_exclude_sections_hides_geometry_rows(qapp):
    dlg = GranularSettingsDialog(
        None,
        _edited_cfg(),
        "current settings",
        ask_name=True,
        exclude_sections=frozenset({"Crop", "Rotation"}),
    )
    assert "Manual Crop" not in {row.label for _box, row, _edited, _line in dlg._checks}
    assert {row.label for row in dlg.selected()} == {"Print Density"}


def test_set_name_prefills_and_enables(qapp):
    dlg = GranularSettingsDialog(None, _edited_cfg(), "Portra", ask_name=True)
    dlg.set_name("Portra")
    assert dlg.name() == "Portra"
    assert dlg.apply_btn.isEnabled()


def test_scope_current_and_apply_mode(qapp):
    dlg = GranularSettingsDialog(
        None, _edited_cfg(), "P", show_scope=True, show_current=True, show_apply_mode=True, sel_count=2, roll_count=3
    )
    assert dlg.current_radio.isChecked()
    assert dlg.apply_mode() == "overlay"
    dlg._on_apply()
    assert dlg.scope() == "current"

    dlg.replace_radio.setChecked(True)
    dlg.sel_radio.setChecked(True)
    dlg._on_apply()
    assert dlg.scope() == "selection"
    assert dlg.apply_mode() == "replace"


def _crop_offset_box(dlg):
    return next(box for box, row, _edited, _line in dlg._checks if row.label == "Crop Offset")


def test_default_valued_row_is_built_hidden_and_unselected(qapp):
    # #656: source frame back at the default offset must still be applicable.
    dlg = GranularSettingsDialog(None, _edited_cfg(), "IMG.cr2")
    box = _crop_offset_box(dlg)
    assert not box.isChecked()
    assert "Crop Offset" not in {row.label for row in dlg.selected()}

    dlg._show_unchanged.setChecked(True)
    box.setChecked(True)
    assert "Crop Offset" in {row.label for row in dlg.selected()}


def test_hiding_unchanged_rows_unchecks_them(qapp):
    dlg = GranularSettingsDialog(None, _edited_cfg(), "IMG.cr2")
    dlg._show_unchanged.setChecked(True)
    dlg._set_all_checked(True)
    assert "Crop Offset" in {row.label for row in dlg.selected()}

    dlg._show_unchanged.setChecked(False)
    assert not _crop_offset_box(dlg).isChecked()
    assert {row.label for row in dlg.selected()} == {"Print Density", "Manual Crop"}


def test_check_all_skips_hidden_unchanged_rows(qapp):
    dlg = GranularSettingsDialog(None, _edited_cfg(), "IMG.cr2")
    dlg._set_all_checked(True)
    assert {row.label for row in dlg.selected()} == {"Print Density", "Manual Crop"}


def test_default_mode_unchanged(qapp):
    dlg = GranularSettingsDialog(None, _edited_cfg(), "clipboard")
    assert dlg.windowTitle() == "Paste Settings"
    assert dlg.apply_btn.text() == "Apply"
    assert dlg.name() == ""
    assert dlg.apply_btn.isEnabled()
