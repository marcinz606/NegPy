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
    assert {row.label for _box, row in dlg._checks} == {"Print Density"}


def test_set_name_prefills_and_enables(qapp):
    dlg = GranularSettingsDialog(None, _edited_cfg(), "Portra", ask_name=True)
    dlg.set_name("Portra")
    assert dlg.name() == "Portra"
    assert dlg.apply_btn.isEnabled()


def test_default_mode_unchanged(qapp):
    dlg = GranularSettingsDialog(None, _edited_cfg(), "clipboard")
    assert dlg.windowTitle() == "Paste Settings"
    assert dlg.apply_btn.text() == "Apply"
    assert dlg.name() == ""
    assert dlg.apply_btn.isEnabled()
