"""GranularSettingsDialog pick mode (the Persistent Settings picker)."""

from PyQt6.QtCore import Qt

from negpy.desktop.settings_catalog import all_rows
from negpy.desktop.view.widgets.granular_settings_dialog import GranularSettingsDialog
from negpy.domain.models import WorkspaceConfig


def _dlg(preselect):
    return GranularSettingsDialog(None, WorkspaceConfig(), "", preselect_ids=frozenset(preselect))


def test_pick_mode_titles_and_preselection(qapp):
    dlg = _dlg({"exposure.density", "lab.saturation"})
    assert dlg.windowTitle() == "Persistent Settings"
    assert set(dlg.selected_ids()) == {"exposure.density", "lab.saturation"}


def test_pick_mode_lists_every_row_even_at_default(qapp):
    """A setting still at its default must stay tickable, or it could never be made sticky."""
    dlg = _dlg(set())
    assert len(dlg._checks) == len(all_rows())
    assert not any(line.isHidden() for _b, _r, _e, line in dlg._checks)
    assert dlg._show_unchanged.isHidden()


def test_pick_mode_allows_selecting_nothing(qapp):
    dlg = _dlg({"exposure.density"})
    dlg._set_all_checked(False)
    assert dlg.selected_ids() == []
    # Ticking nothing is a valid choice: nothing carries over.
    assert dlg.apply_btn.isEnabled()


def test_check_all_reaches_every_row(qapp):
    dlg = _dlg(set())
    dlg._set_all_checked(True)
    assert len(dlg.selected_ids()) == len(all_rows())


def test_section_checkbox_is_tristate_and_toggles_its_rows(qapp):
    dlg = _dlg({"lab.saturation"})
    section, row_ids = next((s, ids) for s, ids in dlg._section_rows if "lab.saturation" in ids)
    assert section.select_box.checkState() == Qt.CheckState.PartiallyChecked

    section._on_select_clicked()
    assert set(row_ids) <= set(dlg.selected_ids())
    assert section.select_box.checkState() == Qt.CheckState.Checked

    section._on_select_clicked()
    assert not (set(row_ids) & set(dlg.selected_ids()))
    assert section.select_box.checkState() == Qt.CheckState.Unchecked


def test_paste_mode_keeps_no_section_checkboxes(qapp):
    """The existing callers must be untouched by pick mode."""
    dlg = GranularSettingsDialog(None, WorkspaceConfig(), "clipboard")
    assert dlg._section_rows == []
    assert all(s.select_box is None for s, _n in dlg._sections)
    assert not dlg._show_unchanged.isHidden()
