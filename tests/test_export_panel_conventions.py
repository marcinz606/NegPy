from conftest import FakeController, FakeRepo
from PyQt6.QtWidgets import QCheckBox, QPushButton

from negpy.desktop.view.sidebar.export import ExportSidebar


def _sidebar() -> ExportSidebar:
    return ExportSidebar(FakeController(repo=FakeRepo()))


def test_booleans_are_toggle_buttons():
    assert _sidebar().findChildren(QCheckBox) == []


def test_export_is_the_one_primary_action():
    sidebar = _sidebar()
    primary = [b for b in sidebar.findChildren(QPushButton) if b.property("primary")]
    assert primary == [sidebar.export_main_btn]


def test_soft_proof_settings_follow_the_master_toggle():
    sidebar = _sidebar()
    sidebar.soft_proof_btn.setChecked(False)
    sidebar.state.soft_proof_enabled = False
    sidebar._sync_proof_controls()
    assert not sidebar.proof_gamut_btn.isEnabled()
    assert not sidebar.proof_profile_combo.isEnabled()


def test_sync_keeps_a_pending_form_edit():
    sidebar = _sidebar()
    sidebar.form.mode_btn.setCurrentIndex(2)
    assert sidebar.update_timer.isActive()
    sidebar.sync_ui()
    assert sidebar.form.mode_btn.currentIndex() == 2


def test_preview_export_size_drives_the_view_and_unlights_when_refused():
    sidebar = _sidebar()
    toggle = sidebar.controller.toggle_export_size_view

    def accept(force):
        sidebar.state.export_size_view = force

    toggle.side_effect = accept
    sidebar.export_size_btn.click()
    toggle.assert_called_once_with(force=True)
    assert sidebar.export_size_btn.isChecked()

    # No frame open: the controller refuses and signals nothing.
    sidebar.state.export_size_view = False
    toggle.side_effect = None
    sidebar.export_size_btn.setChecked(False)
    sidebar.export_size_btn.click()
    assert not sidebar.export_size_btn.isChecked()


def test_preview_export_size_follows_the_print_menu():
    sidebar = _sidebar()
    sidebar._on_export_size_view_changed(True)
    assert sidebar.export_size_btn.isChecked()
    sidebar.controller.toggle_export_size_view.assert_not_called()


def test_preview_export_size_sits_above_paper_ratio():
    sidebar = _sidebar()
    box = sidebar.form._size_box
    row = box.itemAt(box.indexOf(sidebar.form._ratio_row_widget) - 1).layout()
    assert row is not None and row.indexOf(sidebar.export_size_btn) >= 0
