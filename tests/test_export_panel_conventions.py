"""The Export panel follows the panel conventions: toggles, one call to action, one label column."""

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
