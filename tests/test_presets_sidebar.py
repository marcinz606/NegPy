from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog

from negpy.desktop.settings_catalog import all_rows
from negpy.desktop.view.sidebar.presets import PresetsSidebar
from negpy.domain.models import WorkspaceConfig
from negpy.kernel.system.config import APP_CONFIG
from negpy.services.assets.presets import Presets


def _sidebar() -> PresetsSidebar:
    controller = SimpleNamespace(state=SimpleNamespace(config=WorkspaceConfig(), current_file_hash=None))
    return PresetsSidebar(controller)


def test_list_populates_with_summary_tooltip(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "presets_dir", str(tmp_path))
    Presets.save_preset("Portra", {"density": 1.5})
    sb = _sidebar()
    assert [sb.preset_list.item(i).text() for i in range(sb.preset_list.count())] == ["Portra"]
    assert "Print Density" in sb.preset_list.item(0).toolTip()


def test_edit_preset_renames_and_keeps_values(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "presets_dir", str(tmp_path))
    Presets.save_preset("Old", {"density": 1.5})
    sb = _sidebar()
    sb.preset_list.setCurrentRow(0)

    mock_dlg = MagicMock()
    mock_dlg.exec.return_value = QDialog.DialogCode.Accepted
    mock_dlg.name.return_value = "New"
    mock_dlg.selected.return_value = [next(r for r in all_rows() if r.label == "Print Density")]
    with patch("negpy.desktop.view.sidebar.presets.GranularSettingsDialog", return_value=mock_dlg):
        sb._on_edit_clicked()

    assert Presets.list_presets() == ["New"]
    assert Presets.load_preset("New") == {"density": 1.5}


def test_sync_ui_skips_rebuild_when_names_unchanged(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "presets_dir", str(tmp_path))
    Presets.save_preset("Portra", {"density": 1.5})
    sb = _sidebar()
    sb.preset_list.item(0).setData(Qt.ItemDataRole.UserRole, "keep")
    sb.sync_ui()
    assert sb.preset_list.item(0).data(Qt.ItemDataRole.UserRole) == "keep"
