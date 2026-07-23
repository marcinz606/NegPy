from dataclasses import replace
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


def _sidebar_with_file(config: WorkspaceConfig) -> PresetsSidebar:
    controller = SimpleNamespace(
        state=SimpleNamespace(config=config, current_file_hash="h1"),
        session=MagicMock(),
        request_render=MagicMock(),
    )
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


def test_apply_overlay_merges_non_defaults_only(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "presets_dir", str(tmp_path))
    Presets.save_preset("P", {"density": 1.5, "saturation": 1.0})
    base = WorkspaceConfig()
    cfg = replace(base, lab=replace(base.lab, saturation=1.4, vibrance=1.2))
    sb = _sidebar_with_file(cfg)
    sb._do_apply("P", "overlay")
    new_cfg = sb.controller.session.update_config.call_args[0][0]
    assert new_cfg.exposure.density == 1.5
    assert new_cfg.lab.saturation == 1.4
    assert new_cfg.lab.vibrance == 1.2
    sb.controller.request_render.assert_called_once()


def test_apply_replace_resets_look_keeps_frame_state(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "presets_dir", str(tmp_path))
    Presets.save_preset("P", {"density": 1.5})
    base = WorkspaceConfig()
    cfg = replace(
        base,
        lab=replace(base.lab, vibrance=1.2),
        geometry=replace(base.geometry, manual_crop_rect=(0.1, 0.1, 0.9, 0.9)),
    )
    sb = _sidebar_with_file(cfg)
    sb._do_apply("P", "replace")
    new_cfg = sb.controller.session.update_config.call_args[0][0]
    assert new_cfg.exposure.density == 1.5
    assert new_cfg.lab.vibrance == base.lab.vibrance
    assert new_cfg.geometry.manual_crop_rect == (0.1, 0.1, 0.9, 0.9)


def test_sync_ui_skips_rebuild_when_names_unchanged(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "presets_dir", str(tmp_path))
    Presets.save_preset("Portra", {"density": 1.5})
    sb = _sidebar()
    sb.preset_list.item(0).setData(Qt.ItemDataRole.UserRole, "keep")
    sb.sync_ui()
    assert sb.preset_list.item(0).data(Qt.ItemDataRole.UserRole) == "keep"
