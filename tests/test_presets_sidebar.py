from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog

from negpy.desktop.session import DesktopSessionManager
from negpy.desktop.settings_catalog import all_rows
from negpy.desktop.view.sidebar.presets import PresetsSidebar
from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.storage.repository import StorageRepository
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


def _session() -> DesktopSessionManager:
    repo = MagicMock(spec=StorageRepository)
    repo.get_global_setting.return_value = None
    repo.load_file_settings.return_value = None
    repo.load_file_settings_by_path.return_value = None
    repo.get_max_history_index.return_value = 0
    mgr = DesktopSessionManager(repo)
    mgr.state.uploaded_files = [{"name": f"f{i}.tif", "path": f"/tmp/f{i}.tif", "hash": f"h{i}"} for i in range(3)]
    mgr.asset_model.refresh()
    mgr.state.selected_file_idx = 0
    mgr.state.selected_indices = [0, 1]
    mgr.state.current_file_hash = "h0"
    mgr.state.current_file_path = "/tmp/f0.tif"
    return mgr


def _density_row():
    return next(r for r in all_rows() if r.label == "Print Density")


def _preset_cfg(density: float) -> WorkspaceConfig:
    base = WorkspaceConfig()
    return replace(base, exposure=replace(base.exposure, density=density))


def test_apply_preset_fields_current_scope(qapp):
    mgr = _session()
    base = WorkspaceConfig()
    mgr.state.config = replace(base, lab=replace(base.lab, vibrance=1.2))
    assert mgr.apply_preset_fields(_preset_cfg(1.5), [_density_row()], "current") == 1
    assert mgr.state.config.exposure.density == 1.5
    assert mgr.state.config.lab.vibrance == 1.2
    saved_hashes = [c.args[0] for c in mgr.repo.save_file_settings.call_args_list]
    assert saved_hashes == ["h0"]


def test_apply_preset_fields_selection_scope(qapp):
    mgr = _session()
    assert mgr.apply_preset_fields(_preset_cfg(1.5), [_density_row()], "selection") == 2
    saved = {c.args[0]: c.args[1] for c in mgr.repo.save_file_settings.call_args_list}
    assert saved["h1"].exposure.density == 1.5
    assert "h2" not in saved
    assert mgr.state.config.exposure.density == 1.5


def test_apply_preset_fields_roll_scope(qapp):
    mgr = _session()
    assert mgr.apply_preset_fields(_preset_cfg(1.5), [_density_row()], "roll") == 3
    saved = {c.args[0]: c.args[1] for c in mgr.repo.save_file_settings.call_args_list}
    assert saved["h1"].exposure.density == 1.5
    assert saved["h2"].exposure.density == 1.5
    assert mgr.state.config.exposure.density == 1.5


def test_apply_dialog_routes_scope_and_mode(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "presets_dir", str(tmp_path))
    Presets.save_preset("P", {"density": 1.5})
    controller = SimpleNamespace(
        state=SimpleNamespace(config=WorkspaceConfig(), current_file_hash="h0", selected_indices=[0]),
        session=MagicMock(),
        request_render=MagicMock(),
    )
    controller.session.asset_model.visible_actual_indices.return_value = [0, 1, 2]
    controller.session.apply_preset_fields.return_value = 3
    sb = PresetsSidebar(controller)
    sb.preset_list.setCurrentRow(0)

    mock_dlg = MagicMock()
    mock_dlg.exec.return_value = QDialog.DialogCode.Accepted
    mock_dlg.selected.return_value = [_density_row()]
    mock_dlg.apply_mode.return_value = "replace"
    mock_dlg.scope.return_value = "roll"
    with patch("negpy.desktop.view.sidebar.presets.GranularSettingsDialog", return_value=mock_dlg):
        sb._apply_preset()

    (src_cfg, rows, scope), _ = controller.session.apply_preset_fields.call_args
    assert scope == "roll"
    assert src_cfg.exposure.density == 1.5
    assert len(rows) > 1
    controller.request_render.assert_called_once()


def test_sync_ui_skips_rebuild_when_names_unchanged(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "presets_dir", str(tmp_path))
    Presets.save_preset("Portra", {"density": 1.5})
    sb = _sidebar()
    sb.preset_list.item(0).setData(Qt.ItemDataRole.UserRole, "keep")
    sb.sync_ui()
    assert sb.preset_list.item(0).data(Qt.ItemDataRole.UserRole) == "keep"
