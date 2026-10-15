"""Batch and preset export resolve a frame's params the way the canvas does, roll defaults included."""

from dataclasses import replace

from negpy.desktop.controller import AppController
from negpy.desktop.session import DesktopSessionManager
from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets import rolls


def test_a_saved_frame_exports_with_its_rolls_defaults(tmp_path):
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    session = DesktopSessionManager(repo)
    path = str(tmp_path / "b.tif")
    roll_id = rolls.create_virtual_roll(repo, "R", [path])
    session.state.active_roll_id = roll_id
    rolls.set_roll_defaults(repo, roll_id, hue_trim=12.0)
    repo.save_file_settings("hb", WorkspaceConfig(), file_path=path)
    asset = {"name": "b.tif", "path": path, "hash": "hb"}

    ctrl = AppController.__new__(AppController)
    ctrl.session = session
    ctrl.state = session.state
    ctrl.state.current_file_hash = "other"

    params = AppController._batch_params_for(ctrl, asset)

    assert params.process.hue_trim == 12.0
    assert params == session.config_for_asset(asset)


def test_an_unsaved_frame_still_exports_with_the_session_config(tmp_path):
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    session = DesktopSessionManager(repo)
    ctrl = AppController.__new__(AppController)
    ctrl.session = session
    ctrl.state = session.state
    ctrl.state.current_file_hash = "other"
    cfg = WorkspaceConfig()
    ctrl.state.config = replace(cfg, exposure=replace(cfg.exposure, density=0.42))

    params = AppController._batch_params_for(ctrl, {"name": "c.tif", "path": str(tmp_path / "c.tif"), "hash": "hc"})

    assert params.exposure.density == 0.42


def _roll_session(tmp_path):
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    session = DesktopSessionManager(repo)
    paths = [str(tmp_path / "a.tif"), str(tmp_path / "b.tif")]
    roll_id = rolls.create_virtual_roll(repo, "R", paths)
    rolls.set_roll_defaults(repo, roll_id, hue_trim=12.0)
    session.state.active_roll_id = roll_id
    session.state.uploaded_files = [{"name": "a.tif", "path": paths[0], "hash": "ha"}, {"name": "b.tif", "path": paths[1], "hash": "hb"}]
    session.asset_model.refresh()
    session.select_file(0)
    return session


def test_a_preset_applied_to_the_roll_sticks_on_a_roll_card(tmp_path):
    from negpy.desktop.settings_catalog import all_rows

    session = _roll_session(tmp_path)
    row = next(r for r in all_rows() if r.label == "Hue Trim")
    cfg = WorkspaceConfig()
    preset = replace(cfg, process=replace(cfg.process, hue_trim=5.0))

    session.apply_preset_fields(preset, [row], scope="roll")

    assert session.state.config.process.hue_trim == 5.0
    assert session.config_for_asset(session.state.uploaded_files[1]).process.hue_trim == 5.0
    session.select_file(1)
    session.select_file(0)
    assert session.state.config.process.hue_trim == 5.0
