"""One Undo takes back a change to several frames, from any frame."""

from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import DesktopSessionManager
from negpy.desktop.settings_catalog import rows_for_fields
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG


def _session(tmp_path):
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    session = DesktopSessionManager(repo)
    state = session.state
    state.uploaded_files = [{"name": f"{h}.tif", "path": f"/r/{h}.tif", "hash": h} for h in ("h1", "h2", "h3")]
    for h in ("h2", "h3"):
        repo.save_file_settings(h, DEFAULT_WORKSPACE_CONFIG, file_path=f"/r/{h}.tif")
    state.selected_file_idx = 0
    state.selected_indices = [0, 1, 2]
    state.current_file_hash = "h1"
    state.current_file_path = "/r/h1.tif"
    state.config = replace(DEFAULT_WORKSPACE_CONFIG, exposure=replace(DEFAULT_WORKSPACE_CONFIG.exposure, density=1.7))
    session.asset_model = MagicMock()
    session.asset_model.visible_actual_indices_ordered.return_value = [0, 1, 2]
    return session, repo


def test_an_apply_to_several_frames_is_undone_in_one_step(tmp_path):
    session, repo = _session(tmp_path)

    session.sync_selected_settings(rows_for_fields(("density",)), scope="selection")
    assert repo.load_file_settings("h2").exposure.density == 1.7
    assert session.last_batch is not None

    assert session.undo_last_batch() == 2
    assert repo.load_file_settings("h2").exposure.density == DEFAULT_WORKSPACE_CONFIG.exposure.density
    assert repo.load_file_settings("h3").exposure.density == DEFAULT_WORKSPACE_CONFIG.exposure.density
    assert session.last_batch is None


def test_an_edit_after_the_batch_withdraws_its_undo(tmp_path):
    """Undoing then would take the later edit with it."""
    session, _repo = _session(tmp_path)
    session.sync_selected_settings(rows_for_fields(("density",)), scope="selection")

    session.update_config(replace(session.state.config, exposure=replace(session.state.config.exposure, grade=90.0)), persist=True)

    assert session.last_batch is None


def test_a_batch_inside_a_batch_is_one_undo(tmp_path):
    session, repo = _session(tmp_path)
    with session.recording_batch("Outer"):
        session.sync_selected_settings(rows_for_fields(("density",)), scope="selection")

    assert session.last_batch["label"] == "Outer"
    session.undo_last_batch()
    assert repo.load_file_settings("h2").exposure.density == DEFAULT_WORKSPACE_CONFIG.exposure.density
