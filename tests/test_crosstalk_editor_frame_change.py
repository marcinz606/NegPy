"""The Crosstalk editor previews on the frame it opened on. Before that frame's live config is
stored (a frame change, a roll fork) the editor takes its preview back and closes, so no frame
keeps a previewed matrix."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gc
import sys
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtWidgets import QApplication

from negpy.desktop.controller import AppController
from negpy.desktop.session import DesktopSessionManager
from negpy.desktop.view.sidebar.sensor import SensorSidebar
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.kernel.system.config import APP_CONFIG
from negpy.services.assets import rolls
from negpy.services.assets.crosstalk import CrosstalkProfiles
from negpy.services.rendering.preview_manager import PreviewManager

_FIELDS = ("crosstalk_profile", "crosstalk_matrix", "crosstalk_strength", "crosstalk_process", "local_floors", "local_ceils")


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def controller(_app, tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(tmp_path / "ct"))
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    session = DesktopSessionManager(repo)
    paths = [str(tmp_path / "a.tif"), str(tmp_path / "b.tif")]
    session.state.uploaded_files = [{"name": os.path.basename(p), "path": p, "hash": h} for p, h in zip(paths, ("ha", "hb"))]
    with (
        patch("negpy.desktop.controller.RenderWorker") as render_worker,
        patch("negpy.desktop.controller.PreviewManager") as preview_manager,
    ):
        render_worker.return_value = MagicMock()
        preview_manager.side_effect = lambda: MagicMock(spec=PreviewManager)
        ctrl = AppController(session)
    # No decode or render: the test reads configs, not pixels.
    session.file_selected.disconnect(ctrl._on_file_selected_load)
    ctrl.request_render = MagicMock()
    ctrl.set_status = MagicMock()
    session.state.active_roll_id = rolls.create_virtual_roll(repo, "Roll", paths)
    yield ctrl
    for thread in (
        ctrl.render_thread,
        ctrl.export_thread,
        ctrl.thumb_thread,
        ctrl.norm_thread,
        ctrl.discovery_thread,
        ctrl.preview_load_thread,
        ctrl.prefetch_load_thread,
        ctrl.scan_thread,
    ):
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait()
    del ctrl
    gc.collect()


def _fields(config) -> tuple:
    return tuple(getattr(config.process, f) for f in _FIELDS)


def _saved(session, idx: int) -> tuple:
    return _fields(session.config_for_asset(session.state.uploaded_files[idx]))


def _frame_a_with_bounds(ctrl) -> SensorSidebar:
    """Frame a active, with a saved edit and bounds of its own, which a preview clears."""
    session = ctrl.session
    session.select_file(1)
    session.select_file(0)
    cfg = session.state.config
    session.update_config(
        replace(cfg, process=replace(cfg.process, local_floors=(0.1, 0.1, 0.1), local_ceils=(0.9, 0.9, 0.9))), persist=True
    )
    return SensorSidebar(ctrl)


def _preview_a_bundled_matrix(w: SensorSidebar) -> None:
    w._open_crosstalk_editor()
    dlg = w._crosstalk_dialog
    name = next(n for n in sorted(CrosstalkProfiles.scan_bundled()) if n != CrosstalkProfiles.DEFAULT_NAME)
    dlg.profile_list.setCurrentRow([dlg.profile_list.item(i).text() for i in range(dlg.profile_list.count())].index(name))


def test_a_frame_change_ends_the_preview_on_both_frames(controller):
    session = controller.session
    w = _frame_a_with_bounds(controller)
    before = (_saved(session, 0), _saved(session, 1))
    _preview_a_bundled_matrix(w)
    dlg = w._crosstalk_dialog
    assert _fields(session.state.config)[1] != before[0][1]

    session.select_file(1)

    assert (_saved(session, 0), _saved(session, 1)) == before
    assert _fields(session.state.config) == before[1]
    assert not dlg.isVisible()
    assert "sensor" not in controller._previewed_meter_cards


def test_a_persisted_edit_during_the_preview_keeps_the_user_change(controller):
    session = controller.session
    w = _frame_a_with_bounds(controller)
    _preview_a_bundled_matrix(w)
    w._on_crosstalk_strength_changed(0.6, persist=True)

    session.select_file(1)

    saved = session.config_for_asset(session.state.uploaded_files[0]).process
    assert saved.crosstalk_matrix is None
    assert saved.crosstalk_strength == 0.6
    assert saved.local_floors == (0.0, 0.0, 0.0)  # re-metered, not the stale open-time bounds


def test_a_roll_fork_seeds_without_the_preview(controller):
    session = controller.session
    w = _frame_a_with_bounds(controller)
    before = _saved(session, 0)
    _preview_a_bundled_matrix(w)

    controller.request_fork_edit_for_roll()

    assert session.state.current_file_hash.endswith("#roll:" + session.state.active_roll_id)
    assert _fields(session.state.config) == before
    assert _fields(session.repo.load_file_settings("ha")) == before


def test_a_second_open_raises_the_editor_already_open(controller):
    w = _frame_a_with_bounds(controller)
    w._open_crosstalk_editor()
    first = w._crosstalk_dialog
    w._open_crosstalk_editor()
    assert w._crosstalk_dialog is first


def test_an_emptied_session_closes_the_editor(controller):
    w = _frame_a_with_bounds(controller)
    _preview_a_bundled_matrix(w)
    dlg = w._crosstalk_dialog

    controller.session.remove_files([0, 1])

    assert not dlg.isVisible()
    assert w._crosstalk_dialog is None


def test_cancel_still_restores_the_frame(controller):
    session = controller.session
    w = _frame_a_with_bounds(controller)
    before = _saved(session, 0)
    _preview_a_bundled_matrix(w)

    w._crosstalk_dialog.reject()

    assert _saved(session, 0)[:4] == before[:4]
