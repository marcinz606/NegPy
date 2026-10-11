"""Reset Crop and Auto Crop are saved edits and undo steps, not only a render."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from negpy.desktop.controller import AppController
from negpy.desktop.session import DesktopSessionManager
from negpy.infrastructure.storage.repository import StorageRepository


@pytest.fixture
def ctrl(tmp_path):
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    session = DesktopSessionManager(repo)
    session.state.uploaded_files = [{"name": "a.tif", "path": str(tmp_path / "a.tif"), "hash": "ha"}]
    session.select_file(0)
    cfg = session.state.config
    session.update_config(replace(cfg, geometry=replace(cfg.geometry, crop_rect=(0.1, 0.1, 0.9, 0.9))), persist=True)
    c = AppController.__new__(AppController)
    c.session = session
    c.state = session.state
    c._active_batch = None
    c.request_render = MagicMock()
    c.tool_sync_requested = MagicMock()
    c.loading_started = MagicMock()
    c._peek_sections = None
    return c


@pytest.mark.parametrize("action", ["reset_crop", "apply_auto_crop"])
def test_crop_clear_is_saved_and_undoable(ctrl, action):
    getattr(AppController, action)(ctrl)

    assert ctrl.session.repo.load_file_settings("ha").geometry.crop_rect is None
    ctrl.session.undo()
    assert ctrl.state.config.geometry.crop_rect == (0.1, 0.1, 0.9, 0.9)
