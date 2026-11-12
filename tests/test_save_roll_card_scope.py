"""Saving the open frames as a roll turns every Roll-tab card of the open frame to Roll.

A real ControlsPanel on a real AppController: the scope comes through config_updated, the
panel's sync debounce and _sync_scope_buttons, so stubs would test none of the wiring.
"""

import gc
import os
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import QApplication

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState, DesktopSessionManager
from negpy.desktop.view.sidebar.controls_panel import ControlsPanel
from negpy.services.rendering.preview_manager import PreviewManager

_FRAMES = ("a1.tif", "a2.tif")


@pytest.fixture
def controller():
    session = MagicMock(spec=DesktopSessionManager)
    session.state = AppState()
    session.repo = MagicMock()
    session.repo.load_settings_by_path.return_value = {}
    session.repo.load_file_marks_by_path.return_value = {}
    store: dict = {}
    session.repo.get_global_setting.side_effect = lambda key, default=None: store.get(key, default)
    session.repo.save_global_setting.side_effect = lambda key, value: store.__setitem__(key, value)
    session.asset_model = MagicMock()
    with (
        patch("negpy.desktop.controller.RenderWorker", return_value=MagicMock()),
        patch("negpy.desktop.controller.PreviewManager", return_value=MagicMock(spec=PreviewManager)),
    ):
        controller = AppController(session)
    yield controller
    for thread in (
        controller.render_thread,
        controller.export_thread,
        controller.thumb_thread,
        controller.norm_thread,
        controller.discovery_thread,
        controller.preview_load_thread,
        controller.prefetch_load_thread,
        controller.scan_thread,
    ):
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait()
    del controller
    gc.collect()


@pytest.fixture
def panel(controller):
    panel = ControlsPanel(controller)
    yield panel
    panel.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


@pytest.mark.parametrize(
    "save",
    [
        pytest.param(lambda c: c.save_to_folder_roll(), id="folder"),
        pytest.param(lambda c: c.create_roll_from_session("Picks"), id="virtual"),
    ],
)
def test_saving_the_frames_as_a_roll_turns_every_roll_card_to_roll(controller, panel, tmp_path, save):
    for name in _FRAMES:
        (tmp_path / name).touch()
    controller.state.active_roll_id = None
    controller.state.duplicate_paths = set()
    controller.state.uploaded_files = [{"name": n, "path": os.path.join(tmp_path, n), "hash": n} for n in _FRAMES]
    controller.state.current_file_hash = _FRAMES[0]
    panel._sync_all_sidebars()
    sections = panel._roll_sections()
    assert not any(section.roll_btn.isEnabled() for _key, section in sections)

    with patch.object(controller.session, "persist_session"):
        assert save(controller)

    assert panel._sync_debounce.isActive(), "the save asks the panel to resync"
    panel._sync_debounce.stop()
    panel._sync_all_sidebars()
    for key, section in sections:
        assert section.roll_btn.isEnabled() and section.roll_btn.isChecked(), key
