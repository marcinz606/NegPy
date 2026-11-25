from collections.abc import Iterator
from dataclasses import replace
from unittest.mock import patch

import pytest
from PyQt6.QtCore import QPoint, QPointF

from negpy.desktop.controller import AppController
from negpy.desktop.session import DesktopSessionManager, ToolMode
from negpy.desktop.view.canvas.toolbar import ActionToolbar
from negpy.desktop.view.canvas.widget import ImageCanvas
from negpy.infrastructure.storage.repository import StorageRepository


@pytest.fixture
def controller(tmp_path, qapp) -> Iterator[AppController]:
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    session = DesktopSessionManager(repo)
    session.state.uploaded_files = [{"name": "f0.arw", "path": str(tmp_path / "f0.arw"), "hash": "frame0"}]
    session.state.selected_file_idx = 0
    session.state.current_file_hash = "frame0"
    session.state.current_file_path = str(tmp_path / "f0.arw")
    with (
        patch("negpy.desktop.controller.RenderWorker"),
        patch("negpy.desktop.controller.PreviewManager"),
    ):
        ctrl = AppController(session)
    try:
        yield ctrl
    finally:
        ctrl.cleanup()


def _stroke(x: float) -> list:
    return [[[x, x]], 6.0, 0.0, 0.0]


def _clone(x: float) -> list:
    return [[[x, x]], 6.0, 0.1, 0.1, 1.0, 0.5, True]


def test_toolbar_undo_and_redo_round_trip_spot_heals_while_tool_is_active(controller, qapp) -> None:
    tb = ActionToolbar(controller)
    state = controller.state
    state.active_tool = ToolMode.DUST_PICK

    controller._commit_heal_stroke([(0.2, 0.2)])
    controller._commit_heal_stroke([(0.4, 0.4)])
    assert len(state.config.retouch.manual_heal_strokes) == 2
    assert tb.btn_tool_undo.isEnabled()
    assert not tb.btn_tool_redo.isEnabled()

    tb.btn_tool_undo.click()
    assert len(state.config.retouch.manual_heal_strokes) == 1
    assert tb.btn_tool_redo.isEnabled()
    assert tb.btn_redo.isEnabled()

    tb.btn_tool_redo.click()
    assert len(state.config.retouch.manual_heal_strokes) == 2
    assert not tb.btn_tool_redo.isEnabled()


def test_sidebar_undo_last_retouch_and_clone_keep_redo_available(controller) -> None:
    session = controller.session
    state = controller.state
    state.active_tool = ToolMode.DUST_PICK

    # Two strokes in a row exercise tuple-vs-JSON-list round-tripping in edit_history,
    # alongside post-render record_history=False meter, bounds, and fade measurements.
    controller._commit_heal_stroke([(0.3, 0.3)])
    controller._commit_heal_stroke([(0.6, 0.6)])
    measured = replace(
        state.config,
        process=replace(
            state.config.process,
            local_ceils=(0.91, 0.92, 0.93),
            meter_rect=(0.05, 0.05, 0.95, 0.95),
            meter_key="m1",
            fade_white_r=-0.02,
            fade_white_space="s1",
            fade_crossover_table=(0.0, 0.5, 1.0),
            fade_crossover_space="c1",
        ),
    )
    session.update_config(measured, persist=True, render=False, record_history=False)

    controller.undo_last_retouch()
    assert len(state.config.retouch.manual_heal_strokes) == 1
    assert state.undo_index == 1
    assert state.max_history_index == 2

    controller.undo_last_retouch()
    assert state.config.retouch.manual_heal_strokes == []
    assert state.undo_index == 0
    assert state.max_history_index == 2

    session.redo()
    assert len(state.config.retouch.manual_heal_strokes) == 1
    session.redo()
    assert len(state.config.retouch.manual_heal_strokes) == 2

    # Clone sidebar/menu undo likewise steps back across multiple strokes.
    state.active_tool = ToolMode.CLONE
    controller._set_clone_strokes([tuple(_clone(0.5))])
    controller._set_clone_strokes([tuple(_clone(0.5)), tuple(_clone(0.7))])
    assert state.undo_index == 4
    controller.undo_last_clone()
    assert len(state.config.retouch.clone_strokes) == 1
    assert state.undo_index == 3
    assert state.max_history_index == 4
    session.redo()
    assert len(state.config.retouch.clone_strokes) == 2


def test_sidebar_undo_last_retouch_after_later_edit_preserves_later_edit_and_is_undoable(controller) -> None:
    session = controller.session
    state = controller.state
    state.active_tool = ToolMode.DUST_PICK

    controller._commit_heal_stroke([(0.25, 0.25)])
    brighter = replace(state.config, exposure=replace(state.config.exposure, density=0.35))
    session.update_config(brighter, persist=True, render=False)
    assert state.undo_index == 2

    # Sidebar/context-menu "Undo Last Heal" pops only the heal and keeps the density edit.
    controller.undo_last_retouch()
    assert state.config.retouch.manual_heal_strokes == []
    assert state.config.exposure.density == 0.35
    assert state.undo_index == 3

    # Session undo restores the popped heal, and redo removes it again.
    session.undo()
    assert len(state.config.retouch.manual_heal_strokes) == 1
    assert state.config.exposure.density == 0.35
    session.redo()
    assert state.config.retouch.manual_heal_strokes == []


def test_undo_after_delete_or_clear_heals_restores_heals_while_tool_is_active(controller) -> None:
    session = controller.session
    state = controller.state
    state.active_tool = ToolMode.DUST_PICK

    s1, s2 = _stroke(0.2), _stroke(0.6)
    session.update_config(
        replace(state.config, retouch=replace(state.config.retouch, manual_heal_strokes=[s1, s2])),
        persist=True,
        render=False,
    )
    controller.delete_heal("stroke", 0)
    assert state.config.retouch.manual_heal_strokes == [s2]

    # Pressing Undo while Heal is active restores the deleted heal instead of deleting s2.
    session.undo()
    assert state.config.retouch.manual_heal_strokes == [s1, s2]
    session.redo()
    assert state.config.retouch.manual_heal_strokes == [s2]

    with patch("negpy.desktop.view.confirm.confirm_clear_heals", return_value=True):
        controller.clear_retouch()
    assert state.config.retouch.manual_heal_strokes == []
    session.undo()
    assert state.config.retouch.manual_heal_strokes == [s2]
    session.redo()
    assert state.config.retouch.manual_heal_strokes == []


def test_undo_last_retouch_unwinds_latest_repair_across_tool_switches_and_falls_back_by_tool(controller) -> None:
    session = controller.session
    state = controller.state
    line = [0.5, 0.1, 0.5, 0.9, 4.0]

    # Add a heal stroke, then a scratch line, then press Esc (ToolMode.NONE): sidebar Undo
    # must still remove the scratch line (the most recent edit) and preserve Redo.
    state.active_tool = ToolMode.DUST_PICK
    controller._commit_heal_stroke([(0.3, 0.3)])
    state.active_tool = ToolMode.SCRATCH_LINE
    session.update_config(
        replace(state.config, retouch=replace(state.config.retouch, scratch_lines=[tuple(line)])),
        persist=True,
        render=False,
    )
    state.active_tool = ToolMode.NONE
    controller.undo_last_retouch()
    assert len(state.config.retouch.manual_heal_strokes) == 1
    assert state.config.retouch.scratch_lines == []
    assert state.undo_index == 1 and state.max_history_index == 2

    session.redo()
    assert state.config.retouch.scratch_lines == [line]

    # Interleave a non-retouch edit so undo_last_retouch uses its active-tool fallback.
    session.update_config(
        replace(state.config, exposure=replace(state.config.exposure, density=0.1)),
        persist=True,
        render=False,
    )
    state.active_tool = ToolMode.DUST_PICK
    controller.undo_last_retouch()
    assert state.config.retouch.manual_heal_strokes == []
    assert state.config.retouch.scratch_lines == [line]

    state.active_tool = ToolMode.SCRATCH_LINE
    controller.undo_last_retouch()
    assert state.config.retouch.scratch_lines == []


def test_retouch_context_menu_enables_scratch_line_undo_and_redo(controller, qapp) -> None:
    from PyQt6.QtWidgets import QMenu

    state = controller.state
    canvas = ImageCanvas(state)
    canvas._controller = controller
    state.active_tool = ToolMode.SCRATCH_LINE
    state.right_click_excludes = True
    canvas.overlay.set_tool_mode(ToolMode.SCRATCH_LINE)
    assert not canvas.overlay._right_click_excludes()

    line = [0.5, 0.1, 0.5, 0.9, 4.0]
    controller.session.update_config(
        replace(state.config, retouch=replace(state.config.retouch, scratch_lines=[tuple(line)])),
        persist=True,
        render=False,
    )
    controller.undo_last_retouch()
    assert state.undo_index == 0 and state.max_history_index == 1

    captured = []

    def grab_exec(menu_self, *_a):
        captured.extend(menu_self.actions())

    with patch.object(QMenu, "exec", grab_exec):
        canvas.show_canvas_menu(QPointF(10, 10), QPoint(10, 10))

    by_text = {a.text().split("  ")[0]: a for a in captured if not a.isSeparator()}
    assert "Undo Last Heal" in by_text
    assert not by_text["Undo Last Heal"].isEnabled()
    assert by_text["Redo"].isEnabled()
    by_text["Redo"].trigger()
    assert state.config.retouch.scratch_lines == [line]
