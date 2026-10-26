from collections.abc import Iterator
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PyQt6.QtTest import QTest

from negpy.desktop.session import DesktopSessionManager, ToolMode
from negpy.desktop.controller import AppController
from negpy.desktop.view.sidebar.finish import FinishSidebar
from negpy.desktop.view.sidebar.lab import LabSidebar
from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.storage.repository import StorageRepository


@pytest.fixture
def session(tmp_path, qapp) -> Iterator[DesktopSessionManager]:
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    manager = DesktopSessionManager(repo)
    manager.state.current_file_hash = "frame"
    yield manager


def _chroma(config: WorkspaceConfig, value: float) -> WorkspaceConfig:
    return replace(config, lab=replace(config.lab, saturation=value))


def _preview_and_commit(session: DesktopSessionManager, config: WorkspaceConfig) -> None:
    session.update_config(config, persist=False, render=False)
    session.update_config(config, persist=True, render=False)


def test_previewed_edits_in_different_sections_undo_one_at_a_time(session) -> None:
    initial = session.state.config
    flipped = replace(initial, geometry=replace(initial.geometry, flip_horizontal=True))
    session.update_config(flipped, persist=True)
    colored = _chroma(flipped, 1.3)
    _preview_and_commit(session, colored)
    finished = replace(colored, finish=replace(colored.finish, vignette_stops=-0.5))
    _preview_and_commit(session, finished)

    assert session.state.undo_index == 3
    for expected in (colored, flipped, initial):
        session.undo()
        assert session.state.config == expected
    for expected in (flipped, colored, finished):
        session.redo()
        assert session.state.config == expected


def test_one_drag_preserves_the_config_before_its_first_preview(session) -> None:
    initial = session.state.config
    for value in (1.1, 1.2, 1.3):
        session.update_config(_chroma(initial, value))
    session.update_config(session.state.config, persist=True)

    assert session.state.undo_index == 1
    session.undo()
    assert session.state.config == initial
    session.redo()
    assert session.state.config == _chroma(initial, 1.3)


def test_real_sidebar_slider_commits_keep_separate_undo_steps(session, qapp) -> None:
    def apply_config(config, persist=False, readback_metrics=True):
        session.update_config(config, persist=persist, render=False)

    controller = SimpleNamespace(
        state=session.state, session=session, apply_config=apply_config, image_updated=SimpleNamespace(connect=lambda _slot: None)
    )
    lab = LabSidebar(controller)
    finish = FinishSidebar(controller)
    initial = session.state.config
    expected = [initial]
    for control, value in ((lab.saturation_slider, 1.3), (finish.vignette_burn_slider, -0.5)):
        control.slider.setSliderDown(True)
        control.slider.setValue(control._to_int(value))
        QTest.qWait(100)
        control.slider.setSliderDown(False)
        expected.append(session.state.config)

    assert session.state.undo_index == 2
    session.undo()
    assert session.state.config == expected[1]
    session.undo()
    assert session.state.config == expected[0]
    session.redo()
    session.redo()
    assert session.state.config == expected[2]


@pytest.mark.parametrize("field,value", [("crop_rect", (0.1, 0.1, 0.9, 0.9)), ("fine_rotation", 2.5)])
def test_geometry_preview_undo_keeps_an_earlier_flip(session, field, value) -> None:
    initial = session.state.config
    flipped = replace(initial, geometry=replace(initial.geometry, flip_horizontal=True))
    session.update_config(flipped, persist=True)
    edited = replace(flipped, geometry=replace(flipped.geometry, **{field: value}))
    _preview_and_commit(session, edited)

    session.undo()
    assert session.state.config == flipped
    session.redo()
    assert session.state.config == edited


def test_returning_a_preview_to_its_start_keeps_redo(session) -> None:
    initial = session.state.config
    edited = _chroma(initial, 1.3)
    session.update_config(edited, persist=True)
    session.undo()
    session.update_config(_chroma(initial, 1.2))
    _preview_and_commit(session, initial)

    assert session.state.undo_index == 0
    assert session.state.max_history_index == 1
    session.redo()
    assert session.state.config == edited


def test_previewed_edit_after_undo_replaces_only_the_redo_branch(session) -> None:
    initial = session.state.config
    first = _chroma(initial, 1.2)
    second = _chroma(initial, 1.3)
    session.update_config(first, persist=True)
    session.update_config(second, persist=True)
    session.undo()
    replacement = replace(first, finish=replace(first.finish, vignette_stops=-0.5))
    _preview_and_commit(session, replacement)

    assert session.state.undo_index == session.state.max_history_index == 2
    session.undo()
    assert session.state.config == first
    session.redo()
    assert session.state.config == replacement


def test_automatic_persist_during_a_preview_does_not_end_the_edit(session) -> None:
    initial = session.state.config
    session.update_config(_chroma(initial, 1.2))
    measured = replace(session.state.config, process=replace(initial.process, local_floors=(0.1, 0.1, 0.1)))
    session.update_config(measured, persist=True, render=False, record_history=False)
    final = _chroma(measured, 1.3)
    session.update_config(final, persist=True)

    assert session.state.undo_index == 1
    session.undo()
    assert session.state.config == initial
    session.redo()
    assert session.state.config == final


def test_undo_during_a_preview_keeps_the_previous_completed_edit(session) -> None:
    initial = session.state.config
    first = _chroma(initial, 1.2)
    session.update_config(first, persist=True)
    session.update_config(_chroma(first, 1.3))
    session.undo()

    assert session.state.config == first
    session.redo()
    assert session.state.config == _chroma(first, 1.3)


def test_switching_frames_finishes_the_previous_preview(session, monkeypatch) -> None:
    monkeypatch.setattr("negpy.infrastructure.loaders.helpers.read_exif_from_file", lambda _: {})
    session.state.uploaded_files = [
        {"hash": "frame", "path": "/first.tif", "name": "first.tif"},
        {"hash": "other", "path": "/other.tif", "name": "other.tif"},
    ]
    session.select_file(0)
    initial = session.state.config
    _preview_and_commit(session, _chroma(initial, 1.2))
    final = _chroma(initial, 1.3)
    session.update_config(final)
    session.select_file(1)
    other = session.state.config
    _preview_and_commit(session, replace(other, finish=replace(other.finish, vignette_stops=-0.5)))
    session.undo()
    assert session.state.config == other

    session.select_file(0)
    assert session.state.config == final
    session.undo()
    assert session.state.config == _chroma(initial, 1.2)
    session.undo()
    assert session.state.config == initial


def test_crop_controller_drag_undo_keeps_the_earlier_edit(session) -> None:
    initial = session.state.config
    previous = replace(_chroma(initial, 1.3), process=replace(initial.process, local_floors=(0.1, 0.2, 0.3)))
    session.update_config(previous, persist=True)
    session.state.active_tool = ToolMode.CROP_MANUAL
    controller = SimpleNamespace(
        state=session.state,
        session=session,
        crop_edits_blocked=lambda: False,
        _render_debounce=SimpleNamespace(stop=lambda: None),
        _lock_roll_card=lambda _: None,
        reset_all_peeks=lambda: None,
        request_render=lambda: None,
        tool_sync_requested=SimpleNamespace(emit=lambda: None),
        loading_started=SimpleNamespace(emit=lambda: None),
        _carry_peek=lambda _: None,
    )
    AppController.handle_crop_rect_changed(controller, 0.1, 0.1, 0.8, 0.8, False)
    AppController.handle_crop_rect_changed(controller, 0.1, 0.1, 0.9, 0.9, False)
    AppController.handle_crop_rect_changed(controller, 0.1, 0.1, 0.9, 0.9, True)
    AppController.set_active_tool(controller, ToolMode.NONE)
    cropped = session.state.config

    assert session.state.undo_index == 2
    session.undo()
    assert session.state.config == previous
    session.redo()
    assert session.state.config == cropped


def test_history_jump_finishes_a_preview_and_preserves_its_redo(session) -> None:
    initial = session.state.config
    first = _chroma(initial, 1.2)
    session.update_config(first, persist=True)
    final = _chroma(initial, 1.3)
    session.update_config(final)
    session.jump_to_step(1)

    assert session.state.config == first
    session.redo()
    assert session.state.config == final


def test_non_history_updates_do_not_create_an_edit_of_their_own(session) -> None:
    initial = session.state.config
    measured = replace(initial, process=replace(initial.process, local_floors=(0.1, 0.1, 0.1)))
    session.update_config(measured, persist=True, record_history=False)
    _preview_and_commit(session, _chroma(measured, 1.3))

    assert session.state.undo_index == 1
    session.undo()
    assert session.state.config == measured


def test_unloading_a_frame_discards_its_pending_preview(session) -> None:
    initial = session.state.config
    session.update_config(_chroma(initial, 1.2))
    session._reset_active_image_state()
    session.state.current_file_hash = "other"
    other = session.state.config
    _preview_and_commit(session, _chroma(other, 1.3))

    assert session.state.undo_index == 1
    session.undo()
    assert session.state.config == other


def test_reanalyzing_bounds_does_not_consume_an_undo_step(session) -> None:
    initial = replace(session.state.config, process=replace(session.state.config.process, local_floors=(0.1, 0.2, 0.3)))
    session.update_config(initial, record_history=False)
    session.update_config(_chroma(initial, 1.3), persist=True)
    controller = SimpleNamespace(state=session.state, session=session, request_render=lambda: None)
    AppController.reanalyze_current_file(controller)

    session.undo()
    assert session.state.config == initial
