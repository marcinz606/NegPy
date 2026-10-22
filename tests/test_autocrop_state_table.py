"""The Auto Crop state table: one test per row the other suites do not already cover."""

import gc
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from negpy.desktop.controller import AppController, _autocrop_fingerprint
from negpy.desktop.session import AppState, DesktopSessionManager, ToolMode
from negpy.desktop.workers.render import BatchAutoCropResult
from negpy.domain.models import WorkspaceConfig
from negpy.features.geometry.logic import autocrop_detection_key
from negpy.features.geometry.models import AutocropMode
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets import rolls
from negpy.services.rendering.preview_manager import PreviewManager

RECT = (0.1, 0.1, 0.9, 0.9)


def _cfg(crop: str = "none", **geometry) -> WorkspaceConfig:
    """A config in one crop state: none, armed, auto or hand-drawn."""
    base = WorkspaceConfig()
    fields = {
        "none": {},
        "armed": {"crop_from_auto": True},
        "auto": {"crop_from_auto": True, "crop_rect": RECT, "crop_detect_key": "k"},
        "hand-drawn": {"crop_rect": RECT},
    }[crop]
    return replace(base, geometry=replace(base.geometry, **{**fields, **geometry}))


def _kind(config: WorkspaceConfig) -> str:
    geo = config.geometry
    if geo.crop_from_auto:
        return "auto" if geo.crop_rect is not None else "armed"
    return "hand-drawn" if geo.crop_rect is not None else "none"


@pytest.fixture
def repo(tmp_path):
    r = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    r.initialize()
    return r


# --- Opening a frame (resolution) ----------------------------------------------------------


def test_a_rolls_old_crop_values_never_reach_a_frame(repo):
    """Crop is a frame card: values a roll stored for it before never overlay a frame."""
    roll = rolls.create_virtual_roll(repo, "Roll", [])
    rolls.set_roll_defaults(repo, roll, crop_from_auto=True, autocrop_ratio="4:3", autocrop_offset=12)
    for crop in ("none", "auto", "hand-drawn"):
        assert rolls.resolve_roll_config(repo, roll, "h1", _cfg(crop)) == _cfg(crop)


def test_R1_with_no_roll_a_frame_keeps_its_own_crop(repo):
    for crop in ("none", "armed", "auto", "hand-drawn"):
        assert rolls.resolve_roll_config(repo, None, "h1", _cfg(crop)) == _cfg(crop)


# --- Controller rows ---------------------------------------------------------------------


@pytest.fixture
def ctl(repo):
    """A controller over a real repo: one roll, frames h1 (active) and h2. update_config
    writes state.config and, when persisted, the active frame's saved edit, as the session does."""
    session = MagicMock(spec=DesktopSessionManager)
    session.state = AppState()
    session.repo = repo
    session.asset_model = MagicMock()
    with (
        patch("negpy.desktop.controller.RenderWorker") as render_worker,
        patch("negpy.desktop.controller.PreviewManager") as preview_manager,
    ):
        render_worker.return_value = MagicMock()
        preview_manager.side_effect = lambda: MagicMock(spec=PreviewManager)
        controller = AppController(session)
    controller.batch_autocrop_requested.disconnect(controller.batch_autocrop_worker.process)
    tasks: list = []
    controller.batch_autocrop_requested.connect(tasks.append)
    controller.request_render = MagicMock()
    controller.set_status = MagicMock()

    state = session.state

    def update_config(cfg, persist=False, **_kwargs):
        state.config = cfg
        if persist and state.current_file_hash:
            repo.save_file_settings(state.current_file_hash, cfg, file_path=f"/r/{state.current_file_hash}.tif")

    session.update_config.side_effect = update_config
    session.config_for_asset.side_effect = lambda asset: rolls.resolve_roll_config(
        repo, state.active_roll_id, asset["hash"], repo.load_file_settings(asset["hash"]) or WorkspaceConfig()
    )
    roll = rolls.create_virtual_roll(repo, "Roll", ["/r/h1.tif", "/r/h2.tif"])
    state.active_roll_id = roll
    state.uploaded_files = [{"name": f"{h}.tif", "path": f"/r/{h}.tif", "hash": h} for h in ("h1", "h2")]
    state.selected_file_idx = 0
    state.current_file_hash = "h1"
    state.current_file_path = "/r/h1.tif"
    yield SimpleNamespace(controller=controller, session=session, state=state, repo=repo, roll=roll, tasks=tasks)

    controller.batch_autocrop_worker.cancel()
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


def _locks(t) -> set:
    return rolls.frame_override_cards(t.repo, t.roll, "h1")


def _status(t) -> str:
    return t.controller.set_status.call_args.args[0]


def test_E1_wand_on_arms_a_hand_drawn_frame_and_saves(ctl):
    ctl.state.config = _cfg("hand-drawn")

    ctl.controller.apply_auto_crop()

    assert _kind(ctl.state.config) == "armed"
    assert _kind(ctl.repo.load_file_settings("h1")) == "armed"
    assert _locks(ctl) == set()


def test_E2_reset_clears_an_auto_crop_and_saves(ctl):
    ctl.state.config = _cfg("auto")

    ctl.controller.reset_crop()

    assert _kind(ctl.repo.load_file_settings("h1")) == "none"


@pytest.mark.parametrize(
    "row, change, redetects",
    [
        ("E6 Ratio", {"autocrop_ratio": "4:3"}, True),
        ("E7 Mode", {"autocrop_mode": AutocropMode.FILM}, True),
        ("E7 Rebate Trim", {"autocrop_rebate_trim": 0.5}, True),
        ("E8 Crop Offset", {"autocrop_offset": 12}, False),
    ],
)
def test_crop_shape_edits_on_an_auto_crop(ctl, row, change, redetects):
    geo = _cfg("auto").geometry
    ctl.state.config = _cfg("auto", crop_detect_key=autocrop_detection_key(geo))

    if "autocrop_ratio" in change:
        ctl.controller.set_crop_ratio(change["autocrop_ratio"])
    else:
        ctl.controller.set_crop_settings(**change)

    after = ctl.state.config.geometry
    assert after.crop_rect == RECT, row
    assert (autocrop_detection_key(after) != after.crop_detect_key) is redetects, row
    assert ctl.repo.load_file_settings("h1").geometry == after, row
    assert _locks(ctl) == set(), row


# --- Auto-crop all frames ----------------------------------------------------------------


def _finish(t, results) -> None:
    token = t.controller._begin_batch("autocrop", "Auto-cropping all frames", True)
    t.controller._autocrop_batch_token = token
    t.controller._autocrop_dispatched = len(results)
    t.controller._autocrop_preflight_skipped = 0
    t.controller._autocrop_started_auto = {r.file_info["hash"]: True for r in results}
    t.controller._on_batch_autocrop_finished(results)


def _result(t, file_hash: str, config: WorkspaceConfig) -> BatchAutoCropResult:
    asset = next(f for f in t.state.uploaded_files if f["hash"] == file_hash)
    return BatchAutoCropResult(asset, _autocrop_fingerprint(config, t.state.workspace_color_space), (0.2, 0.2, 0.8, 0.8), 0.0, 0.9, False)


def test_A3_all_frames_excluded_reports_nothing_to_analyze(ctl):
    ctl.state.config = _cfg("hand-drawn")
    ctl.repo.save_file_settings("h2", _cfg("hand-drawn"), file_path="/r/h2.tif")

    ctl.controller.request_batch_auto_crop()

    assert ctl.tasks == []
    assert _status(ctl).startswith("Auto Crop kept 2 frames; nothing to analyze")


def test_A5_a_frame_whose_detection_settings_changed_mid_run_is_kept(ctl):
    started = _cfg("armed")
    ctl.repo.save_file_settings("h2", _cfg("armed", rotation=1), file_path="/r/h2.tif")

    _finish(ctl, [_result(ctl, "h2", started)])

    assert _kind(ctl.repo.load_file_settings("h2")) == "armed"


def test_A11_a_frame_cropped_by_hand_mid_run_is_kept(ctl):
    ctl.repo.save_file_settings("h2", _cfg("hand-drawn"), file_path="/r/h2.tif")

    _finish(ctl, [_result(ctl, "h2", _cfg("armed"))])

    assert ctl.repo.load_file_settings("h2").geometry.crop_rect == RECT


def test_A13_every_crop_edit_is_refused_while_the_run_goes(ctl):
    ctl.state.config = _cfg("auto", autocrop_ratio="3:2")
    rolls.set_section_push(ctl.repo, ctl.roll, "autocrop", {"autocrop_ratio": "5:4"})
    before = ctl.state.config
    ctl.controller._begin_batch("autocrop", "Auto-cropping all frames", True)
    c = ctl.controller

    for edit in (
        c.apply_auto_crop,
        c.reset_crop,
        lambda: c.set_crop_ratio("4:3"),
        c.detect_aspect_ratio,
        lambda: c.set_crop_settings(autocrop_offset=20),
        lambda: c.revert_to_roll(("autocrop",)),
        lambda: c.set_active_tool(ToolMode.CROP_MANUAL),
    ):
        c.set_status.reset_mock()
        edit()
        assert _status(ctl) == "Crop is locked while Auto Crop runs"

    assert ctl.state.config == before
    assert ctl.state.active_tool != ToolMode.CROP_MANUAL
    assert ctl.repo.load_file_settings("h1") is None


def test_A13_a_drag_during_the_run_writes_nothing(ctl):
    ctl.state.config = _cfg("auto")
    ctl.state.active_tool = ToolMode.CROP_MANUAL
    ctl.controller._begin_batch("autocrop", "Auto-cropping all frames", True)

    ctl.controller.handle_crop_rect_changed(0.3, 0.3, 0.7, 0.7, persist=True)

    assert ctl.state.config == _cfg("auto")


def test_A14_the_run_closes_the_crop_tool(ctl):
    ctl.state.config = _cfg("none")
    ctl.state.active_tool = ToolMode.CROP_MANUAL

    ctl.controller.request_batch_auto_crop()

    assert ctl.tasks and ctl.state.active_tool == ToolMode.NONE


def test_A14_crop_controls_are_disabled_while_the_run_goes(qapp):
    from negpy.desktop.view.sidebar.controls_panel import ControlsPanel

    controller = MagicMock()
    controller.state = AppState()
    controller.crop_edits_blocked.return_value = True
    panel = ControlsPanel(controller)
    widgets = (panel.autocrop_sidebar, panel.autocrop_section.reset_btn, panel.autocrop_section.roll_revert_btn)

    panel._sync_crop_block()
    assert not any(w.isEnabled() for w in widgets)

    controller.crop_edits_blocked.return_value = False
    panel._sync_crop_block()
    assert all(w.isEnabled() for w in widgets)


def test_A13_reset_to_roll_settings_during_the_run_resets_all_but_crop(ctl):
    rolls.set_roll_defaults(ctl.repo, ctl.roll, hue_trim=4.0)
    rolls.set_section_push(ctl.repo, ctl.roll, "autocrop", {"autocrop_ratio": "5:4"})
    ctl.state.config = replace(_cfg("auto", autocrop_ratio="3:2"), process=replace(WorkspaceConfig().process, hue_trim=-6.0))
    rolls.set_frame_override(ctl.repo, ctl.roll, "h1", "sensor", True)
    ctl.controller._begin_batch("autocrop", "Auto-cropping all frames", True)

    moved = ctl.controller.revert_frame_to_roll()

    assert moved == 1
    assert ctl.state.config.process.hue_trim == 4.0
    assert ctl.state.config.geometry.autocrop_ratio == "3:2"
    assert _status(ctl) == "Reset to the roll: Calibration; Crop waits for Auto Crop"


# --- Copying settings (paste, apply dialog, presets, carried settings) ---------------------


def _rows(*ids):
    from negpy.desktop.settings_catalog import rows_by_id

    by_id = rows_by_id()
    return [by_id[i] for i in ids]


@pytest.mark.parametrize(
    "row, source, target, expected",
    [
        ("E15 auto rect onto Auto off", "auto", "none", "auto"),
        ("E15 hand-drawn rect onto auto", "hand-drawn", "auto", "hand-drawn"),
    ],
)
def test_E15_the_crop_row_brings_its_own_auto_flag(row, source, target, expected):
    from negpy.desktop.settings_catalog import apply_selected_fields

    out = apply_selected_fields(_cfg(source), _cfg(target), _rows("geometry.crop_rect"))

    assert _kind(out) == expected, row
    assert out.geometry.crop_rect == RECT, row


@pytest.mark.parametrize(
    "row, source, target, expected, rect",
    [
        ("E16 Auto on over hand-drawn", "auto", "hand-drawn", "armed", None),
        ("E16 Auto off over auto", "none", "auto", "none", None),
        ("E16 Auto unchanged keeps the crop", "armed", "auto", "auto", RECT),
    ],
)
def test_E16_the_auto_crop_row_alone_switches_through_with_auto_crop(row, source, target, expected, rect):
    from negpy.desktop.settings_catalog import apply_selected_fields

    out = apply_selected_fields(_cfg(source), _cfg(target), _rows("geometry.crop_from_auto"))

    assert _kind(out) == expected, row
    assert out.geometry.crop_rect == rect, row


def test_E17_a_recorded_crop_carries_its_auto_flag():
    from negpy.desktop.settings_catalog import selected_flat_dict

    flat = selected_flat_dict(_cfg("auto"), _rows("geometry.crop_rect"))

    assert flat["crop_from_auto"] is True and flat["crop_rect"] == RECT


def test_E17_crop_reset_to_roll_switches_auto_through_with_auto_crop(ctl):
    rolls.set_section_push(ctl.repo, ctl.roll, "autocrop", {"crop_from_auto": True})
    ctl.state.config = _cfg("hand-drawn")

    ctl.controller.revert_to_roll(("autocrop",))

    assert _kind(ctl.state.config) == "armed"


def test_E18_crop_reset_to_roll_waits_for_the_run(ctl):
    rolls.set_section_push(ctl.repo, ctl.roll, "autocrop", {"crop_from_auto": True})
    ctl.state.config = _cfg("hand-drawn")
    ctl.controller._begin_batch("autocrop", "Auto-cropping all frames", True)

    assert ctl.controller.revert_to_roll(("autocrop",)) == 0

    assert _kind(ctl.state.config) == "hand-drawn"
    assert _status(ctl) == "Crop is locked while Auto Crop runs"


def _session_with_roll(repo):
    """A real session over a real repo: one roll of two frames, h1 active."""
    session = DesktopSessionManager(repo)
    roll = rolls.create_virtual_roll(repo, "Roll", ["/r/h1.tif", "/r/h2.tif"])
    state = session.state
    state.active_roll_id = roll
    state.uploaded_files = [{"name": f"{h}.tif", "path": f"/r/{h}.tif", "hash": h} for h in ("h1", "h2")]
    state.selected_file_idx = 0
    state.selected_indices = [0, 1]
    state.current_file_hash = "h1"
    state.current_file_path = "/r/h1.tif"
    session.asset_model = MagicMock()
    session.asset_model.visible_actual_indices_ordered.return_value = [0, 1]
    return session, roll


def test_E19_a_copied_auto_crop_survives_a_reopen(repo):
    session, _roll = _session_with_roll(repo)
    session.state.config = _cfg("auto")

    assert session.sync_selected_settings(_rows("geometry.crop_rect"), scope="selection") == 1

    reopened = session.config_for_asset(session.state.uploaded_files[1])
    assert _kind(reopened) == "auto" and reopened.geometry.crop_rect == RECT


# --- No roll open: locks follow the roll each frame resolves through (P3, P6, D35) ---------


def _search_results(t):
    """Search results: no active roll; h1 and h2 still belong to t.roll."""
    t.state.active_roll_id = None


def _reopen_in_home_roll(t, file_hash):
    return rolls.resolve_roll_config(
        t.repo, rolls.home_roll(t.repo, f"/r/{file_hash}.tif"), file_hash, t.repo.load_file_settings(file_hash)
    )


def test_D35_a_roll_card_edit_on_a_search_result_holds_in_its_own_roll(ctl):
    rolls.set_roll_defaults(ctl.repo, ctl.roll, hue_trim=4.0)
    _search_results(ctl)

    ctl.controller.set_roll_default("sensor", hue_trim=-6.0)

    assert "sensor" in _locks(ctl)
    assert _reopen_in_home_roll(ctl, "h1").process.hue_trim == -6.0


@pytest.mark.parametrize("settings", [{}, {"autocrop_ratio": "Free"}], ids=["Auto Crop button", "Crop card reset"])
@pytest.mark.parametrize("crop, carried", [("hand-drawn", True), ("auto", False)])
def test_E2_reset_crop_keeps_the_carried_auto_crop_only_for_a_hand_drawn_crop(repo, tmp_path, crop, carried, settings):
    session = DesktopSessionManager(repo)
    session.state.uploaded_files = [{"name": "a.tif", "path": str(tmp_path / "a.tif"), "hash": "ha"}]
    session.select_file(0)
    session.update_config(replace(session.state.config, geometry=replace(session.state.config.geometry, crop_from_auto=True)), persist=True)
    session.update_config(replace(session.state.config, geometry=_cfg(crop, autocrop_ratio="4:3").geometry), persist=True)
    controller = AppController.__new__(AppController)
    controller.session = session
    controller.state = session.state
    controller._active_batch = None
    controller.request_render = MagicMock()
    controller.loading_started = MagicMock()
    controller.tool_sync_requested = MagicMock()
    controller._peek_sections = None

    AppController.reset_crop(controller, **settings)

    assert _kind(session.state.config) == "none"
    assert session.state.config.geometry.autocrop_ratio == settings.get("autocrop_ratio", "4:3")
    assert repo.get_global_setting("sticky_config")["crop_from_auto"] is carried
    controller.request_render.assert_called_once()


def test_the_roll_run_saves_the_keystone_with_its_own_rotation(ctl):
    """Tilt and Swing from the fit, rotation from the run's own deskew, and the arm marked squared."""
    from dataclasses import replace as dc_replace

    started = _cfg("armed", autocrop_square=True)
    ctl.repo.save_file_settings("h2", started, file_path="/r/h2.tif")
    result = dc_replace(_result(ctl, "h2", started), correction_angle=0.3, keystone=(0.5, -0.3))

    _finish(ctl, [result])

    geometry = ctl.repo.load_file_settings("h2").geometry
    assert (round(geometry.fine_rotation, 6), geometry.converge_v, geometry.converge_h) == (0.3, 0.5, -0.3)
    assert geometry.autocrop_squared and geometry.crop_rect == (0.2, 0.2, 0.8, 0.8)
