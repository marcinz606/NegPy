"""One Undo takes back a change to several frames, from any frame."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

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

    assert session.last_batch.label == "Outer"
    session.undo_last_batch()
    assert repo.load_file_settings("h2").exposure.density == DEFAULT_WORKSPACE_CONFIG.exposure.density


def _locked_frame(tmp_path, card, field, roll_value, own_value):
    """h1 follows a roll that holds every field of *card*; h2 is locked on it with its own *field*."""
    from negpy.services.assets import rolls

    session, repo = _session(tmp_path)
    roll_id = rolls.create_virtual_roll(repo, "Roll", [f["path"] for f in session.state.uploaded_files])
    session.state.active_roll_id = roll_id
    section, names = rolls.ROLL_DEFAULT_FIELDS[card]

    def with_card(config, value):
        values = {n: getattr(getattr(DEFAULT_WORKSPACE_CONFIG, section), n) for n in names}
        values[field] = value
        return replace(config, **{section: replace(getattr(config, section), **values)}), values

    session.state.config, roll_values = with_card(session.state.config, roll_value)
    rolls.set_roll_defaults(repo, roll_id, **roll_values)
    own, _ = with_card(DEFAULT_WORKSPACE_CONFIG, own_value)
    repo.save_file_settings("h2", own, file_path="/r/h2.tif")
    rolls.set_frame_override(repo, roll_id, "h2", card, True)
    session.state.selected_indices = [0, 1]
    return session, repo, roll_id


_LOCK_CASES = [
    ("metadata_process", "developer", "HC-110", "Rodinal"),
    ("sensor", "crosstalk_strength", 0.5, 0.8),
    ("fade", "fade_neutrals_mix", 0.3, 0.6),
]


@pytest.mark.parametrize("card,field,roll_value,own_value", _LOCK_CASES)
def test_an_undone_apply_to_frames_gives_a_frame_its_card_lock_back(tmp_path, card, field, roll_value, own_value):
    from negpy.desktop.settings_catalog import roll_card_rows
    from negpy.services.assets import rolls

    session, repo, roll_id = _locked_frame(tmp_path, card, field, roll_value, own_value)
    section = rolls.ROLL_DEFAULT_FIELDS[card][0]
    session.sync_selected_settings(roll_card_rows(card), scope="selection")
    assert card not in rolls.frame_override_cards(repo, roll_id, "h2")

    session.undo_last_batch()

    assert card in rolls.frame_override_cards(repo, roll_id, "h2")
    h2 = session.config_for_asset(session.state.uploaded_files[1])
    assert getattr(getattr(h2, section), field) == own_value


@pytest.mark.parametrize("reset", ["selection", "assets"])
def test_an_undone_reset_frames_gives_a_frame_its_card_lock_back(tmp_path, reset):
    from negpy.services.assets import rolls

    session, repo, roll_id = _locked_frame(tmp_path, "metadata_process", "developer", "HC-110", "Rodinal")
    if reset == "selection":
        session.reset_roll_settings(scope="selection")
    else:
        session.reset_roll(session.state.uploaded_files[:2])
    assert not rolls.frame_override_cards(repo, roll_id, "h2")

    session.undo_last_batch()

    assert "metadata_process" in rolls.frame_override_cards(repo, roll_id, "h2")
    assert session.config_for_asset(session.state.uploaded_files[1]).metadata.developer == "Rodinal"


def test_written_frame_fields_leave_the_rest_and_undo_in_one_step(tmp_path):
    session, repo = _session(tmp_path)
    own = replace(DEFAULT_WORKSPACE_CONFIG.process, fade_crossover=0.4, fade_crossover_table=(0.1,) * 24)
    repo.save_file_settings("h2", replace(DEFAULT_WORKSPACE_CONFIG, process=own), file_path="/r/h2.tif")
    with session.recording_batch("Estimate Roll"):
        written = session.write_frame_fields(session.state.uploaded_files[1:], "process", {"fade_ratio_g": 0.8})

    assert written == ["h2", "h3"]
    h2 = repo.load_file_settings("h2").process
    assert h2.fade_ratio_g == 0.8 and h2.fade_crossover == 0.4 and h2.fade_crossover_table == (0.1,) * 24
    session.undo_last_batch()
    assert repo.load_file_settings("h2").process.fade_ratio_g == DEFAULT_WORKSPACE_CONFIG.process.fade_ratio_g


def _fade_roll(tmp_path, locked: str):
    """A controller over a real session: a roll that holds every Fade field at Green 0.9, slides h1
    (open), h2 and h3, a negative, and *locked* with its own Green 0.7."""
    from unittest.mock import patch

    from negpy.desktop.controller import AppController
    from negpy.features.process.models import ProcessMode
    from negpy.services.assets import rolls

    session, repo = _session(tmp_path)
    state = session.state
    state.uploaded_files.append({"name": "neg.tif", "path": "/r/neg.tif", "hash": "neg"})
    slide = replace(DEFAULT_WORKSPACE_CONFIG, process=replace(DEFAULT_WORKSPACE_CONFIG.process, process_mode=ProcessMode.E6))
    roll_id = rolls.create_virtual_roll(repo, "Slides", [f["path"] for f in state.uploaded_files])
    state.active_roll_id = roll_id
    rolls.set_roll_defaults(repo, roll_id, **{n: getattr(replace(slide.process, fade_ratio_g=0.9), n) for n in rolls.card_fields("fade")})
    own = replace(slide, process=replace(slide.process, fade_ratio_g=0.7))
    for f in state.uploaded_files:
        config = DEFAULT_WORKSPACE_CONFIG if f["hash"] == "neg" else own if f["hash"] == locked else slide
        repo.save_file_settings(f["hash"], config, file_path=f["path"])
    rolls.set_frame_override(repo, roll_id, locked, "fade", True)
    state.selected_indices = [0]
    state.config = session.config_for_asset(state.uploaded_files[0])
    with patch("negpy.desktop.controller.RenderWorker"), patch("negpy.desktop.controller.PreviewManager"):
        controller = AppController(session)
    return controller, session, repo, roll_id


def _stop(controller) -> None:
    import gc

    for name in ("render", "export", "thumb", "norm", "discovery", "preview_load", "prefetch_load", "scan"):
        thread = getattr(controller, f"{name}_thread")
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait()
    del controller
    gc.collect()


_ROLL_ESTIMATE = dict(
    fade_ratio_r=1.0, fade_ratio_g=0.8, fade_ratio_b=1.0, fade_density_shift=0.0, fade_auto_red=False, fade_white_anchor=False
)


def test_undoing_estimate_roll_gives_a_frame_its_own_fade_and_lock_back(tmp_path):
    from negpy.services.assets import rolls

    controller, session, repo, roll_id = _fade_roll(tmp_path, locked="h2")
    try:
        h2, h3 = session.state.uploaded_files[1:3]
        hints = []
        controller.fade_roll_hint.connect(lambda *args: hints.append(args[0]))

        controller._apply_fade_roll(dict(_ROLL_ESTIMATE), "Estimated from 3 of 3 slides.")
        assert session.config_for_asset(h2).process.fade_ratio_g == 0.8
        assert session.config_for_asset(h3).process.fade_ratio_g == 0.8
        assert repo.load_file_settings("neg") == DEFAULT_WORKSPACE_CONFIG

        controller.undo_last_batch()

        assert rolls.frame_override_cards(repo, roll_id, "h2") == {"fade"}
        assert session.config_for_asset(h2).process.fade_ratio_g == 0.7
        assert session.config_for_asset(h3).process.fade_ratio_g == 0.9
        assert session.state.config.process.fade_ratio_g == 0.9
        assert hints == ["Estimated from 3 of 3 slides.", ""]
    finally:
        _stop(controller)


def test_undoing_estimate_roll_gives_the_open_frame_its_own_fade_and_lock_back(tmp_path):
    from negpy.services.assets import rolls

    controller, session, repo, roll_id = _fade_roll(tmp_path, locked="h1")
    try:
        controller._apply_fade_roll(dict(_ROLL_ESTIMATE), "")
        assert session.state.config.process.fade_ratio_g == 0.8

        controller.undo_last_batch()

        assert rolls.frame_override_cards(repo, roll_id, "h1") == {"fade"}
        assert session.state.config.process.fade_ratio_g == 0.7
        assert session.config_for_asset(session.state.uploaded_files[0]).process.fade_ratio_g == 0.7
    finally:
        _stop(controller)


def test_undoing_share_crossover_gives_the_open_frame_its_lock_back(tmp_path):
    from negpy.services.assets import rolls

    controller, session, repo, roll_id = _fade_roll(tmp_path, locked="h1")
    try:
        rolls.set_roll_defaults(repo, roll_id, fade_ratio_g=0.7)
        table = tuple(0.01 * i for i in range(16))
        controller.share_crossover(table, "space")
        assert tuple(rolls.roll_defaults(repo, roll_id)["fade_crossover_table"]) == table
        assert rolls.frame_override_cards(repo, roll_id, "h1") == set()

        controller.undo_last_batch()

        assert rolls.roll_defaults(repo, roll_id).get("fade_crossover_table") is None
        assert rolls.frame_override_cards(repo, roll_id, "h1") == {"fade"}
    finally:
        _stop(controller)
