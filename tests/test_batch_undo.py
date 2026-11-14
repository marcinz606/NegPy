"""One Undo takes back a change to several frames, from any frame."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from negpy.desktop.session import DesktopSessionManager
from negpy.desktop.settings_catalog import rows_for_fields
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG
from negpy.services.assets.rolls import same_value


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

    assert session.undo_last_batch() == (2, 0)
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


def _write_offscreen(session, repo, file_hash, change):
    """A write to a closed frame outside any batch, as a rotation, Measure Again or a roll average refresh makes."""
    saved = repo.load_file_settings(file_hash)
    updated = change(saved)
    session.push_external_history(file_hash, saved, updated)
    repo.save_file_settings(file_hash, updated, file_path=f"/r/{file_hash}.tif")


def test_undo_keeps_a_later_rotation_and_puts_back_the_applied_fields(tmp_path):
    session, repo = _session(tmp_path)
    session.sync_selected_settings(rows_for_fields(("density",)), scope="selection")
    session.state.selected_indices = [0, 1]
    session.rotate_selected_frames(1)
    rotation = repo.load_file_settings("h2").geometry.rotation
    assert rotation != DEFAULT_WORKSPACE_CONFIG.geometry.rotation

    assert session.undo_last_batch() == (2, 0)
    h2 = repo.load_file_settings("h2")
    assert h2.geometry.rotation == rotation
    assert h2.exposure.density == DEFAULT_WORKSPACE_CONFIG.exposure.density
    assert repo.load_file_settings("h3").exposure.density == DEFAULT_WORKSPACE_CONFIG.exposure.density


def test_a_roll_average_refresh_after_the_batch_keeps_the_undo_whole(tmp_path):
    """The refresh writes new bounds with no action from the user, so no frame counts as kept."""
    session, repo = _session(tmp_path)
    session.sync_selected_settings(rows_for_fields(("density",)), scope="selection")
    floors = (0.1, 0.1, 0.1)
    _write_offscreen(session, repo, "h2", lambda c: replace(c, process=replace(c.process, locked_floors=floors)))

    assert session.last_batch is not None
    assert session.undo_last_batch() == (2, 0)
    h2 = repo.load_file_settings("h2")
    assert h2.exposure.density == DEFAULT_WORKSPACE_CONFIG.exposure.density
    assert h2.process.locked_floors == floors


def test_values_read_back_from_the_database_are_no_later_change(tmp_path):
    """The open frame's heal spots are tuples in memory and lists once read back from the database."""
    session, repo = _session(tmp_path)
    config = session.state.config
    session.update_config(
        replace(
            config,
            process=replace(config.process, fade_crossover_table=(0.1, 0.2, 0.3, 0.4)),
            geometry=replace(config.geometry, crop_rect=(0.1, 0.1, 0.9, 0.9)),
            retouch=replace(config.retouch, manual_dust_spots=[(0.5, 0.5, 0.01)]),
        ),
        persist=True,
    )
    preset = replace(DEFAULT_WORKSPACE_CONFIG, exposure=replace(DEFAULT_WORKSPACE_CONFIG.exposure, grade=90.0))
    session.apply_preset_fields(preset, rows_for_fields(("grade",)), scope="selection")
    session.state.config = repo.load_file_settings("h1")

    assert session.undo_last_batch() == (3, 0)
    h1 = session.state.config
    assert h1.exposure.grade == DEFAULT_WORKSPACE_CONFIG.exposure.grade
    assert h1.process.fade_crossover_table == (0.1, 0.2, 0.3, 0.4)
    assert h1.geometry.crop_rect == (0.1, 0.1, 0.9, 0.9)
    assert same_value(h1.retouch.manual_dust_spots, [(0.5, 0.5, 0.01)])


def test_a_later_change_to_an_applied_field_keeps_its_value(tmp_path):
    session, repo = _session(tmp_path)
    session.sync_selected_settings(rows_for_fields(("density",)), scope="selection")
    _write_offscreen(session, repo, "h2", lambda c: replace(c, exposure=replace(c.exposure, density=2.2)))

    assert session.undo_last_batch() == (1, 1)
    assert repo.load_file_settings("h2").exposure.density == 2.2
    assert repo.load_file_settings("h3").exposure.density == DEFAULT_WORKSPACE_CONFIG.exposure.density


def test_a_card_with_a_kept_later_change_settles_its_lock(tmp_path):
    """h2 follows the roll; the batch locks its card, a later change to the card's field stays, so the lock stays."""
    from negpy.services.assets import rolls

    session, repo = _session(tmp_path)
    roll_id = rolls.create_virtual_roll(repo, "Roll", [f["path"] for f in session.state.uploaded_files])
    session.state.active_roll_id = roll_id
    rolls.set_roll_defaults(repo, roll_id, crosstalk_strength=0.5)
    process = session.state.config.process
    session.state.config = replace(session.state.config, process=replace(process, crosstalk_strength=0.8))
    session.state.selected_indices = [0, 1]
    session.sync_selected_settings(rows_for_fields(("crosstalk_strength",)), scope="selection")
    assert "sensor" in rolls.frame_override_cards(repo, roll_id, "h2")
    _write_offscreen(session, repo, "h2", lambda c: replace(c, process=replace(c.process, crosstalk_strength=0.9)))

    assert session.undo_last_batch() == (0, 1)

    assert "sensor" in rolls.frame_override_cards(repo, roll_id, "h2")
    assert session.config_for_asset(session.state.uploaded_files[1]).process.crosstalk_strength == 0.9


def test_undo_takes_back_a_roll_average_refresh_of_the_bounds_it_applied(tmp_path):
    """Apply to Frames gives h2 the roll's baseline in place of its own picked one; the refresh then
    moves it with the roll. Undo gives h2 its own baseline back, and no frame counts as kept."""
    from types import SimpleNamespace

    from negpy.desktop.controller import AppController
    from negpy.services.assets import rolls

    session, repo = _session(tmp_path)
    roll_id = rolls.create_virtual_roll(repo, "Roll", [f["path"] for f in session.state.uploaded_files])
    session.state.active_roll_id = roll_id
    riding = {"use_luma_average": True, "use_color_average": True}
    process = session.state.config.process
    session.state.config = replace(
        session.state.config,
        process=replace(process, **riding, locked_floors=(0.2,) * 3, locked_ceils=(0.8,) * 3, baseline_source=f"roll:{roll_id}"),
    )
    own = replace(
        DEFAULT_WORKSPACE_CONFIG.process, **riding, locked_floors=(0.1,) * 3, locked_ceils=(0.9,) * 3, baseline_source="frame:h9.tif"
    )
    repo.save_file_settings("h2", replace(DEFAULT_WORKSPACE_CONFIG, process=own), file_path="/r/h2.tif")
    session.state.selected_indices = [0, 1]
    session.sync_selected_settings([], bounds_flags=(True, True), scope="selection")
    session.select_file(2)
    controller = SimpleNamespace(session=session, state=session.state, _normalization_targets=lambda _s: [], request_render=lambda: None)
    assert AppController._refresh_riding_frames(controller, None, (0.3,) * 3, (0.7,) * 3, None) == 1

    assert session.undo_last_batch() == (1, 0)
    h2 = repo.load_file_settings("h2").process
    assert (h2.locked_floors, h2.locked_ceils, h2.baseline_source) == (own.locked_floors, own.locked_ceils, own.baseline_source)


def test_opening_a_frame_after_the_batch_is_no_later_change(tmp_path):
    """The open frame shows overlays that its saved edit does not hold; they are no later change."""
    from negpy.services.assets import rolls

    session, repo, roll_id = _locked_frame(tmp_path, "metadata_process", "developer", "HC-110", "Rodinal")
    session.reset_roll_settings(scope="selection")
    session.select_file(1)

    assert session.undo_last_batch() == (2, 0)
    assert "metadata_process" in rolls.frame_override_cards(repo, roll_id, "h2")
    assert session.state.config.metadata.developer == "Rodinal"


def test_a_frame_the_batch_did_not_write_keeps_a_later_change(tmp_path):
    """The source of Apply to Frames is rotated from another frame after the batch; Undo leaves it alone."""
    session, repo = _session(tmp_path)
    repo.save_file_settings("h1", session.state.config, file_path="/r/h1.tif")
    session.sync_selected_settings(rows_for_fields(("density",)), scope="selection")
    session.select_file(1, selection_override=[1, 0])
    session.rotate_selected_frames(1)
    rotation = repo.load_file_settings("h1").geometry.rotation

    assert session.undo_last_batch() == (2, 0)
    assert repo.load_file_settings("h1").geometry.rotation == rotation
    assert repo.load_file_settings("h2").exposure.density == DEFAULT_WORKSPACE_CONFIG.exposure.density
