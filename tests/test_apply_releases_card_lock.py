"""An apply that writes a Roll tab card releases a frame that now matches the roll on it, changed or not."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from negpy.desktop.session import DesktopSessionManager
from negpy.desktop.settings_catalog import roll_card_rows, rows_for_fields
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG
from negpy.services.assets import rolls

CARD, SECTION, FIELD = "sensor", "process", "crosstalk_strength"
ROLL_VALUE, OWN_VALUE = 0.5, 0.8


def _with(config, value):
    return replace(config, **{SECTION: replace(getattr(config, SECTION), **{FIELD: value})})


def _roll(tmp_path, h2_value):
    """h1 is open at the roll's value. h2 is locked on CARD and holds *h2_value*."""
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    session = DesktopSessionManager(repo)
    state = session.state
    state.uploaded_files = [{"name": f"{h}.tif", "path": f"/r/{h}.tif", "hash": h} for h in ("h1", "h2", "h3")]
    roll_id = rolls.create_virtual_roll(repo, "Roll", [f["path"] for f in state.uploaded_files])
    state.active_roll_id = roll_id
    section, names = rolls.ROLL_DEFAULT_FIELDS[CARD]
    rolls.set_roll_defaults(repo, roll_id, **{n: getattr(getattr(_with(DEFAULT_WORKSPACE_CONFIG, ROLL_VALUE), section), n) for n in names})
    for h, value in (("h2", h2_value), ("h3", ROLL_VALUE)):
        repo.save_file_settings(h, _with(DEFAULT_WORKSPACE_CONFIG, value), file_path=f"/r/{h}.tif")
    rolls.set_frame_override(repo, roll_id, "h2", CARD, True)
    state.selected_file_idx = 0
    state.selected_indices = [0, 1, 2]
    state.current_file_hash = "h1"
    state.current_file_path = "/r/h1.tif"
    state.config = _with(DEFAULT_WORKSPACE_CONFIG, ROLL_VALUE)
    session.asset_model = MagicMock()
    session.asset_model.visible_actual_indices_ordered.return_value = [0, 1, 2]
    return session, repo, roll_id


def _locked(repo, roll_id, file_hash="h2"):
    return CARD in rolls.frame_override_cards(repo, roll_id, file_hash)


@pytest.mark.parametrize("scope", ["selection", "roll"])
def test_sync_releases_a_frame_pinned_at_the_rolls_value(tmp_path, scope):
    session, repo, roll_id = _roll(tmp_path, ROLL_VALUE)

    session.sync_selected_settings(roll_card_rows(CARD), scope=scope)

    assert not _locked(repo, roll_id)


def test_sync_releases_a_frame_whose_value_the_roll_took_later(tmp_path):
    session, repo, roll_id = _roll(tmp_path, OWN_VALUE)
    rolls.set_roll_defaults(repo, roll_id, **{FIELD: OWN_VALUE})
    session.state.config = _with(session.state.config, OWN_VALUE)

    session.sync_selected_settings(roll_card_rows(CARD), scope="roll")

    assert not _locked(repo, roll_id)


def test_apply_settings_releases_a_frame_pinned_at_the_rolls_value(tmp_path):
    session, repo, roll_id = _roll(tmp_path, ROLL_VALUE)

    session.apply_preset_fields(session.state.config, roll_card_rows(CARD), scope="roll")

    assert not _locked(repo, roll_id)


def test_a_paste_onto_the_selection_releases_a_frame_pinned_at_the_rolls_value(tmp_path):
    session, repo, roll_id = _roll(tmp_path, ROLL_VALUE)
    session.state.clipboard = session.state.config

    session.apply_pasted_fields(roll_card_rows(CARD), include_bounds=False)

    assert not _locked(repo, roll_id)


def test_a_paste_onto_the_open_frame_alone_keeps_its_pin(tmp_path):
    """The card's Roll button frees it there, as one undo step."""
    session, repo, roll_id = _roll(tmp_path, ROLL_VALUE)
    rolls.set_frame_override(repo, roll_id, "h1", CARD, True)
    session.state.selected_indices = [0]
    session.state.clipboard = session.state.config

    session.apply_pasted_fields(roll_card_rows(CARD), include_bounds=False)

    assert _locked(repo, roll_id, "h1")


@pytest.mark.parametrize("scope", ["current", "selection", "roll"])
def test_a_release_on_the_open_frame_alone_is_one_undo(tmp_path, scope):
    session, repo, roll_id = _roll(tmp_path, ROLL_VALUE)
    rolls.set_frame_override(repo, roll_id, "h1", CARD, True)
    session.state.selected_indices = [0]
    session.asset_model.visible_actual_indices_ordered.return_value = [0]

    session.apply_preset_fields(session.state.config, roll_card_rows(CARD), scope=scope)
    assert not _locked(repo, roll_id, "h1")

    assert session.undo_last_batch() == (1, 0)
    assert _locked(repo, roll_id, "h1")


def test_a_card_paste_through_the_picker_releases_the_frame(qapp, tmp_path, monkeypatch):
    """Copy Card Settings on a Roll tab card, then Paste with the picker as it opens."""
    from PyQt6.QtWidgets import QDialog

    from negpy.desktop.view.widgets import granular_settings_dialog as picker

    session, repo, roll_id = _roll(tmp_path, ROLL_VALUE)
    session.copy_card_settings(roll_card_rows(CARD), whole_card=True)
    monkeypatch.setattr(picker.GranularSettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)

    picker.open_paste_dialog(None, MagicMock(session=session))

    assert not _locked(repo, roll_id)
    session.undo_last_batch()
    assert _locked(repo, roll_id)


def test_a_card_copy_without_whole_card_starts_with_the_edited_rows_only(qapp, tmp_path):
    from negpy.desktop.view.widgets.granular_settings_dialog import GranularSettingsDialog

    session, _repo, _roll_id = _roll(tmp_path, ROLL_VALUE)
    rows = roll_card_rows(CARD)
    session.copy_card_settings(rows)
    dlg = GranularSettingsDialog(None, session.state.clipboard, "clipboard")
    dlg.limit_to_rows([r.id for r in rows])

    assert not session.state.clipboard_whole_card
    assert len(dlg.selected()) < len(rows)


def test_an_apply_to_the_roll_releases_the_open_frame_and_undo_gives_it_back(tmp_path):
    session, repo, roll_id = _roll(tmp_path, ROLL_VALUE)
    rolls.set_frame_override(repo, roll_id, "h1", CARD, True)

    session.apply_preset_fields(session.state.config, roll_card_rows(CARD), scope="roll")
    assert not _locked(repo, roll_id, "h1")
    session.undo_last_batch()

    assert _locked(repo, roll_id, "h1")
    assert _locked(repo, roll_id)


def test_an_apply_of_part_of_a_card_keeps_the_pin(tmp_path):
    session, repo, roll_id = _roll(tmp_path, ROLL_VALUE)
    part = rows_for_fields(("hue_trim",))
    assert part and {f for r in part for f in r.fields} < set(rolls.card_fields(CARD))

    session.sync_selected_settings(part, scope="roll")

    assert _locked(repo, roll_id)


def test_an_apply_of_another_card_keeps_the_pin(tmp_path):
    session, repo, roll_id = _roll(tmp_path, ROLL_VALUE)

    session.sync_selected_settings(rows_for_fields(("density",)), scope="roll")

    assert _locked(repo, roll_id)


def test_an_apply_that_leaves_the_frame_off_the_roll_keeps_the_lock(tmp_path):
    session, repo, roll_id = _roll(tmp_path, ROLL_VALUE)
    session.state.config = _with(session.state.config, OWN_VALUE)

    session.sync_selected_settings(roll_card_rows(CARD), scope="roll")

    assert _locked(repo, roll_id)
    assert _locked(repo, roll_id, "h3")


def test_undo_of_the_apply_gives_the_pin_back(tmp_path):
    session, repo, roll_id = _roll(tmp_path, ROLL_VALUE)
    session.sync_selected_settings(roll_card_rows(CARD), scope="roll")

    session.undo_last_batch()

    assert _locked(repo, roll_id)
