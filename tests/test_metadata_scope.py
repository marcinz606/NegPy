"""The Metadata tab's cards are roll cards: gear, capture, development and scanning are
facts about the roll, so they follow it until a frame is given something of its own."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from negpy.desktop.controller import AppController
from negpy.desktop.view.sidebar.metadata import MetadataSidebar
from negpy.features.metadata.models import MetadataConfig
from negpy.services.assets.rolls import ROLL_DEFAULT_FIELDS

_CARDS = AppController.METADATA_CARDS


def _stub(*, active_roll_id="roll1", locked_cards=(), metadata=None) -> MagicMock:
    panel = MagicMock()
    panel.state.active_roll_id = active_roll_id
    panel.state.config.metadata = metadata or MetadataConfig()
    panel.controller.locked_roll_cards.side_effect = lambda: set(locked_cards)
    panel._scope_sections = lambda: MetadataSidebar._scope_sections(panel)
    return panel


def _scope(section) -> str:
    return section.set_scope_buttons.call_args[0][1]


@pytest.mark.parametrize("card_key", _CARDS)
def test_every_metadata_card_has_roll_default_fields(card_key: str):
    section, fields = ROLL_DEFAULT_FIELDS[card_key]
    assert section == "metadata"
    assert fields


def test_no_two_metadata_cards_claim_the_same_field():
    seen: dict[str, str] = {}
    for key in _CARDS:
        for field in ROLL_DEFAULT_FIELDS[key][1]:
            assert field not in seen, f"{field} claimed by both {seen.get(field)} and {key}"
            seen[field] = key


def test_the_frame_number_is_never_roll_wide():
    """A frame number is unique to one frame, so no card may push it."""
    every = {f for key in _CARDS for f in ROLL_DEFAULT_FIELDS[key][1]}
    assert "capture_frame" not in every
    assert "protect_original_metadata" not in every
    assert "description_fields" not in every


def test_cards_read_roll_until_one_is_locked():
    panel = _stub(locked_cards={"metadata_process"})

    MetadataSidebar._sync_scope_buttons(panel)

    assert _scope(panel.process_section) == "frame"
    assert _scope(panel.gear_section) == "roll"
    assert _scope(panel.scanning_section) == "roll"


def test_the_hint_names_every_overridden_card():
    panel = _stub(locked_cards={"metadata_gear", "metadata_scanning"})

    MetadataSidebar._sync_scope_buttons(panel)

    panel.metadata_scope_hint.setText.assert_called_with("This frame overrides: Analog Gear, Scanning")


def test_the_hint_is_blank_with_nothing_overridden():
    panel = _stub()

    MetadataSidebar._sync_scope_buttons(panel)

    panel.metadata_scope_hint.setText.assert_called_with("")


def test_the_pair_reads_frame_with_roll_disabled_when_no_roll_spans_the_frames():
    """Metadata is roll-wide by default, but frames that are not one roll have no shared
    camera or stock to read: each is the frame's own. The pair stays visible so that is
    stated rather than left unsaid, with only the unusable half turned off."""
    panel = _stub(active_roll_id=None)

    MetadataSidebar._sync_scope_buttons(panel)

    call = panel.gear_section.set_scope_buttons.call_args
    assert call[0][0] is True
    assert call[0][1] == "frame"
    assert call.kwargs["roll_enabled"] is False


def test_a_cards_count_covers_only_its_own_fields():
    panel = _stub(metadata=replace(MetadataConfig(), developer="HC-110"))

    MetadataSidebar._sync_scope_buttons(panel)

    assert panel.process_section.set_modified.call_args[0][0] == 1
    assert panel.gear_section.set_modified.call_args[0][0] == 0


def test_a_click_routes_through_the_controller():
    panel = _stub()

    MetadataSidebar._on_scope_selected(panel, "metadata_gear", "roll")

    panel.controller.set_card_scope.assert_called_once_with("metadata_gear", "roll")
