"""The Frame/Roll pair on a section header: which catalog rows each frame card owns, and
the picker limited to them."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from negpy.desktop.session import AppState
from negpy.desktop.settings_catalog import CATALOG, all_rows, rows_for_fields, rows_for_section
from negpy.desktop.view.sidebar.controls_panel import _APPLY_FIELDS, ControlsPanel
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.desktop.view.widgets.granular_settings_dialog import GranularSettingsDialog
from negpy.domain.models import WorkspaceConfig


def _rows_for(key: str):
    fields = _APPLY_FIELDS[key]
    return rows_for_fields(fields) if fields else rows_for_section(key)


def _edited_cfg() -> WorkspaceConfig:
    c = WorkspaceConfig()
    return replace(
        c,
        exposure=replace(c.exposure, density=1.5, wb_cyan=0.2),
        geometry=replace(c.geometry, crop_rect=(0.1, 0.1, 0.9, 0.9)),
    )


@pytest.mark.parametrize("key", sorted(_APPLY_FIELDS))
def test_every_frame_card_resolves_to_rows(key: str):
    assert _rows_for(key), f"{key} would open an empty picker"


def test_no_two_cards_claim_the_same_row():
    seen: dict[str, str] = {}
    for key in _APPLY_FIELDS:
        for row in _rows_for(key):
            assert row.id not in seen, f"{row.id} claimed by both {seen.get(row.id)} and {key}"
            seen[row.id] = key


def test_tone_carries_its_print_rows_but_not_the_tonal_range():
    """Tone is the catalog's own "Tone" plus Contrast Mask, which no card's field tuple
    names. White/Black Point belong to Normalization."""
    labels = {r.label for r in _rows_for("tone")}
    assert {"Print Density", "Contrast Mask"} <= labels
    assert {"White Point", "Black Trim"} & labels == set()


def test_geometry_excludes_the_rows_that_moved_to_the_roll_tab():
    labels = {r.label for r in _rows_for("geometry")}
    assert "Fine Rotation" in labels
    assert {"Crop Offset", "Rebate Trim", "Crop Ratio", "Lens Correction", "Embedded CA"} & labels == set()


def test_a_partially_named_row_still_travels_whole():
    """Crosstalk's row carries its baked matrix, which no card's field tuple names: a
    row matched on all its fields instead of any would leave the matrix behind."""
    row = next(r for r in rows_for_fields(("crosstalk_strength",)) if r.label == "Crosstalk")
    assert "crosstalk_matrix" in row.fields


def test_limit_to_rows_leaves_only_the_named_rows(qapp):
    dlg = GranularSettingsDialog(None, _edited_cfg(), "IMG_0001.cr2", show_scope=True, sel_count=1, roll_count=3)
    wanted = [r.id for r in _rows_for("tone")]

    dlg.limit_to_rows(wanted)

    assert {row.id for _box, row, _e, _line in dlg._checks} == set(wanted)
    dlg._on_apply()
    assert {r.label for r in dlg.selected()} == {"Print Density"}


def test_limit_to_rows_hides_a_section_it_emptied(qapp):
    """A section left with nothing edited stops showing, through the same path that
    hides an unedited section on a full picker."""
    dlg = GranularSettingsDialog(None, _edited_cfg(), "IMG_0001.cr2", show_scope=True, sel_count=1, roll_count=3)

    dlg.limit_to_rows([r.id for r in _rows_for("tone")])

    shown = {s.title_label.text() for s, count in dlg._sections if count}
    assert shown == {"Tone · 1"}
    assert all(not s.isVisible() for s, count in dlg._sections if not count)


def test_limit_to_rows_keeps_a_dropped_row_from_enabling_apply(qapp):
    dlg = GranularSettingsDialog(None, _edited_cfg(), "IMG_0001.cr2", show_scope=True, sel_count=1, roll_count=3)

    dlg.limit_to_rows([r.id for r in _rows_for("finish")])

    assert dlg.apply_btn.isEnabled() is False
    assert dlg.selected() == []


def test_every_catalog_row_a_frame_card_claims_exists():
    known = {r.id for r in all_rows()}
    for key in _APPLY_FIELDS:
        assert {r.id for r in _rows_for(key)} <= known


def test_an_unlimited_dialog_still_lists_every_section(qapp):
    dlg = GranularSettingsDialog(None, _edited_cfg(), "IMG_0001.cr2", show_scope=True, sel_count=1, roll_count=3)
    assert len(dlg._checks) == len(all_rows())


def test_every_catalog_section_is_named_after_its_panel_card(qapp):
    controller = MagicMock()
    controller.state = AppState()
    panel = ControlsPanel(controller)
    cards = {w._title_text for w in vars(panel).values() if isinstance(w, CollapsibleSection)}
    # Metadata and Export are cards on the right panel, outside ControlsPanel.
    assert {title for title, _rows in CATALOG} - {"Metadata", "Export"} <= cards
