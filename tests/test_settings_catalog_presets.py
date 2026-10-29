"""Preset save/apply building blocks: selected_flat_dict row atomicity,
the flat-dict merge apply path, and preset_summary tooltips."""

from dataclasses import replace

import pytest

from negpy.desktop.settings_catalog import all_rows, apply_selected_fields, preset_summary, roll_card_rows, selected_flat_dict
from negpy.domain.models import WorkspaceConfig
from negpy.features.process.models import DemosaicMode, ProcessMode
from negpy.services.assets.rolls import card_fields


def _row(label: str):
    return next(r for r in all_rows() if r.label == label)


def _merge(cfg: WorkspaceConfig, preset: dict) -> WorkspaceConfig:
    d = cfg.to_dict()
    d.update(preset)
    return WorkspaceConfig.from_flat_dict(d)


def test_channel_grouped_row_stores_all_channels():
    base = WorkspaceConfig()
    cfg = replace(base, process=replace(base.process, white_point_trim_red=0.1))
    data = selected_flat_dict(cfg, [_row("White Point Trim")])
    assert set(data) == {"white_point_trim_red", "white_point_trim_green", "white_point_trim_blue"}
    assert data["white_point_trim_red"] == 0.1
    assert data["white_point_trim_green"] == base.process.white_point_trim_green


def test_crosstalk_row_bundles_fields():
    data = selected_flat_dict(WorkspaceConfig(), [_row("Crosstalk")])
    assert set(data) == {"crosstalk_strength", "crosstalk_profile", "crosstalk_matrix", "crosstalk_process"}


def test_positive_source_row_is_catalogued():
    """Regression: a source-decode toggle with no catalog row can't be copied, cloned
    onto a selection, or saved into a preset — Positive needs one to batch onto
    a roll of already-positive frames rather than being set one frame at a time."""
    base = WorkspaceConfig()
    cfg = replace(base, process=replace(base.process, process_mode=ProcessMode.E6, positive_source=True))
    data = selected_flat_dict(cfg, [_row("Positive")])
    assert data == {"positive_source": True}


@pytest.mark.parametrize("card", ["film", "sensor", "demosaic", "process", "baseline", "lens", "flatfield"])
def test_every_roll_card_field_has_a_row(card):
    """Copy Card Settings, Paste and the Roll tab's Apply offer only catalog rows."""
    listed = {f for r in roll_card_rows(card) for f in r.fields}
    assert set(card_fields(card)) <= listed


def test_pasting_highlight_recovery_remeters():
    """It rebuilds a slide's brightest pixels, which the white point is measured from."""
    base = WorkspaceConfig()
    src = replace(base, process=replace(base.process, process_mode=ProcessMode.E6, highlight_reconstruction=5))
    target = replace(
        src, process=replace(base.process, process_mode=ProcessMode.E6, local_floors=(0.1, 0.1, 0.1), local_ceils=(0.9, 0.9, 0.9))
    )
    out = apply_selected_fields(src, target, [_row("Highlight Recovery")])
    assert out.process.highlight_reconstruction == 5
    assert not out.process.is_local_initialized


def test_pasting_demosaic_preview_remeters():
    """The preview decode is what the meters read."""
    base = WorkspaceConfig()
    src = replace(base, process=replace(base.process, demosaic_preview=DemosaicMode.DCB))
    target = replace(base, process=replace(base.process, local_floors=(0.1, 0.1, 0.1), local_ceils=(0.9, 0.9, 0.9)))
    out = apply_selected_fields(src, target, [_row("Demosaic Preview")])
    assert out.process.demosaic_preview == DemosaicMode.DCB
    assert not out.process.is_local_initialized


@pytest.mark.parametrize(("level", "label"), [(0, "Off"), (2, "Blend"), (5, "Reconstruct"), (7, "Reconstruct"), (1, "Off"), (42, "Off")])
def test_highlight_recovery_shows_the_card_choice(level, label):
    from negpy.desktop.settings_catalog import _format

    assert _format(_row("Highlight Recovery"), (level,)) == label


def test_pasting_raw_decode_carries_its_values_and_remeters_on_white_balance():
    base = WorkspaceConfig()
    src = replace(
        base,
        process=replace(base.process, linear_raw=True, demosaic_export=DemosaicMode.DCB, highlight_reconstruction=2),
    )
    target = replace(base, process=replace(base.process, local_floors=(0.1, 0.1, 0.1), local_ceils=(0.9, 0.9, 0.9)))
    out = apply_selected_fields(src, target, roll_card_rows("demosaic"))
    assert (out.process.linear_raw, out.process.demosaic_export, out.process.highlight_reconstruction) == (True, DemosaicMode.DCB, 2)
    assert not out.process.is_local_initialized


def test_overlay_apply_preserves_unrelated_edits():
    base = WorkspaceConfig()
    cfg = replace(base, lab=replace(base.lab, saturation=1.4))
    merged = _merge(cfg, {"density": 2.2})
    assert merged.lab.saturation == 1.4
    assert merged.exposure.density == 2.2


def test_legacy_preset_key_migrates():
    merged = _merge(WorkspaceConfig(), {"true_black": False})
    assert merged.exposure.paper_black is True


def test_full_snapshot_preset_applies():
    base = WorkspaceConfig()
    cfg = replace(base, lab=replace(base.lab, saturation=1.4))
    merged = _merge(cfg, WorkspaceConfig().to_dict())
    assert merged == WorkspaceConfig()


def test_preset_summary_lists_stored_settings():
    s = preset_summary({"density": 1.5, "wb_cyan": 0.2, "bogus": 1})
    assert s == "Tone: Print Density\nFiltration: Cyan"


def test_preset_summary_lists_a_deliberately_default_value():
    # A preset may store a default on purpose (#656), so presence decides, not value.
    assert preset_summary({"density": WorkspaceConfig().exposure.density}) == "Tone: Print Density"


def test_preset_summary_empty_for_no_data():
    assert preset_summary({}) == ""
