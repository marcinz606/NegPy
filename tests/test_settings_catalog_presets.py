"""Preset save/apply building blocks: selected_flat_dict row atomicity,
the flat-dict merge apply path, and preset_summary tooltips."""

from dataclasses import replace

from negpy.desktop.settings_catalog import all_rows, preset_summary, selected_flat_dict
from negpy.domain.models import WorkspaceConfig


def _row(label: str):
    return next(r for r in all_rows() if r.label == label)


def _merge(cfg: WorkspaceConfig, preset: dict) -> WorkspaceConfig:
    d = cfg.to_dict()
    d.update(preset)
    return WorkspaceConfig.from_flat_dict(d)


def test_channel_grouped_row_stores_all_channels():
    base = WorkspaceConfig()
    cfg = replace(base, process=replace(base.process, white_point_trim_red=0.1))
    data = selected_flat_dict(cfg, [_row("White Trim")])
    assert set(data) == {"white_point_trim_red", "white_point_trim_green", "white_point_trim_blue"}
    assert data["white_point_trim_red"] == 0.1
    assert data["white_point_trim_green"] == base.process.white_point_trim_green


def test_crosstalk_row_bundles_fields():
    data = selected_flat_dict(WorkspaceConfig(), [_row("Crosstalk")])
    assert set(data) == {"crosstalk_strength", "crosstalk_profile", "crosstalk_matrix"}


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


def test_preset_summary_lists_non_default_settings():
    s = preset_summary({"density": 1.5, "wb_cyan": 0.2, "bogus": 1})
    assert s == "Tone: Print Density\nColour: Cyan"


def test_preset_summary_skips_defaults_and_empty():
    assert preset_summary({"density": WorkspaceConfig().exposure.density}) == ""
    assert preset_summary({}) == ""
