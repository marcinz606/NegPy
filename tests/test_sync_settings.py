"""Granular settings catalog + the per-field merge behind copy/paste and apply-to-many."""

from dataclasses import replace

import pytest

from negpy.desktop.session import _source_effective_bounds
from negpy.desktop.settings_catalog import (
    CATALOG,
    all_rows,
    apply_selected_fields,
    catalog_sections,
)
from negpy.domain.models import WorkspaceConfig

_ROWS = {r.label: r for r in all_rows()}


def _row(label: str):
    return _ROWS[label]


# ── Catalog integrity ────────────────────────────────────────────────────────


def test_every_catalog_field_exists_on_its_section():
    default = WorkspaceConfig()
    for _title, rows in CATALOG:
        for row in rows:
            fields = getattr(default, row.section).__dataclass_fields__
            for f in row.fields:
                assert f in fields, f"{row.label}: {row.section}.{f} is not a real field"


# ── catalog_sections ─────────────────────────────────────────────────────────


def test_catalog_sections_lists_every_row_even_for_a_default_config():
    sections = catalog_sections(WorkspaceConfig())
    assert [t for t, _rows in sections] == [t for t, _rows in CATALOG]
    assert sum(len(rows) for _t, rows in sections) == len(all_rows())
    assert not any(edited for _t, rows in sections for _r, _v, edited in rows)


def test_catalog_sections_flags_only_the_changed_row_as_edited():
    c = WorkspaceConfig()
    cfg = replace(c, exposure=replace(c.exposure, density=1.4))
    edited = [(t, r.label, val) for t, rows in catalog_sections(cfg) for r, val, is_edited in rows if is_edited]
    assert edited == [("Tone", "Print Density", "1.4")]


def test_catalog_sections_offers_a_default_valued_row():  # #656: needed to reset a roll back to 0
    c = WorkspaceConfig()
    cfg = replace(c, geometry=replace(c.geometry, autocrop_offset=0))
    crop = dict((r.label, (val, edited)) for t, rows in catalog_sections(cfg) for r, val, edited in rows if t == "Crop")
    assert crop["Crop Offset"] == ("0", False)


def test_catalog_sections_groups_trim_channels_into_one_row():
    c = WorkspaceConfig()
    exp = replace(c.exposure, grade_trim_red=1.0, grade_trim_blue=-2.0)
    rows = next(rows for title, rows in catalog_sections(replace(c, exposure=exp)) if title == "Tone")
    by_label = {r.label: v for r, v, _edited in rows}
    # one grouped "Grade Trim" row, value shows all three channels
    assert by_label["Grade Trim"] == "R1 G0 B-2"


# ── apply_selected_fields ────────────────────────────────────────────────────


def test_apply_copies_only_selected_rows():
    c = WorkspaceConfig()
    src = replace(c, exposure=replace(c.exposure, density=2.0), lab=replace(c.lab, saturation=1.7))
    out = apply_selected_fields(src, c, [_row("Print Density")])
    assert out.exposure.density == 2.0
    assert out.lab.saturation == c.lab.saturation  # not selected → untouched


def test_apply_grouped_trim_copies_all_channels():
    c = WorkspaceConfig()
    src = replace(c, exposure=replace(c.exposure, grade_trim_red=1.0, grade_trim_green=2.0, grade_trim_blue=3.0))
    out = apply_selected_fields(src, c, [_row("Grade Trim")])
    assert (out.exposure.grade_trim_red, out.exposure.grade_trim_green, out.exposure.grade_trim_blue) == (1.0, 2.0, 3.0)


def test_apply_preserves_target_only_fields():
    c = WorkspaceConfig()
    src = replace(c, exposure=replace(c.exposure, density=2.0))
    tgt = replace(
        c,
        retouch=replace(c.retouch, manual_dust_spots=[(0.5, 0.5, 0.01)]),
        process=replace(c.process, local_floors=(0.05, 0.05, 0.05), local_ceils=(0.95, 0.95, 0.95)),
    )
    out = apply_selected_fields(src, tgt, [_row("Print Density")])
    assert out.exposure.density == 2.0
    # per-frame fields never listed in the catalog → stay the target's own
    assert out.retouch.manual_dust_spots == [(0.5, 0.5, 0.01)]
    assert out.process.local_floors == (0.05, 0.05, 0.05)


def test_apply_crosstalk_copies_strength_profile_and_matrix_together():
    c = WorkspaceConfig()
    src = replace(
        c, process=replace(c.process, crosstalk_strength=0.4, crosstalk_profile="Portra", crosstalk_matrix=(1, 0, 0, 0, 1, 0, 0, 0, 1))
    )
    out = apply_selected_fields(src, c, [_row("Crosstalk")])
    assert out.process.crosstalk_strength == 0.4
    assert out.process.crosstalk_profile == "Portra"
    assert out.process.crosstalk_matrix == (1, 0, 0, 0, 1, 0, 0, 0, 1)


# ── metering inputs clear the target's per-frame bounds ──────────────────────


def _metered_target():
    c = WorkspaceConfig()
    return replace(c, process=replace(c.process, local_floors=(0.1, 0.2, 0.3), local_ceils=(0.9, 0.8, 0.7)))


@pytest.mark.parametrize("label", ["Analysis Buffer", "Mode", "Range", "Colour", "Crosstalk", "Sensor Calibration", "Manual Crop"])
def test_apply_metering_row_clears_local_bounds(label):
    tgt = _metered_target()
    out = apply_selected_fields(WorkspaceConfig(), tgt, [_row(label)])
    assert out.process.local_floors == (0.0, 0.0, 0.0)
    assert out.process.local_ceils == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("label", ["White Point", "Black Trim", "Crop Ratio", "Rotation", "Chroma", "Dye Mute"])
def test_apply_non_metering_row_keeps_local_bounds(label):
    tgt = _metered_target()
    out = apply_selected_fields(WorkspaceConfig(), tgt, [_row(label)])
    assert out.process.local_floors == (0.1, 0.2, 0.3)
    assert out.process.local_ceils == (0.9, 0.8, 0.7)


def test_apply_metering_row_respects_target_bounds_lock():
    tgt = _metered_target()
    tgt = replace(tgt, process=replace(tgt.process, lock_bounds=True))
    out = apply_selected_fields(WorkspaceConfig(), tgt, [_row("Analysis Buffer")])
    assert out.process.local_floors == (0.1, 0.2, 0.3)
    assert out.process.local_ceils == (0.9, 0.8, 0.7)


def test_apply_accepts_a_one_shot_iterable():
    # The bounds check makes a second pass over rows; a generator must not be consumed away.
    c = WorkspaceConfig()
    src = replace(c, process=replace(c.process, analysis_buffer=0.2))
    out = apply_selected_fields(src, _metered_target(), iter([_row("Analysis Buffer")]))
    assert out.process.analysis_buffer == 0.2
    assert out.process.local_floors == (0.0, 0.0, 0.0)


# ── _source_effective_bounds (roll-baseline broadcast) ───────────────────────


def test_source_effective_bounds_prefers_per_frame_meter():
    p = replace(WorkspaceConfig().process, local_floors=(0.1, 0.2, 0.3), local_ceils=(0.9, 0.8, 0.7))
    assert _source_effective_bounds(p) == ((0.1, 0.2, 0.3), (0.9, 0.8, 0.7))


def test_source_effective_bounds_uses_roll_baseline_when_active():
    p = replace(
        WorkspaceConfig().process,
        locked_floors=(0.4, 0.5, 0.6),
        locked_ceils=(0.7, 0.6, 0.5),
        use_luma_average=True,
    )
    assert _source_effective_bounds(p) == ((0.4, 0.5, 0.6), (0.7, 0.6, 0.5))


def test_source_effective_bounds_none_when_unanalysed():
    assert _source_effective_bounds(WorkspaceConfig().process) is None
