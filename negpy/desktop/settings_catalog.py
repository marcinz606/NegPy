"""Declarative catalog of copyable settings for the granular copy/paste/apply UI.

Each SettingRow maps a human label to one or more fields of a WorkspaceConfig
sub-config. A row is "edited" when any of its fields differs from the default
config; the picker offers unedited rows too, so a roll can be reset back to a
default value. Grouped rows (per-channel trims, linked metadata) copy their fields as a
unit so linked values can't drift apart. Excluded fields (per-frame bounds/dust/
heal/masks, machine paths, derived caches) are simply not listed here.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Iterable, Mapping, Optional

from negpy.domain.models import WorkspaceConfig
from negpy.features.metadata.models import PUSH_PULL_LABELS
from negpy.features.process.models import invalidate_local_bounds


class SettingRow:
    """One copyable setting. `fields` are config-field names on `section`;
    `channels` gives per-channel letters for grouped numeric trims (e.g. "RGB")."""

    __slots__ = ("label", "section", "fields", "channels", "fmt")

    def __init__(
        self,
        label: str,
        section: str,
        fields: tuple[str, ...],
        channels: str = "",
        fmt: Optional[Callable[[tuple], str]] = None,
    ):
        self.label = label
        self.section = section
        self.fields = fields
        self.channels = channels
        self.fmt = fmt


def _fmt_scalar(v) -> str:
    if isinstance(v, bool):
        return "on" if v else "off"
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (tuple, list)):
        return "set"
    return str(v)


def _format(row: SettingRow, values: tuple) -> str:
    if row.fmt is not None:
        return row.fmt(values)
    if len(values) == 1:
        return _fmt_scalar(values[0])
    if row.channels and len(row.channels) == len(values):
        return " ".join(f"{c}{_fmt_scalar(v)}" for c, v in zip(row.channels, values))
    return " / ".join(_fmt_scalar(v) for v in values)


def _row(label, section, *fields, channels="", fmt=None) -> SettingRow:
    return SettingRow(label, section, tuple(fields), channels, fmt)


# fmt: off
CATALOG: list[tuple[str, tuple[SettingRow, ...]]] = [
    ("Process", (
        _row("Mode", "process", "process_mode"),
        _row("Analysis Buffer", "process", "analysis_buffer"),
        _row("Range", "process", "luma_range_clip"),
        _row("Colour", "process", "color_range_clip"),
        _row("White Point", "process", "white_point_offset"),
        _row("White Trim", "process", "white_point_trim_red", "white_point_trim_green", "white_point_trim_blue", channels="RGB"),
        _row("Black Point", "process", "black_point_offset"),
        _row("Black Trim", "process", "black_point_trim_red", "black_point_trim_green", "black_point_trim_blue", channels="RGB"),
        # Strength + profile + baked matrix copy atomically: strength alone would
        # leave the target on a stale/None matrix.
        _row("Crosstalk", "process", "crosstalk_strength", "crosstalk_profile", "crosstalk_matrix", fmt=lambda v: _fmt_scalar(v[0])),
        _row("Trichrome Calibration", "process", "sensor_profile", "sensor_matrix", fmt=lambda v: _fmt_scalar(v[0])),
        # Absent from _BOUNDS_INPUT_FIELDS: it acts after inversion, so it never feeds the meters.
        _row("Hue Trim", "process", "hue_trim"),
    )),
    ("Crop", (
        _row("Auto Crop", "geometry", "auto_crop_enabled"),
        _row("Crop Offset", "geometry", "autocrop_offset"),
        _row("Rebate Trim", "geometry", "autocrop_rebate_trim"),
        _row("Crop Ratio", "geometry", "autocrop_ratio"),
        _row("Crop Mode", "geometry", "autocrop_mode"),
        _row("Manual Crop", "geometry", "manual_crop_rect"),
    )),
    ("Rotation", (
        _row("Rotation", "geometry", "rotation"),
        _row("Fine Rotation", "geometry", "fine_rotation"),
        _row("Flip Horizontal", "geometry", "flip_horizontal"),
        _row("Flip Vertical", "geometry", "flip_vertical"),
    )),
    ("Tone", (
        _row("Print Density", "exposure", "density"),
        _row("ISO-R Grade", "exposure", "grade"),
        _row("Grade Trim", "exposure", "grade_trim_red", "grade_trim_green", "grade_trim_blue", channels="RGB"),
        _row("Paper Black", "exposure", "paper_black"),
        _row("Paper Dmin", "exposure", "paper_dmin"),
        _row("Shadows Density", "exposure", "shadow_density"),
        _row("Highlights Density", "exposure", "highlight_density"),
        _row("Shadows Grade", "exposure", "shadow_grade"),
        _row("Highlights Grade", "exposure", "highlight_grade"),
        _row("Shadows Grade Trim", "exposure", "shadow_grade_trim_red", "shadow_grade_trim_green", "shadow_grade_trim_blue", channels="RGB"),
        _row("Highlights Grade Trim", "exposure", "highlight_grade_trim_red", "highlight_grade_trim_green", "highlight_grade_trim_blue", channels="RGB"),
        _row("Snap", "exposure", "midtone_gamma"),
        _row("Snap Trim", "exposure", "midtone_gamma_trim_red", "midtone_gamma_trim_green", "midtone_gamma_trim_blue", channels="RGB"),
        _row("Toe", "exposure", "toe"),
        _row("Toe Width", "exposure", "toe_width"),
        _row("Toe Trim", "exposure", "toe_trim_red", "toe_trim_green", "toe_trim_blue", channels="RGB"),
        _row("Toe Width Trim", "exposure", "toe_width_trim_red", "toe_width_trim_green", "toe_width_trim_blue", channels="RGB"),
        _row("Shoulder", "exposure", "shoulder"),
        _row("Shoulder Width", "exposure", "shoulder_width"),
        _row("Shoulder Trim", "exposure", "shoulder_trim_red", "shoulder_trim_green", "shoulder_trim_blue", channels="RGB"),
        _row("Shoulder Width Trim", "exposure", "shoulder_width_trim_red", "shoulder_width_trim_green", "shoulder_width_trim_blue", channels="RGB"),
        _row("Dye Separation", "exposure", "dye_separation"),
        _row("Dye Separation Trim", "exposure", "dye_separation_trim_red", "dye_separation_trim_green", "dye_separation_trim_blue", channels="RGB"),
        _row("Separation Damping", "exposure", "separation_damping"),
        _row("Auto Exposure", "exposure", "auto_exposure"),
        _row("Auto Contrast", "exposure", "auto_normalize_contrast"),
        _row("Paper Profile", "exposure", "paper_profile"),
    )),
    ("Colour", (
        _row("Cyan", "exposure", "wb_cyan"),
        _row("Magenta", "exposure", "wb_magenta"),
        _row("Yellow", "exposure", "wb_yellow"),
        _row("Shadow CMY", "exposure", "shadow_cyan", "shadow_magenta", "shadow_yellow", channels="CMY"),
        _row("Highlight CMY", "exposure", "highlight_cyan", "highlight_magenta", "highlight_yellow", channels="CMY"),
        _row("Cast Removal", "exposure", "cast_removal_strength"),
    )),
    ("Lab", (
        _row("Chroma", "lab", "saturation"),
        _row("Skin Protection", "lab", "skin_protection"),
        _row("CLAHE", "lab", "clahe_strength"),
        _row("Sharpening", "lab", "sharpen"),
        _row("Sharpen Method", "lab", "sharpen_method"),
        _row("Radius", "lab", "sharpen_radius"),
        _row("Masking", "lab", "sharpen_masking"),
        _row("Chroma Denoise", "lab", "chroma_denoise"),
        _row("Glow", "lab", "glow_amount"),
        _row("Halation", "lab", "halation_strength"),
    )),
    ("Toning", (
        _row("Selenium", "toning", "selenium_strength"),
        _row("Sepia", "toning", "sepia_strength"),
        _row("Gold", "toning", "gold_strength"),
        _row("Iron Blue", "toning", "blue_strength"),
        _row("Copper", "toning", "copper_strength"),
        _row("Vanadium", "toning", "vanadium_strength"),
        _row("Shadow Hue", "toning", "shadow_tint_hue"),
        _row("Shadow Strength", "toning", "shadow_tint_strength"),
        _row("Highlight Hue", "toning", "highlight_tint_hue"),
        _row("Highlight Strength", "toning", "highlight_tint_strength"),
    )),
    ("Finish", (
        _row("Vignette Burn", "finish", "vignette_stops"),
        _row("Vignette Size", "finish", "vignette_size"),
        _row("Vignette Roundness", "finish", "vignette_roundness"),
        _row("Carrier Width", "finish", "carrier_width"),
        _row("Carrier Roughness", "finish", "carrier_rough"),
        _row("Carrier Flare", "finish", "carrier_flare"),
        _row("Carrier Corners", "finish", "carrier_corner"),
        _row("Border Width", "finish", "border_size"),
        _row("Border Colour", "finish", "border_color"),
        _row("Border Bottom Weight", "finish", "border_bottom_weight"),
        _row("Border Match Paper", "finish", "border_match_paper"),
    )),
    ("Retouch", (
        _row("Dust Removal", "retouch", "dust_remove"),
        _row("Dust Threshold", "retouch", "dust_threshold"),
        _row("Dust Size", "retouch", "dust_size"),
        _row("IR Removal", "retouch", "ir_dust_remove"),
        _row("IR Threshold", "retouch", "ir_threshold"),
        _row("IR Method", "retouch", "ir_method"),
        _row("IR Attenuation", "retouch", "ir_attenuation"),
    )),
    ("Metadata", (
        _row("Camera", "metadata", "camera_make", "camera_model", fmt=lambda v: " ".join(str(x) for x in v if x) or "—"),
        _row("Lens", "metadata", "lens_make", "lens_model", fmt=lambda v: " ".join(str(x) for x in v if x) or "—"),
        _row("Focal Length", "metadata", "focal_length_mm"),
        _row("Max Aperture", "metadata", "max_aperture"),
        _row("Film", "metadata", "film"),
        _row("Film ISO", "metadata", "film_iso"),
        _row("Film Manufacturer", "metadata", "film_manufacturer"),
        _row("Film Colour Type", "metadata", "film_color_type"),
        _row("Format", "metadata", "format", "format_other", fmt=lambda v: (v[1] if v[0] == "Other" and v[1] else v[0]) or "—"),
        _row("Developer", "metadata", "developer"),
        _row("Push/Pull", "metadata", "push_pull", fmt=lambda v: PUSH_PULL_LABELS.get(v[0], str(v[0]))),
        _row("Scanning", "metadata", "scanning"),
        _row("Exposure Override", "metadata", "exposure_override"),
        _row("Protect Original Metadata", "metadata", "protect_original_metadata"),
        _row(
            "Description Fields",
            "metadata",
            "description_fields",
            fmt=lambda v: (", ".join(str(x) for x in v[0]) if v[0] else "—"),
        ),
    )),
    ("Export", (
        _row("Format", "export", "export_fmt"),
        _row("JPEG Quality", "export", "jpeg_quality"),
        _row("JXL Lossless", "export", "jxl_lossless"),
        _row("JXL Distance", "export", "jxl_distance"),
        _row("JXL Effort", "export", "jxl_effort"),
        _row("WebP Quality", "export", "webp_quality"),
        _row("WebP Lossless", "export", "webp_lossless"),
        _row("WebP Method", "export", "webp_method"),
        _row("Resolution Mode", "export", "export_resolution_mode"),
        _row("Aspect Ratio", "export", "paper_aspect_ratio"),
        _row("Print Size", "export", "export_print_size"),
        _row("DPI", "export", "export_dpi"),
        _row("Target Long Edge", "export", "export_target_long_edge_px"),
        _row("Colour Space", "export", "export_color_space"),
        _row("Filename Pattern", "export", "filename_pattern"),
        _row("Overwrite", "export", "overwrite"),
        _row("Output Mode", "export", "output_mode"),
        _row("Sidecars", "export", "export_sidecars_enabled"),
    )),
]
# fmt: on

_DEFAULT = WorkspaceConfig()

# Fields that decide how a frame is metered. Pasting one must drop the target's
# cached per-frame bounds, or resolve_bounds_detailed keeps short-circuiting on
# them and the new value never reaches the render. Mirrors what every sidebar
# handler for these already does. Excluded on purpose: autocrop_ratio (see
# AppController.set_crop_ratio), rotation, and the white/black points — those
# apply after the bounds rather than feeding them.
_BOUNDS_INPUT_FIELDS = frozenset(
    {
        "process_mode",
        "analysis_buffer",
        "luma_range_clip",
        "color_range_clip",
        "crosstalk_strength",
        "crosstalk_profile",
        "crosstalk_matrix",
        "sensor_profile",
        "sensor_matrix",
        "auto_crop_enabled",
        "autocrop_offset",
        "autocrop_rebate_trim",
        "autocrop_mode",
        "manual_crop_rect",
    }
)


def all_rows() -> list[SettingRow]:
    return [r for _title, rows in CATALOG for r in rows]


def _row_edited(row: SettingRow, cfg: WorkspaceConfig) -> bool:
    src = getattr(cfg, row.section)
    dfl = getattr(_DEFAULT, row.section)
    return any(getattr(src, f) != getattr(dfl, f) for f in row.fields)


def catalog_sections(cfg: WorkspaceConfig) -> list[tuple[str, list[tuple[SettingRow, str, bool]]]]:
    """Every display section with all its rows, each paired with a formatted value
    and whether it differs from default. Rows at their default are included so they
    can still be applied (resetting a roll back to a default value) — the picker
    hides them behind a toggle."""
    out: list[tuple[str, list[tuple[SettingRow, str, bool]]]] = []
    for title, rows in CATALOG:
        entries = []
        for r in rows:
            values = tuple(getattr(getattr(cfg, r.section), f) for f in r.fields)
            entries.append((r, _format(r, values), _row_edited(r, cfg)))
        out.append((title, entries))
    return out


def apply_selected_fields(source: WorkspaceConfig, target: WorkspaceConfig, rows: Iterable[SettingRow]) -> WorkspaceConfig:
    """Overlay only the chosen rows' fields from source onto target (one replace
    per section). Fields not listed — per-frame bounds, dust spots, heal strokes,
    masks — stay the target's own."""
    rows = list(rows)
    by_section: dict[str, dict] = {}
    for row in rows:
        src_section = getattr(source, row.section)
        changes = by_section.setdefault(row.section, {})
        for f in row.fields:
            changes[f] = getattr(src_section, f)
    out = target
    for section, changes in by_section.items():
        out = replace(out, **{section: replace(getattr(out, section), **changes)})
    if any(f in _BOUNDS_INPUT_FIELDS for row in rows for f in row.fields):
        out = replace(out, process=replace(out.process, **invalidate_local_bounds(out.process)))
    return out


def selected_flat_dict(cfg: WorkspaceConfig, rows: Iterable[SettingRow]) -> dict[str, Any]:
    """Flat dict of the chosen rows' fields (flat keys are field names). A row's
    fields travel as a unit, default-valued ones included."""
    return {f: getattr(getattr(cfg, r.section), f) for r in rows for f in r.fields}


def preset_summary(data: Mapping[str, Any]) -> str:
    """One line per display section listing the settings a preset stores, e.g.
    "Tone: Print Density, Snap". Presence, not non-defaultness: a preset may
    deliberately store a default value. Unknown keys are skipped."""
    lines = []
    for title, rows in CATALOG:
        labels = [r.label for r in rows if any(f in data for f in r.fields)]
        if labels:
            lines.append(f"{title}: {', '.join(labels)}")
    return "\n".join(lines)
