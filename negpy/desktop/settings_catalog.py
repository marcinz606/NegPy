"""Declarative catalog of copyable settings for the granular copy/paste/apply UI.

Each SettingRow maps a human label to one or more fields of a WorkspaceConfig
sub-config. A row is "edited" when any of its fields differs from the default
config; the picker offers unedited rows too, so a roll can be reset back to a
default value. Grouped rows (per-channel trims, linked metadata) copy their fields as a
unit so linked values can't drift apart. Excluded fields (per-frame bounds/dust/
heal/masks, machine paths, derived caches) are simply not listed here.
"""

from __future__ import annotations

from negpy.kernel.system.i18n import mark, tr

from dataclasses import fields as dataclass_fields, replace
from typing import Any, Callable, Iterable, Mapping, Optional

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.papers import PAPER_PROFILES
from negpy.features.geometry.models import with_auto_crop_field
from negpy.features.metadata.capture import place_summary
from negpy.features.metadata.models import GEAR_FIELDS, PROCESS_FIELDS, PUSH_PULL_LABELS, SCANNING_FIELDS
from negpy.features.process.logic import VALID_HIGHLIGHT_LEVELS
from negpy.features.process.models import SOURCE_ENCODING_LABELS, SourceEncoding, invalidate_local_bounds, with_film_fields
from negpy.services.assets.flatfield import FlatFieldProfiles
from negpy.services.assets.presets import preset_fields
from negpy.services.assets.rolls import card_fields


class SettingRow:
    """One copyable setting. `fields` are config-field names on `section`;
    `channels` gives per-channel letters for grouped numeric trims (e.g. "RGB").
    `sticky` marks the rows that carry onto a freshly-opened file by default."""

    __slots__ = ("label", "section", "fields", "channels", "fmt", "sticky")

    def __init__(
        self,
        label: str,
        section: str,
        fields: tuple[str, ...],
        channels: str = "",
        fmt: Optional[Callable[[tuple], str]] = None,
        sticky: bool = False,
    ):
        self.label = label
        self.section = section
        self.fields = fields
        self.channels = channels
        self.fmt = fmt
        self.sticky = sticky

    @property
    def id(self) -> str:
        """Stable identity for persistence. Unique because no field is listed twice."""
        return f"{self.section}.{self.fields[0]}"


def _fmt_scalar(v) -> str:
    if isinstance(v, bool):
        return tr("On") if v else tr("Off")
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{round(v, 4):g}"
    if isinstance(v, (tuple, list)):
        return tr("Set")
    return str(v)


def _format(row: SettingRow, values: tuple) -> str:
    if row.fmt is not None:
        return row.fmt(values)
    if len(values) == 1:
        return _fmt_scalar(values[0])
    if row.channels and len(row.channels) == len(values):
        return " ".join(f"{c}{_fmt_scalar(v)}" for c, v in zip(row.channels, values))
    return " / ".join(_fmt_scalar(v) for v in values)


def _row(label, section, *fields, channels="", fmt=None, sticky=False) -> SettingRow:
    return SettingRow(label, section, tuple(fields), channels, fmt, sticky)


def _fmt_source_encoding(values: tuple) -> str:
    return tr(SOURCE_ENCODING_LABELS[SourceEncoding(values[0])])


def _fmt_print_stock(values: tuple) -> str:
    profile = PAPER_PROFILES.get(values[0])
    return profile.label if profile else str(values[0])


def _fmt_process(values: tuple) -> str:
    developer, dilution, push_pull = values[0], values[1], values[2]
    recipe = " ".join(p for p in (developer, dilution) if p)
    push = PUSH_PULL_LABELS.get(push_pull, str(push_pull))
    if not recipe:
        return push if push_pull else "—"
    return recipe if not push_pull else f"{recipe} · {push}"


def _fmt_gear(values: tuple) -> str:
    v = dict(zip(GEAR_FIELDS, values))
    film_format = v["format_other"] if v["format"] == "Other" and v["format_other"] else v["format"]
    parts = [
        " ".join(str(x) for x in (v["camera_make"], v["camera_model"]) if x),
        " ".join(str(x) for x in (v["lens_make"], v["lens_model"]) if x),
        str(v["film"] or ""),
        str(film_format or ""),
    ]
    return " · ".join(p for p in parts if p) or "—"


def _fmt_flatfield(values: tuple) -> str:
    apply, profile_id = values
    if not apply:
        return _fmt_scalar(False)
    profile = FlatFieldProfiles.get(profile_id) if profile_id else None
    return profile.name if profile else _fmt_scalar(True)


def _fmt_white_balance(values: tuple) -> str:
    return tr("Linear RAW") if values[0] else tr("Camera WB")


# Highlight Recovery's choices: rawpy's HighlightMode collapsed to three, Reconstruct pinned
# to libraw's own default level.
HIGHLIGHT_LEVELS = (
    (0, tr("Off")),
    (2, tr("Blend")),
    (5, tr("Reconstruct")),
)


def highlight_bucket(level: int) -> int:
    """The HIGHLIGHT_LEVELS entry a stored level shows under. Any other valid level (3-9) is a
    Reconstruct level. An invalid one is Off, as effective_highlight_reconstruction decodes it."""
    if level not in VALID_HIGHLIGHT_LEVELS or level == 0:
        return 0
    return 1 if level == 2 else 2


def _fmt_highlights(values: tuple) -> str:
    return HIGHLIGHT_LEVELS[highlight_bucket(values[0])][1]


# fmt: off
CATALOG: list[tuple[str, tuple[SettingRow, ...]]] = [
    (mark("Setup"), (
        _row(mark("Film Mode"), "process", "process_mode", sticky=True),
        _row(mark("Positive"), "process", "positive_source", sticky=True),
        _row(mark("Encoding"), "process", "source_encoding", sticky=True, fmt=_fmt_source_encoding),
    )),
    (mark("Metering"), (
        _row(mark("Analysis Buffer"), "process", "analysis_buffer", sticky=True),
        _row(mark("Luma Range Clip"), "process", "luma_range_clip", sticky=True),
        _row(mark("Color Clip"), "process", "color_range_clip", sticky=True),
        _row(mark("White Point"), "process", "white_point_offset"),
        _row(mark("White Point Trim"), "process", "white_point_trim_red", "white_point_trim_green", "white_point_trim_blue", channels="RGB"),
        _row(mark("Black Point"), "process", "black_point_offset"),
        _row(mark("Black Point Trim"), "process", "black_point_trim_red", "black_point_trim_green", "black_point_trim_blue", channels="RGB"),
    )),
    (mark("Roll Analysis"), (
        _row(mark("Use Average Luma"), "process", "use_luma_average"),
        _row(mark("Use Average Color"), "process", "use_color_average"),
        # The axis travels with the toggle: the toggle alone reads a baseline the target may not have.
        _row(mark("Use Average Cast"), "process", "use_cast_average", "locked_neutral_axis", fmt=lambda v: _fmt_scalar(v[0])),
    )),
    (mark("Calibration"), (
        # Strength, profile, the baked matrix and the process it was baked for copy atomically:
        # strength alone would leave the target on a stale or None matrix.
        _row(mark("Crosstalk"), "process", "crosstalk_strength", "crosstalk_profile", "crosstalk_matrix", "crosstalk_process", fmt=lambda v: _fmt_scalar(v[0]), sticky=True),
        _row(mark("Sensor Crosstalk Profile"), "process", "sensor_profile", "sensor_matrix", "sensor_unmix", fmt=lambda v: _fmt_scalar(v[0]), sticky=True),
        # Absent from BOUNDS_INPUT_FIELDS: it acts after inversion, so it never feeds the meters.
        _row(mark("Hue Trim"), "process", "hue_trim", sticky=True),
    )),
    (mark("Raw Decode"), (
        _row(mark("White Balance"), "process", "linear_raw", fmt=_fmt_white_balance),
        _row(mark("Demosaic Preview"), "process", "demosaic_preview"),
        _row(mark("Demosaic Export"), "process", "demosaic_export"),
        _row(mark("Highlight Recovery"), "process", "highlight_reconstruction", fmt=_fmt_highlights),
    )),
    (mark("Crop"), (
        _row(mark("Auto Crop"), "geometry", "crop_from_auto", sticky=True),
        _row(mark("Crop Offset"), "geometry", "autocrop_offset", sticky=True),
        _row(mark("Rebate Trim"), "geometry", "autocrop_rebate_trim", sticky=True),
        _row(mark("Crop Ratio"), "geometry", "autocrop_ratio", sticky=True),
        _row(mark("Crop Mode"), "geometry", "autocrop_mode", sticky=True),
        _row(mark("Square First"), "geometry", "autocrop_square", sticky=True),
        # Rect and key copy together: without the key a copied auto rect looks freshly
        # detected on the target.
        _row(mark("Crop"), "geometry", "crop_rect", "crop_detect_key", fmt=lambda v: _fmt_scalar(v[0])),
    )),
    (mark("Straighten"), (
        _row(mark("Rotation"), "geometry", "rotation"),
        _row(mark("Fine Rotation"), "geometry", "fine_rotation"),
        _row(mark("Tilt"), "geometry", "converge_v"),
        _row(mark("Swing"), "geometry", "converge_h"),
        _row(mark("Flip Horizontal"), "geometry", "flip_horizontal", sticky=True),
        _row(mark("Flip Vertical"), "geometry", "flip_vertical", sticky=True),
    )),
    (mark("Optics"), (
        _row(mark("Distortion Correction"), "geometry", "distortion_k1", "lens_distortion_from_metadata", sticky=True),
        _row(mark("Embedded CA"), "geometry", "lens_ca_from_metadata", sticky=True),
        _row(mark("Flat Field"), "flatfield", "apply", "profile_id", fmt=_fmt_flatfield),
    )),
    (mark("Tone"), (
        _row(mark("Print Density"), "exposure", "density"),
        _row(mark("ISO-R Grade"), "exposure", "grade"),
        _row(mark("Grade Trim"), "exposure", "grade_trim_red", "grade_trim_green", "grade_trim_blue", channels="RGB"),
        _row(mark("Shadows Density"), "exposure", "shadow_density"),
        _row(mark("Hold Black"), "exposure", "hold_black"),
        _row(mark("Highlights Density"), "exposure", "highlight_density"),
        _row(mark("Shadows Grade"), "exposure", "shadow_grade"),
        _row(mark("Highlights Grade"), "exposure", "highlight_grade"),
        _row(mark("Shadows Grade Trim"), "exposure", "shadow_grade_trim_red", "shadow_grade_trim_green", "shadow_grade_trim_blue", channels="RGB"),
        _row(mark("Highlights Grade Trim"), "exposure", "highlight_grade_trim_red", "highlight_grade_trim_green", "highlight_grade_trim_blue", channels="RGB"),
        _row(mark("Contrast Mask"), "exposure", "contrast_mask_strength"),
        _row(mark("Diffusion"), "exposure", "diffusion"),
        _row(mark("Preflash"), "exposure", "preflash"),
        _row(mark("Auto Density"), "exposure", "auto_exposure", sticky=True),
        _row(mark("Auto Grade"), "exposure", "auto_normalize_contrast", sticky=True),
    )),
    (mark("Paper Response"), (
        _row(mark("Print Stock"), "exposure", "paper_profile", fmt=_fmt_print_stock, sticky=True),
        _row(mark("Paper Black"), "exposure", "paper_black", sticky=True),
        _row(mark("Paper White"), "exposure", "paper_dmin", sticky=True),
        _row(mark("Snap"), "exposure", "midtone_gamma"),
        _row(mark("Snap Trim"), "exposure", "midtone_gamma_trim_red", "midtone_gamma_trim_green", "midtone_gamma_trim_blue", channels="RGB"),
        _row(mark("Toe"), "exposure", "toe"),
        _row(mark("Toe Width"), "exposure", "toe_width"),
        _row(mark("Toe Trim"), "exposure", "toe_trim_red", "toe_trim_green", "toe_trim_blue", channels="RGB"),
        _row(mark("Toe Width Trim"), "exposure", "toe_width_trim_red", "toe_width_trim_green", "toe_width_trim_blue", channels="RGB"),
        _row(mark("Shoulder"), "exposure", "shoulder"),
        _row(mark("Shoulder Width"), "exposure", "shoulder_width"),
        _row(mark("Shoulder Trim"), "exposure", "shoulder_trim_red", "shoulder_trim_green", "shoulder_trim_blue", channels="RGB"),
        _row(mark("Shoulder Width Trim"), "exposure", "shoulder_width_trim_red", "shoulder_width_trim_green", "shoulder_width_trim_blue", channels="RGB"),
        _row(mark("Dye Separation"), "exposure", "dye_separation"),
        _row(mark("Dye Separation Trim"), "exposure", "dye_separation_trim_red", "dye_separation_trim_green", "dye_separation_trim_blue", channels="RGB"),
        _row(mark("Separation Damping"), "exposure", "separation_damping"),
    )),
    (mark("Filtration"), (
        _row(mark("Cyan"), "exposure", "wb_cyan"),
        _row(mark("Magenta"), "exposure", "wb_magenta"),
        _row(mark("Yellow"), "exposure", "wb_yellow"),
        _row(mark("Shadows CMY"), "exposure", "shadow_cyan", "shadow_magenta", "shadow_yellow", channels="CMY"),
        _row(mark("Highlights CMY"), "exposure", "highlight_cyan", "highlight_magenta", "highlight_yellow", channels="CMY"),
        _row(mark("Cast Removal"), "exposure", "cast_removal_strength", sticky=True),
        _row(mark("Filter"), "exposure", "bw_filter"),
    )),
    (mark("Lab"), (
        _row(mark("Chroma"), "lab", "saturation", sticky=True),
        _row(mark("Skin Protection"), "lab", "skin_protection", sticky=True),
        _row(mark("CLAHE"), "lab", "clahe_strength", sticky=True),
        _row(mark("Sharpening"), "lab", "sharpen", sticky=True),
        _row(mark("Sharpen Method"), "lab", "sharpen_method", sticky=True),
        _row(mark("Sharpen Radius"), "lab", "sharpen_radius", sticky=True),
        _row(mark("Sharpen Masking"), "lab", "sharpen_masking", sticky=True),
        _row(mark("Auto Masking"), "lab", "sharpen_auto_masking", sticky=True),
        _row(mark("Chroma Denoise"), "lab", "chroma_denoise", sticky=True),
        _row(mark("Glow"), "lab", "glow_amount", sticky=True),
        _row(mark("Halation"), "lab", "halation_strength", sticky=True),
    )),
    (mark("Alternative Processes"), (
        _row(mark("Process"), "altproc", "alt_process"),
        _row(mark("Lith Exposure"), "altproc", "lith_exposure"),
        _row(mark("Snatch Point"), "altproc", "lith_snatch"),
        _row(mark("Abruptness"), "altproc", "lith_abruptness"),
        _row(mark("Sensitizer"), "altproc", "cyano_sensitizer"),
        _row(mark("Cyanotype Exposure"), "altproc", "cyano_exposure"),
        _row(mark("Exposure Scale"), "altproc", "cyano_scale"),
        _row(mark("Bleach"), "altproc", "cyano_bleach"),
        _row(mark("Tannin"), "altproc", "cyano_tannin"),
        _row(mark("Sabattier Strength"), "altproc", "sabattier_strength"),
        _row(mark("Re-exposure"), "altproc", "sabattier_reexposure"),
        _row(mark("Agitation"), "altproc", "sabattier_agitation"),
    )),
    (mark("Toning"), (
        _row(mark("Selenium"), "toning", "selenium_strength"),
        _row(mark("Sepia"), "toning", "sepia_strength"),
        _row(mark("Gold"), "toning", "gold_strength"),
        _row(mark("Iron Blue"), "toning", "blue_strength"),
        _row(mark("Copper"), "toning", "copper_strength"),
        _row(mark("Vanadium"), "toning", "vanadium_strength"),
        _row(mark("Shadow Hue"), "toning", "shadow_tint_hue"),
        _row(mark("Shadow Strength"), "toning", "shadow_tint_strength"),
        _row(mark("Highlight Hue"), "toning", "highlight_tint_hue"),
        _row(mark("Highlight Strength"), "toning", "highlight_tint_strength"),
    )),
    (mark("Finishing"), (
        _row(mark("Vignette Burn"), "finish", "vignette_stops"),
        _row(mark("Vignette Size"), "finish", "vignette_size"),
        _row(mark("Vignette Roundness"), "finish", "vignette_roundness"),
        _row(mark("Carrier Width"), "finish", "carrier_width"),
        _row(mark("Carrier Roughness"), "finish", "carrier_rough"),
        _row(mark("Carrier Flare"), "finish", "carrier_flare"),
        _row(mark("Carrier Corners"), "finish", "carrier_corner"),
        _row(mark("Border Width"), "finish", "border_size"),
        _row(mark("Border Color"), "finish", "border_color"),
        _row(mark("Border Bottom Weight"), "finish", "border_bottom_weight"),
        _row(mark("Border Paper White"), "finish", "border_match_paper"),
    )),
    (mark("Light Leak"), (
        _row(mark("Reduce Light Leak"), "leak", "leak_remove"),
        _row(mark("Film Dmax"), "leak", "leak_dmax"),
        _row(mark("Recover Layers"), "leak", "leak_recover"),
        _row(mark("Leak Grain"), "leak", "leak_grain"),
        _row(mark("Leak Contrast"), "leak", "leak_contrast"),
    )),
    (mark("Retouch"), (
        _row(mark("Optical Removal"), "retouch", "dust_remove", sticky=True),
        _row(mark("Optical Spot Threshold"), "retouch", "dust_threshold"),
        _row(mark("Optical Hair Threshold"), "retouch", "dust_hair_threshold"),
        _row(mark("Optical Size"), "retouch", "dust_size"),
        _row(mark("IR Removal"), "retouch", "ir_dust_remove"),
        _row(mark("IR Threshold"), "retouch", "ir_threshold"),
        _row(mark("IR Attenuation"), "retouch", "ir_attenuation"),
    )),
    (mark("Metadata"), (
        _row(mark("Analog Gear"), "metadata", *GEAR_FIELDS, fmt=_fmt_gear),
        _row(mark("Capture Date"), "metadata", "capture_date"),
        _row(
            mark("Place"),
            "metadata",
            "location_city",
            "location_state",
            "location_country",
            "gps_latitude",
            "gps_longitude",
            fmt=lambda v: place_summary(v[0], v[1], v[2], v[3], v[4]) or "—",
        ),
        _row(mark("Process"), "metadata", *PROCESS_FIELDS, fmt=_fmt_process),
        _row(mark("Scanning"), "metadata", *SCANNING_FIELDS, fmt=lambda v: _fmt_scalar(v[0])),
        # capture_frame is deliberately absent: a frame number is unique to one frame.
        _row(mark("Roll"), "metadata", "capture_roll"),
        _row(mark("Exposure"), "metadata", "exposure_override"),
        _row(mark("Protect Original Metadata"), "metadata", "protect_original_metadata", sticky=True),
        _row(
            mark("Description Fields"),
            "metadata",
            "description_fields",
            fmt=lambda v: (", ".join(str(x) for x in v[0]) if v[0] else "—"),
            sticky=True,
        ),
    )),
    (mark("Export"), (
        _row(mark("Format"), "export", "export_fmt", sticky=True),
        _row(mark("Bit Depth"), "export", "export_bit_depth", sticky=True),
        _row(mark("JPEG Quality"), "export", "jpeg_quality", sticky=True),
        _row(mark("JPEG Progressive"), "export", "jpeg_progressive", sticky=True),
        _row(mark("TIFF Compression"), "export", "tiff_compression", sticky=True),
        _row(mark("PNG Compression"), "export", "png_compress_level", sticky=True),
        _row(mark("JXL Lossless"), "export", "jxl_lossless", sticky=True),
        _row(mark("JXL Distance"), "export", "jxl_distance", sticky=True),
        _row(mark("JXL Effort"), "export", "jxl_effort", sticky=True),
        _row(mark("WebP Quality"), "export", "webp_quality", sticky=True),
        _row(mark("WebP Lossless"), "export", "webp_lossless", sticky=True),
        _row(mark("WebP Method"), "export", "webp_method", sticky=True),
        _row(mark("Size Mode"), "export", "export_resolution_mode", sticky=True),
        _row(mark("Paper Ratio"), "export", "paper_aspect_ratio", sticky=True),
        _row(mark("Print Size"), "export", "export_print_size", sticky=True),
        _row(mark("DPI"), "export", "export_dpi", sticky=True),
        _row(mark("Long Edge"), "export", "export_target_long_edge_px", sticky=True),
        _row(mark("Export Profile"), "export", "export_color_space", sticky=True),
        _row(mark("Filename Pattern"), "export", "filename_pattern", sticky=True),
        _row(mark("Overwrite Existing Files"), "export", "overwrite", sticky=True),
        _row(mark("Output Intent"), "export", "output_mode", sticky=True),
        _row(mark("Save Sidecars on Export"), "export", "export_sidecars_enabled", sticky=True),
    )),
]
# fmt: on

_DEFAULT = WorkspaceConfig()

# Fields that decide how a frame is metered. Pasting one must drop the target's cached
# per-frame bounds, or resolve_bounds_detailed keeps short-circuiting on them and the new
# value never reaches the render. Mirrors what every sidebar handler for these already
# does. Excluded on purpose: autocrop_ratio (see AppController.set_crop_ratio), rotation,
# and the white and black points, which apply after the bounds rather than feeding them.
BOUNDS_INPUT_FIELDS = frozenset(
    {
        "process_mode",
        "analysis_buffer",
        "luma_range_clip",
        "color_range_clip",
        "use_luma_average",
        "use_color_average",
        "crosstalk_strength",
        "crosstalk_profile",
        "crosstalk_matrix",
        "sensor_profile",
        "sensor_matrix",
        "sensor_unmix",
        # Each changes the decoded preview the meters read.
        "linear_raw",
        "source_encoding",
        "highlight_reconstruction",
        "demosaic_preview",
        "crop_from_auto",
        "autocrop_offset",
        "autocrop_rebate_trim",
        "autocrop_mode",
        "crop_rect",
        "crop_detect_key",
    }
)


def all_rows() -> list[SettingRow]:
    return [r for _title, rows in CATALOG for r in rows]


def rows_by_id() -> dict[str, SettingRow]:
    return {r.id: r for r in all_rows()}


def rows_for_fields(fields: Iterable[str]) -> list[SettingRow]:
    """Every row touching one of *fields*. Any overlap counts, because a row travels
    whole: Crosstalk has to come along on its strength field alone, or the matrix it
    carries is left behind."""
    wanted = set(fields)
    return [r for r in all_rows() if any(f in wanted for f in r.fields)]


def section_of_field() -> dict[str, str]:
    """field name -> the WorkspaceConfig section it lives on, over every catalog row."""
    return {f: r.section for r in all_rows() for f in r.fields}


def rows_for_section(section: str) -> list[SettingRow]:
    """Every row on one WorkspaceConfig section, for a card that owns the whole of it."""
    return [r for r in all_rows() if r.section == section]


_LISTED = frozenset((r.section, f) for r in all_rows() for f in r.fields)
# Written by a render, not by the user, so never a change of its own.
_MEASURED = frozenset({("process", "local_floors"), ("process", "local_ceils"), ("process", "meter_key"), ("process", "source_color")})

# The card a change lives on when no catalog row lists its field (a crop rectangle, a heal, a mask).
_UNLISTED_CARD = {
    "process": mark("Metering"),
    "exposure": mark("Export"),
    "flatfield": mark("Optics"),
    "rgbscan": mark("Frame Assembly"),
    "stitch": mark("Stitch"),
    "hdr": mark("HDR"),
    "geometry": mark("Crop"),
    "local": mark("Dodge & Burn"),
    "retouch": mark("Retouch"),
    "metadata": mark("Metadata"),
    "export": mark("Export"),
}


def changed_settings(prev: WorkspaceConfig, config: WorkspaceConfig) -> list[tuple[str, str]]:
    """(card, control) for each setting that differs between two configs, in catalog order. A
    change no row lists names its card alone, with an empty control."""
    changed = [
        (title, row.label)
        for title, rows in CATALOG
        for row in rows
        if any(getattr(getattr(prev, row.section), f) != getattr(getattr(config, row.section), f) for f in row.fields)
    ]
    for section in (f.name for f in dataclass_fields(config)):
        before, after = getattr(prev, section), getattr(config, section)
        if before == after:
            continue
        for f in dataclass_fields(after):
            key = (section, f.name)
            if key not in _LISTED and key not in _MEASURED and getattr(before, f.name) != getattr(after, f.name):
                changed.append((_UNLISTED_CARD.get(section, section), ""))
    return changed


# Filtration, Tone and Paper Response split ExposureConfig for modified counts and scoped resets.
# render_intent (flat-master output) is in none.
COLOR_FIELDS = (
    "wb_cyan",
    "wb_magenta",
    "wb_yellow",
    "shadow_cyan",
    "shadow_magenta",
    "shadow_yellow",
    "highlight_cyan",
    "highlight_magenta",
    "highlight_yellow",
    "cast_removal_strength",
    "bw_filter",
)

GEOMETRY_FIELDS = (
    "rotation",
    "fine_rotation",
    "flip_horizontal",
    "flip_vertical",
    "converge_v",
    "converge_h",
    "crop_to_valid",
)

CROP_FIELDS = (
    "crop_rect",
    "crop_from_auto",
    "crop_detect_key",
    "autocrop_mode",
    "autocrop_offset",
    "autocrop_rebate_trim",
    "autocrop_ratio",
    "autocrop_square",
)

TONE_FIELDS = (
    "density",
    "grade",
    "grade_trim_red",
    "grade_trim_green",
    "grade_trim_blue",
    "shadow_density",
    "hold_black",
    "highlight_density",
    "shadow_grade",
    "highlight_grade",
    "shadow_grade_trim_red",
    "shadow_grade_trim_green",
    "shadow_grade_trim_blue",
    "highlight_grade_trim_red",
    "highlight_grade_trim_green",
    "highlight_grade_trim_blue",
    "auto_exposure",
    "auto_normalize_contrast",
    "contrast_mask_strength",
    "diffusion",
    "preflash",
)

PAPER_FIELDS = (
    "paper_profile",
    "paper_black",
    "paper_dmin",
    "midtone_gamma",
    "midtone_gamma_trim_red",
    "midtone_gamma_trim_green",
    "midtone_gamma_trim_blue",
    "toe",
    "toe_width",
    "toe_trim_red",
    "toe_trim_green",
    "toe_trim_blue",
    "toe_width_trim_red",
    "toe_width_trim_green",
    "toe_width_trim_blue",
    "shoulder",
    "shoulder_width",
    "shoulder_trim_red",
    "shoulder_trim_green",
    "shoulder_trim_blue",
    "shoulder_width_trim_red",
    "shoulder_width_trim_green",
    "shoulder_width_trim_blue",
    "dye_separation",
    "dye_separation_trim_red",
    "dye_separation_trim_green",
    "dye_separation_trim_blue",
    "separation_damping",
)

# Frame cards whose settings can be pushed to other frames, and the fields each owns. A
# card keyed by its own config section needs no tuple. Roll-tab cards drive roll defaults
# instead, and Dodge & Burn has no catalog row: a mask means nothing on the next frame.
FRAME_CARD_FIELDS: dict[str, tuple | None] = {
    "autocrop": CROP_FIELDS,
    "geometry": GEOMETRY_FIELDS,
    "color": COLOR_FIELDS,
    "tone": TONE_FIELDS,
    "paper": PAPER_FIELDS,
    "lab": None,
    "altproc": None,
    "toning": None,
    "retouch": None,
    "leak": None,
    "finish": None,
}


def roll_card_rows(*card_keys: str) -> list[SettingRow]:
    """The catalog rows of one or more Roll-tab cards (ROLL_DEFAULT_FIELDS keys)."""
    return rows_for_fields(f for card in card_keys for f in card_fields(card))


def frame_card_rows(card_key: str) -> list[SettingRow]:
    """A frame card's catalog rows: its own field tuple, or its whole config section."""
    fields = FRAME_CARD_FIELDS.get(card_key, ())
    if fields is None:
        return rows_for_section(card_key)
    return rows_for_fields(fields) if fields else []


# Everything but the Metadata rows, for the pickers that offer metadata alone.
NON_METADATA_SECTIONS: frozenset[str] = frozenset(title for title, _rows in CATALOG) - {"Metadata"}

# What carries onto a fresh file out of the box. The user's own choice overrides it.
DEFAULT_STICKY_IDS: frozenset[str] = frozenset(r.id for r in all_rows() if r.sticky)

# Config attrs whose rows also carry onto a file that already has a saved edit. Everything
# else is a per-frame decision and the saved edit wins.
GLOBAL_TIER_SECTIONS: frozenset[str] = frozenset({"export", "metadata"})


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
    # Film Mode and Positive carry their own defaults; a field chosen alongside still wins below.
    out = with_film_fields(out, by_section.get("process", {}))
    # A crop rect arrives with its own Auto flag; Auto Crop alone switches through with_auto_crop.
    geometry = by_section.get("geometry", {})
    if "crop_rect" in geometry:
        geometry["crop_from_auto"] = source.geometry.crop_from_auto
    elif "crop_from_auto" in geometry:
        out = with_auto_crop_field(out, geometry)
    for section, changes in by_section.items():
        out = replace(out, **{section: replace(getattr(out, section), **changes)})
    if any(f in BOUNDS_INPUT_FIELDS for row in rows for f in row.fields):
        out = replace(out, process=replace(out.process, **invalidate_local_bounds(out.process)))
    return out


def selected_flat_dict(cfg: WorkspaceConfig, rows: Iterable[SettingRow]) -> dict[str, Any]:
    """Flat dict of the chosen rows' fields (flat keys are field names). A row's
    fields travel as a unit, default-valued ones included; a crop rect brings its Auto flag."""
    flat = {f: getattr(getattr(cfg, r.section), f) for r in rows for f in r.fields}
    if "crop_rect" in flat:
        flat["crop_from_auto"] = cfg.geometry.crop_from_auto
    return flat


def preset_config(data: Mapping[str, Any]) -> WorkspaceConfig:
    """A preset's stored fields over defaults, so pickers show the preset's own
    values rather than the current image's."""
    base = WorkspaceConfig().to_dict()
    base.update(preset_fields(data))
    return WorkspaceConfig.from_flat_dict(base)


def rows_for_keys(data: Mapping[str, Any], section: str = "") -> list[SettingRow]:
    """The rows a preset stores, optionally narrowed to one config section."""
    fields = preset_fields(data)
    return [r for r in all_rows() if (not section or r.section == section) and any(f in fields for f in r.fields)]


def preset_values(data: Mapping[str, Any], section: str = "") -> list[tuple[str, str]]:
    """Each row a preset stores, as (label, formatted value)."""
    cfg = preset_config(data)
    rows = rows_for_keys(data, section)
    return [(r.label, _format(r, tuple(getattr(getattr(cfg, r.section), f) for f in r.fields))) for r in rows]


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
