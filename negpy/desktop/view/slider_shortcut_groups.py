"""Slider shortcut groups: paired inc/dec actions with keyboard step defaults."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SliderShortcutGroup:
    id: str
    label: str
    inc_action: str
    dec_action: str
    default_step: float
    category: str
    step_decimals: int = 2
    step_suffix: str = ""
    inc_sign: float = 1.0
    dec_sign: float = -1.0


def _g(
    id: str,
    label: str,
    inc_action: str,
    dec_action: str,
    default_step: float,
    category: str,
    *,
    step_decimals: int = 2,
    step_suffix: str = "",
    inc_sign: float = 1.0,
    dec_sign: float = -1.0,
) -> SliderShortcutGroup:
    return SliderShortcutGroup(
        id=id,
        label=label,
        inc_action=inc_action,
        dec_action=dec_action,
        default_step=default_step,
        category=category,
        step_decimals=step_decimals,
        step_suffix=step_suffix,
        inc_sign=inc_sign,
        dec_sign=dec_sign,
    )


SLIDER_GROUPS: tuple[SliderShortcutGroup, ...] = (
    _g("cyan", "Cyan ↑/↓", "cyan_inc", "cyan_dec", 0.01, "Exposure"),
    _g("magenta", "Magenta ↑/↓", "magenta_up", "magenta_down", 0.01, "Exposure"),
    _g("yellow", "Yellow ↑/↓", "yellow_up", "yellow_down", 0.01, "Exposure"),
    _g(
        "temperature",
        "Temperature ↑/↓",
        "temp_warm",
        "temp_cool",
        50.0,
        "Exposure",
        step_decimals=0,
        step_suffix=" K",
        inc_sign=-1.0,
        dec_sign=1.0,
    ),
    _g("density", "Density ↑/↓", "density_up", "density_down", 0.01, "Exposure"),
    _g(
        "grade",
        "Grade ↑/↓",
        "grade_up",
        "grade_down",
        10.0,
        "Exposure",
        step_decimals=0,
        step_suffix=" ISO-R",
        inc_sign=-1.0,
        dec_sign=1.0,
    ),
    _g("toe", "Toe ↑/↓", "toe_inc", "toe_dec", 0.01, "Exposure"),
    _g("toe_width", "Toe width ↑/↓", "toe_width_inc", "toe_width_dec", 0.01, "Exposure"),
    _g("shoulder", "Shoulder ↑/↓", "shoulder_inc", "shoulder_dec", 0.01, "Exposure"),
    _g("shoulder_width", "Shoulder width ↑/↓", "shoulder_width_inc", "shoulder_width_dec", 0.01, "Exposure"),
    _g("snap", "Snap ↑/↓", "snap_inc", "snap_dec", 0.01, "Exposure"),
    _g("shadow_density", "Shadows density ↑/↓", "shadow_density_inc", "shadow_density_dec", 0.01, "Exposure"),
    _g("highlight_density", "Highlights density ↑/↓", "highlight_density_inc", "highlight_density_dec", 0.01, "Exposure"),
    _g("shadow_grade", "Shadows grade ↑/↓", "shadow_grade_inc", "shadow_grade_dec", 1.0, "Exposure", step_decimals=0),
    _g("highlight_grade", "Highlights grade ↑/↓", "highlight_grade_inc", "highlight_grade_dec", 1.0, "Exposure", step_decimals=0),
    _g("offset", "Crop offset ↑/↓", "offset_inc", "offset_dec", 1.0, "Geometry", step_decimals=0, step_suffix=" px"),
    _g("fine_rot", "Fine rotation ↑/↓", "fine_rot_inc", "fine_rot_dec", 0.01, "Geometry", step_suffix="°"),
    _g("analysis_buffer", "Analysis buffer ↑/↓", "analysis_buffer_inc", "analysis_buffer_dec", 0.01, "Process"),
    _g("luma_range_clip", "Luma range clip ↑/↓", "luma_range_clip_inc", "luma_range_clip_dec", 1.0, "Process", step_decimals=0),
    _g("color_range_clip", "Colour range clip ↑/↓", "color_range_clip_inc", "color_range_clip_dec", 1.0, "Process", step_decimals=0),
    _g("white_point", "White point ↑/↓", "white_point_inc", "white_point_dec", 0.01, "Process"),
    _g("black_point", "Black point ↑/↓", "black_point_inc", "black_point_dec", 0.01, "Process"),
    _g("separation", "Crosstalk ↑/↓", "separation_inc", "separation_dec", 0.01, "Process"),
    _g("chroma_denoise", "Denoise ↑/↓", "chroma_denoise_inc", "chroma_denoise_dec", 0.01, "Lab"),
    _g("saturation", "Saturation ↑/↓", "saturation_inc", "saturation_dec", 0.01, "Lab"),
    _g("chroma_damping", "Dye Mute ↑/↓", "chroma_damping_inc", "chroma_damping_dec", 0.01, "Lab"),
    _g("vibrance", "Vibrance ↑/↓", "vibrance_inc", "vibrance_dec", 0.01, "Lab"),
    _g("clahe", "CLAHE ↑/↓", "clahe_inc", "clahe_dec", 0.01, "Lab"),
    _g("sharpen", "Sharpening ↑/↓", "sharpen_inc", "sharpen_dec", 0.01, "Lab"),
    _g("glow", "Glow ↑/↓", "glow_inc", "glow_dec", 0.01, "Lab"),
    _g("halation", "Halation ↑/↓", "halation_inc", "halation_dec", 0.01, "Lab"),
    _g("threshold", "Threshold ↑/↓", "threshold_inc", "threshold_dec", 0.01, "Retouch"),
    _g("auto_size", "Auto size ↑/↓", "auto_size_inc", "auto_size_dec", 1.0, "Retouch", step_decimals=0, step_suffix=" px"),
    _g("manual_size", "Brush size ↑/↓", "manual_size_inc", "manual_size_dec", 1.0, "Retouch", step_decimals=0, step_suffix=" px"),
    _g("selenium", "Selenium ↑/↓", "selenium_inc", "selenium_dec", 0.01, "Toning"),
    _g("sepia", "Sepia ↑/↓", "sepia_inc", "sepia_dec", 0.01, "Toning"),
    _g("shadow_hue", "Shadow hue ↑/↓", "shadow_hue_inc", "shadow_hue_dec", 0.01, "Toning"),
    _g("shadow_strength", "Shadow strength ↑/↓", "shadow_strength_inc", "shadow_strength_dec", 0.01, "Toning"),
    _g("highlight_hue", "Highlight hue ↑/↓", "highlight_hue_inc", "highlight_hue_dec", 0.01, "Toning"),
    _g("highlight_strength", "Highlight strength ↑/↓", "highlight_strength_inc", "highlight_strength_dec", 0.01, "Toning"),
    _g("vignette_str", "Vignette burn ↑/↓", "vignette_str_inc", "vignette_str_dec", 0.01, "Finishing"),
    _g("vignette_size", "Vignette size ↑/↓", "vignette_size_inc", "vignette_size_dec", 0.01, "Finishing"),
    _g("border_size", "Border width ↑/↓", "border_size_inc", "border_size_dec", 0.01, "Finishing"),
)

SLIDER_GROUP_BY_ID: dict[str, SliderShortcutGroup] = {group.id: group for group in SLIDER_GROUPS}

SLIDER_GROUP_BY_ACTION: dict[str, SliderShortcutGroup] = {}
for _group in SLIDER_GROUPS:
    SLIDER_GROUP_BY_ACTION[_group.inc_action] = _group
    SLIDER_GROUP_BY_ACTION[_group.dec_action] = _group


def sign_for_action(action_id: str) -> float:
    group = SLIDER_GROUP_BY_ACTION[action_id]
    if action_id == group.inc_action:
        return group.inc_sign
    return group.dec_sign
