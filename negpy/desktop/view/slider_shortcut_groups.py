"""Slider shortcut groups: paired inc/dec actions with keyboard step defaults."""

from __future__ import annotations

from negpy.kernel.system.i18n import tr

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
    _g("cyan", tr("Cyan ↑/↓"), "cyan_inc", "cyan_dec", 0.01, tr("Exposure")),
    _g("magenta", tr("Magenta ↑/↓"), "magenta_up", "magenta_down", 0.01, tr("Exposure")),
    _g("yellow", tr("Yellow ↑/↓"), "yellow_up", "yellow_down", 0.01, tr("Exposure")),
    _g(
        "temperature",
        tr("Temperature ↑/↓"),
        "temp_warm",
        "temp_cool",
        50.0,
        tr("Exposure"),
        step_decimals=0,
        step_suffix=" K",
        inc_sign=-1.0,
        dec_sign=1.0,
    ),
    _g("density", tr("Print Density ↑/↓"), "density_up", "density_down", 0.01, tr("Exposure")),
    _g(
        "grade",
        tr("ISO-R Grade ↑/↓"),
        "grade_up",
        "grade_down",
        10.0,
        tr("Exposure"),
        step_decimals=0,
        step_suffix=" ISO-R",
        inc_sign=-1.0,
        dec_sign=1.0,
    ),
    _g("toe", tr("Toe ↑/↓"), "toe_inc", "toe_dec", 0.01, tr("Exposure")),
    _g("toe_width", tr("Toe Width ↑/↓"), "toe_width_inc", "toe_width_dec", 0.01, tr("Exposure")),
    _g("shoulder", tr("Shoulder ↑/↓"), "shoulder_inc", "shoulder_dec", 0.01, tr("Exposure")),
    _g("shoulder_width", tr("Shoulder Width ↑/↓"), "shoulder_width_inc", "shoulder_width_dec", 0.01, tr("Exposure")),
    _g("snap", tr("Snap ↑/↓"), "snap_inc", "snap_dec", 0.01, tr("Exposure")),
    _g("shadow_density", tr("Shadows Density ↑/↓"), "shadow_density_inc", "shadow_density_dec", 0.01, tr("Exposure")),
    _g("highlight_density", tr("Highlights Density ↑/↓"), "highlight_density_inc", "highlight_density_dec", 0.01, tr("Exposure")),
    _g("shadow_grade", tr("Shadows Grade ↑/↓"), "shadow_grade_inc", "shadow_grade_dec", 1.0, tr("Exposure"), step_decimals=0),
    _g("highlight_grade", tr("Highlights Grade ↑/↓"), "highlight_grade_inc", "highlight_grade_dec", 1.0, tr("Exposure"), step_decimals=0),
    _g("dye_separation", tr("Dye Separation ↑/↓"), "dye_separation_inc", "dye_separation_dec", 0.01, tr("Exposure")),
    _g("separation_damping", tr("Separation Damping ↑/↓"), "separation_damping_inc", "separation_damping_dec", 0.01, tr("Exposure")),
    _g("contrast_mask", tr("Contrast Mask ↑/↓"), "contrast_mask_inc", "contrast_mask_dec", 0.01, tr("Exposure")),
    _g("mask_spacer", tr("Mask Spacer ↑/↓"), "mask_spacer_inc", "mask_spacer_dec", 0.1, tr("Exposure"), step_decimals=1, step_suffix="%"),
    _g("diffusion", tr("Diffusion ↑/↓"), "diffusion_inc", "diffusion_dec", 0.02, tr("Exposure")),
    _g("preflash", tr("Preflash ↑/↓"), "preflash_inc", "preflash_dec", 0.01, tr("Exposure")),
    _g("offset", tr("Crop Offset ↑/↓"), "offset_inc", "offset_dec", 1.0, tr("Geometry"), step_decimals=0, step_suffix=" px"),
    _g("fine_rot", tr("Fine Rotation ↑/↓"), "fine_rot_inc", "fine_rot_dec", 0.01, tr("Geometry"), step_suffix="°"),
    _g("converge_v", tr("Tilt ↑/↓"), "converge_v_inc", "converge_v_dec", 0.1, tr("Geometry"), step_suffix="%"),
    _g("converge_h", tr("Swing ↑/↓"), "converge_h_inc", "converge_h_dec", 0.1, tr("Geometry"), step_suffix="%"),
    _g("analysis_buffer", tr("Analysis Buffer ↑/↓"), "analysis_buffer_inc", "analysis_buffer_dec", 0.01, tr("Process")),
    _g("luma_range_clip", tr("Luma Range Clip ↑/↓"), "luma_range_clip_inc", "luma_range_clip_dec", 1.0, tr("Process"), step_decimals=0),
    _g("color_range_clip", tr("Color Clip ↑/↓"), "color_range_clip_inc", "color_range_clip_dec", 1.0, tr("Process"), step_decimals=0),
    _g("white_point", tr("White Point ↑/↓"), "white_point_inc", "white_point_dec", 0.01, tr("Exposure")),
    _g("black_point", tr("Black Point ↑/↓"), "black_point_inc", "black_point_dec", 0.01, tr("Exposure")),
    _g("render_ev", tr("Render Exposure ↑/↓"), "render_ev_inc", "render_ev_dec", 0.05, tr("Process"), step_suffix=" EV"),
    _g("separation", tr("Crosstalk ↑/↓"), "separation_inc", "separation_dec", 0.01, tr("Process")),
    _g("chroma_denoise", tr("Chroma Denoise ↑/↓"), "chroma_denoise_inc", "chroma_denoise_dec", 0.01, tr("Lab")),
    _g("saturation", tr("Chroma ↑/↓"), "saturation_inc", "saturation_dec", 0.01, tr("Lab")),
    _g("clahe", tr("CLAHE ↑/↓"), "clahe_inc", "clahe_dec", 0.01, tr("Lab")),
    _g("sharpen", tr("Sharpening ↑/↓"), "sharpen_inc", "sharpen_dec", 0.01, tr("Lab")),
    _g("glow", tr("Glow ↑/↓"), "glow_inc", "glow_dec", 0.01, tr("Lab")),
    _g("halation", tr("Halation ↑/↓"), "halation_inc", "halation_dec", 0.01, tr("Lab")),
    _g("threshold", tr("Spot Threshold ↑/↓"), "threshold_inc", "threshold_dec", 0.01, tr("Retouch")),
    _g("hair_threshold", tr("Hair Threshold ↑/↓"), "hair_threshold_inc", "hair_threshold_dec", 0.01, tr("Retouch")),
    _g("auto_size", tr("Size ↑/↓"), "auto_size_inc", "auto_size_dec", 1.0, tr("Retouch"), step_decimals=0, step_suffix=" px"),
    _g("manual_size", tr("Brush Size ↑/↓"), "manual_size_inc", "manual_size_dec", 1.0, tr("Retouch"), step_decimals=0, step_suffix=" px"),
    _g("lith_exposure", tr("Lith Exposure ↑/↓"), "lith_exposure_inc", "lith_exposure_dec", 0.1, tr("Alt Process")),
    _g("lith_snatch", tr("Snatch Point ↑/↓"), "lith_snatch_inc", "lith_snatch_dec", 0.01, tr("Alt Process")),
    _g("lith_abruptness", tr("Lith Abruptness ↑/↓"), "lith_abruptness_inc", "lith_abruptness_dec", 0.01, tr("Alt Process")),
    _g("cyano_exposure", tr("Cyanotype Exposure ↑/↓"), "cyano_exposure_inc", "cyano_exposure_dec", 0.1, tr("Alt Process")),
    _g("cyano_scale", tr("Exposure Scale ↑/↓"), "cyano_scale_inc", "cyano_scale_dec", 0.05, tr("Alt Process")),
    _g("cyano_bleach", tr("Bleach ↑/↓"), "cyano_bleach_inc", "cyano_bleach_dec", 0.01, tr("Alt Process")),
    _g("cyano_tannin", tr("Tannin ↑/↓"), "cyano_tannin_inc", "cyano_tannin_dec", 0.01, tr("Alt Process")),
    _g("sabattier_strength", tr("Sabattier Strength ↑/↓"), "sabattier_strength_inc", "sabattier_strength_dec", 0.05, tr("Alt Process")),
    _g("sabattier_reexposure", tr("Re-exposure ↑/↓"), "sabattier_reexposure_inc", "sabattier_reexposure_dec", 0.05, tr("Alt Process")),
    _g("sabattier_agitation", tr("Agitation ↑/↓"), "sabattier_agitation_inc", "sabattier_agitation_dec", 0.05, tr("Alt Process")),
    _g("selenium", tr("Selenium ↑/↓"), "selenium_inc", "selenium_dec", 0.01, tr("Toning")),
    _g("sepia", tr("Sepia ↑/↓"), "sepia_inc", "sepia_dec", 0.01, tr("Toning")),
    _g("shadow_hue", tr("Shadow hue ↑/↓"), "shadow_hue_inc", "shadow_hue_dec", 0.01, tr("Toning")),
    _g("shadow_strength", tr("Shadow strength ↑/↓"), "shadow_strength_inc", "shadow_strength_dec", 0.01, tr("Toning")),
    _g("highlight_hue", tr("Highlight hue ↑/↓"), "highlight_hue_inc", "highlight_hue_dec", 0.01, tr("Toning")),
    _g("highlight_strength", tr("Highlight strength ↑/↓"), "highlight_strength_inc", "highlight_strength_dec", 0.01, tr("Toning")),
    _g("vignette_str", tr("Vignette burn ↑/↓"), "vignette_str_inc", "vignette_str_dec", 0.01, tr("Finishing")),
    _g("vignette_size", tr("Vignette size ↑/↓"), "vignette_size_inc", "vignette_size_dec", 0.01, tr("Finishing")),
    _g("border_size", tr("Border width ↑/↓"), "border_size_inc", "border_size_dec", 0.01, tr("Finishing")),
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
