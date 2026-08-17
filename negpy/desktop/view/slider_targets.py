"""Slider id -> live widget, the one registry of controls addressable by id.

Keyed by `SliderShortcutGroup.id`; both the keyboard shortcuts (inc/dec actions) and the
Favourites panel resolve their widgets through here."""

from __future__ import annotations

from collections.abc import Callable
from operator import attrgetter

SLIDER_ATTRS: dict[str, str] = {
    "cyan": "color_sidebar.cyan_slider",
    "magenta": "color_sidebar.magenta_slider",
    "yellow": "color_sidebar.yellow_slider",
    "temperature": "color_sidebar.temp_slider",
    "density": "tone_sidebar.density_slider",
    "grade": "tone_sidebar.grade_slider",
    "toe": "tone_sidebar.toe_slider",
    "toe_width": "tone_sidebar.toe_w_slider",
    "shoulder": "tone_sidebar.sh_slider",
    "shoulder_width": "tone_sidebar.sh_w_slider",
    "snap": "tone_sidebar.midtone_gamma_slider",
    "shadow_density": "tone_sidebar.shadow_density_slider",
    "highlight_density": "tone_sidebar.highlight_density_slider",
    "shadow_grade": "tone_sidebar.shadow_grade_slider",
    "highlight_grade": "tone_sidebar.highlight_grade_slider",
    "dye_separation": "tone_sidebar.dye_separation_slider",
    "separation_damping": "tone_sidebar.separation_damping_slider",
    "contrast_mask": "tone_sidebar.contrast_mask_slider",
    "mask_blur": "tone_sidebar.mask_blur_slider",
    "offset": "geometry_sidebar.offset_slider",
    "fine_rot": "geometry_sidebar.fine_rot_slider",
    "converge_v": "geometry_sidebar.converge_v_slider",
    "converge_h": "geometry_sidebar.converge_h_slider",
    "analysis_buffer": "process_sidebar.analysis_buffer_slider",
    "luma_range_clip": "process_sidebar.luma_range_clip_slider",
    "color_range_clip": "process_sidebar.color_range_clip_slider",
    "white_point": "process_sidebar.white_point_slider",
    "black_point": "process_sidebar.black_point_slider",
    "render_ev": "process_sidebar.render_ev_slider",
    "separation": "sensor_sidebar.crosstalk_strength_slider",
    "chroma_denoise": "lab_sidebar.chroma_denoise_slider",
    "saturation": "lab_sidebar.saturation_slider",
    "clahe": "lab_sidebar.clahe_slider",
    "sharpen": "lab_sidebar.sharpen_slider",
    "glow": "lab_sidebar.glow_slider",
    "halation": "lab_sidebar.halation_slider",
    "threshold": "retouch_sidebar.threshold_slider",
    "auto_size": "retouch_sidebar.auto_size_slider",
    "manual_size": "retouch_sidebar.manual_size_slider",
    "lith_exposure": "altproc_sidebar.exposure_slider",
    "lith_snatch": "altproc_sidebar.snatch_slider",
    "lith_abruptness": "altproc_sidebar.abruptness_slider",
    "cyano_exposure": "altproc_sidebar.cyano_exposure_slider",
    "cyano_scale": "altproc_sidebar.cyano_scale_slider",
    "cyano_bleach": "altproc_sidebar.cyano_bleach_slider",
    "cyano_tannin": "altproc_sidebar.cyano_tannin_slider",
    "selenium": "toning_sidebar.selenium_slider",
    "sepia": "toning_sidebar.sepia_slider",
    "shadow_hue": "toning_sidebar.shadow_hue_slider",
    "shadow_strength": "toning_sidebar.shadow_str_slider",
    "highlight_hue": "toning_sidebar.highlight_hue_slider",
    "highlight_strength": "toning_sidebar.highlight_str_slider",
    "vignette_str": "finish_sidebar.vignette_burn_slider",
    "vignette_size": "finish_sidebar.vignette_size_slider",
    "border_size": "finish_sidebar.border_slider",
}


def slider_widget_map(controls) -> dict[str, Callable[[], object]]:
    """Resolve lazily: sidebars rebuild their widgets, so a getter must re-read the
    attribute rather than capture the instance."""
    return {slider_id: _bind(controls, path) for slider_id, path in SLIDER_ATTRS.items()}


def _bind(controls, path: str) -> Callable[[], object]:
    getter = attrgetter(path)
    return lambda: getter(controls)
