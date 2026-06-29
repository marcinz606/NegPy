import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.exposure.logic import (
    apply_characteristic_curve,
    apply_flat_curve,
    flat_curve_params,
    normalized_neutral_axis,
    normalized_shadow_refs,
    per_channel_curve_params,
)
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig, RenderIntent
from negpy.features.exposure.papers import effective_paper_profile
from negpy.features.exposure.normalization import (
    LogNegativeBounds,
    analyze_log_exposure_bounds,
    luma_source_bounds,
    luminance_density_range,
    measure_anchor_from_log,
    measure_neutral_axis_from_log,
    measure_shadow_refs_from_log,
    measure_textural_range_from_log,
    normalize_log_image,
    resolve_bounds_detailed,
)
from negpy.features.process.models import ProcessConfig, ProcessMode
from negpy.kernel.image.logic import get_luminance


class NormalizationProcessor:
    """
    Converts linear RAW to normalized log-density.
    """

    def __init__(self, config: ProcessConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        epsilon = 1e-6
        img_log = np.log10(np.clip(np.nan_to_num(image, nan=epsilon, posinf=1.0, neginf=epsilon), epsilon, 1.0))

        def analyze_base() -> LogNegativeBounds:
            cached_buffer = context.metrics.get("log_bounds_buffer_val")
            cached_norm = context.metrics.get("log_bounds_norm_val")
            cached_mode = context.metrics.get("log_bounds_mode_val")

            cached_clip = context.metrics.get("log_bounds_clip_val")
            cached_color_clip = context.metrics.get("log_bounds_color_clip_val")
            needs_reanalysis = (
                "log_bounds" not in context.metrics
                or cached_buffer is None
                or abs(cached_buffer - self.config.analysis_buffer) > 1e-5
                or cached_clip is None
                or abs(cached_clip - self.config.luma_range_clip) > 1e-6
                or cached_color_clip is None
                or abs(cached_color_clip - self.config.color_range_clip) > 1e-6
                or cached_norm != self.config.e6_normalize
                or cached_mode != context.process_mode
            )

            if not needs_reanalysis:
                return context.metrics["log_bounds"]

            analyzed = analyze_log_exposure_bounds(
                image,
                context.active_roi,
                self.config.analysis_buffer,
                process_mode=context.process_mode,
                e6_normalize=self.config.e6_normalize,
                percentile_clip=self.config.luma_range_clip,
                color_clip=self.config.color_range_clip,
            )
            context.metrics["log_bounds"] = analyzed
            context.metrics["log_bounds_buffer_val"] = self.config.analysis_buffer
            context.metrics["log_bounds_clip_val"] = self.config.luma_range_clip
            context.metrics["log_bounds_color_clip_val"] = self.config.color_range_clip
            context.metrics["log_bounds_norm_val"] = self.config.e6_normalize
            context.metrics["log_bounds_mode_val"] = context.process_mode
            return analyzed

        bounds, base_bounds = resolve_bounds_detailed(self.config, analyze_base)
        context.metrics["log_bounds_base"] = base_bounds

        context.metrics["norm_density_range"] = luminance_density_range(bounds)

        if context.process_mode == ProcessMode.C41:
            cached_ref_buffer = context.metrics.get("shadow_refs_buffer_val")
            if (
                "shadow_log_refs" not in context.metrics
                or cached_ref_buffer is None
                or abs(cached_ref_buffer - self.config.analysis_buffer) > 1e-5
            ):
                context.metrics["shadow_log_refs"] = measure_shadow_refs_from_log(
                    img_log,
                    context.active_roi,
                    self.config.analysis_buffer,
                )
                context.metrics["shadow_refs_buffer_val"] = self.config.analysis_buffer

        if self.config.white_point_offset != 0.0 or self.config.black_point_offset != 0.0:
            wp_offset = self.config.white_point_offset
            bp_offset = self.config.black_point_offset

            if context.process_mode == ProcessMode.E6:
                wp_offset = -wp_offset
                bp_offset = -bp_offset

            adj_floors = (
                bounds.floors[0] + wp_offset,
                bounds.floors[1] + wp_offset,
                bounds.floors[2] + wp_offset,
            )
            adj_ceils = (
                bounds.ceils[0] + bp_offset,
                bounds.ceils[1] + bp_offset,
                bounds.ceils[2] + bp_offset,
            )
            bounds = LogNegativeBounds(floors=adj_floors, ceils=adj_ceils)

        res = normalize_log_image(img_log, bounds)

        # Neutral axis for the two-point Cast Removal gray balance (C-41 only).
        if context.process_mode == ProcessMode.C41:
            context.metrics["neutral_axis_refs"] = measure_neutral_axis_from_log(
                img_log, bounds, context.active_roi, self.config.analysis_buffer
            )

        # Per-frame exposure anchor, measured against the same final bounds the
        # image is normalized with. Stored unconditionally (cheap, block-grid);
        # PhotometricProcessor uses it only when auto_exposure is on.
        anchor_bounds = luma_source_bounds(self.config, base_bounds)
        context.metrics["metered_anchor"] = measure_anchor_from_log(img_log, anchor_bounds, context.active_roi, self.config.analysis_buffer)
        context.metrics["textural_range"] = measure_textural_range_from_log(img_log, context.active_roi, self.config.analysis_buffer)

        context.metrics["final_bounds"] = bounds
        context.metrics["normalized_log"] = res
        return res


class PhotometricProcessor:
    """
    Applies H&D curve simulation.
    """

    def __init__(self, config: ExposureConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        if self.config.render_intent == RenderIntent.FLAT:
            return self._process_flat(image, context)

        paper = effective_paper_profile(self.config.paper_profile, context.process_mode)
        d_min = paper.d_min if self.config.paper_dmin else 0.0
        anchor = context.metrics.get("metered_anchor") if self.config.auto_exposure else None
        lum_range = context.metrics.get("norm_density_range")
        final_bounds = context.metrics.get("final_bounds")
        shadow_refs_norm = normalized_shadow_refs(final_bounds, context.metrics.get("shadow_log_refs"))
        neutral_axis_norm = normalized_neutral_axis(final_bounds, context.metrics.get("neutral_axis_refs"))
        slopes, pivots = per_channel_curve_params(
            self.config.grade,
            self.config.density,
            self.config.auto_normalize_contrast,
            self.config.cast_removal,
            lum_range,
            shadow_refs_norm,
            context.metrics.get("textural_range"),
            d_min=d_min,
            anchor=anchor,
            paper=paper,
            neutral_axis_norm=neutral_axis_norm,
        )

        c = EXPOSURE_CONSTANTS
        cmy_max = c["cmy_max_density"]
        tint = paper.base_tint_cmy
        cmy_offsets = (
            self.config.wb_cyan * cmy_max + tint[0],
            self.config.wb_magenta * cmy_max + tint[1],
            self.config.wb_yellow * cmy_max + tint[2],
        )
        # Manual shadow CMY only; auto neutralization is Cast Removal (slope balance).
        shadow_cmy = (
            self.config.shadow_cyan * cmy_max,
            self.config.shadow_magenta * cmy_max,
            self.config.shadow_yellow * cmy_max,
        )
        highlight_cmy = (
            self.config.highlight_cyan * cmy_max,
            self.config.highlight_magenta * cmy_max,
            self.config.highlight_yellow * cmy_max,
        )

        # Grade-coupled baseline: hard grades (VC paper) physically have snappier
        # toes and compressed shoulders. slope_norm=0 at softest grade, 1 at hardest.
        slope_norm = (slopes[1] - float(c["slope_min"])) / (float(c["slope_max"]) - float(c["slope_min"]))
        slope_norm = min(max(slope_norm, 0.0), 1.0)
        toe_eff = self.config.toe + float(c["toe_grade_strength"]) * slope_norm
        shoulder_eff = self.config.shoulder + float(c["shoulder_grade_strength"]) * slope_norm

        if context.process_mode == ProcessMode.BW:
            # Panchromatic response: collapse to a single density BEFORE the
            # curve, so the curve shapes one channel instead of mixing three.
            lum = get_luminance(image)
            image = np.stack([lum, lum, lum], axis=-1)

        img_pos = apply_characteristic_curve(
            image,
            params_r=(pivots[0], slopes[0]),
            params_g=(pivots[1], slopes[1]),
            params_b=(pivots[2], slopes[2]),
            toe=toe_eff,
            toe_width=self.config.toe_width,
            shoulder=shoulder_eff,
            shoulder_width=self.config.shoulder_width,
            shadow_cmy=shadow_cmy,
            highlight_cmy=highlight_cmy,
            cmy_offsets=cmy_offsets,
            d_min=d_min,
            flare=EXPOSURE_CONSTANTS["flare_fraction"] if self.config.flare else 0.0,
            surround_gamma=EXPOSURE_CONSTANTS["target_system_gamma"] if self.config.surround else 1.0,
            paper=paper,
        )

        if context.process_mode == ProcessMode.BW:
            res = get_luminance(img_pos)
            res = np.stack([res, res, res], axis=-1)
            return res

        return img_pos

    def _process_flat(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        """
        Flat log-master render: emits the normalized log signal directly (a flat,
        milky log-video look), dropping all creative print decisions — no auto
        density/grade, cast removal, toe/shoulder, surround/flare. A fixed gain/lift
        keeps the master consistent across a roll and holds maximal editing latitude.

        Manual global white balance (the WB picker / CMY global) is still honoured
        because it is an explicit, per-roll-consistent user choice, not automatic
        grading.
        """
        gain, lift = flat_curve_params()

        cmy_max = EXPOSURE_CONSTANTS["cmy_max_density"]
        cmy_offsets = (
            self.config.wb_cyan * cmy_max,
            self.config.wb_magenta * cmy_max,
            self.config.wb_yellow * cmy_max,
        )

        is_bw = context.process_mode == ProcessMode.BW

        if is_bw:
            lum = get_luminance(image)
            image = np.stack([lum, lum, lum], axis=-1)

        img_pos = apply_flat_curve(image, gain, lift, cmy_offsets=cmy_offsets)

        if is_bw:
            res = get_luminance(img_pos)
            return np.stack([res, res, res], axis=-1)

        return img_pos
