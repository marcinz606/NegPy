import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.exposure.logic import (
    apply_characteristic_curve,
    apply_flat_curve,
    effective_cast_strength,
    filtration_offsets,
    flat_curve_params,
    grade_coupled_shape,
    normalized_neutral_axis,
    normalized_shadow_refs,
    per_channel_curve_params,
)
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig, RenderIntent
from negpy.features.exposure.papers import effective_paper_profile
from negpy.features.exposure.normalization import (
    LogNegativeBounds,
    analyze_log_exposure_bounds_from_log,
    luma_source_bounds,
    luminance_density_range,
    measure_anchor_from_log,
    measure_clip_fractions,
    measure_neutral_axis_from_log,
    measure_shadow_refs_from_log,
    measure_textural_range_from_log,
    normalize_log_image,
    prefilter_log_grid,
    resolve_bounds_detailed,
    resolve_crosstalk_matrix,
    unmix_log_image,
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
        # No upper clamp: mirrors normalization.wgsl (only the low side is clamped);
        # values above 1.0 only occur with flat-field gain and must match the GPU.
        img_log = np.log10(np.clip(np.nan_to_num(image, nan=epsilon, posinf=1.0, neginf=epsilon), epsilon, None))
        # Shared prefilter, once for all five meters (ROI/buffer applied here).
        prefiltered = prefilter_log_grid(image, context.active_roi, self.config.analysis_buffer)
        context.metrics["scan_clip_fractions"] = measure_clip_fractions(image, context.active_roi, self.config.analysis_buffer)

        # Capture-side dye unmix on the negative densities, before any metering,
        # so bounds/anchor/cast refs all read the unmixed film.
        unmix = resolve_crosstalk_matrix(self.config.crosstalk_strength, self.config.crosstalk_matrix)
        img_log = unmix_log_image(img_log, unmix)
        prefiltered = unmix_log_image(prefiltered, unmix)

        def analyze_base() -> LogNegativeBounds:
            cached_buffer = context.metrics.get("log_bounds_buffer_val")
            cached_norm = context.metrics.get("log_bounds_norm_val")
            cached_mode = context.metrics.get("log_bounds_mode_val")

            cached_clip = context.metrics.get("log_bounds_clip_val")
            cached_color_clip = context.metrics.get("log_bounds_color_clip_val")
            cached_unmix = context.metrics.get("log_bounds_crosstalk_val")
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
                or cached_unmix != (self.config.crosstalk_strength, self.config.crosstalk_matrix)
            )

            if not needs_reanalysis:
                return context.metrics["log_bounds"]

            analyzed = analyze_log_exposure_bounds_from_log(
                prefiltered,
                None,
                0.0,
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
            context.metrics["log_bounds_crosstalk_val"] = (self.config.crosstalk_strength, self.config.crosstalk_matrix)
            return analyzed

        bounds, base_bounds = resolve_bounds_detailed(self.config, analyze_base)
        context.metrics["log_bounds_base"] = base_bounds

        context.metrics["norm_density_range"] = luminance_density_range(bounds)

        if context.process_mode == ProcessMode.C41:
            cached_ref_buffer = context.metrics.get("shadow_refs_buffer_val")
            cached_ref_unmix = context.metrics.get("shadow_refs_crosstalk_val")
            if (
                "shadow_log_refs" not in context.metrics
                or cached_ref_buffer is None
                or abs(cached_ref_buffer - self.config.analysis_buffer) > 1e-5
                or cached_ref_unmix != (self.config.crosstalk_strength, self.config.crosstalk_matrix)
            ):
                context.metrics["shadow_log_refs"] = measure_shadow_refs_from_log(
                    prefiltered,
                    None,
                    0.0,
                )
                context.metrics["shadow_refs_buffer_val"] = self.config.analysis_buffer
                context.metrics["shadow_refs_crosstalk_val"] = (self.config.crosstalk_strength, self.config.crosstalk_matrix)

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
            context.metrics["neutral_axis_refs"] = measure_neutral_axis_from_log(prefiltered, bounds, None, 0.0)

        # Per-frame exposure anchor, measured against the same final bounds the
        # image is normalized with. Stored unconditionally (cheap, block-grid);
        # PhotometricProcessor uses it only when auto_exposure is on.
        anchor_bounds = luma_source_bounds(self.config, base_bounds)
        context.metrics["metered_anchor"] = measure_anchor_from_log(prefiltered, anchor_bounds, None, 0.0)
        context.metrics["textural_range"] = measure_textural_range_from_log(prefiltered, None, 0.0)

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
        neutral_axis_refs = context.metrics.get("neutral_axis_refs")
        neutral_axis_norm = normalized_neutral_axis(final_bounds, neutral_axis_refs)
        confidence = neutral_axis_refs[3] if neutral_axis_refs is not None else None
        strength = effective_cast_strength(self.config.cast_removal_strength, self.config.auto_cast_removal, confidence)
        slopes, pivots, curvatures = per_channel_curve_params(
            self.config.grade,
            self.config.density,
            self.config.auto_normalize_contrast,
            strength,
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
        cmy_offsets = filtration_offsets(
            (self.config.wb_cyan, self.config.wb_magenta, self.config.wb_yellow),
            final_bounds,
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

        toe_eff, shoulder_eff = grade_coupled_shape(slopes[1], self.config.toe, self.config.shoulder)

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
            curvatures=curvatures,
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

        cmy_offsets = filtration_offsets(
            (self.config.wb_cyan, self.config.wb_magenta, self.config.wb_yellow),
            context.metrics.get("final_bounds"),
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
