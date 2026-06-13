import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.exposure.logic import (
    apply_characteristic_curve,
    normalized_shadow_refs,
    per_channel_curve_params,
)
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig
from negpy.features.exposure.normalization import (
    LogNegativeBounds,
    analyze_log_exposure_bounds,
    luminance_density_range,
    measure_anchor_from_log,
    measure_shadow_refs_from_log,
    measure_textural_range_from_log,
    normalize_log_image,
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

        if self.config.use_roll_average and self.config.is_locked_initialized:
            bounds = LogNegativeBounds(
                floors=self.config.locked_floors,
                ceils=self.config.locked_ceils,
            )
        elif self.config.is_local_initialized:
            bounds = LogNegativeBounds(
                floors=self.config.local_floors,
                ceils=self.config.local_ceils,
            )
        else:
            cached_buffer = context.metrics.get("log_bounds_buffer_val")
            cached_norm = context.metrics.get("log_bounds_norm_val")
            cached_mode = context.metrics.get("log_bounds_mode_val")

            cached_clip = context.metrics.get("log_bounds_clip_val")
            needs_reanalysis = (
                "log_bounds" not in context.metrics
                or cached_buffer is None
                or abs(cached_buffer - self.config.analysis_buffer) > 1e-5
                or cached_clip is None
                or abs(cached_clip - self.config.drange_clip) > 1e-6
                or cached_norm != self.config.e6_normalize
                or cached_mode != context.process_mode
            )

            if not needs_reanalysis:
                bounds = context.metrics["log_bounds"]
            else:
                bounds = analyze_log_exposure_bounds(
                    image,
                    context.active_roi,
                    self.config.analysis_buffer,
                    process_mode=context.process_mode,
                    e6_normalize=self.config.e6_normalize,
                    percentile_clip=self.config.drange_clip,
                )
                context.metrics["log_bounds"] = bounds
                context.metrics["log_bounds_buffer_val"] = self.config.analysis_buffer
                context.metrics["log_bounds_clip_val"] = self.config.drange_clip
                context.metrics["log_bounds_norm_val"] = self.config.e6_normalize
                context.metrics["log_bounds_mode_val"] = context.process_mode

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

        # Per-frame exposure anchor, measured against the same final bounds the
        # image is normalized with. Stored unconditionally (cheap, block-grid);
        # PhotometricProcessor uses it only when auto_exposure is on.
        context.metrics["metered_anchor"] = measure_anchor_from_log(img_log, bounds, context.active_roi, self.config.analysis_buffer)
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
        d_min = EXPOSURE_CONSTANTS["d_min"] if self.config.paper_dmin else 0.0
        anchor = context.metrics.get("metered_anchor") if self.config.auto_exposure else None
        lum_range = context.metrics.get("norm_density_range")
        shadow_refs_norm = normalized_shadow_refs(context.metrics.get("final_bounds"), context.metrics.get("shadow_log_refs"))
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
        )

        cmy_max = EXPOSURE_CONSTANTS["cmy_max_density"]
        cmy_offsets = (
            self.config.wb_cyan * cmy_max,
            self.config.wb_magenta * cmy_max,
            self.config.wb_yellow * cmy_max,
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

        mode_val = 0
        if context.process_mode == ProcessMode.BW:
            mode_val = 1
        elif context.process_mode == ProcessMode.E6:
            mode_val = 2

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
            toe=self.config.toe,
            toe_width=self.config.toe_width,
            shoulder=self.config.shoulder,
            shoulder_width=self.config.shoulder_width,
            shadow_cmy=shadow_cmy,
            highlight_cmy=highlight_cmy,
            cmy_offsets=cmy_offsets,
            d_min=d_min,
            flare=EXPOSURE_CONSTANTS["flare_fraction"] if self.config.flare else 0.0,
            surround_gamma=EXPOSURE_CONSTANTS["target_system_gamma"] if self.config.surround else 1.0,
            mode=mode_val,
        )

        if context.process_mode == ProcessMode.BW:
            res = get_luminance(img_pos)
            res = np.stack([res, res, res], axis=-1)
            return res

        return img_pos
