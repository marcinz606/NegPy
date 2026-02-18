import numpy as np
from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.exposure.models import ExposureConfig, EXPOSURE_CONSTANTS
from negpy.features.process.models import ProcessConfig, ProcessMode
from negpy.features.exposure.logic import apply_characteristic_curve
from negpy.kernel.image.logic import get_luminance
from negpy.features.exposure.normalization import (
    normalize_log_image,
    analyze_log_exposure_bounds,
    LogNegativeBounds,
)
from negpy.features.exposure.shadows import analyze_shadow_cast, apply_shadow_cast_correction


class NormalizationProcessor:
    """
    Converts linear RAW to normalized log-density.
    """

    def __init__(self, config: ProcessConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        epsilon = 1e-6
        img_log = np.log10(np.clip(image, epsilon, 1.0))

        if self.config.use_roll_average and self.config.is_locked_initialized:
            bounds = LogNegativeBounds(floors=self.config.locked_floors, ceils=self.config.locked_ceils)
        elif self.config.is_local_initialized:
            bounds = LogNegativeBounds(floors=self.config.local_floors, ceils=self.config.local_ceils)
        else:
            cached_buffer = context.metrics.get("log_bounds_buffer_val")
            cached_norm = context.metrics.get("log_bounds_norm_val")

            needs_reanalysis = (
                "log_bounds" not in context.metrics
                or cached_buffer is None
                or abs(cached_buffer - self.config.analysis_buffer) > 1e-5
                or cached_norm != self.config.e6_normalize
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
                )
                context.metrics["log_bounds"] = bounds
                context.metrics["log_bounds_buffer_val"] = self.config.analysis_buffer
                context.metrics["log_bounds_norm_val"] = self.config.e6_normalize
        
        if self.config.white_point_offset != 0.0 or self.config.black_point_offset != 0.0:
            adj_floors = (
                bounds.floors[0] + self.config.white_point_offset,
                bounds.floors[1] + self.config.white_point_offset,
                bounds.floors[2] + self.config.white_point_offset,
            )
            adj_ceils = (
                bounds.ceils[0] + self.config.black_point_offset,
                bounds.ceils[1] + self.config.black_point_offset,
                bounds.ceils[2] + self.config.black_point_offset,
            )
            bounds = LogNegativeBounds(floors=adj_floors, ceils=adj_ceils)

        res = normalize_log_image(img_log, bounds)

        cast = (0.0, 0.0, 0.0)
        if self.config.shadow_cast_strength > 0:
            if self.config.use_roll_average:
                cast = self.config.locked_shadow_cast
            elif any(v != 0.0 for v in self.config.local_shadow_cast):
                cast = self.config.local_shadow_cast
            else:
                cast = analyze_shadow_cast(res, self.config.shadow_cast_threshold)
            
            res = apply_shadow_cast_correction(res, cast, self.config.shadow_cast_strength)

        context.metrics["normalized_log"] = res
        context.metrics["shadow_cast"] = cast
        return res


class PhotometricProcessor:
    """
    Applies H&D curve simulation.
    """

    def __init__(self, config: ExposureConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        master_ref = 1.0
        exposure_shift = 0.1 + (self.config.density * EXPOSURE_CONSTANTS["density_multiplier"])
        slope = 1.0 + (self.config.grade * EXPOSURE_CONSTANTS["grade_multiplier"])

        pivots = [master_ref - exposure_shift] * 3

        cmy_max = EXPOSURE_CONSTANTS["cmy_max_density"]
        cmy_offsets = (
            self.config.wb_cyan * cmy_max,
            self.config.wb_magenta * cmy_max,
            self.config.wb_yellow * cmy_max,
        )
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

        img_pos = apply_characteristic_curve(
            image,
            params_r=(pivots[0], slope),
            params_g=(pivots[1], slope),
            params_b=(pivots[2], slope),
            toe=self.config.toe,
            toe_width=self.config.toe_width,
            toe_hardness=self.config.toe_hardness,
            shoulder=self.config.shoulder,
            shoulder_width=self.config.shoulder_width,
            shoulder_hardness=self.config.shoulder_hardness,
            shadows=self.config.shadows,
            highlights=self.config.highlights,
            shadow_cmy=shadow_cmy,
            highlight_cmy=highlight_cmy,
            cmy_offsets=cmy_offsets,
            mode=mode_val,
        )

        if context.process_mode == ProcessMode.BW:
            res = get_luminance(img_pos)
            res = np.stack([res, res, res], axis=-1)
            return res

        return img_pos
