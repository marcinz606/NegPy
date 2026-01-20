import numpy as np
from src.domain.interfaces import PipelineContext
from src.domain.types import ImageBuffer
from src.features.exposure.models import ExposureConfig, EXPOSURE_CONSTANTS
from src.features.exposure.logic import apply_characteristic_curve
from src.kernel.image.logic import get_luminance
from src.features.exposure.normalization import (
    measure_log_negative_bounds,
    normalize_log_image,
    get_analysis_crop,
)
from src.domain.models import ProcessMode


class NormalizationProcessor:
    """
    Converts linear RAW to normalized log-density.
    """

    def __init__(self, config: ExposureConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        epsilon = 1e-6
        img_log = np.log10(np.clip(image, epsilon, 1.0))

        # Check if bounds are cached and valid for current buffer config
        cached_buffer = context.metrics.get("log_bounds_buffer_val")
        if (
            "log_bounds" in context.metrics
            and cached_buffer is not None
            and abs(cached_buffer - self.config.analysis_buffer) < 1e-5
        ):
            bounds = context.metrics["log_bounds"]
        else:
            analysis_img = img_log
            if context.active_roi:
                y1, y2, x1, x2 = context.active_roi
                analysis_img = img_log[y1:y2, x1:x2]

            if self.config.analysis_buffer > 0:
                analysis_img = get_analysis_crop(
                    analysis_img, self.config.analysis_buffer
                )

            bounds = measure_log_negative_bounds(analysis_img)
            context.metrics["log_bounds"] = bounds
            context.metrics["log_bounds_buffer_val"] = self.config.analysis_buffer

        return normalize_log_image(img_log, bounds)


class PhotometricProcessor:
    """
    Applies H&D curve simulation.
    """

    def __init__(self, config: ExposureConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        master_ref = 1.0
        exposure_shift = 0.1 + (
            self.config.density * EXPOSURE_CONSTANTS["density_multiplier"]
        )
        slope = 1.0 + (self.config.grade * EXPOSURE_CONSTANTS["grade_multiplier"])

        pivots = [master_ref - exposure_shift] * 3

        cmy_max = EXPOSURE_CONSTANTS["cmy_max_density"]
        cmy_offsets = (
            self.config.wb_cyan * cmy_max,
            self.config.wb_magenta * cmy_max,
            self.config.wb_yellow * cmy_max,
        )

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
            cmy_offsets=cmy_offsets,
        )

        if context.process_mode == ProcessMode.BW:
            res = get_luminance(img_pos)
            res = np.stack([res, res, res], axis=-1)
            return res

        return img_pos
