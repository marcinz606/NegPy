import numpy as np
from src.core.interfaces import IProcessor, PipelineContext
from src.core.types import ImageBuffer
from src.core.constants import PIPELINE_CONSTANTS
from src.features.exposure.models import ExposureConfig
from src.features.exposure.logic import apply_characteristic_curve
from src.features.exposure.normalization import (
    measure_log_negative_bounds,
    normalize_log_image,
)


class NormalizationProcessor(IProcessor):
    """
    Step 1 of Exposure: Log-Normalization.
    """

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        epsilon = 1e-6
        # 1. Convert to Log Space
        img_log = np.log10(np.clip(image, epsilon, 1.0))

        # 2. Analyze bounds (using active ROI if available)
        analysis_img = img_log
        if context.active_roi:
            y1, y2, x1, x2 = context.active_roi
            analysis_img = img_log[y1:y2, x1:x2]

        bounds = measure_log_negative_bounds(analysis_img)

        # Store bounds in context for other processors if needed
        context.metrics["log_bounds"] = bounds

        # 3. Normalize
        return normalize_log_image(img_log, bounds)


class PhotometricProcessor(IProcessor):
    """
    Step 2 of Exposure: Inversion & Grading.
    """

    def __init__(self, config: ExposureConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        # Check if bounds were calculated (required for consistent behavior, though
        # apply_characteristic_curve works on 0-1 normalized data)
        # In this architecture, 'image' coming in IS normalized log data from NormalizationProcessor.

        master_ref = 1.0
        exposure_shift = 0.1 + (
            self.config.density * PIPELINE_CONSTANTS["density_multiplier"]
        )
        slope = 1.0 + (self.config.grade * PIPELINE_CONSTANTS["grade_multiplier"])

        # Base pivot (Midpoint - Exposure)
        pivots = [master_ref - exposure_shift] * 3

        # Convert CMY sliders to density offsets
        # cmy_max_density usually e.g. 0.2
        cmy_max = PIPELINE_CONSTANTS["cmy_max_density"]

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

        # Handle B&W mode here or in a separate step?
        # Original code did it in PhotometricProcessor.
        if self.config.process_mode == "B&W":
            # Simple luminance extraction
            res = (
                0.2126 * img_pos[..., 0]
                + 0.7152 * img_pos[..., 1]
                + 0.0722 * img_pos[..., 2]
            )
            # Restore channel dim
            res = np.stack([res, res, res], axis=-1)
            return res

        return img_pos
