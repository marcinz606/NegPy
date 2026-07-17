import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.exposure.logic import grade_chroma_damping
from negpy.features.lab.logic import (
    apply_chroma_denoise,
    apply_glow_and_halation,
    apply_output_sharpening,
    apply_saturation,
    apply_vibrance,
)
from negpy.features.lab.models import LabConfig


class PhotoLabProcessor:
    def __init__(self, config: LabConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        """
        Apply effects from logic.py in sequence
        """
        img = image

        if self.config.chroma_denoise > 0:
            img = apply_chroma_denoise(img, self.config.chroma_denoise, context.scale_factor)

        if self.config.vibrance != 1.0:
            img = apply_vibrance(img, self.config.vibrance)

        slopes = context.metrics.get("print_slopes")
        damp = 1.0 if slopes is None else grade_chroma_damping(slopes[1], self.config.chroma_damping)
        eff_sat = self.config.saturation * damp
        if eff_sat != 1.0:
            img = apply_saturation(img, eff_sat)

        if self.config.sharpen > 0:
            img = apply_output_sharpening(img, self.config.sharpen, context.scale_factor)

        if self.config.glow_amount > 0 or self.config.halation_strength > 0:
            img = apply_glow_and_halation(img, self.config.glow_amount, self.config.halation_strength, context.scale_factor)

        return np.clip(img, 0, 1)
