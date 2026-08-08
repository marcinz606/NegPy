import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.lab.logic import (
    apply_chroma_denoise,
    apply_glow_and_halation,
    apply_output_sharpening,
    apply_rl_sharpening,
    apply_saturation,
)
from negpy.features.lab.models import LabConfig, SharpenMethod


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

        if self.config.saturation != 1.0 or self.config.skin_protection > 0:
            img = apply_saturation(img, self.config.saturation, self.config.skin_protection)

        if self.config.sharpen > 0:
            sharpen = apply_rl_sharpening if self.config.sharpen_method == SharpenMethod.RL else apply_output_sharpening
            img = sharpen(
                img,
                self.config.sharpen,
                radius=self.config.sharpen_radius,
                masking=self.config.sharpen_masking,
            )

        if self.config.glow_amount > 0 or self.config.halation_strength > 0:
            img = apply_glow_and_halation(img, self.config.glow_amount, self.config.halation_strength, context.scale_factor)

        return np.clip(img, 0, 1)
