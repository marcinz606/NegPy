import negpy_core
import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.lab.logic import (
    apply_chroma_denoise,
    chroma_denoise_radius,
    apply_glow_and_halation,
    apply_lab_sharpening,
    apply_saturation,
    auto_masking,
)
from negpy.features.lab.models import LabConfig


class PhotoLabProcessor:
    def __init__(self, config: LabConfig, mono: bool = False):
        self.config = config
        self.mono = mono

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        """
        Apply effects from logic.py in sequence
        """
        img = image

        chroma_radius = chroma_denoise_radius(self.config.chroma_denoise, self.mono)
        if chroma_radius > 0:
            img = apply_chroma_denoise(img, chroma_radius, context.scale_factor)

        if self.config.saturation != 1.0 or self.config.skin_protection > 0:
            img = apply_saturation(img, self.config.saturation, self.config.skin_protection)

        if self.config.sharpen > 0:
            auto = auto_masking(img, self.config)
            if auto is not None:
                context.metrics["auto_masking"] = auto
            img = apply_lab_sharpening(img, self.config, auto)

        if self.config.glow_amount > 0 or self.config.halation_strength > 0:
            img = apply_glow_and_halation(
                img,
                self.config.glow_amount,
                self.config.halation_strength,
                bw=self.mono,
            )

        return negpy_core.clip01(np.ascontiguousarray(img, dtype=np.float32))
