import negpy_core
import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.lab.logic import (
    apply_chroma_denoise,
    apply_glow_and_halation,
    apply_lab_sharpening,
    apply_saturation,
    auto_masking,
)
from negpy.features.lab.models import LabConfig
from negpy.features.process.models import ProcessMode


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
            auto = auto_masking(img, self.config)
            if auto is not None:
                context.metrics["auto_masking"] = auto
            img = apply_lab_sharpening(img, self.config, auto)

        if self.config.glow_amount > 0 or self.config.halation_strength > 0:
            img = apply_glow_and_halation(
                img,
                self.config.glow_amount,
                self.config.halation_strength,
                bw=context.process_mode == ProcessMode.BW,
            )

        return negpy_core.clip01(np.ascontiguousarray(img, dtype=np.float32))
