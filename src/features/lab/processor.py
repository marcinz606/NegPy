import numpy as np
from src.core.interfaces import IProcessor, PipelineContext
from src.core.types import ImageBuffer
from src.features.lab.models import LabConfig
from src.features.lab.logic import (
    apply_spectral_crosstalk,
    apply_hypertone,
    apply_chroma_noise_removal,
    apply_output_sharpening,
)


class PhotoLabProcessor(IProcessor):
    def __init__(self, config: LabConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        img = image
        scale_factor = context.scale_factor

        # 1. Spectral Crosstalk (Needs Density Space)
        # Original logic: crosstalk_strength = max(0.0, settings.color_separation - 1.0)
        c_strength = max(0.0, self.config.color_separation - 1.0)
        if c_strength > 0:
            epsilon = 1e-6
            img_dens = -np.log10(np.clip(img, epsilon, 1.0))
            img_dens = apply_spectral_crosstalk(
                img_dens, c_strength, self.config.crosstalk_matrix
            )
            img = np.power(10.0, -img_dens)

        # 2. Scanner Emulation (Hypertone)
        if self.config.hypertone_strength > 0:
            img = apply_hypertone(img, self.config.hypertone_strength)

        # 3. Chroma Noise Removal
        if self.config.c_noise_strength > 0:
            img = apply_chroma_noise_removal(
                img, self.config.c_noise_strength, scale_factor
            )

        # 4. Output Sharpening
        if self.config.sharpen > 0:
            img = apply_output_sharpening(img, self.config.sharpen)

        return np.clip(img, 0, 1)
