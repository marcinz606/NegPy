import numpy as np
from src.core.interfaces import IProcessor, PipelineContext
from src.core.types import ImageBuffer
from src.features.lab.models import LabConfig
from src.features.lab.logic import (
    apply_spectral_crosstalk,
    apply_hypertone,
    apply_output_sharpening,
)


class PhotoLabProcessor(IProcessor):
    def __init__(self, config: LabConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        """
        Apply effects from logic.py in sequence
        """
        img = image

        c_strength = max(0.0, self.config.color_separation - 1.0)
        if c_strength > 0:
            epsilon = 1e-6
            img_dens = -np.log10(np.clip(img, epsilon, 1.0))
            img_dens = apply_spectral_crosstalk(
                img_dens, c_strength, self.config.crosstalk_matrix
            )
            img = np.power(10.0, -img_dens)

        if self.config.hypertone_strength > 0:
            img = apply_hypertone(img, self.config.hypertone_strength)

        if self.config.sharpen > 0:
            img = apply_output_sharpening(img, self.config.sharpen)

        return np.clip(img, 0, 1)
