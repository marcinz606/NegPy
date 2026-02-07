import numpy as np
from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.process.models import ProcessMode
from negpy.features.lab.models import LabConfig
from negpy.features.lab.logic import (
    apply_spectral_crosstalk,
    apply_clahe,
    apply_output_sharpening,
    apply_saturation,
)


class PhotoLabProcessor:
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

            matrix = self.config.crosstalk_matrix
            if matrix is None:
                if context.process_mode == ProcessMode.E6:
                    matrix = self.config.E6_MATRIX
                else:
                    matrix = self.config.C41_MATRIX

            img_dens = apply_spectral_crosstalk(img_dens, c_strength, matrix)
            img = np.power(10.0, -img_dens)

        if self.config.saturation != 1.0:
            img = apply_saturation(img, self.config.saturation)

        if self.config.clahe_strength > 0:
            img = apply_clahe(img, self.config.clahe_strength, context.scale_factor)

        if self.config.sharpen > 0:
            img = apply_output_sharpening(img, self.config.sharpen, context.scale_factor)

        return np.clip(img, 0, 1)
