from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.vignette.logic import apply_luminance_vignette
from negpy.features.vignette.models import VignetteConfig


class VignetteProcessor:
    def __init__(self, config: VignetteConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        if self.config.strength != 0.0:
            image = apply_luminance_vignette(image, self.config.strength, self.config.feather, self.config.roundness)
        return image
