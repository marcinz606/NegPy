from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.finish.logic import apply_vignette
from negpy.features.finish.models import FinishConfig


class FinishProcessor:
    def __init__(self, config: FinishConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        if self.config.vignette_strength == 0.0:
            return image
        return apply_vignette(image, self.config.vignette_strength, self.config.vignette_size)
