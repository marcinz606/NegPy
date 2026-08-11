from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.exposure.papers import PaperProfile, effective_constants
from negpy.features.lith.logic import apply_lith
from negpy.features.lith.models import LithConfig
from negpy.features.process.models import ProcessMode


class LithProcessor:
    def __init__(self, config: LithConfig, paper: PaperProfile):
        self.config = config
        self.paper = paper

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        if context.process_mode != ProcessMode.BW:
            return image

        return apply_lith(
            image,
            self.paper.lith_path,
            float(effective_constants(self.paper)["d_max"]),
            enabled=self.config.lith_enabled,
            exposure=self.config.lith_exposure,
            snatch=self.config.lith_snatch,
            abruptness=self.config.lith_abruptness,
        )
