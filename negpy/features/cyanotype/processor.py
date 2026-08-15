from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.altprocess.models import AltProcess, AltProcessConfig
from negpy.features.cyanotype.logic import apply_cyanotype
from negpy.features.process.models import ProcessMode


class CyanotypeProcessor:
    def __init__(self, config: AltProcessConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        if context.process_mode != ProcessMode.BW:
            return image

        return apply_cyanotype(
            image,
            self.config.cyano_sensitizer,
            enabled=self.config.alt_process == AltProcess.CYANOTYPE,
            exposure=self.config.cyano_exposure,
            scale=self.config.cyano_scale,
            bleach=self.config.cyano_bleach,
            tannin=self.config.cyano_tannin,
        )
