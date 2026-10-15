from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.altprocess.models import AltProcess, AltProcessConfig
from negpy.features.exposure.papers import PaperProfile, effective_constants
from negpy.features.process.models import ProcessMode
from negpy.features.sabattier.logic import apply_sabattier, line_sigma_px


class SabattierProcessor:
    def __init__(self, config: AltProcessConfig, paper: PaperProfile):
        self.config = config
        self.paper = paper

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        if context.process_mode != ProcessMode.BW:
            return image

        return apply_sabattier(
            image,
            float(effective_constants(self.paper)["d_max"]),
            enabled=self.config.alt_process == AltProcess.SABATTIER,
            strength=self.config.sabattier_strength,
            reexposure=self.config.sabattier_reexposure,
            line_sigma_px=line_sigma_px(self.config.sabattier_agitation, image.shape[:2]),
        )
