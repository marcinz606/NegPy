from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.altprocess.models import AltProcess
from negpy.features.toning.models import ToningConfig
from negpy.features.toning.logic import apply_chemical_toning, apply_split_toning
from negpy.features.process.models import ProcessMode


class ToningProcessor:
    def __init__(self, config: ToningConfig, alt_process: str = AltProcess.NONE):
        self.config = config
        self.alt_process = alt_process

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        img = image

        # A cyanotype holds no silver, so none of the six baths have anything to
        # react with. Split toning is a dye stain and still applies.
        if context.process_mode == ProcessMode.BW and self.alt_process != AltProcess.CYANOTYPE:
            img = apply_chemical_toning(
                img,
                selenium_strength=self.config.selenium_strength,
                sepia_strength=self.config.sepia_strength,
                gold_strength=self.config.gold_strength,
                blue_strength=self.config.blue_strength,
                copper_strength=self.config.copper_strength,
                vanadium_strength=self.config.vanadium_strength,
                lith_active=self.alt_process == AltProcess.LITH,
            )

        img = apply_split_toning(
            img,
            shadow_hue=self.config.shadow_tint_hue,
            shadow_strength=self.config.shadow_tint_strength,
            highlight_hue=self.config.highlight_tint_hue,
            highlight_strength=self.config.highlight_tint_strength,
        )

        return img
