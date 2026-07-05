import numpy as np
from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.toning.models import ToningConfig
from negpy.features.toning.logic import apply_chemical_toning, apply_split_toning
from negpy.kernel.image.logic import get_luminance, working_oetf_decode, working_oetf_encode
from negpy.features.process.models import ProcessMode


def apply_chromaticity_preserving_black_point(img: ImageBuffer, percentile: float) -> ImageBuffer:
    lum = get_luminance(img)
    bp = np.percentile(lum, percentile)
    res = (img - bp) / (1.0 - bp + 1e-6)
    return np.clip(res, 0.0, 1.0).astype(np.float32)  # type: ignore


class ToningProcessor:
    def __init__(self, config: ToningConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        img = image

        if context.process_mode == ProcessMode.BW:
            # Density-driven toning reads the linear print; the black point
            # keeps its display-domain bracket.
            img = apply_chemical_toning(
                img,
                selenium_strength=self.config.selenium_strength,
                sepia_strength=self.config.sepia_strength,
            )
            p = working_oetf_encode(img)
            p = apply_chromaticity_preserving_black_point(p, 0.05)
            img = working_oetf_decode(p)

        img = apply_split_toning(
            img,
            shadow_hue=self.config.shadow_tint_hue,
            shadow_strength=self.config.shadow_tint_strength,
            highlight_hue=self.config.highlight_tint_hue,
            highlight_strength=self.config.highlight_tint_strength,
        )

        return img
