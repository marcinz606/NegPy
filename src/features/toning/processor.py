import numpy as np
from src.domain.interfaces import PipelineContext
from src.domain.types import ImageBuffer
from src.features.toning.models import ToningConfig
from src.features.toning.logic import simulate_paper_substrate, apply_chemical_toning
from src.kernel.image.logic import get_luminance


# We need to port this helper locally or into logic as well
def apply_chromaticity_preserving_black_point(
    img: ImageBuffer, percentile: float
) -> ImageBuffer:
    lum = get_luminance(img)
    bp = np.percentile(lum, percentile)
    res = (img - bp) / (1.0 - bp + 1e-6)
    return np.clip(res, 0.0, 1.0).astype(np.float32)  # type: ignore


class ToningProcessor:
    def __init__(self, config: ToningConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        img = image
        img = simulate_paper_substrate(img, self.config.paper_profile)

        if context.process_mode == "B&W":
            img = apply_chemical_toning(
                img,
                selenium_strength=self.config.selenium_strength,
                sepia_strength=self.config.sepia_strength,
            )

            img = apply_chromaticity_preserving_black_point(img, 0.05)

        return img
