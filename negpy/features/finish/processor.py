from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.finish.logic import apply_carrier, apply_vignette
from negpy.features.finish.models import FinishConfig


def carrier_width_px(carrier_width_mm: float, print_size_cm: float, long_edge_px: float) -> float:
    """Rebate width in image pixels for a given print long edge in cm."""
    return (carrier_width_mm / max(print_size_cm * 10.0, 1.0)) * long_edge_px


class FinishProcessor:
    def __init__(self, config: FinishConfig, print_size_cm: float = 30.0):
        self.config = config
        self.print_size_cm = print_size_cm

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        if self.config.vignette_stops != 0.0:
            image = apply_vignette(image, self.config.vignette_stops, self.config.vignette_size, self.config.vignette_roundness)
        if self.config.carrier_width > 0.0:
            width = carrier_width_px(self.config.carrier_width, self.print_size_cm, float(max(image.shape[:2])))
            image = apply_carrier(image, width, self.config.carrier_rough)
        return image
