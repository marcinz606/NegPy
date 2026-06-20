from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.local.logic import apply_local_adjustments
from negpy.features.local.models import LocalAdjustmentsConfig


class LocalProcessor:
    def __init__(self, config: LocalAdjustmentsConfig):
        self.config = config

    def process(self, img: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        if not self.config.masks:
            return img

        geo = context.metrics.get("geometry_params", {})
        return apply_local_adjustments(
            img,
            self.config,
            orig_shape=context.original_size,
            rotation=geo.get("rotation", 0),
            fine_rotation=geo.get("fine_rotation", 0.0),
            flip_horizontal=geo.get("flip_horizontal", False),
            flip_vertical=geo.get("flip_vertical", False),
        )
