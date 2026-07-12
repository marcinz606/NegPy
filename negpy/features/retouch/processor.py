from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.retouch.models import RetouchConfig
from negpy.features.retouch.logic import apply_manual_heals, build_heal_regions


class RetouchProcessor:
    """
    Membrane-clones heal regions (manual + synthesized auto/IR injected
    upstream in ImageProcessor._augment_retouch).
    """

    def __init__(self, config: RetouchConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        if not (self.config.manual_heal_strokes or self.config.manual_dust_spots):
            return image

        orig_h, orig_w = context.original_size
        rot_params = context.metrics.get(
            "geometry_params",
            {
                "rotation": 0,
                "fine_rotation": 0.0,
                "flip_horizontal": False,
                "flip_vertical": False,
            },
        )
        distortion_k1 = context.metrics.get("distortion_k1", 0.0)

        heal_regions = build_heal_regions(
            self.config.manual_heal_strokes,
            self.config.manual_dust_spots,
            (orig_h, orig_w),
            rot_params.get("rotation", 0),
            rot_params.get("fine_rotation", 0.0),
            rot_params.get("flip_horizontal", False),
            rot_params.get("flip_vertical", False),
            distortion_k1,
            (image.shape[1], image.shape[0]),
        )
        return apply_manual_heals(image, *heal_regions)
