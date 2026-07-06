from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.retouch.models import RetouchConfig
from negpy.features.retouch.logic import apply_dust_removal, build_heal_regions


class RetouchProcessor:
    """
    Applies healing and automatic dust removal.
    """

    def __init__(self, config: RetouchConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        img = image
        scale_factor = context.scale_factor

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

        heal_regions = None
        if self.config.manual_heal_strokes or self.config.manual_dust_spots:
            heal_regions = build_heal_regions(
                self.config.manual_heal_strokes,
                self.config.manual_dust_spots,
                (orig_h, orig_w),
                rot_params.get("rotation", 0),
                rot_params.get("fine_rotation", 0.0),
                rot_params.get("flip_horizontal", False),
                rot_params.get("flip_vertical", False),
                distortion_k1,
                scale_factor,
                (img.shape[1], img.shape[0]),
            )

        ir_post_geometry = context.metrics.get("ir_post_geometry")

        return apply_dust_removal(
            img,
            self.config.dust_remove,
            self.config.dust_threshold,
            self.config.dust_size,
            heal_regions,
            scale_factor,
            ir_buffer=ir_post_geometry,
            ir_dust_remove=self.config.ir_dust_remove,
            ir_threshold=1.0 - self.config.ir_threshold,
            ir_inpaint_radius=self.config.ir_inpaint_radius,
        )
