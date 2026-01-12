from src.domain.interfaces import PipelineContext
from src.domain.types import ImageBuffer
from src.features.retouch.models import RetouchConfig, LocalAdjustmentConfig
from src.features.retouch.logic import apply_dust_removal, apply_local_adjustments
from src.features.geometry.logic import map_coords_to_geometry


class RetouchProcessor:
    """
    Applies healing and dodge/burn.
    """

    def __init__(self, config: RetouchConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        img = image
        scale_factor = context.scale_factor

        # Original size is needed for coordinate mapping
        # context.original_size is (Height, Width)
        orig_h, orig_w = context.original_size

        rot_params = context.metrics.get(
            "geometry_params", {"rotation": 0, "fine_rotation": 0.0}
        )
        rotation = rot_params["rotation"]
        fine_rotation = rot_params["fine_rotation"]

        mapped_spots = []
        if self.config.manual_dust_spots:
            for nx, ny, size in self.config.manual_dust_spots:
                mnx, mny = map_coords_to_geometry(
                    nx,
                    ny,
                    (orig_h, orig_w),
                    rotation,
                    fine_rotation,
                )
                mapped_spots.append((mnx, mny, size))

        mapped_adjustments = []
        if self.config.local_adjustments:
            for adj in self.config.local_adjustments:
                new_points = []
                for nx, ny in adj.points:
                    mnx, mny = map_coords_to_geometry(
                        nx,
                        ny,
                        (orig_h, orig_w),
                        rotation,
                        fine_rotation,
                    )
                    new_points.append((mnx, mny))

                mapped_adj = LocalAdjustmentConfig(
                    points=new_points,
                    strength=adj.strength,
                    radius=adj.radius,
                    feather=adj.feather,
                    luma_range=adj.luma_range,
                    luma_softness=adj.luma_softness,
                )
                mapped_adjustments.append(mapped_adj)

        img = apply_dust_removal(
            img,
            self.config.dust_remove,
            self.config.dust_threshold,
            self.config.dust_size,
            mapped_spots,
            scale_factor,
        )

        if mapped_adjustments:
            img = apply_local_adjustments(img, mapped_adjustments, scale_factor)

        return img
