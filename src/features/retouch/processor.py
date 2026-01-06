from src.core.interfaces import IProcessor, PipelineContext
from src.core.types import ImageBuffer
from src.features.retouch.models import RetouchConfig, LocalAdjustmentConfig
from src.features.retouch.logic import apply_dust_removal, apply_local_adjustments

# We need the geometry mapper to transform raw coordinates to the current rotated image space
from src.features.geometry.logic import map_coords_to_geometry


class RetouchProcessor(IProcessor):
    def __init__(self, config: RetouchConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        img = image
        scale_factor = context.scale_factor

        # Original size is needed for coordinate mapping
        # context.original_size is (Height, Width)
        orig_h, orig_w = context.original_size

        # We need to know the current rotation state to map coordinates correctly.
        # Ideally, this should be passed in the context or the points should be mapped
        # by the UI before reaching here.
        # However, following the original engine design, the 'settings' (now Config)
        # contains Raw/Original coordinates, and the Processor maps them to the
        # current image state (which might be rotated by GeometryProcessor).

        # Limitation: The RetouchConfig doesn't know about Rotation.
        # We must either:
        # 1. Pass Rotation info in PipelineContext (added by GeometryProcessor).
        # 2. Inject Rotation info into RetouchProcessor (coupling).

        # Let's check context.metrics. GeometryProcessor could store rotation there?
        # For now, we'll rely on the fact that if we want this to work strictly,
        # we need the rotation parameters.
        # But wait! 'map_coords_to_geometry' requires rotation/fine_rotation params.
        # These are in GeometryConfig, not RetouchConfig.

        # Solution: The Engine should probably resolve coordinates or we pass them in context.
        # For this refactor step, we will check if context has 'geometry_state'.
        # If not, we skip mapping (or assume 0 rotation) which might be a regression
        # unless we update GeometryProcessor to write to context.

        # Let's assume GeometryProcessor writes 'rotation_params' to context.metrics.
        rot_params = context.metrics.get(
            "geometry_params", {"rotation": 0, "fine_rotation": 0.0}
        )
        rotation = rot_params["rotation"]
        fine_rotation = rot_params["fine_rotation"]

        # 1. Map Manual Dust Spots
        mapped_spots = []
        if self.config.manual_dust_spots:
            for nx, ny, size in self.config.manual_dust_spots:
                mnx, mny = map_coords_to_geometry(
                    nx, ny, (orig_h, orig_w), rotation, fine_rotation, roi=None
                )
                mapped_spots.append((mnx, mny, size))

        # 2. Map Local Adjustments
        mapped_adjustments = []
        if self.config.local_adjustments:
            for adj in self.config.local_adjustments:
                # Create a copy with mapped points
                new_points = []
                for nx, ny in adj.points:
                    mnx, mny = map_coords_to_geometry(
                        nx, ny, (orig_h, orig_w), rotation, fine_rotation, roi=None
                    )
                    new_points.append((mnx, mny))

                # We use replace() or manual copy since dataclass is frozen?
                # RetouchConfig is frozen, LocalAdjustmentConfig is not frozen in models.py (default).
                # But let's be safe and create new instance.
                mapped_adj = LocalAdjustmentConfig(
                    points=new_points,
                    strength=adj.strength,
                    radius=adj.radius,
                    feather=adj.feather,
                    luma_range=adj.luma_range,
                    luma_softness=adj.luma_softness,
                )
                mapped_adjustments.append(mapped_adj)

        # 3. Apply Dust Removal
        img = apply_dust_removal(
            img,
            self.config.dust_remove,
            self.config.dust_threshold,
            self.config.dust_size,
            mapped_spots,
            scale_factor,
        )

        # 4. Apply Local Adjustments
        if mapped_adjustments:
            img = apply_local_adjustments(img, mapped_adjustments, scale_factor)

        return img
