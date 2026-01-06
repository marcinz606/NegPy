import numpy as np
from src.core.interfaces import IProcessor, PipelineContext
from src.core.types import ImageBuffer
from src.features.geometry.models import GeometryConfig
from src.features.geometry.logic import apply_fine_rotation, get_autocrop_coords


class GeometryProcessor(IProcessor):
    """
    Applies Rotation and detects ROI.
    """

    def __init__(self, config: GeometryConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        img = image

        # 1. 90-degree Rotations
        if self.config.rotation != 0:
            img = np.rot90(img, k=self.config.rotation)

        # 2. Fine Rotation
        if self.config.fine_rotation != 0.0:
            img = apply_fine_rotation(img, self.config.fine_rotation)

        # Store rotation state for downstream processors (like Retouch) that map coordinates
        context.metrics["geometry_params"] = {
            "rotation": self.config.rotation,
            "fine_rotation": self.config.fine_rotation,
        }

        # 3. Detect ROI (Autocrop)
        if self.config.autocrop:
            roi = get_autocrop_coords(
                img,
                offset_px=self.config.autocrop_offset,
                scale_factor=context.scale_factor,
                target_ratio_str=self.config.autocrop_ratio,
            )
            context.active_roi = roi
        else:
            context.active_roi = None

        return img


class CropProcessor(IProcessor):
    """
    Applies the active ROI crop.
    """

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        if context.active_roi:
            y1, y2, x1, x2 = context.active_roi
            return image[y1:y2, x1:x2]
        return image
