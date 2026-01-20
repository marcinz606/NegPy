import numpy as np
from src.domain.interfaces import PipelineContext
from src.domain.types import ImageBuffer
from src.features.geometry.models import GeometryConfig
from src.features.geometry.logic import (
    apply_fine_rotation,
    get_autocrop_coords,
    get_manual_rect_coords,
)


class GeometryProcessor:
    """
    Rotates and detects crop.
    """

    def __init__(self, config: GeometryConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        orig_shape = (image.shape[0], image.shape[1])
        img = image

        if self.config.rotation != 0:
            img = np.rot90(img, k=self.config.rotation)

        if self.config.flip_horizontal:
            img = np.ascontiguousarray(np.fliplr(img))

        if self.config.flip_vertical:
            img = np.ascontiguousarray(np.flipud(img))

        if self.config.fine_rotation != 0.0:
            img = apply_fine_rotation(img, self.config.fine_rotation)

        context.metrics["geometry_params"] = {
            "rotation": self.config.rotation,
            "fine_rotation": self.config.fine_rotation,
            "flip_horizontal": self.config.flip_horizontal,
            "flip_vertical": self.config.flip_vertical,
        }

        if self.config.manual_crop_rect:
            roi = get_manual_rect_coords(
                img,
                self.config.manual_crop_rect,
                orig_shape=orig_shape,
                rotation_k=self.config.rotation,
                fine_rotation=self.config.fine_rotation,
                flip_horizontal=self.config.flip_horizontal,
                flip_vertical=self.config.flip_vertical,
                offset_px=0,
                scale_factor=context.scale_factor,
            )
            context.active_roi = roi
        else:
            roi = get_autocrop_coords(
                img,
                offset_px=self.config.autocrop_offset,
                scale_factor=context.scale_factor,
                target_ratio_str=self.config.autocrop_ratio,
            )
            context.active_roi = roi

        context.metrics["active_roi"] = context.active_roi
        return img


class CropProcessor:
    """
    Executes final crop.
    """

    def __init__(self, config: GeometryConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        if context.active_roi:
            y1, y2, x1, x2 = context.active_roi
            return image[y1:y2, x1:x2]
        return image
