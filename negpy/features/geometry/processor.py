import numpy as np
from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.geometry.models import GeometryConfig
from negpy.features.geometry.logic import (
    apply_fine_rotation,
    apply_margin_to_roi,
    apply_radial_distortion,
    get_autocrop_coords,
    get_manual_rect_coords,
)


class GeometryProcessor:
    """
    Rotates and detects crop.
    """

    def __init__(self, config: GeometryConfig, distortion_k1: float = 0.0):
        self.config = config
        self.distortion_k1 = distortion_k1

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        img = image

        if self.config.rotation != 0:
            img = np.rot90(img, k=self.config.rotation)

        if self.config.flip_horizontal:
            img = np.ascontiguousarray(np.fliplr(img))

        if self.config.flip_vertical:
            img = np.ascontiguousarray(np.flipud(img))

        if self.config.fine_rotation != 0.0:
            img = apply_fine_rotation(img, self.config.fine_rotation)

        if self.distortion_k1 != 0.0:
            img = apply_radial_distortion(img, self.distortion_k1)

        context.metrics["geometry_params"] = {
            "rotation": self.config.rotation,
            "fine_rotation": self.config.fine_rotation,
            "flip_horizontal": self.config.flip_horizontal,
            "flip_vertical": self.config.flip_vertical,
        }
        # Downstream coordinate mappers (retouch/local) need the same correction.
        context.metrics["distortion_k1"] = self.distortion_k1

        if self.config.manual_crop_rect:
            roi = get_manual_rect_coords(
                img,
                self.config.manual_crop_rect,
                offset_px=self.config.autocrop_offset,
                scale_factor=context.scale_factor,
            )
            context.active_roi = roi
        elif self.config.auto_crop_enabled:
            roi = get_autocrop_coords(
                img,
                offset_px=self.config.autocrop_offset,
                scale_factor=context.scale_factor,
                target_ratio_str=self.config.autocrop_ratio,
                mode=self.config.autocrop_mode,
            )
            context.active_roi = roi
        elif self.config.autocrop_offset > 0:
            h_img, w_img = img.shape[:2]
            margin = self.config.autocrop_offset * context.scale_factor
            context.active_roi = apply_margin_to_roi((0, h_img, 0, w_img), h_img, w_img, margin)
        else:
            context.active_roi = None

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
