import numpy as np
import cv2
import streamlit as st
from PIL import Image
from typing import List, Tuple, Optional
from src.features.retouch.logic import generate_local_mask, calculate_luma_mask
from src.features.geometry.models import GeometryConfig
from src.kernel.image.validation import ensure_image
from src.domain.types import ImageBuffer


class Overlays:
    """
    Draws UI overlays (Masks, Spot patches).
    """

    @staticmethod
    def apply_adjustment_mask(
        pil_img: Image.Image,
        img_raw: np.ndarray,
        points: List[Tuple[float, float]],
        radius: float,
        feather: float,
        luma_range: Tuple[float, float],
        luma_softness: float,
        geo_conf: GeometryConfig,
        roi: Optional[Tuple[int, int, int, int]],
        border_px: int = 0,
    ) -> Image.Image:
        """
        Red overlay for dodge/burn masks.
        """
        rh_orig, rw_orig = img_raw.shape[:2]
        orig_w = pil_img.size[0] - 2 * border_px
        orig_h = pil_img.size[1] - 2 * border_px

        mask = generate_local_mask(rh_orig, rw_orig, points, radius, feather, 1.0)

        # use normalized positive
        base_pos = st.session_state.get("base_positive")
        if base_pos is not None and base_pos.shape[:2] == (rh_orig, rw_orig):
            luma_mask = calculate_luma_mask(
                ensure_image(base_pos), luma_range, luma_softness
            )
        else:
            img_pos_lin = 1.0 - np.clip(img_raw, 0, 1)
            luma_mask = calculate_luma_mask(
                ensure_image(img_pos_lin), luma_range, luma_softness
            )

        final_vis_mask = mask * luma_mask

        final_vis_mask = Overlays._transform_mask(
            final_vis_mask, geo_conf, roi, orig_w, orig_h
        )

        mask_u8 = (final_vis_mask * 180).astype(np.uint8)
        if border_px > 0:
            mask_bordered = cv2.copyMakeBorder(
                mask_u8,
                border_px,
                border_px,
                border_px,
                border_px,
                cv2.BORDER_CONSTANT,
                value=0,
            )
            mask_u8 = np.asarray(mask_bordered, dtype=np.uint8)

        mask_pil = Image.fromarray(mask_u8, mode="L")
        overlay = Image.new("RGBA", pil_img.size, (255, 0, 0, 0))
        red_fill = Image.new("RGBA", pil_img.size, (255, 75, 75, 255))

        if pil_img.mode != "RGBA":
            pil_img = pil_img.convert("RGBA")

        return Image.alpha_composite(
            pil_img, Image.composite(red_fill, overlay, mask_pil)
        ).convert("RGB")

    @staticmethod
    def apply_dust_patches(
        pil_img: Image.Image,
        manual_spots: List[Tuple[float, float, float]],
        img_raw_shape: Tuple[int, int],
        geo_conf: GeometryConfig,
        roi: Optional[Tuple[int, int, int, int]],
        border_px: int = 0,
        alpha: int = 75,
    ) -> Image.Image:
        """
        Green patches for manual spots.
        """
        if not manual_spots:
            return pil_img

        rh_orig, rw_orig = img_raw_shape
        orig_w = pil_img.size[0] - 2 * border_px
        orig_h = pil_img.size[1] - 2 * border_px

        mask_manual: ImageBuffer = np.zeros((rh_orig, rw_orig), dtype=np.float32)
        for rx, ry, size in manual_spots:
            px = int(rx * rw_orig)
            py = int(ry * rh_orig)
            cv2.circle(mask_manual, (px, py), int(size), 1.0, -1)

        # Apply same feathering as in logic.py
        # inpaint_rad = 3, feather = 3
        mask_manual = ensure_image(cv2.GaussianBlur(mask_manual, (3, 3), 0))

        mask_manual = Overlays._transform_mask(
            mask_manual, geo_conf, roi, orig_w, orig_h
        )

        mask_u8 = (mask_manual * 255).astype(np.uint8)
        if border_px > 0:
            mask_bordered = cv2.copyMakeBorder(
                mask_u8,
                border_px,
                border_px,
                border_px,
                border_px,
                cv2.BORDER_CONSTANT,
                value=0,
            )
            mask_final = np.asarray(mask_bordered, dtype=np.uint8)
        else:
            mask_final = mask_u8

        mask_pil = Image.fromarray(mask_final, mode="L")
        green_fill = Image.new("RGBA", pil_img.size, (0, 255, 0, alpha))
        transparent = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))

        if pil_img.mode != "RGBA":
            pil_img = pil_img.convert("RGBA")

        overlay = Image.composite(green_fill, transparent, mask_pil)
        return Image.alpha_composite(pil_img, overlay).convert("RGB")

    @staticmethod
    def _transform_mask(
        mask: np.ndarray,
        geo_conf: GeometryConfig,
        roi: Optional[Tuple[int, int, int, int]],
        target_w: int,
        target_h: int,
    ) -> np.ndarray:
        """
        Geometry pipeline for masks (Rot -> Crop -> Resize).
        """
        if geo_conf.rotation % 4 != 0:
            mask = np.rot90(mask, k=geo_conf.rotation % 4)

        if geo_conf.fine_rotation != 0.0:
            h, w = mask.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), geo_conf.fine_rotation, 1.0)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR)

        # ROI Crop - skip if keep_full_frame is enabled
        if roi and not geo_conf.keep_full_frame:
            y1, y2, x1, x2 = roi
            mask = mask[y1:y2, x1:x2]

        # Resize to match viewport
        if mask.shape[:2] != (target_h, target_w):
            mask = cv2.resize(
                mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )

        return mask
