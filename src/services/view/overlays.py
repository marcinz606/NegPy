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
        content_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> Image.Image:
        """
        Overlays red mask for local adjustments.
        """
        canvas_w, canvas_h = pil_img.size

        if content_rect:
            cx, cy, cw, ch = content_rect
        else:
            cx, cy, cw, ch = 0, 0, canvas_w, canvas_h

        rh_orig, rw_orig = img_raw.shape[:2]

        mask = generate_local_mask(rh_orig, rw_orig, points, radius, feather, 1.0)

        retouch_source = st.session_state.get("retouch_source")

        if retouch_source is not None:
            h_rot, w_rot = retouch_source.shape[:2]

            mask_rot = Overlays._transform_mask(
                mask, geo_conf, roi=None, target_w=w_rot, target_h=h_rot
            )

            luma_mask = calculate_luma_mask(
                ensure_image(retouch_source), luma_range, luma_softness
            )

            final_vis_mask = mask_rot * luma_mask

            final_vis_mask = Overlays._transform_mask(
                final_vis_mask,
                GeometryConfig(rotation=0),
                roi,
                cw,
                ch,
            )
        else:
            final_vis_mask = Overlays._transform_mask(mask, geo_conf, roi, cw, ch)

        mask_u8: np.ndarray = (final_vis_mask * 180).astype(np.uint8)

        pad_top = cy
        pad_left = cx
        pad_bottom = max(0, canvas_h - (cy + ch))
        pad_right = max(0, canvas_w - (cx + cw))

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            mask_u8 = cv2.copyMakeBorder(
                mask_u8,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=0,
            )

        if mask_u8.shape[:2] != (canvas_h, canvas_w):
            mask_u8 = cv2.resize(
                mask_u8, (canvas_w, canvas_h), interpolation=cv2.INTER_NEAREST
            )

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
        content_rect: Optional[Tuple[int, int, int, int]] = None,
        alpha: int = 75,
    ) -> Image.Image:
        """
        Overlays green markers for manual dust removal.
        """
        if not manual_spots:
            return pil_img

        canvas_w, canvas_h = pil_img.size

        if content_rect:
            cx, cy, cw, ch = content_rect
        else:
            cx, cy, cw, ch = 0, 0, canvas_w, canvas_h

        rh_orig, rw_orig = img_raw_shape

        mask_manual: ImageBuffer = np.zeros((rh_orig, rw_orig), dtype=np.float32)
        for rx, ry, size in manual_spots:
            px = int(rx * rw_orig)
            py = int(ry * rh_orig)
            cv2.circle(mask_manual, (px, py), int(size), 1.0, -1)

        mask_manual = ensure_image(cv2.GaussianBlur(mask_manual, (3, 3), 0))
        mask_manual = Overlays._transform_mask(mask_manual, geo_conf, roi, cw, ch)

        mask_u8: np.ndarray = (mask_manual * 255).astype(np.uint8)

        pad_top = cy
        pad_left = cx
        pad_bottom = max(0, canvas_h - (cy + ch))
        pad_right = max(0, canvas_w - (cx + cw))

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            mask_u8 = cv2.copyMakeBorder(
                mask_u8,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=0,
            )

        if mask_u8.shape[:2] != (canvas_h, canvas_w):
            mask_u8 = cv2.resize(
                mask_u8, (canvas_w, canvas_h), interpolation=cv2.INTER_NEAREST
            )

        mask_pil = Image.fromarray(mask_u8, mode="L")
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
        Transforms mask to match display viewport geometry.
        """
        if geo_conf.rotation % 4 != 0:
            mask = np.rot90(mask, k=geo_conf.rotation % 4)

        if geo_conf.flip_horizontal:
            mask = np.fliplr(mask)

        if geo_conf.flip_vertical:
            mask = np.flipud(mask)

        if geo_conf.fine_rotation != 0.0:
            h, w = mask.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), geo_conf.fine_rotation, 1.0)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR)

        if roi and not geo_conf.keep_full_frame:
            y1, y2, x1, x2 = roi
            mask = mask[y1:y2, x1:x2]

        if mask.shape[:2] != (target_h, target_w):
            mask = cv2.resize(
                mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )

        return mask
