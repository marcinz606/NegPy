from PIL import Image
import cv2
import numpy as np
from typing import Tuple
from src.domain.models import ExportConfig


class PrintService:
    """
    Handles layout, scaling and padding for print exports and previews.
    """

    @staticmethod
    def apply_preview_layout_to_pil(
        pil_img: Image.Image,
        paper_aspect_ratio: str,
        border_size_cm: float,
        print_size_cm: float,
        border_color_hex: str,
        preview_size_px: float,
    ) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """
        Pads a PIL image to match a specific paper aspect ratio for UI preview.
        Returns (Image, (content_x, content_y, content_w, content_h)).
        """
        img_np = np.array(pil_img).astype(np.float32) / 255.0

        virtual_dpi = int((preview_size_px * 2.54) / max(0.1, print_size_cm))

        config = ExportConfig(
            paper_aspect_ratio=paper_aspect_ratio,
            export_border_size=border_size_cm,
            export_print_size=print_size_cm,
            export_border_color=border_color_hex,
            export_dpi=virtual_dpi,
            use_original_res=False,
        )

        result_np, content_rect = PrintService.apply_layout(img_np, config)
        result_uint8 = (np.clip(result_np, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(result_uint8), content_rect

    @staticmethod
    def calculate_paper_px(
        print_size_cm: float, dpi: int, aspect_ratio_str: str, img_w: int, img_h: int
    ) -> Tuple[int, int]:
        """
        Calculates target paper dimensions in pixels.
        """
        long_edge_px = int((print_size_cm / 2.54) * dpi)

        if aspect_ratio_str == "Original":
            if img_w >= img_h:
                return long_edge_px, int(long_edge_px * (img_h / img_w))
            else:
                return int(long_edge_px * (img_w / img_h)), long_edge_px

        try:
            w_r, h_r = map(float, aspect_ratio_str.split(":"))
            ratio = w_r / h_r
        except (ValueError, ZeroDivisionError):
            ratio = 1.0

        if ratio >= 1.0:
            paper_w = long_edge_px
            paper_h = int(paper_w / ratio)
        else:
            paper_h = long_edge_px
            paper_w = int(paper_h * ratio)

        return paper_w, paper_h

    @staticmethod
    def apply_layout(
        img: np.ndarray, export_settings: ExportConfig
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Scales and pads image to fit paper aspect ratio and border requirements.
        Returns (ImageBuffer, (content_x, content_y, content_w, content_h)).
        """
        img_h, img_w = img.shape[:2]
        img_aspect = img_w / img_h
        dpi = export_settings.export_dpi
        border_px = int((export_settings.export_border_size / 2.54) * dpi)

        if export_settings.paper_aspect_ratio == "Original":
            if export_settings.use_original_res:
                target_w, target_h = img_w, img_h
                img_scaled = img
            else:
                paper_long_px = int((export_settings.export_print_size / 2.54) * dpi)
                content_long_px = max(10, paper_long_px - 2 * border_px)
                if img_w >= img_h:
                    target_w = content_long_px
                    target_h = int(target_w / img_aspect)
                else:
                    target_h = content_long_px
                    target_w = int(target_h * img_aspect)
                img_scaled = cv2.resize(
                    img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4
                )

            paper_w = target_w + 2 * border_px
            paper_h = target_h + 2 * border_px
        else:
            try:
                w_r, h_r = map(float, export_settings.paper_aspect_ratio.split(":"))
                paper_ratio = w_r / h_r
            except Exception:
                paper_ratio = img_aspect

            if export_settings.use_original_res:
                target_w, target_h = img_w, img_h
                img_scaled = img

                min_paper_w = target_w + 2 * border_px
                min_paper_h = target_h + 2 * border_px

                if (min_paper_w / min_paper_h) > paper_ratio:
                    paper_w = min_paper_w
                    paper_h = int(paper_w / paper_ratio)
                else:
                    paper_h = min_paper_h
                    paper_w = int(paper_h * paper_ratio)
            else:
                paper_w, paper_h = PrintService.calculate_paper_px(
                    export_settings.export_print_size,
                    dpi,
                    export_settings.paper_aspect_ratio,
                    img_w,
                    img_h,
                )

                max_content_w = max(10, paper_w - 2 * border_px)
                max_content_h = max(10, paper_h - 2 * border_px)

                if img_aspect > (max_content_w / max_content_h):
                    target_w = max_content_w
                    target_h = int(target_w / img_aspect)
                else:
                    target_h = max_content_h
                    target_w = int(target_h * img_aspect)

                img_scaled = cv2.resize(
                    img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4
                )

        color_hex = export_settings.export_border_color.lstrip("#")
        r, g, b = tuple(int(color_hex[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

        channels = img_scaled.shape[2] if img_scaled.ndim == 3 else 1
        paper_shape = (
            (paper_h, paper_w, channels) if channels > 1 else (paper_h, paper_w)
        )
        paper = np.full(
            paper_shape,
            (r, g, b) if channels > 1 else (r,),
            dtype=img_scaled.dtype,
        )

        offset_x = (paper_w - target_w) // 2
        offset_y = (paper_h - target_h) // 2

        h_copy = min(target_h, paper_h - offset_y)
        w_copy = min(target_w, paper_w - offset_x)

        if channels > 1:
            paper[offset_y : offset_y + h_copy, offset_x : offset_x + w_copy, :] = (
                img_scaled[:h_copy, :w_copy, :]
            )
        else:
            paper[offset_y : offset_y + h_copy, offset_x : offset_x + w_copy] = (
                img_scaled[:h_copy, :w_copy]
            )

        return paper, (offset_x, offset_y, w_copy, h_copy)
