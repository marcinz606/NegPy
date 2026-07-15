from PIL import Image
import cv2
import numpy as np
from typing import Optional, Tuple
from negpy.domain.models import ExportConfig, AspectRatio, ExportResolutionMode
from negpy.features.finish.models import FinishConfig
from negpy.features.toning.logic import apply_chemical_toning, apply_split_toning
from negpy.features.toning.models import ToningConfig

# Keyline sits 30% of the border width away from the image edge.
KEYLINE_GAP_FRACTION = 0.3
# Fixed drugstore-print corner geometry; mirrored as literals in layout.wgsl.
ROUNDED_RADIUS_MM = 4.0
DECKLE_AMP_MM = 1.5
_TAU = 2.0 * np.pi


def _deckle_profile(s: np.ndarray, phase: float) -> np.ndarray:
    """0..1 scallop along a normalized edge coordinate; mirrored in layout.wgsl."""
    return 0.5 + 0.3 * np.sin(_TAU * 5.0 * s + 0.7 + phase) + 0.2 * np.sin(_TAU * 13.0 * s + 2.9 + phase)


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
        finish: Optional[FinishConfig] = None,
    ) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """
        Pads a PIL image to match a specific paper aspect ratio for UI preview.
        Returns (Image, (content_x, content_y, content_w, content_h)).
        """
        img_np = np.array(pil_img).astype(np.float32) / 255.0

        virtual_dpi = int((preview_size_px * 2.54) / max(0.1, print_size_cm))

        config = ExportConfig(
            paper_aspect_ratio=paper_aspect_ratio,
            export_print_size=print_size_cm,
            export_dpi=virtual_dpi,
            export_resolution_mode=ExportResolutionMode.PRINT.value,
        )

        result_np, content_rect = PrintService.apply_layout(
            img_np, config, border_size=border_size_cm, border_color=border_color_hex, finish=finish
        )
        result_uint8 = (np.clip(result_np, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(result_uint8), content_rect

    @staticmethod
    def effective_border_color(finish: FinishConfig, toning: ToningConfig) -> str:
        """Mat colour: the picked hex, or the toned paper white when matching."""
        if not finish.border_match_paper:
            return finish.border_color
        white = np.full((1, 1, 3), 1.0, dtype=np.float32)
        tinted = apply_chemical_toning(
            white,
            selenium_strength=toning.selenium_strength,
            sepia_strength=toning.sepia_strength,
            gold_strength=toning.gold_strength,
            blue_strength=toning.blue_strength,
            copper_strength=toning.copper_strength,
            vanadium_strength=toning.vanadium_strength,
        )
        tinted = apply_split_toning(
            tinted,
            shadow_hue=toning.shadow_tint_hue,
            shadow_strength=toning.shadow_tint_strength,
            highlight_hue=toning.highlight_tint_hue,
            highlight_strength=toning.highlight_tint_strength,
        )
        r, g, b = (int(round(float(c) * 255.0)) for c in tinted[0, 0])
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def paper_long_edge_px(export_settings: ExportConfig) -> int:
        """Long-edge paper size in pixels for non-ORIGINAL modes."""
        if export_settings.export_resolution_mode == ExportResolutionMode.TARGET_PX:
            return max(1, int(export_settings.export_target_long_edge_px))
        return int((export_settings.export_print_size / 2.54) * export_settings.export_dpi)

    @staticmethod
    def effective_dpi(export_settings: ExportConfig) -> int:
        """Physical DPI for PRINT/ORIGINAL; virtual DPI derived from target px in TARGET_PX (for border calc)."""
        if export_settings.export_resolution_mode == ExportResolutionMode.TARGET_PX:
            denom = max(0.1, export_settings.export_print_size) / 2.54
            return max(1, int(export_settings.export_target_long_edge_px / denom))
        return export_settings.export_dpi

    @staticmethod
    def paper_dims_from_long_edge(long_edge_px: int, aspect_ratio_str: str, img_w: int, img_h: int) -> Tuple[int, int]:
        """Paper dims in pixels given a paper long edge and aspect ratio."""
        if aspect_ratio_str == AspectRatio.ORIGINAL:
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
    def calculate_paper_px(print_size_cm: float, dpi: int, aspect_ratio_str: str, img_w: int, img_h: int) -> Tuple[int, int]:
        """Paper dims in pixels from cm + dpi (PRINT-mode wrapper)."""
        long_edge_px = int((print_size_cm / 2.54) * dpi)
        return PrintService.paper_dims_from_long_edge(long_edge_px, aspect_ratio_str, img_w, img_h)

    @staticmethod
    def weighted_offset_y(paper_h: int, target_h: int, border_px: int, border_bottom_px: int) -> int:
        """Vertical content offset splitting the padding top:bottom like the borders."""
        pad_y = paper_h - target_h
        if border_px <= 0:
            return pad_y // 2
        return int(round(pad_y * border_px / (border_px + border_bottom_px)))

    @staticmethod
    def apply_layout(
        img: np.ndarray,
        export_settings: ExportConfig,
        border_size: float = 0.0,
        border_color: str = "#ffffff",
        finish: Optional[FinishConfig] = None,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Scales and pads image to fit paper aspect ratio and border requirements.
        Returns (ImageBuffer, (content_x, content_y, content_w, content_h)).
        """
        if finish is None:
            finish = FinishConfig()
        weight = max(1.0, float(finish.border_bottom_weight))

        img_h, img_w = img.shape[:2]
        img_aspect = img_w / img_h
        mode = export_settings.export_resolution_mode
        dpi = PrintService.effective_dpi(export_settings)
        border_px = int((border_size / 2.54) * dpi)
        border_bottom_px = int(border_px * weight)
        border_y_px = border_px + border_bottom_px

        if mode == ExportResolutionMode.ORIGINAL:
            target_w, target_h = img_w, img_h
            img_scaled = img

            if export_settings.paper_aspect_ratio == AspectRatio.ORIGINAL:
                paper_w = target_w + 2 * border_px
                paper_h = target_h + border_y_px
            else:
                try:
                    w_r, h_r = map(float, export_settings.paper_aspect_ratio.split(":"))
                    paper_ratio = w_r / h_r
                except Exception:
                    paper_ratio = img_aspect

                min_paper_w = target_w + 2 * border_px
                min_paper_h = target_h + border_y_px

                if (min_paper_w / min_paper_h) > paper_ratio:
                    paper_w = min_paper_w
                    paper_h = int(paper_w / paper_ratio)
                else:
                    paper_h = min_paper_h
                    paper_w = int(paper_h * paper_ratio)
        else:
            paper_long_px = PrintService.paper_long_edge_px(export_settings)

            if export_settings.paper_aspect_ratio == AspectRatio.ORIGINAL:
                if img_w >= img_h:
                    target_w = max(10, paper_long_px - 2 * border_px)
                    target_h = int(target_w / img_aspect)
                else:
                    target_h = max(10, paper_long_px - border_y_px)
                    target_w = int(target_h * img_aspect)
                paper_w = target_w + 2 * border_px
                paper_h = target_h + border_y_px
            else:
                paper_w, paper_h = PrintService.paper_dims_from_long_edge(
                    paper_long_px,
                    export_settings.paper_aspect_ratio,
                    img_w,
                    img_h,
                )

                max_content_w = max(10, paper_w - 2 * border_px)
                max_content_h = max(10, paper_h - border_y_px)

                if img_aspect > (max_content_w / max_content_h):
                    target_w = max_content_w
                    target_h = int(target_w / img_aspect)
                else:
                    target_h = max_content_h
                    target_w = int(target_h * img_aspect)

            img_scaled = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        color_hex = border_color.lstrip("#")
        r, g, b = tuple(int(color_hex[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

        channels = img_scaled.shape[2] if img_scaled.ndim == 3 else 1
        paper_shape = (paper_h, paper_w, channels) if channels > 1 else (paper_h, paper_w)
        paper = np.full(
            paper_shape,
            (r, g, b) if channels > 1 else (r,),
            dtype=img_scaled.dtype,
        )
        bg = np.array((r, g, b) if channels > 1 else (r,), dtype=img_scaled.dtype)

        offset_x = (paper_w - target_w) // 2
        offset_y = PrintService.weighted_offset_y(paper_h, target_h, border_px, border_bottom_px)

        h_copy = min(target_h, paper_h - offset_y)
        w_copy = min(target_w, paper_w - offset_x)

        if channels > 1:
            paper[offset_y : offset_y + h_copy, offset_x : offset_x + w_copy, :] = img_scaled[:h_copy, :w_copy, :]
        else:
            paper[offset_y : offset_y + h_copy, offset_x : offset_x + w_copy] = img_scaled[:h_copy, :w_copy]

        PrintService._apply_corner_style(paper, bg, finish.border_corner_style, dpi, offset_x, offset_y, w_copy, h_copy)
        PrintService._draw_keyline(paper, finish.keyline_width, dpi, border_px, offset_x, offset_y, w_copy, h_copy)

        return paper, (offset_x, offset_y, w_copy, h_copy)

    @staticmethod
    def _draw_keyline(paper: np.ndarray, keyline_mm: float, dpi: int, border_px: int, ox: int, oy: int, cw: int, ch: int) -> None:
        """Thin black rule around the image, offset into the mat. In-place."""
        kw = int(round((keyline_mm / 25.4) * dpi))
        if kw <= 0 or border_px <= 0:
            return
        gap = max(2, int(border_px * KEYLINE_GAP_FRACTION))
        ph, pw = paper.shape[:2]
        x0, x1 = max(ox - gap - kw, 0), min(ox + cw + gap + kw, pw)
        y0, y1 = max(oy - gap - kw, 0), min(oy + ch + gap + kw, ph)
        yt, yb = max(oy - gap, 0), min(oy + ch + gap, ph)
        xl, xr = max(ox - gap, 0), min(ox + cw + gap, pw)
        paper[y0:yt, x0:x1] = 0.0
        paper[yb:y1, x0:x1] = 0.0
        paper[yt:yb, x0:xl] = 0.0
        paper[yt:yb, xr:x1] = 0.0

    @staticmethod
    def _apply_corner_style(paper: np.ndarray, bg: np.ndarray, style: str, dpi: int, ox: int, oy: int, cw: int, ch: int) -> None:
        """Rounded or deckled content edge, alpha-composited against the mat. In-place."""
        if style == "rounded":
            r = (ROUNDED_RADIUS_MM / 25.4) * dpi
            n = int(np.ceil(r)) + 1
            if n * 2 > min(cw, ch):
                return
            yy, xx = np.meshgrid(np.arange(n, dtype=np.float32), np.arange(n, dtype=np.float32), indexing="ij")
            for flip_y, flip_x in ((False, False), (False, True), (True, False), (True, True)):
                lx = (cw - 1) - xx if flip_x else xx
                ly = (ch - 1) - yy if flip_y else yy
                ax = np.maximum(r - np.minimum(lx, cw - 1 - lx), 0.0)
                ay = np.maximum(r - np.minimum(ly, ch - 1 - ly), 0.0)
                alpha = np.where(
                    (ax > 0) & (ay > 0),
                    np.clip(r - np.sqrt(ax * ax + ay * ay) + 0.5, 0.0, 1.0),
                    1.0,
                ).astype(paper.dtype)
                ys = oy + ch - n if flip_y else oy
                xs = ox + cw - n if flip_x else ox
                PrintService._composite_band(paper, bg, alpha, ys, xs)
        elif style == "deckle":
            amp = (DECKLE_AMP_MM / 25.4) * dpi
            band = int(np.ceil(amp)) + 1
            if band * 2 > min(cw, ch):
                return
            sx = (np.arange(cw, dtype=np.float32) / np.float32(cw))[None, :]
            sy = (np.arange(ch, dtype=np.float32) / np.float32(ch))[:, None]
            d = np.arange(band, dtype=np.float32)
            # top / bottom / left / right, each with its own phase
            a_t = np.clip(d[:, None] - amp * _deckle_profile(sx, 0.0) + 0.5, 0.0, 1.0).astype(paper.dtype)
            a_b = np.clip(d[::-1][:, None] - amp * _deckle_profile(sx, 3.4) + 0.5, 0.0, 1.0).astype(paper.dtype)
            a_l = np.clip(d[None, :] - amp * _deckle_profile(sy, 5.1) + 0.5, 0.0, 1.0).astype(paper.dtype)
            a_r = np.clip(d[::-1][None, :] - amp * _deckle_profile(sy, 1.7) + 0.5, 0.0, 1.0).astype(paper.dtype)
            PrintService._composite_band(paper, bg, a_t, oy, ox)
            PrintService._composite_band(paper, bg, a_b, oy + ch - band, ox)
            PrintService._composite_band(paper, bg, a_l, oy, ox)
            PrintService._composite_band(paper, bg, a_r, oy, ox + cw - band)

    @staticmethod
    def _composite_band(paper: np.ndarray, bg: np.ndarray, alpha: np.ndarray, y: int, x: int) -> None:
        h, w = alpha.shape
        region = paper[y : y + h, x : x + w]
        a = alpha[..., None] if region.ndim == 3 else alpha
        region *= a
        region += bg * (1.0 - a)
