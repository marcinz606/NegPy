import math
from functools import lru_cache
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MAX_TILES_PER_SHEET = 38

CELL_PX = 600  # long-edge of a single cell
GAP = 16  # gap between cells
MARGIN = 32  # border around the grid
CAPTION_GAP = 6  # space between a photo and its caption band


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return (0, 0, 0)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


@lru_cache(maxsize=8)
def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Anti-aliased TrueType, trying common faces before PIL's default."""
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _caption_font_size(cell_px: int) -> int:
    return max(13, min(40, round(cell_px * 0.05)))


def _caption_height(cell_px: int) -> int:
    # Band tall enough for the glyphs plus even vertical padding.
    return _caption_font_size(cell_px) + 12


class ContactSheetService:
    """Composites rendered frames into darkroom-style contact sheets on black."""

    @staticmethod
    def grid_dims(n: int) -> Tuple[int, int]:
        """Square-ish (cols, rows) holding n frames."""
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        return cols, rows

    @staticmethod
    def build_sheets(
        tiles: List[np.ndarray],
        *,
        labels: Optional[List[str]] = None,
        show_labels: bool = True,
        background_color: str = "#000000",
        label_color: str = "#ffffff",
        max_tiles: int = MAX_TILES_PER_SHEET,
        cell_px: int = CELL_PX,
        gap: int = GAP,
        margin: int = MARGIN,
    ) -> List[Image.Image]:
        """Paginate tiles (<=max_tiles per sheet) into grids on a solid background."""
        if labels is not None and len(labels) != len(tiles):
            raise ValueError("labels length must match tiles")
        sheets: List[Image.Image] = []
        bg = _hex_to_rgb(background_color)
        fg = _hex_to_rgb(label_color)
        for start in range(0, len(tiles), max_tiles):
            chunk = tiles[start : start + max_tiles]
            chunk_labels = labels[start : start + max_tiles] if labels else None
            sheets.append(
                ContactSheetService._compose_sheet(
                    chunk,
                    chunk_labels,
                    show_labels,
                    bg,
                    fg,
                    cell_px,
                    gap,
                    margin,
                )
            )
        return sheets

    @staticmethod
    def _slot_height(cell_px: int, show_labels: bool) -> int:
        if not show_labels:
            return cell_px
        return cell_px + CAPTION_GAP + _caption_height(cell_px)

    @staticmethod
    def _compose_sheet(
        tiles: List[np.ndarray],
        labels: Optional[List[str]],
        show_labels: bool,
        bg: tuple[int, int, int],
        fg: tuple[int, int, int],
        cell_px: int,
        gap: int,
        margin: int,
    ) -> Image.Image:
        cols, rows = ContactSheetService.grid_dims(len(tiles))
        slot_h = ContactSheetService._slot_height(cell_px, show_labels)
        draw_labels = show_labels and labels is not None

        sheet_w = margin * 2 + cols * cell_px + (cols - 1) * gap
        sheet_h = margin * 2 + rows * slot_h + (rows - 1) * gap
        canvas = np.empty((sheet_h, sheet_w, 3), dtype=np.uint8)
        canvas[:] = bg

        caption_h = _caption_height(cell_px)
        band_rgb = tuple(int(round(0.85 * b + 0.15 * f)) for b, f in zip(bg, fg))
        # Each photo + caption is a card, vertically centered in its slot so the
        # caption always hugs its own image rather than floating in the cell.
        captions: list[tuple[int, int, int, str]] = []  # (band_x, band_y, band_w, text)

        for idx, tile in enumerate(tiles):
            row, col = divmod(idx, cols)
            cell_x = margin + col * (cell_px + gap)
            cell_y = margin + row * (slot_h + gap)

            th_scaled, tw_scaled = ContactSheetService._fit_dims(tile, cell_px)
            if draw_labels:
                card_h = th_scaled + CAPTION_GAP + caption_h
                photo_top = cell_y + (slot_h - card_h) // 2
            else:
                photo_top = cell_y + (cell_px - th_scaled) // 2
            photo_x = cell_x + (cell_px - tw_scaled) // 2

            ContactSheetService._paste_resized(canvas, tile, photo_x, photo_top, tw_scaled, th_scaled)

            if draw_labels:
                band_y = photo_top + th_scaled + CAPTION_GAP
                band_rgb_arr = np.array(band_rgb, dtype=np.uint8)
                canvas[band_y : band_y + caption_h, photo_x : photo_x + tw_scaled] = band_rgb_arr
                captions.append((photo_x, band_y, tw_scaled, labels[idx]))

        image = Image.fromarray(canvas)
        if captions:
            ContactSheetService._draw_captions(image, captions, caption_h, cell_px, fg)
        return image

    @staticmethod
    def _fit_dims(tile: np.ndarray, cell_px: int) -> tuple[int, int]:
        """Scaled (height, width) fitting a cell_px square while keeping aspect."""
        h, w = tile.shape[:2]
        scale = cell_px / max(h, w)
        tw = max(1, int(round(w * scale)))
        th = max(1, int(round(h * scale)))
        return th, tw

    @staticmethod
    def _paste_resized(canvas: np.ndarray, tile: np.ndarray, x: int, y: int, tw: int, th: int) -> None:
        resized = cv2.resize(tile, (tw, th), interpolation=cv2.INTER_AREA)
        canvas[y : y + th, x : x + tw] = resized

    @staticmethod
    def _draw_captions(
        image: Image.Image,
        captions: list[tuple[int, int, int, str]],
        caption_h: int,
        cell_px: int,
        fg: tuple[int, int, int],
    ) -> None:
        draw = ImageDraw.Draw(image)
        font = _load_font(_caption_font_size(cell_px))
        for band_x, band_y, band_w, text in captions:
            if not text:
                continue
            display = ContactSheetService._truncate(draw, text, font, band_w - 8)
            tw = int(round(draw.textlength(display, font=font)))
            x = band_x + max(4, (band_w - tw) // 2)
            y = band_y + caption_h // 2
            draw.text((x, y), display, font=font, fill=fg, anchor="lm")

    @staticmethod
    def _truncate(
        draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_w: int
    ) -> str:
        if draw.textlength(text, font=font) <= max_w:
            return text
        ellipsis = "..."
        display = text
        while display and draw.textlength(display + ellipsis, font=font) > max_w:
            display = display[:-1]
        return (display + ellipsis) if display else text[:1]
