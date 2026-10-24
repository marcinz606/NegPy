"""Darkroom contact print, strips laid 1:1 on paper; draws the dialog preview (`draft`) and the export."""

import math
from dataclasses import replace
from typing import Optional, Sequence

import negpy_core
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from negpy.services.export.contact_sheet_layout import STRIP_GAP, RowPlacement, SheetPlan, StripPlacement
from negpy.services.export.contact_sheet_roll import SheetLook
from negpy.services.film_edge.edge import ascii_upper
from negpy.services.film_edge.paint import RGB, EDGE_CAP_EM, Palette, blend, clip_box, edge_font, paint_strip, px

# Aspect tolerance to fill the film window; a tile further off fits whole inside it.
COVER_TOLERANCE = 0.03
LABEL_CAP = 2.0
LABEL_ABOVE_STRIP = 3.0


def grey(level: int) -> RGB:
    return (level, level, level)


def palette_for(look: SheetLook) -> Palette:
    pal = _film_palette(look)
    if look.white_paper:
        # Pure white, so a home printer lays no ink on the paper.
        return replace(pal, no_film=(255, 255, 255), rim=None, label=grey(look.black))
    return pal


def _film_palette(look: SheetLook) -> Palette:
    b = look.black
    if look.palette == "slide":
        # Reversal paper: bare light prints white, the slide's D-max black.
        white = (241, 240, 236)
        return Palette(white, grey(b), None, white, grey(b))
    if look.palette == "color":
        # The filter pack cancels the orange mask, so bare light prints reddish.
        warm = (236, 222, 188)
        return Palette((b + 8, b + 2, b + 2), grey(b + 5), (b + 15, b + 7, b + 6), warm, warm)
    return Palette(grey(b), grey(b + 5), grey(b + 10), grey(224), grey(224))


def _place_tile(canvas: np.ndarray, tile: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    ww, wh = x1 - x0, y1 - y0
    th, tw = tile.shape[:2]
    if ww <= 0 or wh <= 0 or th <= 0 or tw <= 0:
        return
    tile_aspect, window_aspect = tw / th, ww / wh
    cover = abs(math.log(tile_aspect / window_aspect)) <= math.log(1 + COVER_TOLERANCE)
    scale = max(ww / tw, wh / th) if cover else min(ww / tw, wh / th)
    rw, rh = max(1, int(round(tw * scale))), max(1, int(round(th * scale)))
    resized = negpy_core.resize_u8(np.ascontiguousarray(tile[..., :3], dtype=np.uint8), (rw, rh), cubic=scale >= 1)
    if cover:
        cx, cy = (rw - ww) // 2, (rh - wh) // 2
        resized = resized[max(0, cy) : max(0, cy) + wh, max(0, cx) : max(0, cx) + ww]
        px, py = x0 + max(0, -cx), y0 + max(0, -cy)
    else:
        px, py = x0 + (ww - rw) // 2, y0 + (wh - rh) // 2
    h_img, w_img = canvas.shape[:2]
    cx0, cy0, cx1, cy1 = clip_box(px, py, px + resized.shape[1], py + resized.shape[0], w_img, h_img)
    if cx1 > cx0 and cy1 > cy0:
        canvas[cy0:cy1, cx0:cx1] = resized[cy0 - py : cy1 - py, cx0 - px : cx1 - px]


def _draw_strip(
    canvas: np.ndarray,
    strip: StripPlacement,
    plan: SheetPlan,
    tiles: Sequence[Optional[np.ndarray]],
    turns: Sequence[int],
    s: float,
    look: SheetLook,
    pal: Palette,
    draft: bool,
    numbers: Optional[Sequence[int]],
) -> None:
    geo = plan.geometry
    paint_strip(canvas, strip.x, strip.y, geo, look.edge, plan.frame_count, strip.roll_start, strip.length, s, pal, draft, numbers)
    top_px = strip.y + geo.image_top
    for index in range(strip.first, strip.first + strip.count):
        tile = tiles[index] if index < len(tiles) else None
        if tile is None:
            continue
        k = turns[index] if index < len(turns) else 0
        if k % 4:
            tile = np.rot90(tile, k)
        left_mm = strip.x + geo.frame_center(index) - strip.roll_start - geo.frame_along / 2
        box = (px(left_mm, s), px(top_px, s), px(left_mm + geo.frame_along, s), px(top_px + geo.frame_across, s))
        _place_tile(canvas, tile, box)


def _draw_row(canvas: np.ndarray, row: RowPlacement, tiles: Sequence[Optional[np.ndarray]], s: float, look: SheetLook) -> None:
    """Upright frames on a band of the paper's black, as the Light Table draws a roll."""
    height, width = canvas.shape[:2]
    x0, y0, x1, y1 = clip_box(px(row.x, s), px(row.y, s), px(row.x + row.width, s), px(row.y + row.height, s), width, height)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = grey(look.black)
    for frame in row.frames:
        tile = tiles[frame.slot] if frame.slot < len(tiles) else None
        if tile is not None:
            box = (px(frame.x, s), px(frame.y, s), px(frame.x + frame.width, s), px(frame.y + frame.height, s))
            _place_tile(canvas, tile, box)


def label_caps(text: str) -> str:
    return " · ".join(ascii_upper(part) for part in text.split("·") if part.strip())


def _draw_label(canvas: np.ndarray, plan: SheetPlan, page: int, s: float, look: SheetLook, pal: Palette) -> None:
    block_x, block_y, block_w, _block_h = plan.pages[page].block
    size = max(1, int(round(LABEL_CAP * s / EDGE_CAP_EM)))
    font = edge_font(size)
    stroke = 0.05 * size
    text = label_caps(look.label)
    page_text = f"{page + 1}/{len(plan.pages)}" if len(plan.pages) > 1 else ""
    baseline = (block_y - LABEL_ABOVE_STRIP) * s
    left, right = block_x * s, (block_x + block_w) * s
    height, width = canvas.shape[:2]
    band_top = int(math.floor(baseline - LABEL_CAP * s * 1.6))
    band_bottom = int(math.ceil(baseline + LABEL_CAP * s * 0.5))
    band_top, band_bottom = max(0, band_top), min(height, band_bottom)
    if band_bottom <= band_top:
        return
    layer = Image.new("L", (width, band_bottom - band_top), 0)
    draw = ImageDraw.Draw(layer)
    reserved = draw.textlength(page_text, font=font) + STRIP_GAP * s if page_text else 0.0
    text = _fit_text(draw, text, font, right - left - reserved)
    y = baseline - band_top
    if text:
        draw.text((left, y), text, font=font, fill=255, anchor="ls", stroke_width=stroke, stroke_fill=255)
    if page_text:
        draw.text((right, y), page_text, font=font, fill=255, anchor="rs", stroke_width=stroke, stroke_fill=255)
    alpha = np.asarray(layer, dtype=np.float32) / 255.0
    blend(canvas[band_top:band_bottom, :], alpha, pal.label)


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_width: float) -> str:
    if max_width <= 0:
        return ""
    if draw.textlength(text, font=font) <= max_width:
        return text
    ellipsis = "..."
    while text and draw.textlength(text + ellipsis, font=font) > max_width:
        text = text[:-1]
    return (text.rstrip() + ellipsis) if text else ""


class ContactSheetService:
    @staticmethod
    def render_sheet(
        plan: SheetPlan,
        page: int,
        tiles: Sequence[Optional[np.ndarray]],
        turns: Sequence[int],
        px_per_mm: float,
        look: SheetLook,
        draft: bool = False,
        numbers: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """uint8 sRGB sheet. `tiles` and `turns` are indexed by frame slot; a None tile prints unexposed.
        A row plan (Film Edges off) prints its tiles upright and ignores `turns`."""
        s = float(px_per_mm)
        pal = palette_for(look)
        canvas = np.empty((max(1, px(plan.paper_height, s)), max(1, px(plan.paper_width, s)), 3), np.uint8)
        canvas[:] = pal.no_film
        if not plan.pages or not 0 <= page < len(plan.pages):
            return canvas
        for strip in plan.pages[page].strips:
            _draw_strip(canvas, strip, plan, tiles, turns, s, look, pal, draft, numbers)
        for row in plan.pages[page].rows:
            _draw_row(canvas, row, tiles, s, look)
        if plan.label and look.label:
            _draw_label(canvas, plan, page, s, look, pal)
        return canvas
