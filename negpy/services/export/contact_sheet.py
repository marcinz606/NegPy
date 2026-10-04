"""Darkroom contact print, strips laid 1:1 on paper; draws the dialog preview (`draft`) and the export."""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from negpy.services.export.contact_sheet_edge import (
    DX_CLOCK_TRACK,
    DX_DATA_TRACK,
    DX_MODULE,
    Band,
    EdgeCode,
    EdgeText,
    ascii_upper,
    edge_items,
)
from negpy.services.export.contact_sheet_layout import (
    PERF_ACROSS,
    PERF_ALONG,
    PERF_FROM_EDGE,
    PERF_RADIUS,
    STRIP_GAP,
    SheetPlan,
    StripPlacement,
    perforation_centers,
)
from negpy.services.export.contact_sheet_roll import SheetLook

RGB = tuple[int, int, int]

# Pillow's bundled Aileron, the same on every platform; capitals are 0.70 em tall.
EDGE_CAP_EM = 0.70
# Aspect tolerance to fill the film window; a tile further off fits whole inside it.
COVER_TOLERANCE = 0.03
PERF_RIM = 0.08
LABEL_CAP = 2.0
LABEL_ABOVE_STRIP = 3.0
# A DX clock track runs past the rebate, up between the holes.
DX_BAND = DX_CLOCK_TRACK[1] + 0.1


@dataclass(frozen=True)
class Palette:
    no_film: RGB
    rebate: RGB
    rim: Optional[RGB]
    ink: RGB
    label: RGB


def palette_for(look: SheetLook) -> Palette:
    b = look.black

    def grey(level: int) -> RGB:
        return (level, level, level)

    if look.palette == "slide":
        # Reversal paper: bare light prints white, the slide's D-max black.
        white = (241, 240, 236)
        return Palette(white, grey(b), None, white, grey(b))
    if look.palette == "color":
        # The filter pack cancels the orange mask, so bare light prints reddish.
        warm = (236, 222, 188)
        return Palette((b + 8, b + 2, b + 2), grey(b + 5), (b + 15, b + 7, b + 6), warm, warm)
    return Palette(grey(b), grey(b + 5), grey(b + 10), grey(224), grey(224))


@lru_cache(maxsize=64)
def _edge_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=max(1, size))


def _px(mm: float, s: float) -> int:
    return int(round(mm * s))


@lru_cache(maxsize=8)
def _perforation_sprites(s_key: int) -> tuple[np.ndarray, np.ndarray]:
    """(hole, rim) coverage of one perforation."""
    s = s_key / 100.0
    ss = 4
    w_px, h_px = PERF_ALONG * s, PERF_ACROSS * s
    size_w, size_h = int(math.ceil(w_px)) + 2, int(math.ceil(h_px)) + 2

    def rounded(inset: float) -> np.ndarray:
        big = Image.new("L", (size_w * ss, size_h * ss), 0)
        x0 = ((size_w - w_px) / 2 + inset) * ss
        y0 = ((size_h - h_px) / 2 + inset) * ss
        x1 = ((size_w + w_px) / 2 - inset) * ss
        y1 = ((size_h + h_px) / 2 - inset) * ss
        radius = max(0.0, (PERF_RADIUS * s - inset) * ss)
        if x1 > x0 and y1 > y0:
            ImageDraw.Draw(big).rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=255)
        small = big.resize((size_w, size_h), Image.Resampling.BOX)
        return np.asarray(small, dtype=np.float32) / 255.0

    hole = rounded(0.0)
    rim = np.clip(hole - rounded(PERF_RIM * s), 0.0, 1.0)
    return hole, rim


def _blend(region: np.ndarray, alpha: np.ndarray, color: RGB) -> None:
    if region.size == 0:
        return
    a = alpha[..., None]
    region[:] = np.clip(region * (1.0 - a) + np.asarray(color, np.float32) * a + 0.5, 0, 255).astype(np.uint8)


def _clip_box(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> tuple[int, int, int, int]:
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


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
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(np.ascontiguousarray(tile[..., :3]), (rw, rh), interpolation=interp)
    if cover:
        cx, cy = (rw - ww) // 2, (rh - wh) // 2
        resized = resized[max(0, cy) : max(0, cy) + wh, max(0, cx) : max(0, cx) + ww]
        px, py = x0 + max(0, -cx), y0 + max(0, -cy)
    else:
        px, py = x0 + (ww - rw) // 2, y0 + (wh - rh) // 2
    h_img, w_img = canvas.shape[:2]
    cx0, cy0, cx1, cy1 = _clip_box(px, py, px + resized.shape[1], py + resized.shape[0], w_img, h_img)
    if cx1 > cx0 and cy1 > cy0:
        canvas[cy0:cy1, cx0:cx1] = resized[cy0 - py : cy1 - py, cx0 - px : cx1 - px]


class _Band:
    def __init__(self, width_px: int, height_mm: float, s: float, band: Band) -> None:
        self.s = s
        self.band = band
        self.height_mm = height_mm
        self.image = Image.new("L", (max(1, width_px), max(1, _px(height_mm, s))), 0)
        self.draw = ImageDraw.Draw(self.image)

    def y_of(self, offset_mm: float) -> float:
        from_top = offset_mm if self.band == Band.TOP else self.height_mm - offset_mm
        return from_top * self.s

    def text(self, item: EdgeText, x_px: float) -> None:
        s = self.s
        size = max(1, int(round(item.cap * s / EDGE_CAP_EM)))
        font = _edge_font(size)
        stroke = 0.05 * size
        left_edge, top, right_edge, bottom = font.getbbox(item.text, anchor="ls", stroke_width=stroke)
        glyphs = Image.new("L", (max(1, int(math.ceil(right_edge - left_edge))), max(1, int(math.ceil(bottom - top)))), 0)
        ImageDraw.Draw(glyphs).text((-left_edge, -top), item.text, font=font, fill=255, anchor="ls", stroke_width=stroke, stroke_fill=255)
        width = glyphs.width
        if item.max_width and width > item.max_width * s:
            width = max(1, int(round(item.max_width * s)))
            glyphs = glyphs.resize((width, glyphs.height), Image.Resampling.BOX)
        if item.align == "right":
            left = x_px - width
        elif item.align == "left":
            left = x_px
        else:
            left = x_px - width / 2
        cap_px = item.cap * s
        baseline = self.y_of(item.offset) + cap_px / 2
        box = (int(round(left)), int(round(baseline + top)))
        region = self.image.crop((box[0], box[1], box[0] + glyphs.width, box[1] + glyphs.height))
        self.image.paste(ImageChops.lighter(region, glyphs), box)
        self._marks(item, left, left + width, self.y_of(item.offset), cap_px)

    def _marks(self, item: EdgeText, left: float, right: float, center_y: float, cap_px: float) -> None:
        gap = 0.3 * cap_px
        if item.lead:
            self._mark(item.lead, right_edge=left - gap, center_y=center_y, cap_px=cap_px)
        if item.trail:
            self._mark(item.trail, right_edge=right + gap + 0.9 * cap_px, center_y=center_y, cap_px=cap_px)
        if item.under == "arrow":
            y = self.y_of(0.55)
            self._arrow(left - 0.3 * self.s, right + 0.3 * self.s, y)

    def _arrow(self, x0: float, x1: float, y: float) -> None:
        s = self.s
        thickness = max(1, int(round(0.12 * s)))
        head = 0.35 * s
        self.draw.line((x0, y, x1 - head * 0.5, y), fill=255, width=thickness)
        self.draw.polygon([(x1, y), (x1 - head, y - head * 0.6), (x1 - head, y + head * 0.6)], fill=255)

    def _mark(self, kind: str, right_edge: float, center_y: float, cap_px: float) -> None:
        size = 0.9 * cap_px
        half = size / 2
        if kind == "triangle":
            self.draw.polygon(
                [(right_edge, center_y), (right_edge - size, center_y - half), (right_edge - size, center_y + half)], fill=255
            )
        elif kind == "outline":
            points = [(right_edge, center_y), (right_edge - size, center_y - half), (right_edge - size, center_y + half)]
            self.draw.polygon(points, outline=255, width=max(1, int(round(0.1 * self.s))))
        elif kind == "arrow":
            self._arrow(right_edge - 1.4 * cap_px, right_edge, center_y)
        elif kind == "dots":
            radius = max(0.5, 0.14 * cap_px)
            for k in range(4):
                cx = right_edge - radius - k * 0.5 * cap_px
                self.draw.ellipse((cx - radius, center_y - radius, cx + radius, center_y + radius), fill=255)

    def coverage(self) -> np.ndarray:
        return np.asarray(self.image, dtype=np.float32) / 255.0


def _draw_code(coverage: np.ndarray, band: _Band, code: EdgeCode, x_px: float, draft: bool) -> None:
    s = band.s
    width = coverage.shape[1]
    start = int(math.floor(x_px))
    end = int(math.ceil(x_px + code.length * s))
    c0, c1 = max(0, start), min(width, end)
    if c1 <= c0:
        return
    for bits, (near, far) in ((code.data, DX_DATA_TRACK), (code.clock, DX_CLOCK_TRACK)):
        rows = sorted((band.y_of(near), band.y_of(far)))
        r0, r1 = max(0, int(round(rows[0]))), min(coverage.shape[0], int(round(rows[1])))
        if r1 <= r0:
            continue
        if draft:
            column = np.full(c1 - c0, sum(bits) / len(bits), np.float32)
        else:
            oversample = 8
            xs = (np.arange(c0 * oversample, c1 * oversample) + 0.5) / oversample
            module = np.floor((xs - x_px) / (DX_MODULE * s)).astype(int)
            valid = (module >= 0) & (module < len(bits))
            on = np.zeros_like(xs, dtype=np.float32)
            on[valid] = np.asarray(bits, np.float32)[module[valid]]
            column = on.reshape(-1, oversample).mean(axis=1)
        coverage[r0:r1, c0:c1] = np.maximum(coverage[r0:r1, c0:c1], column[None, :])


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
    height, width = canvas.shape[:2]
    x0, y0 = _px(strip.x, s), _px(strip.y, s)
    x1, y1 = _px(strip.x + strip.length, s), _px(strip.y + geo.width, s)
    cx0, cy0, cx1, cy1 = _clip_box(x0, y0, x1, y1, width, height)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    canvas[cy0:cy1, cx0:cx1] = pal.rebate

    def roll_to_px(roll_x: float) -> float:
        return (roll_x - strip.roll_start) * s

    strip_w = x1 - x0
    items = edge_items(geo, look.edge, plan.frame_count, strip.roll_start, strip.roll_start + strip.length, numbers)
    for band_kind in (Band.TOP, Band.BOTTOM):
        band_mm = max(geo.edge_band, DX_BAND) if (band_kind == Band.BOTTOM and geo.perforated and look.edge.dx) else geo.edge_band
        band = _Band(strip_w, band_mm, s, band_kind)
        codes: list[EdgeCode] = []
        for item in items:
            if isinstance(item, EdgeCode):
                if band_kind == Band.BOTTOM:
                    codes.append(item)
            elif item.band == band_kind:
                band.text(item, roll_to_px(item.x))
        coverage = band.coverage()
        for code in codes:
            _draw_code(coverage, band, code, roll_to_px(code.x), draft)
        band_top = y0 if band_kind == Band.TOP else y1 - coverage.shape[0]
        bx0, by0, bx1, by1 = _clip_box(x0, band_top, x0 + coverage.shape[1], band_top + coverage.shape[0], width, height)
        if bx1 > bx0 and by1 > by0:
            alpha = coverage[by0 - band_top : by1 - band_top, bx0 - x0 : bx1 - x0]
            _blend(canvas[by0:by1, bx0:bx1], alpha, pal.ink)

    if geo.perforated:
        hole, rim = _perforation_sprites(int(round(s * 100)))
        sh, sw = hole.shape
        centers_y = (PERF_FROM_EDGE + PERF_ACROSS / 2, geo.width - PERF_FROM_EDGE - PERF_ACROSS / 2)
        for roll_x in perforation_centers(strip.roll_start, strip.roll_start + strip.length):
            px_center = x0 + roll_to_px(roll_x)
            for center_y in centers_y:
                py_center = (strip.y + center_y) * s
                hx0 = int(round(px_center - sw / 2))
                hy0 = int(round(py_center - sh / 2))
                bx0, by0, bx1, by1 = _clip_box(hx0, hy0, hx0 + sw, hy0 + sh, width, height)
                if bx1 <= bx0 or by1 <= by0:
                    continue
                crop = (slice(by0 - hy0, by1 - hy0), slice(bx0 - hx0, bx1 - hx0))
                _blend(canvas[by0:by1, bx0:bx1], hole[crop], pal.no_film)
                if pal.rim is not None:
                    _blend(canvas[by0:by1, bx0:bx1], rim[crop], pal.rim)

    top_px = strip.y + geo.image_top
    for index in range(strip.first, strip.first + strip.count):
        tile = tiles[index] if index < len(tiles) else None
        if tile is None:
            continue
        k = turns[index] if index < len(turns) else 0
        if k % 4:
            tile = np.rot90(tile, k)
        left_mm = strip.x + geo.frame_center(index) - strip.roll_start - geo.frame_along / 2
        box = (_px(left_mm, s), _px(top_px, s), _px(left_mm + geo.frame_along, s), _px(top_px + geo.frame_across, s))
        _place_tile(canvas, tile, box)


def label_caps(text: str) -> str:
    return " · ".join(ascii_upper(part) for part in text.split("·") if part.strip())


def _draw_label(canvas: np.ndarray, plan: SheetPlan, page: int, s: float, look: SheetLook, pal: Palette) -> None:
    block_x, block_y, block_w, _block_h = plan.pages[page].block
    size = max(1, int(round(LABEL_CAP * s / EDGE_CAP_EM)))
    font = _edge_font(size)
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
    _blend(canvas[band_top:band_bottom, :], alpha, pal.label)


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
        """uint8 sRGB sheet. `tiles` and `turns` are indexed by frame slot; a None tile prints unexposed."""
        s = float(px_per_mm)
        pal = palette_for(look)
        canvas = np.empty((max(1, _px(plan.paper_height, s)), max(1, _px(plan.paper_width, s)), 3), np.uint8)
        canvas[:] = pal.no_film
        if not plan.pages or not 0 <= page < len(plan.pages):
            return canvas
        for strip in plan.pages[page].strips:
            _draw_strip(canvas, strip, plan, tiles, turns, s, look, pal, draft, numbers)
        if plan.label and look.label:
            _draw_label(canvas, plan, page, s, look, pal)
        return canvas
