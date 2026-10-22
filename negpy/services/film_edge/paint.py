"""Film strip painter: rebate, edge print and perforations, into a uint8 RGB canvas."""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from negpy.services.film_edge.barcodes import DX_CLOCK_TRACK
from negpy.services.film_edge.edge import Band, EdgeCode, EdgeStyle, EdgeText, edge_items
from negpy.services.film_edge.geometry import (
    PERF_ACROSS,
    PERF_ALONG,
    PERF_FROM_EDGE,
    PERF_RADIUS,
    FilmGeometry,
    perforation_centers,
)
from negpy.services.film_edge.layouts import KEY_DOT, MARK_GAP

RGB = tuple[int, int, int]

# Pillow's bundled Aileron, the same on every platform; capitals are 0.70 em tall.
EDGE_CAP_EM = 0.70
PERF_RIM = 0.08
# A DX clock track runs past the rebate, up between the holes.
DX_BAND = DX_CLOCK_TRACK[1] + 0.1


@dataclass(frozen=True)
class Palette:
    no_film: RGB
    rebate: RGB
    rim: Optional[RGB]
    ink: RGB
    label: RGB


@lru_cache(maxsize=64)
def edge_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=max(1, size))


def px(mm: float, s: float) -> int:
    return int(round(mm * s))


@lru_cache(maxsize=8)
def _perforation_sprites(s_key: int, rim_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """(hole, rim) coverage of one perforation, the rim `rim_mm` wide, inside the hole's edge."""
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
    rim = np.clip(hole - rounded(rim_mm * s), 0.0, 1.0)
    return hole, rim


def blend(region: np.ndarray, alpha: np.ndarray, color: RGB) -> None:
    if region.size == 0:
        return
    a = alpha[..., None]
    region[:] = np.clip(region * (1.0 - a) + np.asarray(color, np.float32) * a + 0.5, 0, 255).astype(np.uint8)


def clip_box(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> tuple[int, int, int, int]:
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


class _Band:
    def __init__(self, width_px: int, height_mm: float, s: float, band: Band) -> None:
        self.s = s
        self.band = band
        self.height_mm = height_mm
        self.image = Image.new("L", (max(1, width_px), max(1, px(height_mm, s))), 0)
        self.draw = ImageDraw.Draw(self.image)

    def y_of(self, offset_mm: float) -> float:
        from_top = offset_mm if self.band == Band.TOP else self.height_mm - offset_mm
        return from_top * self.s

    def text(self, item: EdgeText, x_px: float) -> None:
        s = self.s
        size = max(1, int(round(item.cap * s / EDGE_CAP_EM)))
        font = edge_font(size)
        stroke = 0.05 * size
        left_edge, top, right_edge, bottom = font.getbbox(item.text, anchor="ls", stroke_width=stroke)
        glyphs = Image.new("L", (max(1, int(math.ceil(right_edge - left_edge))), max(1, int(math.ceil(bottom - top)))), 0)
        ImageDraw.Draw(glyphs).text((-left_edge, -top), item.text, font=font, fill=255, anchor="ls", stroke_width=stroke, stroke_fill=255)
        width = glyphs.width
        if item.char_width:
            glyphs = self._monospaced(item.text, font, stroke, top, glyphs.height, item.char_width * s)
            width = glyphs.width
        elif item.max_width and width > item.max_width * s:
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

    @staticmethod
    def _monospaced(
        text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, stroke: float, top: float, height: int, cell: float
    ) -> Image.Image:
        """Each glyph centered in its own `cell` px, narrowed to fit, as the monospaced edge faces of cine stock."""
        row = Image.new("L", (max(1, int(round(cell * len(text)))), height), 0)
        for i, char in enumerate(text):
            if char.isspace():
                continue
            left_edge, _top, right_edge, _bottom = font.getbbox(char, anchor="ls", stroke_width=stroke)
            glyph = Image.new("L", (max(1, int(math.ceil(right_edge - left_edge))), height), 0)
            ImageDraw.Draw(glyph).text((-left_edge, -top), char, font=font, fill=255, anchor="ls", stroke_width=stroke, stroke_fill=255)
            if glyph.width > 0.85 * cell:
                glyph = glyph.resize((max(1, int(0.85 * cell)), height), Image.Resampling.BOX)
            row.paste(glyph, (int(round((i + 0.5) * cell - glyph.width / 2)), 0))
        return row

    def _marks(self, item: EdgeText, left: float, right: float, center_y: float, cap_px: float) -> None:
        gap = MARK_GAP * cap_px
        if item.lead:
            self._mark(item.lead, right_edge=left - gap, center_y=center_y, cap_px=cap_px)
        if item.trail:
            size = KEY_DOT * self.s if item.trail == "dot" else 0.9 * cap_px
            self._mark(item.trail, right_edge=right + gap + size, center_y=center_y, cap_px=cap_px)
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
        if kind in ("triangle", "wedge"):
            back = right_edge - size * (1.75 if kind == "wedge" else 1.0)
            self.draw.polygon([(right_edge, center_y), (back, center_y - half), (back, center_y + half)], fill=255)
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
        elif kind == "dot":
            radius = KEY_DOT * self.s / 2
            cx = right_edge - radius
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
    for (near, far), bits in code.tracks:
        rows = sorted((band.y_of(near), band.y_of(far)))
        r0, r1 = max(0, int(round(rows[0]))), min(coverage.shape[0], int(round(rows[1])))
        if r1 <= r0:
            continue
        if draft:
            column = np.full(c1 - c0, sum(bits) / len(bits), np.float32)
        else:
            oversample = 8
            xs = (np.arange(c0 * oversample, c1 * oversample) + 0.5) / oversample
            module = np.floor((xs - x_px) / (code.module * s)).astype(int)
            valid = (module >= 0) & (module < len(bits))
            on = np.zeros_like(xs, dtype=np.float32)
            on[valid] = np.asarray(bits, np.float32)[module[valid]]
            column = on.reshape(-1, oversample).mean(axis=1)
        coverage[r0:r1, c0:c1] = np.maximum(coverage[r0:r1, c0:c1], column[None, :])


def paint_strip(
    canvas: np.ndarray,
    x: float,
    y: float,
    geometry: FilmGeometry,
    style: EdgeStyle,
    frame_count: int,
    roll_start: float,
    length: float,
    s: float,
    palette: Palette,
    draft: bool = False,
    numbers: Optional[Sequence[int]] = None,
) -> None:
    """Rebate, edge print and perforations of the roll from `roll_start` for `length` mm, its top-left corner at (x, y) mm.

    `s` is pixels per mm; `numbers` is each frame slot's place on the roll (see `edge_items`)."""
    height, width = canvas.shape[:2]
    x0, y0 = px(x, s), px(y, s)
    x1, y1 = px(x + length, s), px(y + geometry.width, s)
    cx0, cy0, cx1, cy1 = clip_box(x0, y0, x1, y1, width, height)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    canvas[cy0:cy1, cx0:cx1] = palette.rebate

    def roll_topx(roll_x: float) -> float:
        return (roll_x - roll_start) * s

    strip_w = x1 - x0
    items = edge_items(geometry, style, frame_count, roll_start, roll_start + length, numbers)
    for band_kind in (Band.TOP, Band.BOTTOM):
        band_mm = (
            max(geometry.edge_band, DX_BAND) if (band_kind == Band.BOTTOM and geometry.perforated and style.dx) else geometry.edge_band
        )
        band = _Band(strip_w, band_mm, s, band_kind)
        codes: list[EdgeCode] = []
        for item in items:
            if isinstance(item, EdgeCode):
                if band_kind == Band.BOTTOM:
                    codes.append(item)
            elif item.band == band_kind:
                band.text(item, roll_topx(item.x))
        coverage = band.coverage()
        for code in codes:
            _draw_code(coverage, band, code, roll_topx(code.x), draft)
        band_top = y0 if band_kind == Band.TOP else y1 - coverage.shape[0]
        bx0, by0, bx1, by1 = clip_box(x0, band_top, x0 + coverage.shape[1], band_top + coverage.shape[0], width, height)
        if bx1 > bx0 and by1 > by0:
            alpha = coverage[by0 - band_top : by1 - band_top, bx0 - x0 : bx1 - x0]
            blend(canvas[by0:by1, bx0:bx1], alpha, palette.ink)

    if geometry.perforated:
        s_key = int(round(s * 100))
        hole, rim = _perforation_sprites(s_key, PERF_RIM)
        sh, sw = hole.shape
        centers_y = (PERF_FROM_EDGE + PERF_ACROSS / 2, geometry.width - PERF_FROM_EDGE - PERF_ACROSS / 2)
        for roll_x in perforation_centers(roll_start, roll_start + length):
            px_center = x0 + roll_topx(roll_x)
            for center_y in centers_y:
                py_center = (y + center_y) * s
                hx0 = int(round(px_center - sw / 2))
                hy0 = int(round(py_center - sh / 2))
                bx0, by0, bx1, by1 = clip_box(hx0, hy0, hx0 + sw, hy0 + sh, width, height)
                if bx1 <= bx0 or by1 <= by0:
                    continue
                crop = (slice(by0 - hy0, by1 - hy0), slice(bx0 - hx0, bx1 - hx0))
                blend(canvas[by0:by1, bx0:bx1], hole[crop], palette.no_film)
                if palette.rim is not None:
                    blend(canvas[by0:by1, bx0:bx1], rim[crop], palette.rim)
