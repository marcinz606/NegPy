"""The film maker's edge print, laid out in roll millimetres.

135 carries a label every half frame on the bottom edge (`N`, `NA`), each followed by a DX
barcode on DX films, and the stock name or full-frame numbers on the top edge (ISO 1007). 120
has no perforations; its numbers run at the maker's own pitch, not in step with the camera.
Positions are along the roll; `offset` is the distance of an item's center from its film edge.
"""

import unicodedata
import zlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional, Sequence, Union

from negpy.services.export.contact_sheet_layout import HALF_FRAME_PITCH, FilmGeometry, SheetFormat

LABEL_PITCH_135 = HALF_FRAME_PITCH
DX_MODULE = 0.406
DX_MODULES = 31
DX_START = 1.9
DX_DATA_TRACK = (0.0, 1.0)
DX_CLOCK_TRACK = (1.0, 2.2)
# A label group ends this far left of its label's own position, so it clears the code's quiet zone.
DX_LABEL_END = 1.5


class EdgeFamily(StrEnum):
    KODAK = "kodak"
    ILFORD = "ilford"
    FUJI = "fuji"
    GENERIC = "generic"
    CINE = "cine"


class Band(StrEnum):
    TOP = "top"
    BOTTOM = "bottom"


@dataclass(frozen=True)
class EdgeText:
    """One word of edge print. `lead`/`trail` draw a mark before/after it, `under` beneath it
    ("triangle", "outline", "arrow", "dots")."""

    x: float
    band: Band
    text: str
    cap: float
    offset: float
    align: str = "center"
    max_width: float = 0.0
    lead: str = ""
    trail: str = ""
    under: str = ""


@dataclass(frozen=True)
class EdgeCode:
    """A DX film edge barcode starting at `x`; bits read left to right, 1 prints light."""

    x: float
    clock: tuple[int, ...]
    data: tuple[int, ...]

    @property
    def length(self) -> float:
        return DX_MODULES * DX_MODULE


EdgeItem = Union[EdgeText, EdgeCode]

_CINE_WORDS = ("vision", "cinestill", "double-x", "eastman", "eterna", "5207", "5219", "5222", "5203", "5213", "7222")
_ILFORD_MAKERS = ("ILFORD", "KENTMERE", "HARMAN")
_FUJI_MAKERS = ("FUJIFILM", "FUJICOLOR", "FUJICHROME", "FUJI")


def ascii_upper(text: str) -> str:
    """Edge print is plain capitals; accents are folded so every glyph exists in the face."""
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(folded.upper().split())


def edge_family(stock: str, manufacturer: str = "") -> EdgeFamily:
    text = f"{manufacturer} {stock}".lower()
    if any(word in text for word in _CINE_WORDS):
        return EdgeFamily.CINE
    if "kodak" in text:
        return EdgeFamily.KODAK
    if any(maker.lower() in text for maker in _ILFORD_MAKERS):
        return EdgeFamily.ILFORD
    if "fuji" in text:
        return EdgeFamily.FUJI
    return EdgeFamily.GENERIC


@dataclass(frozen=True)
class EdgeStyle:
    family: EdgeFamily = EdgeFamily.GENERIC
    stock: str = ""
    dx: bool = False
    # False prints plain film: the rebate and perforations without any edge print.
    printed: bool = True

    def _split_maker(self, makers: tuple[str, ...]) -> tuple[str, str]:
        words = self.stock.split()
        if words and words[0] in makers:
            return words[0], " ".join(words[1:])
        return makers[0], self.stock

    @property
    def maker_word(self) -> str:
        return self._split_maker(_ILFORD_MAKERS)[0]

    @property
    def product(self) -> str:
        return self._split_maker(_ILFORD_MAKERS)[1]

    @property
    def fuji_code(self) -> str:
        return self._split_maker(_FUJI_MAKERS)[1]

    @property
    def dx_product(self) -> tuple[int, int]:
        """A stable stand-in for the stock's DX product number (never 0, which readers reject)."""
        digest = zlib.crc32(self.stock.encode("ascii", "ignore"))
        return 1 + digest % 127, (digest >> 8) % 16


def label_text(index: int) -> str:
    """The `index`-th half-frame label of a roll: 1, 1A, 2, 2A, …"""
    return f"{index // 2 + 1}{'A' if index % 2 else ''}"


def dx_bits(number: int, half: bool, product: tuple[int, int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(clock, data) tracks of a 31-module DX film edge barcode (ISO 1007, zxing layout)."""
    p1, p2 = product

    def bits(value: int, count: int) -> list[int]:
        return [(value >> shift) & 1 for shift in range(count - 1, -1, -1)]

    payload = [0, *bits(p1, 7), 0, *bits(p2, 4), *bits(number % 64, 6), int(half), 0]
    parity = sum(payload) % 2
    data = (1, 0, 1, 0, 1, *payload, parity, 0, 1, 0, 1)
    clock = (1, 1, 1, 1, 1, *[i % 2 for i in range(23)], 1, 1, 1)
    return clock, data


def edge_items(
    geometry: FilmGeometry,
    style: EdgeStyle,
    frame_count: int,
    x0: float,
    x1: float,
    numbers: Optional[Sequence[int]] = None,
) -> list[EdgeItem]:
    """Every edge-print item that can touch roll positions x0..x1 (items straddling a cut are
    returned too; the strip clips them).

    `numbers` gives each frame slot its place on the original roll, so a frame keeps its own
    edge numbers when others are left out of the sheet; by default the slots run 0, 1, 2 …
    """
    if not style.printed:
        return []
    if geometry.format == SheetFormat.MEDIUM:
        items = _items_120(style, frame_count * geometry.pitch)
    else:
        slots = list(numbers) if numbers is not None else list(range(frame_count))
        items = _items_135(geometry, style, slots[:frame_count])
    reach = 45.0
    return [item for item in items if x0 - reach <= item.x <= x1 + reach]


def _label_slots(geometry: FilmGeometry, slots: Sequence[int]) -> list[tuple[float, int]]:
    """(roll position, label index) of every half-frame label: one per half frame, two per
    full frame (its number over the frame, the A number in the gap after it)."""
    if geometry.format == SheetFormat.HALF_FRAME:
        return [(geometry.frame_center(i), roll_index) for i, roll_index in enumerate(slots)]
    labels: list[tuple[float, int]] = []
    for i, roll_index in enumerate(slots):
        x = geometry.frame_center(i)
        labels += [(x, 2 * roll_index), (x + LABEL_PITCH_135, 2 * roll_index + 1)]
    return labels


def _items_135(geometry: FilmGeometry, style: EdgeStyle, slots: Sequence[int]) -> list[EdgeItem]:
    frame_count = len(slots)
    family = style.family
    items: list[EdgeItem] = []
    top, bottom = Band.TOP, Band.BOTTOM

    for x, j in _label_slots(geometry, slots):
        number = j // 2 + 1
        is_a = bool(j % 2)
        text = label_text(j)

        if family != EdgeFamily.CINE:
            anchor, align = (x + DX_LABEL_END, "right") if style.dx else (x, "center")
            if not is_a:
                cap = 1.4 if family in (EdgeFamily.KODAK, EdgeFamily.GENERIC) else 1.5
                items.append(EdgeText(anchor, bottom, text, cap, 1.0, align))
            elif family == EdgeFamily.KODAK:
                items.append(EdgeText(anchor, bottom, text, 0.9, 1.4, align, under="arrow"))
            elif family == EdgeFamily.FUJI:
                items.append(EdgeText(anchor, bottom, text, 1.5, 1.0, align, lead="arrow"))
            else:
                items.append(EdgeText(anchor, bottom, text, 1.1, 1.0, align, lead="triangle"))
            if style.dx:
                clock, data = dx_bits(number, is_a, style.dx_product)
                items.append(EdgeCode(x + DX_START, clock, data))

        if is_a:
            continue
        if family in (EdgeFamily.KODAK, EdgeFamily.GENERIC):
            items.append(EdgeText(x, top, str(number), 1.2, 1.0))
            if style.stock:
                items.append(EdgeText(x + 15.5, top, style.stock, 1.25, 1.0, max_width=26.0))
        elif family == EdgeFamily.FUJI:
            items.append(EdgeText(x, top, str(number), 1.5, 1.0))
            if style.fuji_code:
                items.append(EdgeText(x + 9.5, top, style.fuji_code, 1.0, 1.0, max_width=7.0))
            items.append(EdgeText(x + LABEL_PITCH_135, top, f"{number}A", 1.5, 1.0))
            items.append(EdgeText(x + 28.5, top, "FUJI", 1.0, 1.0))

    roll_end = frame_count * geometry.pitch
    if family == EdgeFamily.ILFORD:
        x = 12.0
        while x < roll_end + 48.0:
            items.append(EdgeText(x, top, style.maker_word, 1.6, 1.0))
            if style.product:
                items.append(EdgeText(x + 48.0, top, style.product, 1.6, 1.0, max_width=40.0))
            x += 96.0
    elif family == EdgeFamily.CINE and style.stock:
        x = 12.0
        while x < roll_end + 76.0:
            items.append(EdgeText(x, top, style.stock, 1.2, 1.0, max_width=60.0))
            x += 76.0
    return items


def _items_120(style: EdgeStyle, roll_end: float) -> list[EdgeItem]:
    cap, offset = 1.8, 1.3
    top, bottom = Band.TOP, Band.BOTTOM
    family = style.family
    items: list[EdgeItem] = []

    def run(start: float, pitch: float, count: int):
        for k in range(count):
            x = start + pitch * k
            if x > roll_end + pitch:
                return
            yield k, x

    if family == EdgeFamily.KODAK:
        for k, x in run(8.0, 44.5, 15):
            items.append(EdgeText(x, top, str(41 + k), cap, offset))
            if style.stock and k % 2 == 0:
                items.append(EdgeText(x + 22.25, top, style.stock, cap, offset, max_width=36.0))
        for k, x in run(20.0, 60.0, 12):
            items.append(EdgeText(x, bottom, str(1 + k), cap, offset, trail="triangle"))
        return items

    if family == EdgeFamily.FUJI:
        code = style.fuji_code or style.stock
        for _k, x in run(30.0, 84.0, 19):
            items.append(EdgeText(x, top, f"FUJI {code}".strip(), cap, offset, max_width=50.0))
        for k, x in run(8.0, 42.0, 19):
            if code:
                items.append(EdgeText(x - 1.6, bottom, code, cap, offset, "right", max_width=24.0))
            items.append(EdgeText(x, bottom, str(1 + k), cap, offset, "left", lead="outline"))
        return items

    if family == EdgeFamily.ILFORD:
        name = f"{style.maker_word} {style.product}".strip()
        lead = "dots"
    else:
        name = style.stock
        lead = "triangle"
    if name:
        for _k, x in run(30.0, 86.0, 19):
            items.append(EdgeText(x, top, name, cap, offset, max_width=60.0))
    for k, x in run(8.0, 43.0, 19):
        items.append(EdgeText(x, bottom, str(1 + k), cap, offset, lead=lead))
    return items
