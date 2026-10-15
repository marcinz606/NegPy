import numpy as np
import pytest
from negpy.services.export.contact_sheet import PERF_RING, ContactSheetService, label_caps, palette_for
from negpy.services.export.contact_sheet_edge import EdgeFamily, EdgeStyle
from negpy.services.export.contact_sheet_layout import (
    DEFAULT_PAPER,
    FRAME_SIZES_120,
    PERF_ACROSS,
    PERF_FROM_EDGE,
    SheetFormat,
    film_geometry,
    perforation_centers,
    plan_sheets,
)
from negpy.services.export.contact_sheet_roll import SheetLook

DPI = 150
S = DPI / 25.4
FULL = film_geometry(SheetFormat.FULL_FRAME)


def _render(n=6, tiles=None, turns=None, look=None, geometry=FULL, label=False, draft=False):
    plan = plan_sheets(DEFAULT_PAPER.width, DEFAULT_PAPER.height, geometry, n, label=label)
    tiles = tiles if tiles is not None else [np.full((200, 300, 3), 128, np.uint8)] * n
    turns = turns if turns is not None else [0] * n
    look = look or SheetLook("bw", 0, EdgeStyle(EdgeFamily.KODAK, "KODAK TRI-X 400", False))
    return plan, ContactSheetService.render_sheet(plan, 0, tiles, turns, S, look, draft=draft)


def _px(mm):
    return int(round(mm * S))


def test_sheet_is_the_paper_at_the_given_resolution():
    _plan, sheet = _render()
    assert sheet.shape == (_px(DEFAULT_PAPER.height), _px(DEFAULT_PAPER.width), 3)
    assert sheet.dtype == np.uint8


@pytest.mark.parametrize("palette", ["bw", "color", "slide"])
def test_paper_prints_the_no_film_tone(palette):
    look = SheetLook(palette, 0, EdgeStyle())
    _plan, sheet = _render(look=look)
    assert tuple(sheet[3, 3]) == palette_for(look).no_film


@pytest.mark.parametrize("palette", ["bw", "color", "slide"])
def test_white_paper_prints_white_with_a_dark_label(palette):
    look = SheetLook(palette, 0, EdgeStyle(), white_paper=True)
    pal = palette_for(look)
    assert pal.no_film == (255, 255, 255)
    assert pal.label == (0, 0, 0)
    assert pal.rebate == palette_for(SheetLook(palette, 0, EdgeStyle())).rebate
    _plan, sheet = _render(look=look)
    assert tuple(sheet[3, 3]) == (255, 255, 255)


def test_rebate_prints_just_above_the_paper_black():
    plan, sheet = _render()
    strip = plan.pages[0].strips[0]
    # Between the top perforation row (ends 4.8 mm in) and the image (starts 5.5 mm in).
    y = _px(strip.y + 5.15)
    x = _px(strip.x + FULL.frame_center(0) - strip.roll_start)
    assert tuple(sheet[y, x]) == (5, 5, 5)


def test_perforations_print_the_no_film_tone():
    plan, sheet = _render()
    strip = plan.pages[0].strips[0]
    center_x = perforation_centers(strip.roll_start, strip.roll_start + strip.length)[3]
    y = _px(strip.y + PERF_FROM_EDGE + PERF_ACROSS / 2)
    assert tuple(sheet[y, _px(strip.x + center_x)]) == (0, 0, 0)


def test_frames_fill_their_window():
    plan, sheet = _render()
    strip = plan.pages[0].strips[0]
    x = _px(strip.x + FULL.frame_center(1))
    y = _px(strip.y + FULL.width / 2)
    assert tuple(sheet[y, x]) == (128, 128, 128)


def test_a_missing_tile_prints_as_unexposed_film():
    plan, sheet = _render(tiles=[None] * 6)
    strip = plan.pages[0].strips[0]
    assert tuple(sheet[_px(strip.y + FULL.width / 2), _px(strip.x + FULL.frame_center(0))]) == (5, 5, 5)


def test_a_turned_tile_lies_sideways_top_to_the_right():
    tile = np.zeros((300, 200, 3), np.uint8)
    tile[:150] = (255, 0, 0)
    tile[150:] = (0, 0, 255)
    plan, sheet = _render(n=1, tiles=[tile], turns=[3])
    strip = plan.pages[0].strips[0]
    y = _px(strip.y + FULL.width / 2)
    left = _px(strip.x + FULL.frame_center(0) - 9)
    right = _px(strip.x + FULL.frame_center(0) + 9)
    assert tuple(sheet[y, right]) == (255, 0, 0)
    assert tuple(sheet[y, left]) == (0, 0, 255)


def test_a_different_shape_is_fitted_whole():
    square = np.full((300, 300, 3), 200, np.uint8)
    plan, sheet = _render(n=1, tiles=[square])
    strip = plan.pages[0].strips[0]
    y = _px(strip.y + FULL.width / 2)
    assert tuple(sheet[y, _px(strip.x + FULL.frame_center(0) - 16)]) == (5, 5, 5)
    assert tuple(sheet[y, _px(strip.x + FULL.frame_center(0))]) == (200, 200, 200)


def test_edge_print_prints_light_in_its_band():
    plan, sheet = _render()
    strip = plan.pages[0].strips[0]
    band = sheet[_px(strip.y) : _px(strip.y + 2.0), _px(strip.x) : _px(strip.x + strip.length)]
    assert band.max() > 150


def test_dx_codes_print_in_the_bottom_band():
    look = SheetLook("color", 0, EdgeStyle(EdgeFamily.KODAK, "KODAK PORTRA 400", True))
    plan, sheet = _render(look=look)
    strip = plan.pages[0].strips[0]
    x0 = _px(strip.x + FULL.frame_center(0) + 3)
    x1 = _px(strip.x + FULL.frame_center(0) + 13)
    track = sheet[_px(strip.y + FULL.width - 0.5), x0:x1, 0]
    assert track.max() > 150 and track.min() < 60


def test_label_prints_above_the_strips_in_the_edge_ink():
    look = SheetLook("color", 0, EdgeStyle(EdgeFamily.KODAK, "KODAK GOLD 200", True), "ROLL 12 · Gold 200 · 2026-08-11")
    plan, sheet = _render(look=look, label=True)
    _x, block_y, _w, _h = plan.pages[0].block
    band = sheet[_px(block_y - 8) : _px(block_y - 2), :].reshape(-1, 3)
    brightest = band[band.sum(axis=1).argmax()]
    assert tuple(brightest) == palette_for(look).ink


def test_label_is_set_in_edge_print_capitals():
    assert label_caps("Roll 12 · Köln · Kodak Gold 200") == "ROLL 12 · KOLN · KODAK GOLD 200"


def test_plain_film_leaves_the_edge_bands_empty():
    look = SheetLook("bw", 0, EdgeStyle(EdgeFamily.KODAK, "KODAK TRI-X 400", False, printed=False))
    plan, sheet = _render(look=look)
    strip = plan.pages[0].strips[0]
    for top in (strip.y, strip.y + FULL.width - 2.0):
        band = sheet[_px(top) + 1 : _px(top + 2.0) - 1, _px(strip.x) : _px(strip.x + strip.length)]
        assert band.max() <= 10


def test_draft_render_matches_the_layout():
    _plan, full = _render()
    _plan, draft = _render(draft=True)
    assert full.shape == draft.shape


def test_120_has_no_perforations():
    geometry = film_geometry(SheetFormat.MEDIUM, "6×6")
    look = SheetLook("bw", 0, EdgeStyle())
    plan, sheet = _render(n=3, geometry=geometry, tiles=[np.full((300, 300, 3), 128, np.uint8)] * 3, look=look)
    strip = plan.pages[0].strips[0]
    # Where a 135 hole row would start; 120's rebate is 2.5 mm deep and has no stock text here.
    rebate = sheet[_px(strip.y + 2.2), _px(strip.x + 2) : _px(strip.x + strip.length - 2)]
    assert (rebate == 5).all()


def test_a_paper_film_base_prints_white_with_dark_ink_and_rings():
    look = SheetLook("bw", 0, EdgeStyle(EdgeFamily.KODAK, "KODAK TRI-X 400", False), white_paper=True, film_base=False)
    pal = palette_for(look)
    assert pal.no_film == pal.rebate == (255, 255, 255)
    assert pal.ink == pal.label == (0, 0, 0)
    plan, sheet = _render(look=look)
    strip = plan.pages[0].strips[0]
    # The rebate between the perforation row and the image, between two perforations: paper.
    centers = perforation_centers(strip.roll_start, strip.roll_start + strip.length)
    between_x = _px(strip.x + (centers[3] + centers[4]) / 2)
    rebate_y = _px(strip.y + PERF_FROM_EDGE + PERF_ACROSS + 0.35)
    assert tuple(sheet[rebate_y, between_x]) == (255, 255, 255)
    # The perforation's edge: a ring darker than the paper, inside its hole.
    ring_y = _px(strip.y + PERF_FROM_EDGE + PERF_RING / 2)
    hole_y = _px(strip.y + PERF_FROM_EDGE + PERF_ACROSS / 2)
    center_x = _px(strip.x + centers[3])
    assert sheet[ring_y, center_x].max() < 200
    assert tuple(sheet[hole_y, center_x]) == (255, 255, 255)


def test_white_paper_keeps_the_film_black_until_the_base_is_turned_off():
    with_base = palette_for(SheetLook("bw", 0, EdgeStyle(), white_paper=True))
    assert with_base.rebate == palette_for(SheetLook("bw", 0, EdgeStyle())).rebate


def test_a_paper_film_base_keeps_the_strip_outline_and_a_draft_ring():
    look = SheetLook("bw", 0, EdgeStyle(), white_paper=True, film_base=False)
    plan, sheet = _render(look=look)
    strip = plan.pages[0].strips[0]
    # The strip's top edge is a line on the paper; the darkroom look paints no outline.
    assert sheet[_px(strip.y), _px(strip.x + 1.0)].max() < 200
    film_plan, film_sheet = _render()
    assert tuple(film_sheet[_px(strip.y), _px(strip.x + 1.0)]) == palette_for(SheetLook("bw", 0, EdgeStyle())).rebate
    # At a draft scale the ring is still at least a pixel.
    draft_scale = 2.0
    draft = ContactSheetService.render_sheet(plan, 0, [np.full((200, 300, 3), 128, np.uint8)] * 6, [0] * 6, draft_scale, look, draft=True)
    centers = perforation_centers(strip.roll_start, strip.roll_start + strip.length)
    ring_y = int(round((strip.y + PERF_FROM_EDGE + 0.25) * draft_scale))
    assert draft[ring_y, int(round((strip.x + centers[3]) * draft_scale))].max() < 255


_PAPER_LOOK = SheetLook("bw", 0, EdgeStyle(EdgeFamily.KODAK, "KODAK TRI-X 400", False), white_paper=True, film_base=False)
_EVERY_FORMAT = [film_geometry(SheetFormat.FULL_FRAME), film_geometry(SheetFormat.HALF_FRAME)] + [
    film_geometry(SheetFormat.MEDIUM, size) for size in FRAME_SIZES_120
]


@pytest.mark.parametrize("geometry", _EVERY_FORMAT, ids=lambda geo: geo.frame_size or geo.format.value)
@pytest.mark.parametrize("scale, draft", [(300 / 25.4, False), (2.0, True)], ids=["print", "draft"])
def test_a_paper_film_base_outlines_the_strip_on_every_format(geometry, scale, draft):
    plan = plan_sheets(DEFAULT_PAPER.width, DEFAULT_PAPER.height, geometry, 6, label=False)
    tiles = [np.full((200, 300, 3), 128, np.uint8)] * 6
    sheet = ContactSheetService.render_sheet(plan, 0, tiles, [0] * 6, scale, _PAPER_LOOK, draft=draft)
    strip = plan.pages[0].strips[0]

    def px(mm):
        return int(round(mm * scale))

    x0, y0 = px(strip.x), px(strip.y)
    x1, y1 = px(strip.x + strip.length) - 1, px(strip.y + geometry.width) - 1
    mid_y = (y0 + y1) // 2
    ring = palette_for(_PAPER_LOOK).rim
    # The film's four edges are lines in the ring tone, and the rebate inside them is paper.
    assert tuple(sheet[y0, x0]) == tuple(sheet[y1, x1]) == tuple(sheet[mid_y, x0]) == tuple(sheet[mid_y, x1]) == ring
    line = max(1, px(PERF_RING))
    assert tuple(sheet[y0 + line, x0 + line]) == (255, 255, 255)
