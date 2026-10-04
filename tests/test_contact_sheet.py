import numpy as np
import pytest
from negpy.services.export.contact_sheet import ContactSheetService, label_caps, palette_for
from negpy.services.export.contact_sheet_edge import EdgeFamily, EdgeStyle
from negpy.services.export.contact_sheet_layout import (
    DEFAULT_PAPER,
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
