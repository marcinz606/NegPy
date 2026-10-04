import math

import pytest

from negpy.services.export.contact_sheet_layout import (
    DEFAULT_PAPER,
    FRAME_SIZES_120,
    ILFORD_PAPERS,
    PAPER_MAX,
    PERF_ALONG,
    SHEET_MARGIN,
    ContactSheetSettings,
    SheetFormat,
    best_dpi,
    better_orientation,
    clamp_paper,
    cut_strips,
    dpi_allowed,
    film_geometry,
    frame_turns,
    min_paper_size,
    paper_preset,
    perforation_centers,
    plan_capacity,
    plan_sheets,
    sheet_pixels,
    snap_paper,
)

FULL = film_geometry(SheetFormat.FULL_FRAME)
HALF = film_geometry(SheetFormat.HALF_FRAME)


def test_default_paper_is_ilford_9_5_x_12_in_true_size():
    assert DEFAULT_PAPER.label == "24 × 30.5 cm"
    assert (DEFAULT_PAPER.width, DEFAULT_PAPER.height) == (241.3, 304.8)
    assert sheet_pixels(DEFAULT_PAPER.width, DEFAULT_PAPER.height, 300) == (2850, 3600)


def test_default_paper_holds_a_36_exposure_roll_with_the_label():
    assert plan_capacity(DEFAULT_PAPER.width, DEFAULT_PAPER.height, FULL, label=True) == (6, 7)
    plan = plan_sheets(DEFAULT_PAPER.width, DEFAULT_PAPER.height, FULL, 38)
    assert len(plan.pages) == 1
    assert [s.count for s in plan.pages[0].strips] == [6, 6, 6, 6, 6, 6, 2]


def test_default_paper_turned_holds_35_so_portrait_is_the_default():
    assert plan_capacity(DEFAULT_PAPER.height, DEFAULT_PAPER.width, FULL, label=True) == (7, 5)


def test_default_paper_holds_72_half_frames_in_strips_of_12():
    assert plan_capacity(DEFAULT_PAPER.width, DEFAULT_PAPER.height, HALF) == (12, 7)
    assert len(plan_sheets(DEFAULT_PAPER.width, DEFAULT_PAPER.height, HALF, 72).pages) == 1


@pytest.mark.parametrize(
    "size, portrait, roll",
    [("6×4.5", (5, 4), 16), ("6×6", (3, 4), 12), ("6×7", (3, 4), 10), ("6×9", (2, 4), 8), ("6×17", (1, 4), 4)],
)
def test_120_rolls_on_the_default_paper(size, portrait, roll):
    geometry = film_geometry(SheetFormat.MEDIUM, size)
    assert plan_capacity(DEFAULT_PAPER.width, DEFAULT_PAPER.height, geometry) == portrait
    assert portrait[0] * portrait[1] >= roll


@pytest.mark.parametrize("size, roll", [("6×8", 9), ("6×12", 6)])
def test_6x8_and_6x12_need_the_paper_turned(size, roll):
    geometry = film_geometry(SheetFormat.MEDIUM, size)
    per, strips = plan_capacity(DEFAULT_PAPER.width, DEFAULT_PAPER.height, geometry)
    assert per * strips < roll
    turned = better_orientation(DEFAULT_PAPER.width, DEFAULT_PAPER.height, geometry, roll)
    assert turned == (DEFAULT_PAPER.height, DEFAULT_PAPER.width)
    per, strips = plan_capacity(*turned, geometry)
    assert per * strips >= roll


def test_8x10_holds_30_frames():
    assert math.prod(plan_capacity(203.2, 254.0, FULL)) == 30


def test_frame_geometry():
    assert (FULL.frame_along, FULL.frame_across, FULL.pitch, FULL.width) == (36.0, 24.0, 38.0, 35.0)
    assert (HALF.frame_along, HALF.frame_across, HALF.pitch) == (18.0, 24.0, 19.0)
    six_seven = film_geometry(SheetFormat.MEDIUM, "6×7")
    assert (six_seven.frame_along, six_seven.frame_across, six_seven.width, six_seven.pitch) == (70.0, 56.0, 61.0, 74.0)
    assert set(FRAME_SIZES_120) == {"6×4.5", "6×6", "6×7", "6×8", "6×9", "6×12", "6×17"}
    assert film_geometry(SheetFormat.MEDIUM, "nonsense").frame_size == "6×6"


def test_block_is_centered_and_a_short_strip_starts_at_its_left_edge():
    plan = plan_sheets(DEFAULT_PAPER.width, DEFAULT_PAPER.height, FULL, 38)
    page = plan.pages[0]
    x, y, w, h = page.block
    assert x == pytest.approx((DEFAULT_PAPER.width - w) / 2)
    assert w == pytest.approx(6 * 38.0)
    assert y + h / 2 == pytest.approx(DEFAULT_PAPER.height / 2)
    assert all(strip.x == pytest.approx(x) for strip in page.strips)
    assert page.strips[-1].length == pytest.approx(2 * 38.0)
    assert [s.first for s in page.strips] == [0, 6, 12, 18, 24, 30, 36]
    assert [s.roll_start for s in page.strips] == [i * 6 * 38.0 for i in range(7)]


def test_a_single_short_strip_is_centered_on_its_own_length():
    page = plan_sheets(DEFAULT_PAPER.width, DEFAULT_PAPER.height, FULL, 3).pages[0]
    x, _y, w, _h = page.block
    assert w == pytest.approx(3 * 38.0)
    assert x == pytest.approx((DEFAULT_PAPER.width - w) / 2)


def test_a_full_block_leaves_the_label_band_clear():
    plan = plan_sheets(DEFAULT_PAPER.width, 263.0, FULL, 42)
    _x, y, _w, _h = plan.pages[0].block
    assert y >= SHEET_MARGIN + 8.0 - 1e-9


def test_pages_when_the_roll_does_not_fit():
    plan = plan_sheets(203.2, 254.0, FULL, 38)
    assert plan.capacity == 30
    assert [sum(s.count for s in page.strips) for page in plan.pages] == [30, 8]
    assert plan.pages[1].strips[0].first == 30


def test_zero_capacity_has_no_pages_and_says_why():
    plan = plan_sheets(127.0, 177.8, film_geometry(SheetFormat.MEDIUM, "6×17"), 4)
    assert plan.pages == ()
    assert "narrow" in plan.reason


def test_no_perforation_is_cut_by_a_strip_end():
    for geometry, per_strip in ((FULL, 6), (HALF, 12), (FULL, 5), (HALF, 7)):
        for strip in range(6):
            cut = strip * per_strip * geometry.pitch
            for center in perforation_centers(cut - 10, cut + 10):
                assert abs(center - cut) >= PERF_ALONG / 2 + 1.0


def test_perforations_are_symmetric_about_each_frame():
    for i in range(3):
        center = FULL.frame_center(i)
        holes = perforation_centers(center - 19, center + 19)
        assert len(holes) == 8
        assert sum(h - center for h in holes) == pytest.approx(0.0)


def test_snap_paper_finds_ilford_sizes_in_either_orientation():
    assert snap_paper(242.5, 303.9) == (241.3, 304.8)
    assert snap_paper(303.9, 242.5) == (304.8, 241.3)
    assert snap_paper(250.4, 280.6) == (250.0, 281.0)
    assert paper_preset(304.8, 241.3) == DEFAULT_PAPER
    assert paper_preset(250.0, 281.0) is None


def test_paper_is_clamped_to_one_frame_and_the_largest_sheet():
    min_w, min_h = min_paper_size(film_geometry(SheetFormat.MEDIUM, "6×17"))
    assert clamp_paper(10, 10, film_geometry(SheetFormat.MEDIUM, "6×17")) == (min_w, min_h)
    assert clamp_paper(9999, 9999, FULL) == (PAPER_MAX, PAPER_MAX)


def test_dpi_stays_inside_the_pixel_budget():
    assert dpi_allowed(304.8, 406.4, 600)
    assert not dpi_allowed(406.4, 508.0, 600)
    assert best_dpi(406.4, 508.0, 600) == 300
    assert best_dpi(DEFAULT_PAPER.width, DEFAULT_PAPER.height, 600) == 600


class TestFrameTurns:
    def test_matching_orientation_never_turns(self):
        assert frame_turns(1.5, FULL, 0, False, False) == 0
        assert frame_turns(0.75, HALF, 0, False, False) == 0

    def test_square_film_or_picture_never_turns(self):
        assert frame_turns(0.67, film_geometry(SheetFormat.MEDIUM, "6×6"), 1, False, False) == 0
        assert frame_turns(1.0, FULL, 0, False, False) == 0
        assert frame_turns(None, FULL, 1, False, False) == 0

    def test_a_frame_turned_upright_is_turned_back(self):
        # rot90(k=1) made it upright; rot90(k=3) undoes it.
        assert frame_turns(0.67, FULL, 1, False, False) == 3
        assert frame_turns(0.67, FULL, 3, False, False) == 1

    def test_one_mirror_reverses_the_undo(self):
        assert frame_turns(0.67, FULL, 1, True, False) == 1
        assert frame_turns(0.67, FULL, 1, True, True) == 3

    def test_a_vertical_scanned_upright_goes_clockwise(self):
        assert frame_turns(0.67, FULL, 0, False, False) == 3
        assert frame_turns(1.33, HALF, 2, False, False) == 3

    def test_a_vertical_strip_scan_turned_landscape_stays(self):
        assert frame_turns(1.5, FULL, 1, False, False) == 0


class TestSettings:
    def test_round_trip(self):
        settings = ContactSheetSettings(250.0, 300.0, 600, False, False)
        assert ContactSheetSettings.from_dict(settings.to_dict()) == settings

    def test_edge_print_defaults_on_for_records_without_it(self):
        assert ContactSheetSettings.from_dict({"dpi": 600}).edge_print is True

    def test_stale_or_bad_records_fall_back(self):
        assert ContactSheetSettings.from_dict(None) == ContactSheetSettings()
        restored = ContactSheetSettings.from_dict(
            {"paper_width": "wide", "paper_height": 99999, "dpi": 72, "roll_label": 1, "cell_px": 600}
        )
        assert restored.paper_width == DEFAULT_PAPER.width
        assert restored.paper_height == PAPER_MAX
        assert restored.dpi == 300
        assert restored.roll_label is True

    def test_every_ilford_size_is_portrait(self):
        assert all(p.width <= p.height for p in ILFORD_PAPERS)


def test_a_new_scene_starts_a_new_strip():
    assert cut_strips(10, 6, breaks=[4]) == [(0, 4), (4, 6)]
    assert cut_strips(14, 6, breaks=[3, 5]) == [(0, 3), (3, 2), (5, 6), (11, 3)]
    assert cut_strips(8, 6, breaks=[0, 8, 99]) == [(0, 6), (6, 2)]
    plan = plan_sheets(DEFAULT_PAPER.width, DEFAULT_PAPER.height, FULL, 10, breaks=[4])
    strips = plan.pages[0].strips
    assert [(s.first, s.count) for s in strips] == [(0, 4), (4, 6)]
    assert plan.pages[0].block[2] == pytest.approx(6 * 38.0)
    assert strips[0].x == strips[1].x


def test_scene_breaks_paginate_by_strips():
    plan = plan_sheets(DEFAULT_PAPER.width, DEFAULT_PAPER.height, FULL, 16, breaks=list(range(1, 16)))
    assert [len(page.strips) for page in plan.pages] == [7, 7, 2]
