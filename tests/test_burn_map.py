"""Printer's burn map: dodge/burn strengths in darkroom notation plus the frame's printing
instructions. The two things that can go quietly wrong are a label landing outside its own
concave mask, and a near-zero mask reading as real work."""

import numpy as np
import cv2

from negpy.features.exposure.densitometer import print_instructions, stops_label
from negpy.features.exposure.models import ExposureConfig
from negpy.features.local.logic import polygon_label_anchor
from negpy.features.local.models import PolygonMask

# A C shape: its vertex centroid lands in the notch, outside the polygon.
_C_SHAPE = [
    (0.0, 0.0),
    (10.0, 0.0),
    (10.0, 2.0),
    (3.0, 2.0),
    (3.0, 8.0),
    (10.0, 8.0),
    (10.0, 10.0),
    (0.0, 10.0),
]


def test_stops_label_uses_thirds_and_the_printers_sign():
    assert stops_label(0.0) == "0"
    assert stops_label(1.0) == "+1"
    assert stops_label(4 / 3) == "+1⅓"
    assert stops_label(5 / 3) == "+1⅔"
    assert stops_label(1 / 3) == "+⅓"
    assert stops_label(-2.0) == "-2"
    assert stops_label(-2 / 3) == "-⅔"
    assert stops_label(-1 / 3) == "-⅓"


def test_a_negligible_mask_reads_as_zero_not_a_third():
    # 0.05 EV rounds to 0 thirds — it must not print as "+⅓".
    assert stops_label(0.05) == "0"
    assert stops_label(-0.05) == "0"
    # The rounding boundary sits at half a third.
    assert stops_label(0.17) == "+⅓"


def test_the_centroid_is_used_when_it_lands_inside():
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert polygon_label_anchor(square) == (5.0, 5.0)


def test_a_concave_masks_label_lands_inside_the_mask():
    arr = np.asarray(_C_SHAPE, dtype=np.float32)
    cx, cy = float(arr[:, 0].mean()), float(arr[:, 1].mean())
    contour = arr.reshape(-1, 1, 2)
    # Precondition: the plain centroid really is outside, else this proves nothing.
    assert cv2.pointPolygonTest(contour, (cx, cy), False) < 0

    ax, ay = polygon_label_anchor(_C_SHAPE)
    assert cv2.pointPolygonTest(contour, (ax, ay), False) >= 0


def test_a_degenerate_polygon_falls_back_to_the_centroid():
    assert polygon_label_anchor([]) == (0.0, 0.0)
    assert polygon_label_anchor([(2.0, 4.0)]) == (2.0, 4.0)
    assert polygon_label_anchor([(0.0, 0.0), (4.0, 2.0)]) == (2.0, 1.0)


def test_the_card_lists_density_and_grade():
    lines = print_instructions(ExposureConfig(density=1.32, grade=115.0))
    assert lines == ["D 1.32   R 115"]


def test_the_split_line_appears_only_when_a_split_is_dialled_in():
    assert len(print_instructions(ExposureConfig())) == 1  # no split at defaults

    only_shadows = print_instructions(ExposureConfig(shadow_grade=20.0))
    assert only_shadows[1] == "split  shadows +20"

    both = print_instructions(ExposureConfig(shadow_grade=20.0, highlight_grade=-10.0))
    assert both[1] == "split  shadows +20  highlights -10"


def test_the_card_lists_each_mask_in_index_order_with_its_stops():
    masks = (
        PolygonMask(vertices=tuple(_C_SHAPE), strength=4 / 3),
        PolygonMask(vertices=tuple(_C_SHAPE), strength=-2.0),
    )
    lines = print_instructions(ExposureConfig(), masks)
    assert lines[1:] == ["1  dodge  +1⅓", "2  burn  -2"]


def test_hidden_and_degenerate_masks_are_left_off_the_card():
    masks = (
        PolygonMask(vertices=tuple(_C_SHAPE), strength=1.0),
        PolygonMask(vertices=((0.0, 0.0), (1.0, 1.0)), strength=1.0),  # too few vertices
        PolygonMask(vertices=tuple(_C_SHAPE), strength=-1.0),
    )
    lines = print_instructions(ExposureConfig(), masks, hidden={2})
    # Only mask 1 survives: 2 is degenerate, 3 is hidden. Numbering stays the mask's own.
    assert lines[1:] == ["1  dodge  +1"]
