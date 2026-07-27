import numpy as np

from negpy.features.exposure.analysis import ZONE_GRID_CELLS, zone_grid, zone_region_labels
from negpy.kernel.image.logic import working_oetf_encode


def _flat(value: float, h: int = 180, w: int = 240) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.float32)


def _mid() -> float:
    return float(working_oetf_encode(np.asarray([0.18], dtype=np.float32))[0])


def test_grid_is_square_ish_cells_on_the_long_edge():
    zones = zone_grid(_flat(0.5, 180, 240))
    assert zones.shape == (round(ZONE_GRID_CELLS * 180 / 240), ZONE_GRID_CELLS)


def test_mid_gray_reads_zone_five_everywhere():
    assert np.all(zone_grid(_flat(_mid())) == 5)


def test_uniform_frame_is_one_region_labelled_at_its_centre():
    zones = zone_grid(_flat(_mid()))
    rows, cols = zones.shape
    assert zone_region_labels(zones) == [(cols // 2 - 1, rows // 2 - 1, 5)]


def test_black_and_white_halves_are_one_region_each():
    img = _flat(0.0)
    img[:, 120:] = 1.0
    labels = zone_region_labels(zone_grid(img))
    assert len(labels) == 2
    assert {zone for *_, zone in labels} == {0, 10}


def test_grain_does_not_shatter_a_flat_field_into_confetti():
    rng = np.random.default_rng(0)
    grainy = np.clip(_flat(_mid(), 1067, 1600) + rng.normal(0, 0.05, (1067, 1600, 3)), 0.0, 1.0).astype(np.float32)
    # A noisy but tonally uniform field must merge, not read as a grid of identical labels.
    assert len(zone_region_labels(zone_grid(grainy))) <= 3


def test_label_anchor_sits_inside_its_own_region():
    rng = np.random.default_rng(1)
    zones = zone_grid(rng.random((1067, 1600, 3), dtype=np.float32))
    for col, row, zone in zone_region_labels(zones):
        assert zones[row, col] == zone


def test_degenerate_input_has_no_grid():
    assert zone_grid(np.zeros((1, 240, 3), dtype=np.float32)) is None
    assert zone_grid(np.zeros((180, 240), dtype=np.float32)) is None
