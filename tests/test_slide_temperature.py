import math
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pytest

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.color import ColorSidebar
from negpy.features.exposure.logic import filtration_offsets
from negpy.features.exposure.normalization import LogNegativeBounds
from negpy.features.process.models import ProcessMode
from negpy.features.transparency.logic import (
    apply_transfer_curve,
    contrast_free_cmy,
    slide_kelvin_to_wb,
    slide_pick_wb,
    slide_wb_to_kelvin,
)


def test_slide_temperature_roundtrip_and_neutral():
    assert slide_wb_to_kelvin(0.0, 0.0) == 5500.0
    for k in (3200.0, 4500.0, 7000.0, 11000.0):
        c, y = slide_kelvin_to_wb(k, 0.0, 0.0)
        assert abs(slide_wb_to_kelvin(c, y) - k) < 1e-6


def test_warming_lowers_blue_more_than_it_raises_red():
    c, y = slide_kelvin_to_wb(4000.0, 0.0, 0.0)
    assert c < 0.0 < y
    assert 1.8 < y / -c < 2.4


def test_slide_temperature_keeps_off_locus_component():
    c0, y0 = 0.1, 0.05
    c1, y1 = slide_kelvin_to_wb(4500.0, c0, y0)
    c2, y2 = slide_kelvin_to_wb(slide_wb_to_kelvin(c0, y0), c1, y1)
    assert math.isclose(c2, c0, abs_tol=1e-9) and math.isclose(y2, y0, abs_tol=1e-9)


def test_slide_temperature_range_fits_the_sliders():
    for k in (3000.0, 12000.0):
        c, y = slide_kelvin_to_wb(k, 0.0, 0.0)
        assert abs(c) < 1.0 and abs(y) < 1.0


def _slide_sidebar(qapp):
    controller = MagicMock()
    controller.state = AppState()
    controller.session.repo.get_global_setting.return_value = None
    sidebar = ColorSidebar(controller)
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6))
    return controller, sidebar


def test_slide_temperature_writes_cyan_and_yellow_not_magenta(qapp):
    controller, sidebar = _slide_sidebar(qapp)
    sidebar._on_temp_changed(4000.0)
    written = controller.apply_config.call_args.args[0].exposure
    assert written.wb_cyan < 0.0 < written.wb_yellow
    assert written.wb_magenta == 0.0


@pytest.mark.parametrize("grade", [70.0, 115.0, 160.0])
@pytest.mark.parametrize("positive", [False, True])
def test_slide_pick_wb_prints_the_patch_gray(grade, positive):
    window = 3.0
    patch = np.array([0.300, 0.330, 0.270], dtype=np.float32)
    contrast = 115.0 / grade
    c, m, y = slide_pick_wb(patch, (window,) * 3, contrast)
    bounds = LogNegativeBounds(floors=(0.0,) * 3, ceils=(window,) * 3)
    offsets = contrast_free_cmy(filtration_offsets((c, m, y), bounds), contrast)
    img = np.tile(patch, (2, 2, 1)).astype(np.float32)
    px = apply_transfer_curve(img, 0.0, contrast, (0.3,) * 3, (0.3,) * 3, offsets, positive_source=positive)[0, 0]
    assert np.ptp(np.log10(px)) < 1e-5


def test_slide_pick_wb_on_a_neutral_patch_does_nothing():
    assert slide_pick_wb(np.array([0.3, 0.3, 0.3]), (4.0, 4.0, 4.0)) == (0.0, 0.0, 0.0)
