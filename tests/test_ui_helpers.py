import numpy as np
from src.features.exposure.models import EXPOSURE_CONSTANTS
from src.ui.components.sidebar.helpers import apply_wb_gains_to_sliders


def test_apply_wb_gains_identity():
    res = apply_wb_gains_to_sliders(1.0, 1.0, 1.0)
    assert res["wb_cyan"] == 0
    assert res["wb_magenta"] == 0
    assert res["wb_yellow"] == 0


def test_apply_wb_gains_magenta_yellow():
    # log10(1.479) is ~0.17 density
    # Calculate expected slider value based on current config
    max_d = EXPOSURE_CONSTANTS["cmy_max_density"]
    expected = np.log10(1.479) / max_d
    expected = float(np.clip(expected, -1.0, 1.0))

    res = apply_wb_gains_to_sliders(1.0, 1.479, 1.479)
    assert res["wb_cyan"] == 0
    assert round(res["wb_magenta"], 2) == round(expected, 2)
    assert round(res["wb_yellow"], 2) == round(expected, 2)


def test_apply_wb_gains_clamping():
    res = apply_wb_gains_to_sliders(1.0, 10.0, 1.0)
    assert res["wb_magenta"] == 1.0
