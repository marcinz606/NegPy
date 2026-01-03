import numpy as np
from src.config import PIPELINE_CONSTANTS
from src.frontend.components.sidebar.helpers import apply_wb_gains_to_sliders


def test_apply_wb_gains_identity():
    # 1.0, 1.0, 1.0 -> 0, 0, 0
    res = apply_wb_gains_to_sliders(1.0, 1.0, 1.0)
    assert res["wb_cyan"] == 0
    assert res["wb_magenta"] == 0
    assert res["wb_yellow"] == 0


def test_apply_wb_gains_magenta_yellow():
    # log10(1.479) is ~0.17 density
    # Calculate expected slider value based on current config
    max_d = PIPELINE_CONSTANTS["cmy_max_density"]
    expected = np.log10(1.479) / max_d
    expected = float(np.clip(expected, -1.0, 1.0))

    res = apply_wb_gains_to_sliders(1.0, 1.479, 1.479)
    assert res["wb_cyan"] == 0
    assert round(res["wb_magenta"], 2) == round(expected, 2)
    assert round(res["wb_yellow"], 2) == round(expected, 2)


def test_apply_wb_gains_clamping():
    # Gain of 10.0 -> log10(10) = 1.0 density.
    # 1.0 / 0.17 = 5.88 -> clamped to 1.0
    # Even with max_d = 0.2, 1.0/0.2 = 5.0 -> clamped to 1.0
    res = apply_wb_gains_to_sliders(1.0, 10.0, 1.0)
    assert res["wb_magenta"] == 1.0
