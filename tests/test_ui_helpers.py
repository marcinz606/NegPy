from src.frontend.components.sidebar.helpers import apply_wb_gains_to_sliders


def test_apply_wb_gains_identity():
    # 1.0, 1.0, 1.0 -> 0, 0, 0
    res = apply_wb_gains_to_sliders(1.0, 1.0, 1.0)
    assert res["wb_cyan"] == 0
    assert res["wb_magenta"] == 0
    assert res["wb_yellow"] == 0


def test_apply_wb_gains_magenta_yellow():
    # Gain of 10.0 -> log10(10) * 100 = 100
    res = apply_wb_gains_to_sliders(1.0, 10.0, 10.0)
    assert res["wb_cyan"] == 0
    assert res["wb_magenta"] == 100
    assert res["wb_yellow"] == 100


def test_apply_wb_gains_clamping():
    # Gain of 100.0 -> log10(100) * 100 = 200 (clamped to 170)
    res = apply_wb_gains_to_sliders(1.0, 100.0, 1.0)
    assert res["wb_magenta"] == 170
