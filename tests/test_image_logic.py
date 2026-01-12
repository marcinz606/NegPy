import numpy as np
from src.features.exposure.logic import (
    apply_characteristic_curve,
)


def test_apply_film_characteristic_curve_range():
    img = np.array([[[0.1, 0.5, 0.9]]])
    # Params: (pivot, slope)
    params = (-2.5, 1.0)
    res = apply_characteristic_curve(img, params, params, params)
    assert res.shape == img.shape
    assert np.all(res >= 0.0)
    assert np.all(res <= 1.0)


def test_apply_film_characteristic_curve_positive_output():
    # Ensure POSITIVE output (Bright Input -> Dark Output)
    # 0.1 (Highlight) -> Bright Print
    # 0.9 (Shadow) -> Dark Print

    img = np.array([[[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]]])

    params = (-2.0, 1.0)  # Pivot -2.0, Slope 1.0
    res = apply_characteristic_curve(img, params, params, params)

    val_highlight_input = np.mean(res[0, 0])  # Input 0.1
    val_shadow_input = np.mean(res[0, 1])  # Input 0.9

    # Highlight Input (0.1) should result in Bright Output
    # Shadow Input (0.9) should result in Dark Output

    assert val_highlight_input > val_shadow_input
