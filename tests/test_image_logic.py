import numpy as np
from src.backend.image_logic.exposure import apply_contrast, apply_film_characteristic_curve

# REMOVED color imports because cv2 is missing in this environment
# from src.backend.image_logic.color import apply_color_separation, convert_to_monochrome

def test_apply_film_characteristic_curve_range():
    img = np.array([[[0.1, 0.5, 0.9]]])
    # Params: ((pivot, slope), (pivot, slope), (pivot, slope))
    params = (-2.5, 1.0)
    res = apply_film_characteristic_curve(img, params, params, params)
    assert res.shape == img.shape
    assert np.all(res >= 0.0)
    assert np.all(res <= 1.0)


def test_apply_film_characteristic_curve_positive_output():
    # Ensure that the output is POSITIVE (Bright Input -> Dark Output)
    # Input is Negative Scan:
    # 0.1 = Highlight (Clear Neg) -> Low Exposure -> Low Density -> High Transmittance (Bright Print).
    # 0.9 = Shadow (Dense Neg) -> High Exposure -> High Density -> Low Transmittance (Dark Print).
    
    img = np.array([[
        [0.1, 0.1, 0.1],
        [0.9, 0.9, 0.9]
    ]])
    
    params = (-2.0, 1.0) # Pivot -2.0, Slope 1.0
    res = apply_film_characteristic_curve(img, params, params, params)
    
    val_highlight_input = np.mean(res[0, 0]) # Input 0.1
    val_shadow_input = np.mean(res[0, 1])    # Input 0.9
    
    # Highlight Input (0.1) should result in Bright Output
    # Shadow Input (0.9) should result in Dark Output
    
    assert val_highlight_input > val_shadow_input


def test_apply_contrast_neutral():
    img = np.array([0.2, 0.5, 0.8])
    # contrast = 1.0 should be identity
    res = apply_contrast(img, 1.0)
    assert np.allclose(res, img)


def test_apply_contrast_increase():
    img = np.array([0.4, 0.5, 0.6])
    # Increase contrast
    res = apply_contrast(img, 2.0)
    # (0.4 - 0.5) * 2 + 0.5 = 0.3
    # (0.6 - 0.5) * 2 + 0.5 = 0.7
    assert np.allclose(res, [0.3, 0.5, 0.7])