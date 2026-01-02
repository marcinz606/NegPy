import numpy as np
from src.backend.image_logic.gamma import apply_gamma_to_img, calculate_balancing_gammas


def test_apply_gamma_identity():
    img = np.random.rand(10, 10, 3)
    res = apply_gamma_to_img(img, 1.0)
    assert np.allclose(res, img)


def test_apply_gamma_power():
    img = np.array([[[0.25, 0.25, 0.25]]])
    # Gamma 2.0 -> g_inv = 0.5. 0.25^0.5 = 0.5
    res = apply_gamma_to_img(img, 2.0)
    assert np.allclose(res, 0.5)


def test_apply_gamma_per_channel():
    img = np.array([[[0.25, 0.25, 0.25]]])
    # [2.0, 1.0, 0.5] -> [0.25^0.5, 0.25^1.0, 0.25^2.0] = [0.5, 0.25, 0.0625]
    res = apply_gamma_to_img(img, [2.0, 1.0, 0.5])
    assert np.allclose(res, [0.5, 0.25, 0.0625])


def test_calculate_balancing_gammas():
    # Construct an image where:
    # Red P90 = 0.25
    # Green P90 = 0.5
    # Blue P90 = 0.8
    img = np.zeros((10, 1, 3))
    img[9, 0, 0] = (
        0.25  # Simplify percentile by just using one high value in a small array
    )
    img[9, 0, 1] = 0.5
    img[9, 0, 2] = 0.8
    # We want to map Red and Blue to Green (0.5)
    # Red: 0.25^g = 0.5 => g = log(0.5)/log(0.25) = 0.5
    # Green: 0.5^g = 0.5 => g = 1.0
    # Blue: 0.8^g = 0.5 => g = log(0.5)/log(0.8) approx 3.1
    gammas = calculate_balancing_gammas(
        img, 100
    )  # Use 100th percentile for our dummy image
    assert np.isclose(gammas[0], 0.5)
    assert np.isclose(gammas[1], 1.0)
    assert gammas[2] > 3.0
