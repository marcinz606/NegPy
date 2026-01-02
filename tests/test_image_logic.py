import numpy as np
from src.backend.image_logic.exposure import apply_contrast, apply_scan_gain_with_toe
from src.backend.image_logic.color import apply_color_separation, convert_to_monochrome


def test_apply_scan_gain_identity():
    img = np.array([[[0.5, 0.5, 0.5]]])
    # gain=1.0, toe=0, shoulder=0 should be identity (mostly)
    res = apply_scan_gain_with_toe(img, 1.0, 0.0, 0.0)
    assert np.allclose(res, img, atol=1e-5)


def test_apply_scan_gain_with_toe_lift():
    img = np.array([[[0.6, 0.6, 0.6]]])  # > 0.5
    # Shadow toe in this model lifts values ABOVE 0.5 (negative domain)
    res = apply_scan_gain_with_toe(img, 1.0, 0.1, 0.0)
    # 0.5938 < 0.6 (negative domain) -> Lighter in positive print.
    assert res[0, 0, 0] < 0.6


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


def test_convert_to_monochrome():
    img = np.random.rand(10, 10, 3)
    mono = convert_to_monochrome(img)
    assert mono.shape == (10, 10, 3)
    # All channels should be equal
    assert np.allclose(mono[:, :, 0], mono[:, :, 1])
    assert np.allclose(mono[:, :, 1], mono[:, :, 2])


def test_apply_color_separation_identity():
    img = np.random.rand(10, 10, 3)
    res = apply_color_separation(img, 1.0)
    assert np.array_equal(res, img)


def test_apply_color_separation_increase():
    # Mid-tone red-ish pixel
    img = np.array([[[0.6, 0.4, 0.4]]])
    # Increase separation (intensity = 1.5)
    res = apply_color_separation(img, 1.5)
    # Red is above luma, so it should increase
    # Green/Blue are below luma, so they should decrease
    assert res[0, 0, 0] > img[0, 0, 0]
    assert res[0, 0, 1] < img[0, 0, 1]
    assert res[0, 0, 2] < img[0, 0, 2]
