import numpy as np
from src.features.retouch.logic import apply_dust_removal


def test_manual_dust_removal_effect():
    # Use grey background and white dust (inverted film scan scenario)
    img = np.full((100, 100, 3), 0.5, dtype=np.float32)
    img[48:53, 48:53] = 1.0

    orig_mean = np.mean(img)
    manual_spots = [(0.5, 0.5, 10)]

    res = apply_dust_removal(
        img.copy(),
        dust_remove=False,
        dust_threshold=0.75,
        dust_size=2,
        manual_spots=manual_spots,
        scale_factor=1.0,
    )

    res_mean = np.mean(res)
    # The healing should make the white spot darker (closer to 0.5 background)
    assert res_mean < orig_mean

    spot_area = res[48:53, 48:53]
    assert np.mean(spot_area) < 0.9


def test_manual_dust_removal_no_spots():
    img = np.ones((100, 100, 3), dtype=np.float32)
    res = apply_dust_removal(
        img.copy(),
        dust_remove=False,
        dust_threshold=0.75,
        dust_size=2,
        manual_spots=[],
        scale_factor=1.0,
    )
    assert np.array_equal(img, res)


def test_auto_dust_removal_low_res():
    # Simple isolated white pixel on dark background
    img = np.zeros((100, 100, 3), dtype=np.float32)
    img[50, 50] = 1.0

    res = apply_dust_removal(
        img.copy(),
        dust_remove=True,
        dust_threshold=0.5,
        dust_size=2,
        manual_spots=[],
        scale_factor=1.0,
    )

    # The bright pixel should be gone
    assert res[50, 50, 0] < 0.5


def test_auto_dust_removal_high_res():
    # Larger spot at high scale
    img = np.zeros((200, 200, 3), dtype=np.float32)
    img[98:103, 98:103] = 1.0

    res = apply_dust_removal(
        img.copy(),
        dust_remove=True,
        dust_threshold=0.5,
        dust_size=4,
        manual_spots=[],
        scale_factor=2.0,
    )

    # The bright spot should be healed
    assert np.mean(res[98:103, 98:103]) < 0.5


def test_auto_dust_removal_cloud_protection():
    # Soft gradients should NOT be treated as dust
    y, x = np.mgrid[0:100, 0:100]
    img_gray = (np.sin(x / 10.0) * np.cos(y / 10.0) * 0.1) + 0.5
    img = np.stack([img_gray] * 3, axis=-1).astype(np.float32)

    res = apply_dust_removal(
        img.copy(),
        dust_remove=True,
        dust_threshold=0.5,
        dust_size=2,
        manual_spots=[],
        scale_factor=1.0,
    )

    # Soft gradients should remain identical or very close
    np.testing.assert_allclose(img, res, atol=0.01)
