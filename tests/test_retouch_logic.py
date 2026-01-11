import numpy as np
from src.features.retouch.logic import apply_dust_removal


def test_manual_dust_removal_effect():
    img = np.ones((100, 100, 3), dtype=np.float32)
    img[48:53, 48:53] = 0.0

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
    assert res_mean > orig_mean

    spot_area = res[48:53, 48:53]
    assert np.mean(spot_area) > 0.5


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
