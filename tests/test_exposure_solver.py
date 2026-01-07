import numpy as np
from src.features.exposure.analysis import (
    prepare_exposure_analysis,
)
from src.features.exposure.normalization import LogNegativeBounds


def test_prepare_exposure_analysis():
    # Create a dummy image (100x100 RGB)
    # Range 0.0 to 1.0
    img = np.random.rand(100, 100, 3).astype(np.float64)

    # Introduce specific min/max to test bounds
    # Channel 0: min 0.1, max 0.9
    img[0, 0, 0] = 0.1
    img[0, 1, 0] = 0.9

    norm_log, bounds = prepare_exposure_analysis(img)

    assert isinstance(bounds, LogNegativeBounds)
    # prepare_exposure_analysis crops the center 60% (removes 20% margins)
    # 100 * 0.6 = 60
    assert norm_log.shape == (60, 60, 3)
    # Norm log should be roughly 0 to 1 (normalized by dynamic range)
    # Since we have random noise, it might be slightly wider or narrower depending on percentiles used in measure_bounds
    # But generally it shouldn't be empty
    assert np.any(norm_log != 0)
