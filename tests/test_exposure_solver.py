import pytest
import numpy as np
from src.features.exposure.analysis import (
    solve_photometric_exposure,
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


def test_solve_photometric_exposure_neutral_base():
    """
    Test that if the base is already neutral (same min values), CMY offsets are 0.
    """
    # Create an image where the "base" (minimum density) is identical for R, G, B
    img = np.ones((50, 50, 3)) * 0.5
    # Add some variation so it's not flat
    img += np.random.normal(0, 0.01, size=(50, 50, 3))
    img = np.clip(img, 0.1, 0.9)

    # Force "clear base" (low values in input = low density in negative logic if interpreted as density map directly?
    # Wait, the function comments say: "Low values = Clearer Negative".
    # And the logic in solve uses 0.1th percentile.

    # Let's verify the inputs to solve_photometric_exposure.
    # It expects `norm_subject_log`.

    norm_log = np.zeros((50, 50, 3))
    # Set base to 0.05 for all channels
    norm_log[:, :, 0] = 0.05 + np.random.rand(50, 50) * 0.9
    norm_log[:, :, 1] = 0.05 + np.random.rand(50, 50) * 0.9
    norm_log[:, :, 2] = 0.05 + np.random.rand(50, 50) * 0.9

    # Bounds don't matter much for the solver itself as it uses norm_log,
    # but strictly it asks for them.
    bounds = LogNegativeBounds((0, 0, 0), (1, 1, 1))

    c, m, y, density, grade = solve_photometric_exposure(norm_log, bounds)

    # Since bases are roughly equal (0.05), offsets should be close to 0
    assert abs(c) < 1e-6
    assert abs(m) < 0.05  # Allow small noise deviation
    assert abs(y) < 0.05


def test_solve_photometric_exposure_color_cast():
    """
    Test that if Green base is denser (higher value) than Red base,
    it produces a Magenta offset to compensate?

    Wait:
    Solver logic:
    base_r = min(R)
    base_g = min(G)
    magenta_offset = base_r - base_g

    If G has HIGHER base (e.g. 0.2) than R (0.1):
    m_offset = 0.1 - 0.2 = -0.1.

    This offset is ADDED to the channel later:
    d_g = curve(log_exp + offset)
    If offset is negative, we are subtracting density -> Making it lighter?

    Let's check `apply_film_characteristic_curve` in `exposure.py`:
    d_r = curve_r(log_exp[:, :, 0] + cmy_offsets[0])

    If we have excess Green density (Base is 0.2 instead of 0.1), we want to subtract it to align to 0.1.
    So offset should be negative. Correct.
    """
    norm_log = np.zeros((50, 50, 3))
    # R base = 0.1
    norm_log[:, :, 0] = 0.1 + np.random.rand(50, 50) * 0.8
    # G base = 0.2 (Denser base)
    norm_log[:, :, 1] = 0.2 + np.random.rand(50, 50) * 0.8
    # B base = 0.1
    norm_log[:, :, 2] = 0.1 + np.random.rand(50, 50) * 0.8

    bounds = LogNegativeBounds((0, 0, 0), (1, 1, 1))

    c, m, y, density, grade = solve_photometric_exposure(norm_log, bounds)

    # Expected: m approx 0.1 - 0.2 = -0.1
    assert pytest.approx(m, abs=0.02) == -0.1
    assert pytest.approx(y, abs=0.02) == 0.0
    assert c == 0.0


def test_solve_photometric_exposure_contrast():
    """
    Test Auto-Grade.
    Grade = Target (2.1) / Measured_DR.
    """
    norm_log = np.zeros((100, 100, 3))
    # Create a Red channel with known range
    # 0.5th percentile ~ 0.1
    # 99.5th percentile ~ 0.6
    # DR = 0.5.
    # Target = 2.1
    # Expected Grade = 2.1 / 0.5 = 4.2 -> Clamped to 4.0

    # Fill with linspace to ensure distribution
    vals = np.linspace(0.1, 0.6, 100 * 100)
    np.random.shuffle(vals)
    norm_log[:, :, 0] = vals.reshape(100, 100)

    # Fill others randomly
    norm_log[:, :, 1] = vals.reshape(100, 100)
    norm_log[:, :, 2] = vals.reshape(100, 100)

    bounds = LogNegativeBounds((0, 0, 0), (1, 1, 1))

    _, _, _, density, grade = solve_photometric_exposure(norm_log, bounds)

    # Expected Physical Slope = 4.2 -> Clamped to 4.0
    # UI Grade = (4.0 - 1.0) / 1.5 = 2.0
    assert pytest.approx(grade, abs=0.01) == 2.0

    # Test softer contrast
    # DR = 2.1 -> Grade = 1.0
    vals2 = np.linspace(0.0, 2.1, 100 * 100)
    norm_log[:, :, 0] = vals2.reshape(100, 100)
    _, _, _, density, grade = solve_photometric_exposure(norm_log, bounds)

    # Expected Physical Slope = 1.0
    # UI Grade = (1.0 - 1.0) / 1.5 = 0.0
    assert pytest.approx(grade, abs=0.1) == 0.0
