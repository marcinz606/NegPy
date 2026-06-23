import unittest

import numpy as np

from negpy.features.exposure.logic import CharacteristicCurve, compute_pivot, grade_to_slope


def _c41_negative_densities(stops: np.ndarray, gamma: float = 0.55) -> np.ndarray:
    """
    C-41 green-channel densities for scene exposures (in stops around 18% gray),
    with a film toe: linear gamma above -2.5 stops, compressing below.
    """
    lg2 = np.log10(2.0)
    d = np.where(stops >= -2.5, gamma * stops * lg2, gamma * (-2.5) * lg2 + 0.32 * (stops + 2.5) * lg2)
    return np.where(stops >= -3.5, d, gamma * (-2.5) * lg2 - 0.32 * lg2 + 0.15 * (stops + 3.5) * lg2)


class TestToneReproduction(unittest.TestCase):
    """
    Physical tone-reproduction aims for the default conversion (classic print
    sensitometry): 18% gray prints near reflection density 0.74 (LATD aim),
    diffuse white stays near paper white, deep shadows reach near paper black,
    and no zone renders with harsh system contrast.
    """

    def setUp(self):
        self.stops = np.array([-4.5, -3.5, -2.5, -1.5, -0.5, 0.0, 1.0, 2.0, 2.33, 3.0])
        dneg = _c41_negative_densities(self.stops)
        floor, ceil = dneg.max(), dneg.min()
        self.rng = floor - ceil
        x = (dneg - floor) / (ceil - floor)
        slope = grade_to_slope(115.0, self.rng)
        pivot = compute_pivot(slope, density=1.0)
        self.density = np.asarray(CharacteristicCurve(slope, pivot)(x.reshape(-1, 1, 1))).ravel()
        self.i_mid = list(self.stops).index(0.0)
        self.i_white = list(self.stops).index(2.33)

    def test_mid_gray_prints_at_latd_aim(self):
        self.assertGreaterEqual(self.density[self.i_mid], 0.65)
        self.assertLessEqual(self.density[self.i_mid], 0.90)

    def test_diffuse_white_prints_near_paper_white(self):
        # Soft default grade (textural_range_factor) keeps whites slightly
        # toned, like printing a normal negative on soft paper.
        self.assertLessEqual(self.density[self.i_white], 0.24)

    def test_deep_shadow_reaches_near_paper_black(self):
        self.assertGreaterEqual(self.density[0], 1.75)
        self.assertLessEqual(self.density[0], 2.3)

    def test_no_harsh_contrast_zone(self):
        gammas = np.abs(np.gradient(self.density, self.stops * np.log10(2.0)))
        self.assertLessEqual(float(gammas[1:6].max()), 1.5)


if __name__ == "__main__":
    unittest.main()
