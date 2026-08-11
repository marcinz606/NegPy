"""The camera matrix that carries a transparency from sensor primaries to the working space.

Only the E-6 as-captured path uses it (`NormalizationProcessor._process_transparency`, and
the same rows uploaded to normalization.wgsl behind its `is_transfer` gate). The print path
derives colour from measured film density, so C-41 and B&W never touch this.

The construction follows dcraw's `cam_xyz_coeff`: normalize the *forward* working->cam rows,
then invert. Normalizing the rows of the already-inverted matrix sends neutral to neutral
just as well, which is why getting it backwards survived unnoticed — greys look right and
only saturated colour is wrong.
"""

import unittest

import numpy as np

from negpy.features.process.capture_color import _XYZ_TO_WORKING, apply_camera_matrix, camera_to_working_matrix

# Nikon D3300, libraw's rgb_xyz_matrix (XYZ -> camera), first three rows.
D3300 = [
    [0.6988000273704529, -0.13840000331401825, -0.0714000016450882],
    [-0.5630999803543091, 1.340999960899353, 0.24469999969005585],
    [-0.148499995470047, 0.22040000557899475, 0.7318000197410583],
]


class Construction(unittest.TestCase):
    def test_matches_the_rows_validated_against_libraw(self):
        """Golden values, because no structural invariant separates the two orders.

        Row sums do not: normalizing either direction gives M @ (1,1,1) = (1,1,1), and if
        M maps neutral to neutral so does its inverse. The only witness is the coefficients
        themselves. These were checked against libraw's own cam->sRGB on two unclipped
        frames from this sensor, where they reproduce its R/G to within 0.1%; the previous
        order was 5-7% off on those and +119% on a saturated one.
        """
        m = camera_to_working_matrix(D3300)
        self.assertIsNotNone(m)
        expected = [
            [1.255325, -0.113049, -0.142277],
            [-0.144795, 1.544686, -0.399891],
            [0.033506, -0.351512, 1.318006],
        ]
        np.testing.assert_allclose(np.asarray(m, dtype=np.float64), expected, atol=2e-6)

    def test_neutral_is_preserved_either_way_which_is_why_this_hid(self):
        """Both orders map a neutral camera signal to a neutral working signal, so no grey
        chart, and no near-neutral frame, can catch the mistake."""
        m = np.asarray(camera_to_working_matrix(D3300), dtype=np.float64)
        np.testing.assert_allclose(m @ np.ones(3), np.ones(3), atol=1e-5)

    def test_saturated_colour_is_where_the_orders_diverge(self):
        """The error grows with distance from neutral. On a saturated red it was a factor
        of two on both R/G and B/G, which renders as a magenta cast."""
        good = np.asarray(camera_to_working_matrix(D3300), dtype=np.float64)
        wrong = _XYZ_TO_WORKING @ np.linalg.inv(np.asarray(D3300, dtype=np.float64))
        wrong = wrong / wrong.sum(axis=1, keepdims=True)  # the order this used to use

        red = np.array([0.9, 0.3, 0.2])
        g_ratio = (good @ red)[0] / (good @ red)[1]
        w_ratio = (wrong @ red)[0] / (wrong @ red)[1]
        self.assertGreater(w_ratio / g_ratio, 1.5, "fixture must be saturated enough to separate them")

    def test_camera_wb_folds_in_for_a_buffer_decoded_without_it(self):
        """Linear RAW decodes without white balance, so the multipliers are folded into the
        matrix instead; normalized to green, so only the ratios apply and exposure does not."""
        plain = np.asarray(camera_to_working_matrix(D3300), dtype=np.float64)
        folded = np.asarray(camera_to_working_matrix(D3300, camera_wb=[1.945, 1.0, 1.473, 1.0]), dtype=np.float64)
        np.testing.assert_allclose(folded, plain @ np.diag([1.945, 1.0, 1.473]), rtol=1e-5)
        # Green untouched: a WB already normalized to green must not move overall exposure.
        np.testing.assert_allclose(
            np.asarray(camera_to_working_matrix(D3300, camera_wb=[1.0, 1.0, 1.0, 1.0]), dtype=np.float64), plain, rtol=1e-6
        )


class Degenerate(unittest.TestCase):
    def test_unusable_input_yields_no_matrix(self):
        """None means "already in the working space", which is the right reading for a
        scanner TIFF or a JPEG — better than a singular matrix reaching a decode."""
        self.assertIsNone(camera_to_working_matrix(None))
        self.assertIsNone(camera_to_working_matrix([[1, 0, 0], [2, 0, 0], [3, 0, 0]]))  # singular
        self.assertIsNone(camera_to_working_matrix([[1, 0], [0, 1]]))  # wrong shape
        self.assertIsNone(camera_to_working_matrix([[float("nan")] * 3] * 3))

    def test_a_none_matrix_passes_the_buffer_through(self):
        img = np.random.default_rng(0).random((4, 4, 3)).astype(np.float32)
        np.testing.assert_array_equal(apply_camera_matrix(img, None), img)
