import unittest
import numpy as np
from negpy.features.finish.logic import (
    CARRIER_MARGIN,
    CARRIER_SAMPLES,
    apply_carrier,
    apply_vignette,
    carrier_noise,
    carrier_profiles,
)


class TestVignette(unittest.TestCase):
    def _gradient_image(self) -> np.ndarray:
        """100x100 mid-gray image for reliable vignette testing."""
        return np.full((100, 100, 3), 0.5, dtype=np.float32)

    def test_noop_when_stops_zero(self) -> None:
        """Zero stops returns image unchanged."""
        img = self._gradient_image()
        res = apply_vignette(img, stops=0.0, size=0.5)
        np.testing.assert_array_equal(res, img)

    def test_output_shape_and_range(self) -> None:
        """Output keeps same shape and stays in [0, 1]."""
        img = self._gradient_image()
        for stops in [-1.0, 1.0, -2.0, 2.0]:
            for size in [0.0, 0.5, 1.0]:
                res = apply_vignette(img, stops, size)
                self.assertEqual(res.shape, img.shape)
                self.assertGreaterEqual(float(res.min()), 0.0)
                self.assertLessEqual(float(res.max()), 1.0)

    def test_burn_darkens_corners(self) -> None:
        """Positive stops (burn) darkens corners more than center."""
        img = self._gradient_image()
        res = apply_vignette(img, stops=1.0, size=0.5)
        corner_luma = float(res[0, 0].mean())
        center_luma = float(res[50, 50].mean())
        self.assertLess(corner_luma, center_luma)

    def test_dodge_brightens_corners(self) -> None:
        """Negative stops (hold back) brightens corners more than center."""
        img = self._gradient_image()
        res = apply_vignette(img, stops=-1.0, size=0.5)
        corner_luma = float(res[0, 0].mean())
        center_luma = float(res[50, 50].mean())
        self.assertGreater(corner_luma, center_luma)

    def test_burn_is_exposure_exact(self) -> None:
        """A fully-covered corner at +1 stop halves the linear value."""
        img = self._gradient_image()
        res = apply_vignette(img, stops=1.0, size=0.5)
        np.testing.assert_allclose(res[0, 0], img[0, 0] * 0.5, atol=1e-5)

    def test_center_unaffected(self) -> None:
        """Center pixel should be unchanged regardless of stops."""
        img = self._gradient_image()
        for stops in [-2.0, -1.0, 1.0, 2.0]:
            res = apply_vignette(img, stops, size=0.5)
            np.testing.assert_array_almost_equal(res[50, 50], img[50, 50], decimal=5)

    def test_small_size_localizes_to_corners(self) -> None:
        """Small size keeps the center untouched while still burning corners."""
        img = self._gradient_image()
        res = apply_vignette(img, stops=2.0, size=0.1)
        center_luma = float(res[50, 50].mean())
        self.assertAlmostEqual(center_luma, 0.5, delta=0.01)
        corner_luma = float(res[0, 0].mean())
        self.assertLess(corner_luma, center_luma)

    def test_size_one_affects_entire_image(self) -> None:
        """Size=1 means the burn covers the entire image — center is affected too."""
        img = self._gradient_image()
        res = apply_vignette(img, stops=2.0, size=1.0)
        center_luma = float(res[50, 50].mean())
        self.assertLess(center_luma, 0.5)

    def test_non_square_image(self) -> None:
        """Works correctly on non-square images."""
        img = np.full((50, 200, 3), 0.5, dtype=np.float32)
        res = apply_vignette(img, stops=2.0, size=0.5)
        self.assertEqual(res.shape, img.shape)
        self.assertGreaterEqual(float(res.min()), 0.0)
        self.assertLessEqual(float(res.max()), 1.0)

    def test_circular_falloff_invariant(self) -> None:
        """Pixels equidistant from center receive identical vignette weight."""
        h, w = 100, 200
        img = np.full((h, w, 3), 0.5, dtype=np.float32)
        res = apply_vignette(img, stops=2.0, size=0.5)

        c00 = res[0, 0]
        c0w = res[0, w - 1]
        ch0 = res[h - 1, 0]
        chw = res[h - 1, w - 1]
        np.testing.assert_allclose(c00, c0w, atol=1e-5)
        np.testing.assert_allclose(c00, ch0, atol=1e-5)
        np.testing.assert_allclose(c00, chw, atol=1e-5)

        mid_x = w // 2
        mid_y = h // 2
        np.testing.assert_allclose(res[0, mid_x], res[h - 1, mid_x], atol=1e-5)
        np.testing.assert_allclose(res[mid_y, 0], res[mid_y, w - 1], atol=1e-5)

    def test_radial_edges_weaker_than_corners(self) -> None:
        """Roundness 0: edge midpoints sit inside the radial falloff, so they
        burn less than corners."""
        h, w = 100, 100
        img = np.full((h, w, 3), 0.5, dtype=np.float32)
        res = apply_vignette(img, stops=2.0, size=0.5, roundness=0.0)
        self.assertGreater(float(res[0, w // 2].mean()), float(res[0, 0].mean()))

    def test_rectangular_edges_match_corners(self) -> None:
        """Roundness 1: the burn follows the frame — edge midpoints and corners
        receive the same weight (card burn along each edge)."""
        h, w = 100, 200
        img = np.full((h, w, 3), 0.5, dtype=np.float32)
        res = apply_vignette(img, stops=2.0, size=0.5, roundness=1.0)
        np.testing.assert_allclose(res[0, w // 2], res[0, 0], atol=1e-5)
        np.testing.assert_allclose(res[h // 2, 0], res[0, 0], atol=1e-5)


class TestCarrier(unittest.TestCase):
    def _image(self) -> np.ndarray:
        return np.full((100, 150, 3), 0.5, dtype=np.float32)

    def test_noop_when_width_zero(self) -> None:
        img = self._image()
        res = apply_carrier(img, width_px=0.0, rough=0.5)
        np.testing.assert_array_equal(res, img)

    def test_paper_margin_then_black_then_picture(self) -> None:
        """Across each edge: unexposed paper, black rebate, then the picture."""
        img = self._image()
        res = apply_carrier(img, width_px=8.0, rough=0.0)
        for cut in (res[:20, 75, 0], res[-20:, 75, 0][::-1], res[50, :20, 0], res[50, -20:, 0][::-1]):
            self.assertGreater(float(cut[0]), 0.9)
            black = int(np.argmin(cut))
            self.assertEqual(float(cut[black]), 0.0)
            self.assertGreater(black, 0)
            # Past the black run the picture value returns.
            self.assertAlmostEqual(float(cut[-1]), 0.5, places=5)
        np.testing.assert_array_equal(res[50, 75], img[50, 75])

    def test_deterministic(self) -> None:
        img = self._image()
        a = apply_carrier(img, width_px=5.0, rough=1.0)
        b = apply_carrier(img, width_px=5.0, rough=1.0)
        np.testing.assert_array_equal(a, b)

    def test_rough_ragges_the_filed_edge_only(self) -> None:
        """The slider swings the paper-side edge; the picture-side film gate stays put."""
        # Big enough that the sampled columns sit clear of the left/right bands.
        img = np.full((300, 400, 3), 0.5, dtype=np.float32)
        smooth = apply_carrier(img, width_px=16.0, rough=0.0)
        rough = apply_carrier(img, width_px=16.0, rough=1.0)

        def edges(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Per-column row of the paper->black and black->picture boundaries along the top edge."""
            cut = a[:80, 120:280, 0]
            paper = np.argmax(cut < 0.5, axis=0)
            gate = 80 - np.argmax((cut < 0.25)[::-1], axis=0)
            return paper, gate

        p_smooth, g_smooth = edges(smooth)
        p_rough, g_rough = edges(rough)
        self.assertEqual(int(np.ptp(p_smooth)), 0)
        self.assertGreaterEqual(int(np.ptp(p_rough)), 6)
        # The gate wobble is sub-pixel at this width, and the slider must not touch it.
        self.assertLessEqual(int(np.ptp(g_smooth)), 1)
        np.testing.assert_array_equal(g_smooth, g_rough)

    def test_profiles_shape_and_range(self) -> None:
        p = carrier_profiles()
        self.assertEqual(p.shape, (8, CARRIER_SAMPLES))
        self.assertLessEqual(float(np.abs(p).max()), 1.0)
        self.assertEqual(p.dtype, np.float32)

    def test_filed_rows_are_sparse_bites(self) -> None:
        """Rows 4-7 must read as a mostly-clean line the file bit into, not a ripple."""
        p = carrier_profiles()
        self.assertLess(float(np.abs(p[4:]).mean()), float(np.abs(p[:4]).mean()))
        # Marks at file scale, not per-sample teeth: neighbouring samples stay correlated.
        rho = [float(np.corrcoef(r[:-1], r[1:])[0, 1]) for r in p[4:]]
        self.assertGreater(min(rho), 0.99)

    def test_flare_off_by_default(self) -> None:
        img = self._image()
        np.testing.assert_array_equal(
            apply_carrier(img, width_px=5.0, rough=0.5),
            apply_carrier(img, width_px=5.0, rough=0.5, flare=0.0),
        )

    def _rebate_mask(self, plain: np.ndarray) -> np.ndarray:
        """Pixels the rebate multiplied to black, taken off a flare-free render."""
        return plain.max(axis=-1) == 0.0

    def test_flare_rides_the_filed_edge(self) -> None:
        img = self._image()
        # rough=0 keeps the filed edge straight, so a row index is either margin or band.
        plain = apply_carrier(img, width_px=8.0, rough=0.0)
        lit = apply_carrier(img, width_px=8.0, rough=0.0, flare=1.0)
        black = self._rebate_mask(plain)

        self.assertGreater(float(lit.max(axis=-1)[black].max()), 0.05)
        np.testing.assert_array_equal(lit[30:70, 30:120], plain[30:70, 30:120])
        # Top edge, corners excluded: the effect straddles the filed edge rather than
        # sitting on the picture-side gate.
        delta = np.abs(lit - plain).max(axis=-1)[:20, 20:130].mean(axis=1)
        filed, gate = 8.0 * CARRIER_MARGIN, 8.0 * CARRIER_MARGIN + 8.0
        peak = int(np.argmax(delta))
        self.assertLess(abs(peak - filed), abs(peak - gate))

    def test_flare_stains_the_paper_margin(self) -> None:
        """The reflection exposes the paper outside the aperture, so the white picks up a cast."""
        img = self._image()
        plain = apply_carrier(img, width_px=8.0, rough=0.0)
        lit = apply_carrier(img, width_px=8.0, rough=0.0, flare=1.0)
        paper = plain.min(axis=-1) == 1.0
        self.assertTrue(paper.any())
        # Some of that paper is now both darker and no longer neutral.
        self.assertLess(float(lit[paper].min()), 0.98)
        self.assertGreater(float(np.abs(np.diff(lit[paper], axis=-1)).max()), 1e-3)
        # ...but only near the aperture: the outermost paper row is untouched.
        np.testing.assert_array_equal(lit[0, 20:130], plain[0, 20:130])

    def test_corner_slider_rounds_the_corners(self) -> None:
        """You cannot file a sharp inside corner, so Corners pulls the aperture back there."""
        img = np.full((300, 400, 3), 0.5, dtype=np.float32)
        square = apply_carrier(img, width_px=16.0, rough=0.0, corner=0.0)
        round_ = apply_carrier(img, width_px=16.0, rough=0.0, corner=1.0)

        def corner_paper(a: np.ndarray) -> int:
            return int((a[:60, :60].min(axis=-1) == 1.0).sum())

        self.assertGreater(corner_paper(round_), corner_paper(square) * 1.05)

    def test_flare_color_vs_bw(self) -> None:
        img = self._image()
        plain = apply_carrier(img, width_px=8.0, rough=0.4)
        color = apply_carrier(img, width_px=8.0, rough=0.4, flare=1.0)
        mono = apply_carrier(img, width_px=8.0, rough=0.4, flare=1.0, bw=True)
        black = self._rebate_mask(plain)
        self.assertGreater(float(np.abs(np.diff(color[black], axis=-1)).max()), 1e-3)
        self.assertEqual(float(np.abs(np.diff(mono[black], axis=-1)).max()), 0.0)

    def test_flare_deterministic(self) -> None:
        img = self._image()
        a = apply_carrier(img, width_px=8.0, rough=1.0, flare=0.7)
        b = apply_carrier(img, width_px=8.0, rough=1.0, flare=0.7)
        np.testing.assert_array_equal(a, b)

    def test_noise_is_two_dimensional_and_bounded(self) -> None:
        x = np.linspace(0.0, 12.0, 64, dtype=np.float32)[None, :]
        y = np.linspace(0.0, 12.0, 48, dtype=np.float32)[:, None]
        n = carrier_noise(x, y)
        self.assertEqual(n.shape, (48, 64))
        self.assertLessEqual(float(np.abs(n).max()), 1.0)
        np.testing.assert_array_equal(n, carrier_noise(x, y))
        # Genuinely 2-D: rows differ, so the field is not a function of x alone.
        self.assertGreater(float(np.abs(n[0] - n[20]).max()), 0.05)

    def test_filed_edge_is_not_a_height_field(self) -> None:
        """The 2-D field must be able to overhang and shed flecks — a 1-D edge cannot."""
        img = np.full((300, 400, 3), 0.5, dtype=np.float32)
        res = apply_carrier(img, width_px=16.0, rough=1.0)
        black = res[:80, :].min(axis=-1) < 0.02
        # Count columns whose top band holds two or more disjoint black runs.
        breaks = np.diff(black.astype(np.int8), axis=0)
        detached = int((np.count_nonzero(breaks == 1, axis=0) > 1).sum())
        self.assertGreater(detached, 0)

    def test_paper_margin_takes_the_mat_color(self) -> None:
        img = np.full((300, 400, 3), 0.5, dtype=np.float32)
        res = apply_carrier(img, width_px=16.0, rough=0.0, paper=(0.2, 0.4, 0.6))
        np.testing.assert_allclose(res[0, 200], [0.2, 0.4, 0.6], atol=1e-6)
        np.testing.assert_array_equal(res[150, 200], img[150, 200])

    def test_soft_penumbra(self) -> None:
        """The rebate-to-image transition spans multiple pixels, not one."""
        img = np.full((200, 300, 3), 0.5, dtype=np.float32)
        res = apply_carrier(img, width_px=20.0, rough=0.0)
        cut = res[:40, 150, 0]
        partial = np.sum((cut > 0.05) & (cut < 0.45))
        self.assertGreaterEqual(int(partial), 3)

    def test_small_image_does_not_crash(self) -> None:
        img = np.full((4, 4, 3), 0.5, dtype=np.float32)
        res = apply_carrier(img, width_px=10.0, rough=1.0)
        self.assertEqual(res.shape, img.shape)


if __name__ == "__main__":
    unittest.main()
