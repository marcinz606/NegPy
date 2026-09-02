"""Printability read-out: the joint color histogram, the ICC gamut mask built from it,
and the fraction the Analysis panel shows.

The engine measures colors and never learns which profile is being proofed to, so the
two halves are tested apart: binning (CPU vs the WGSL pass) and the gamut mask (against
profile pairs whose answer is known from their primaries).
"""

import unittest

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.analysis import COLOR_HIST_BINS, color_histogram, gamut_fraction
from negpy.features.exposure.stats import negative_statistics
from negpy.infrastructure.display.color_spaces import ColorSpaceRegistry
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.services.rendering.image_processor import ImageProcessor


def _names(rows):
    return [r.name for r in rows]


def _by_name(rows, name):
    return next(r for r in rows if r.name == name)


def _lut(dst_name, src="Adobe RGB"):
    return ImageProcessor.gamut_lut(src, None, ColorSpaceRegistry.get_icc_path(dst_name))


def _cpu_color_histogram(result):
    """What workers/render.py bins for a CPU render."""
    return color_histogram(result)


class TestColorHistogram(unittest.TestCase):
    def test_counts_every_pixel_once(self):
        rng = np.random.default_rng(0)
        buf = rng.random((64, 48, 3)).astype(np.float32)
        self.assertEqual(color_histogram(buf).sum(), 64 * 48)

    def test_a_single_color_lands_in_one_bin(self):
        buf = np.full((8, 8, 3), 0.5, dtype=np.float32)
        hist = color_histogram(buf)
        self.assertEqual(int((hist > 0).sum()), 1)
        self.assertEqual(hist[16, 16, 16], 64)

    def test_out_of_range_values_clamp_into_the_end_bins(self):
        buf = np.stack(
            [np.full((4, 4), -0.5, dtype=np.float32), np.full((4, 4), 2.0, dtype=np.float32), np.zeros((4, 4), np.float32)],
            axis=-1,
        )
        hist = color_histogram(buf)
        self.assertEqual(hist.sum(), 16)
        self.assertEqual(hist[0, COLOR_HIST_BINS - 1, 0], 16)

    def test_a_ready_grid_passes_through(self):
        grid = np.zeros((COLOR_HIST_BINS,) * 3)
        grid[1, 2, 3] = 5.0
        self.assertEqual(color_histogram(grid)[1, 2, 3], 5.0)


class TestGamutLut(unittest.TestCase):
    def test_no_output_profile_is_no_mask(self):
        self.assertIsNone(ImageProcessor.gamut_lut("Adobe RGB", None, None))

    def test_a_profile_cannot_fail_its_own_colors(self):
        """The round-trip tolerance has to clear the transform's own noise, or every
        read-out carries a false floor."""
        self.assertEqual(float(_lut("Adobe RGB").mean()), 0.0)

    def test_an_input_class_source_profile_still_builds(self):
        # RGBScan.icc is the implicit Narrowband Scan input profile: source-only, no B2A
        # table, so the round trip must land in the working space rather than back in it.
        from negpy.kernel.system.paths import get_resource_path

        mask = ImageProcessor.gamut_lut("Adobe RGB", get_resource_path("icc/RGBScan.icc"), ColorSpaceRegistry.get_icc_path("sRGB"))
        self.assertIsNotNone(mask)
        self.assertGreater(float(mask.mean()), 0.0)

    def test_a_wider_destination_clips_nothing(self):
        self.assertEqual(float(ImageProcessor.gamut_lut("sRGB", None, ColorSpaceRegistry.get_icc_path("Rec 2020")).mean()), 0.0)

    def test_a_narrower_destination_clips_saturated_color_but_not_neutrals(self):
        lut = _lut("sRGB")
        self.assertGreater(float(lut.mean()), 0.1)
        axis = np.arange(COLOR_HIST_BINS)
        self.assertFalse(lut[axis, axis, axis].any(), "the neutral axis is common to both gamuts")
        self.assertTrue(lut[0, COLOR_HIST_BINS - 1, 0], "saturated green is outside sRGB")


class TestGamutFraction(unittest.TestCase):
    def test_none_without_a_mask_or_a_histogram(self):
        self.assertIsNone(gamut_fraction(np.zeros((COLOR_HIST_BINS,) * 3), None))
        self.assertIsNone(gamut_fraction(None, np.zeros((COLOR_HIST_BINS,) * 3, dtype=bool)))

    def test_is_the_share_of_pixels_not_of_bins(self):
        hist = np.zeros((COLOR_HIST_BINS,) * 3)
        hist[0, 0, 0] = 30.0
        hist[31, 0, 0] = 10.0
        mask = np.zeros((COLOR_HIST_BINS,) * 3, dtype=bool)
        mask[31, 0, 0] = True
        self.assertAlmostEqual(gamut_fraction(hist, mask), 0.25)

    def test_an_empty_frame_reports_nothing(self):
        self.assertIsNone(gamut_fraction(np.zeros((COLOR_HIST_BINS,) * 3), np.ones((COLOR_HIST_BINS,) * 3, dtype=bool)))

    def test_a_neutral_frame_is_printable_everywhere(self):
        buf = np.full((16, 16, 3), 0.5, dtype=np.float32)
        self.assertEqual(gamut_fraction(color_histogram(buf), _lut("sRGB")), 0.0)


class TestGamutRow(unittest.TestCase):
    def _rows(self, gamut):
        return negative_statistics(1.3, 0.46, 0.0, 0.0, gamut=gamut)

    def test_absent_while_nothing_is_being_proofed(self):
        self.assertNotIn("Gamut", _names(self._rows(None)))

    def test_present_at_zero_once_a_profile_is_proofed_to(self):
        """Zero is an answer: the frame prints. Absent means the question was not asked."""
        self.assertEqual(_by_name(self._rows(0.0), "Gamut").value, "0.0% unprintable")

    def test_warns_above_two_percent(self):
        self.assertTrue(_by_name(self._rows(0.03), "Gamut").warn)
        self.assertFalse(_by_name(self._rows(0.01), "Gamut").warn)


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestColorHistogramParity(unittest.TestCase):
    def test_wgsl_bins_match_the_cpu(self):
        """color_hist.wgsl and color_histogram() bin the same frame; a drift makes the
        printability number depend on which engine rendered."""
        rng = np.random.default_rng(7)
        img = rng.random((240, 180, 3)).astype(np.float32) * 0.9 + 0.05
        proc = ImageProcessor()
        cfg = WorkspaceConfig()

        cpu_out, cpu_metrics = proc.run_pipeline(img, cfg, "parity|cpu", render_size_ref=240, prefer_gpu=False, readback_metrics=True)
        _gpu_out, gpu_metrics = proc.run_pipeline(img, cfg, "parity|gpu", render_size_ref=240, prefer_gpu=True, readback_metrics=True)

        gpu_hist = gpu_metrics.get("histogram_color")
        if gpu_hist is None:
            self.skipTest("render fell back to the CPU engine")
        # A CPU render carries no in-shader histogram; the worker bins the float output.
        cpu_hist = _cpu_color_histogram(cpu_out)
        self.assertEqual(cpu_hist.sum(), gpu_hist.sum())
        a = cpu_hist / cpu_hist.sum()
        b = gpu_hist / gpu_hist.sum()
        self.assertLess(float(np.abs(a - b).sum()), 0.01)


if __name__ == "__main__":
    unittest.main()
