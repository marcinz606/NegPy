import unittest

import numpy as np

from negpy.features.exposure.analysis import DENSITY_HIST_BINS, DENSITY_HIST_RANGE, density_histogram
from negpy.infrastructure.gpu.device import GPUDevice


def _bin_of(val: float) -> int:
    lo, hi = DENSITY_HIST_RANGE
    return min(DENSITY_HIST_BINS - 1, int((val - lo) / (hi - lo) * DENSITY_HIST_BINS))


def _gray(vals) -> np.ndarray:
    v = np.asarray(vals, dtype=np.float32)
    return np.stack([v, v, v], axis=-1).reshape(1, -1, 3)


class TestDensityHistogram(unittest.TestCase):
    def test_bin_placement(self):
        # Bin-center value: exact bin edges are float32-fragile by construction.
        hist = density_histogram(_gray([0.505]))
        self.assertEqual(hist.sum(), 1)
        self.assertEqual(int(np.argmax(hist)), _bin_of(0.505))

    def test_out_of_range_mass_lands_in_edge_bins(self):
        hist = density_histogram(_gray([-0.5, -0.2, 2.0]))
        self.assertEqual(hist[0], 2)
        self.assertEqual(hist[-1], 1)
        self.assertEqual(hist.sum(), 3)

    def test_roi_slicing(self):
        img = np.zeros((4, 4, 3), dtype=np.float32)
        img[1:3, 1:3] = 0.905
        hist = density_histogram(img, roi=(1, 3, 1, 3))
        self.assertEqual(hist.sum(), 4)
        self.assertEqual(hist[_bin_of(0.905)], 4)
        self.assertEqual(hist[_bin_of(0.005)], 0)

    def test_luma_weighting(self):
        img = np.zeros((1, 1, 3), dtype=np.float32)
        img[..., 1] = 1.0
        hist = density_histogram(img)
        self.assertEqual(int(np.argmax(hist)), _bin_of(0.7152))


class TestOutputHistogram(unittest.TestCase):
    def test_bin_array_passthrough(self):
        from negpy.features.exposure.analysis import output_histogram

        bins = np.arange(4 * 256, dtype=np.uint32).reshape(4, 256)
        out = output_histogram(bins)
        self.assertTrue(np.array_equal(out, bins.astype(float)))

    def test_buffer_binning(self):
        from negpy.features.exposure.analysis import output_histogram

        buf = np.full((8, 8, 3), 0.5, dtype=np.float32)
        buf[..., 0] = 0.905
        out = output_histogram(buf)
        self.assertEqual(out.shape, (4, 256))
        self.assertEqual(float(out[0].sum()), 64.0)
        self.assertEqual(int(np.argmax(out[0])), int(0.905 * 256))
        self.assertEqual(int(np.argmax(out[1])), 128)

    def test_rejects_junk(self):
        from negpy.features.exposure.analysis import output_histogram

        self.assertIsNone(output_histogram(None))
        self.assertIsNone(output_histogram(np.zeros((4, 4))))


class TestZones(unittest.TestCase):
    def test_zone_ruler_anchors(self):
        from negpy.features.exposure.analysis import zone_of_encoded
        from negpy.kernel.image.logic import working_oetf_encode

        mid = float(working_oetf_encode(np.asarray([0.18], dtype=np.float32))[0])
        self.assertAlmostEqual(float(zone_of_encoded(0.0)), 0.0)
        self.assertAlmostEqual(float(zone_of_encoded(mid)), 5.0, places=5)
        self.assertAlmostEqual(float(zone_of_encoded(1.0)), 10.0)

    def test_zone_occupancy_placement(self):
        from negpy.features.exposure.analysis import ZONE_COUNT, zone_occupancy

        bins = np.zeros(256)
        bins[0] = 1.0  # paper black → Zone 0
        bins[255] = 3.0  # paper white → Zone IX cell
        occ = zone_occupancy(bins)
        self.assertEqual(occ.shape, (ZONE_COUNT,))
        self.assertAlmostEqual(float(occ[0]), 0.25)
        self.assertAlmostEqual(float(occ[9]), 0.75)
        self.assertAlmostEqual(float(occ.sum()), 1.0)

    def test_zone_warnings_truth_table(self):
        from negpy.features.exposure.analysis import zone_warnings

        blocked = np.zeros(10)
        blocked[0] = 0.30  # extreme loaded, texture zones II+III empty
        self.assertEqual(zone_warnings(blocked), (True, False))
        textured = np.zeros(10)
        textured[0] = 0.30
        textured[3] = 0.10  # texture present → no warning
        self.assertEqual(zone_warnings(textured), (False, False))
        blown = np.zeros(10)
        blown[9] = 0.10
        self.assertEqual(zone_warnings(blown), (False, True))
        self.assertEqual(zone_warnings(np.zeros(10)), (False, False))


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestDensityHistogramGpuParity(unittest.TestCase):
    def test_cpu_gpu_distributions_agree(self):
        from negpy.domain.models import WorkspaceConfig
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

        rng = np.random.default_rng(0)
        h, w = 64, 64
        grad = np.linspace(0.05, 0.9, w, dtype=np.float32)
        img = np.repeat(grad[None, :], h, axis=0)
        img = np.stack([img, img * 0.95, img * 0.9], axis=-1)
        img = np.ascontiguousarray(img + rng.uniform(0, 0.01, img.shape).astype(np.float32))

        settings = WorkspaceConfig()
        _, m_cpu = processor.run_pipeline(img, settings, "dh-parity-cpu", render_size_ref=64.0, prefer_gpu=False)
        _, m_gpu = processor.run_pipeline(img, settings, "dh-parity-gpu", render_size_ref=64.0, prefer_gpu=True)

        hc = np.asarray(m_cpu["histogram_density"], dtype=np.float64)
        hg = np.asarray(m_gpu["histogram_density"], dtype=np.float64)
        self.assertEqual(hc.shape, (DENSITY_HIST_BINS,))
        self.assertEqual(hg.shape, (DENSITY_HIST_BINS,))
        self.assertGreater(hc.sum(), 0)
        self.assertGreater(hg.sum(), 0)
        # Distribution-level agreement: engines sample/round differently.
        l1 = float(np.abs(hc / hc.sum() - hg / hg.sum()).sum())
        self.assertLess(l1, 0.05, f"normalized L1 distance {l1:.4f}")


if __name__ == "__main__":
    unittest.main()
