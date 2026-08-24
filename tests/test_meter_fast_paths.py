"""The meter's percentile and log-density shortcuts must be bit-exact, not close.

Both replace numpy expressions whose exact output is baked into every golden, so these
assert `==`, never approx.
"""

import unittest

import numpy as np

from negpy.features.exposure.normalization import (
    analyze_log_exposure_bounds_from_log,
    measure_shadow_refs_from_log,
    percentile_from_sorted,
    prefilter_log_grid,
    sorted_channel_grid,
    to_log_density,
)
from negpy.features.process.models import ProcessMode

_QS = [0.0, 1e-5, 0.05, 0.35, 1.0, 2.5, 10.0, 33.3333, 49.999, 50.0, 66.6, 90.0, 99.65, 99.99999, 100.0]


def _grid(shape, seed=0, scale=1.0):
    return ((np.random.default_rng(seed).random(shape).astype(np.float32) - 0.5) * scale).copy()


class TestPercentileFromSorted(unittest.TestCase):
    def test_matches_numpy_for_python_and_numpy_quantiles(self):
        """numpy interpolates in float32 for a plain float q and float64 for np.float64;
        both kinds of caller exist in the meters, so both must land on the same bits."""
        for shape in ((533, 800, 3), (37, 41, 3), (2, 3, 3), (1, 1, 3), (4, 4, 3)):
            for scale in (1.0, 1e-3, 1e3):
                g = _grid(shape, scale=scale)
                srt = sorted_channel_grid(g)
                for q in _QS:
                    for qq in (q, np.float64(q)):
                        got = percentile_from_sorted(srt, qq)
                        for ch in range(3):
                            self.assertEqual(
                                float(np.percentile(g[:, :, ch], qq)),
                                float(got[ch]),
                                msg=f"shape={shape} scale={scale} q={q!r} ({type(qq).__name__}) ch={ch}",
                            )


class TestSortedGridMeters(unittest.TestCase):
    def test_bounds_and_shadow_refs_identical_with_and_without_the_sort(self):
        g = prefilter_log_grid(np.clip(_grid((600, 900, 3), seed=3, scale=0.4) + 0.5, 1e-4, 1.0), None, 0.0)
        srt = sorted_channel_grid(g)
        for mode in (ProcessMode.C41, ProcessMode.E6):
            for e6n in (True, False):
                for luma_clip in (-0.5, 0.0, 0.35, 2.5):
                    for color_clip in (0.0, 0.35, 5.0):
                        slow = analyze_log_exposure_bounds_from_log(
                            g, None, 0.0, process_mode=mode, e6_normalize=e6n, percentile_clip=luma_clip, color_clip=color_clip
                        )
                        fast = analyze_log_exposure_bounds_from_log(
                            g,
                            None,
                            0.0,
                            process_mode=mode,
                            e6_normalize=e6n,
                            percentile_clip=luma_clip,
                            color_clip=color_clip,
                            sorted_grid=srt,
                        )
                        self.assertEqual(slow.floors, fast.floors)
                        self.assertEqual(slow.ceils, fast.ceils)
        self.assertEqual(measure_shadow_refs_from_log(g, None, 0.0), measure_shadow_refs_from_log(g, None, 0.0, sorted_grid=srt))

    def test_sorted_grid_is_ignored_when_a_roi_still_has_to_be_applied(self):
        """The sort describes the whole grid, so a caller that still crops must not use it."""
        g = prefilter_log_grid(np.clip(_grid((600, 900, 3), seed=4, scale=0.4) + 0.5, 1e-4, 1.0), None, 0.0)
        roi = (10, 300, 20, 400)
        wrong = sorted_channel_grid(g)
        self.assertEqual(
            measure_shadow_refs_from_log(g, roi, 0.0),
            measure_shadow_refs_from_log(g, roi, 0.0, sorted_grid=wrong),
        )


class TestToLogDensity(unittest.TestCase):
    def test_matches_the_nan_to_num_clip_it_replaces(self):
        eps = 1e-6
        for dtype in (np.float32, np.float64):
            x = (np.random.default_rng(1).random((97, 89, 3)) * 1.4 - 0.2).astype(dtype)
            x[0, 0] = np.nan
            x[1, 1] = np.inf
            x[2, 2] = -np.inf
            x[3, 3] = 0.0
            x[4, 4] = 1.0
            x[5, 5] = eps
            x[6, 6] = -5.0
            x[7, 7] = 1e30
            expected = np.log10(np.clip(np.nan_to_num(x, nan=eps, posinf=1.0, neginf=eps), eps, 1.0))
            got = to_log_density(x)
            self.assertEqual(expected.dtype, got.dtype)
            self.assertTrue(np.array_equal(expected, got), msg=f"{dtype.__name__} differs")


if __name__ == "__main__":
    unittest.main()


class TestBorderlessLayoutEarlyOut(unittest.TestCase):
    """A borderless layout must return exactly what the padded path built."""

    def test_no_border_returns_the_scaled_content_itself(self):
        from negpy.domain.models import AspectRatio, ExportConfig, ExportResolutionMode
        from negpy.services.export.print import PrintService

        img = _grid((120, 180, 3), seed=7, scale=0.5) + 0.5
        export = ExportConfig(export_resolution_mode=ExportResolutionMode.ORIGINAL, paper_aspect_ratio=AspectRatio.ORIGINAL)
        out, rect = PrintService.apply_layout(img, export, border_size=0.0)
        self.assertEqual(rect, (0, 0, 180, 120))
        self.assertTrue(np.array_equal(out, img))

    def test_a_border_still_pads(self):
        from negpy.domain.models import AspectRatio, ExportConfig, ExportResolutionMode
        from negpy.services.export.print import PrintService

        img = _grid((120, 180, 3), seed=7, scale=0.5) + 0.5
        export = ExportConfig(export_resolution_mode=ExportResolutionMode.ORIGINAL, paper_aspect_ratio=AspectRatio.ORIGINAL)
        out, (ox, oy, w, h) = PrintService.apply_layout(img, export, border_size=1.0, border_color="#000000")
        self.assertGreater(out.shape[0], img.shape[0])
        self.assertGreater(out.shape[1], img.shape[1])
        self.assertTrue(np.array_equal(out[oy : oy + h, ox : ox + w], img))
