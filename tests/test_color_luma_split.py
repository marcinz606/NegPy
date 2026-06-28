import unittest

import numpy as np

from dataclasses import replace

from negpy.features.exposure.normalization import (
    LogNegativeBounds,
    analyze_log_exposure_bounds,
    mix_luma_colour_bounds,
    resolve_bounds,
)
from negpy.features.process.models import ProcessConfig


def _offset_image() -> np.ndarray:
    # Gradient base with per-channel linear gain -> per-channel log offset (a colour
    # cast), identical span per channel.
    vals = np.linspace(0.02, 1.0, 10000, dtype=np.float32).reshape(100, 100)
    return np.stack([vals, vals * 0.7, vals * 0.85], axis=-1)


def _mono_image() -> np.ndarray:
    vals = np.linspace(0.02, 1.0, 10000, dtype=np.float32).reshape(100, 100)
    return np.stack([vals, vals, vals], axis=-1)


class TestColorLumaSplit(unittest.TestCase):
    def setUp(self):
        self.img = _offset_image()

    def test_luma_drives_mean_center_and_span(self):
        """Overall (mean) floor/ceil and span come purely from the luma sampling —
        the colour clip only redistributes per-channel deviation, never the mean."""
        p_luma = 0.6
        a = analyze_log_exposure_bounds(self.img, percentile_clip=p_luma, color_clip=5.0)
        b = analyze_log_exposure_bounds(self.img, percentile_clip=p_luma, color_clip=0.5)

        self.assertAlmostEqual(sum(a.floors) / 3.0, sum(b.floors) / 3.0, places=5)
        self.assertAlmostEqual(sum(a.ceils) / 3.0, sum(b.ceils) / 3.0, places=5)
        self.assertAlmostEqual((sum(a.ceils) - sum(a.floors)) / 3.0, (sum(b.ceils) - sum(b.floors)) / 3.0, places=5)

    def test_colour_clip_preserves_luma_span(self):
        """Changing the colour clip must not change the mean span (highlights)."""
        a = analyze_log_exposure_bounds(self.img, percentile_clip=0.0, color_clip=10.0)
        b = analyze_log_exposure_bounds(self.img, percentile_clip=0.0, color_clip=0.01)
        span_a = (sum(a.ceils) - sum(a.floors)) / 3.0
        span_b = (sum(b.ceils) - sum(b.floors)) / 3.0
        self.assertAlmostEqual(span_a, span_b, places=5)

    def test_colour_pass_injects_per_channel_cast(self):
        """The colour pass carries the per-channel cast into the bounds: the more
        attenuated channels (lower linear gain) get a lower floor."""
        bounds = analyze_log_exposure_bounds(self.img, percentile_clip=0.0, color_clip=5.0)
        # Channel gains were 1.0, 0.7, 0.85 -> log floors ordered r > b > g.
        self.assertLess(bounds.floors[1], bounds.floors[2])
        self.assertLess(bounds.floors[2], bounds.floors[0])

    def test_mono_image_has_no_cast(self):
        """Identical channels -> zero colour deviation -> all channels share the luma
        sampling regardless of colour clip."""
        mono = _mono_image()
        a = analyze_log_exposure_bounds(mono, percentile_clip=0.3, color_clip=10.0)
        b = analyze_log_exposure_bounds(mono, percentile_clip=0.3, color_clip=0.01)
        for ch in range(3):
            self.assertAlmostEqual(a.floors[ch], b.floors[ch], places=6)
            self.assertAlmostEqual(a.ceils[ch], b.ceils[ch], places=6)
            self.assertAlmostEqual(a.floors[ch], a.floors[0], places=6)


class TestResolveBounds(unittest.TestCase):
    """The roll baseline can be applied per axis: luma (span) and colour (cast)
    independently take from the roll (locked) or the per-frame (local) bounds."""

    # Distinct luma mean + colour deviation so each source is identifiable per channel.
    LOCKED = LogNegativeBounds((-2.0, -2.2, -2.1), (-0.2, -0.3, -0.1))  # roll
    LOCAL = LogNegativeBounds((-1.0, -1.1, -0.9), (-0.6, -0.5, -0.7))   # per-frame

    def _proc(self, **kw) -> ProcessConfig:
        return replace(
            ProcessConfig(),
            locked_floors=self.LOCKED.floors,
            locked_ceils=self.LOCKED.ceils,
            local_floors=self.LOCAL.floors,
            local_ceils=self.LOCAL.ceils,
            **kw,
        )

    def _boom(self) -> LogNegativeBounds:
        raise AssertionError("analyze_fn must not be called when local is initialized")

    def _assert_close(self, a: LogNegativeBounds, b: LogNegativeBounds) -> None:
        for ch in range(3):
            self.assertAlmostEqual(a.floors[ch], b.floors[ch], places=6)
            self.assertAlmostEqual(a.ceils[ch], b.ceils[ch], places=6)

    def test_mix_identity(self):
        """Mixing a bounds with itself is the identity."""
        self._assert_close(mix_luma_colour_bounds(self.LOCKED, self.LOCKED), self.LOCKED)

    def test_both_on_uses_locked(self):
        proc = self._proc(use_luma_average=True, use_colour_average=True)
        self._assert_close(resolve_bounds(proc, self._boom), self.LOCKED)

    def test_both_off_uses_local(self):
        proc = self._proc(use_luma_average=False, use_colour_average=False)
        self._assert_close(resolve_bounds(proc, self._boom), self.LOCAL)

    def test_luma_only_mixes_locked_luma_with_local_colour(self):
        proc = self._proc(use_luma_average=True, use_colour_average=False)
        self._assert_close(resolve_bounds(proc, self._boom), mix_luma_colour_bounds(self.LOCKED, self.LOCAL))

    def test_colour_only_mixes_local_luma_with_locked_colour(self):
        proc = self._proc(use_luma_average=False, use_colour_average=True)
        self._assert_close(resolve_bounds(proc, self._boom), mix_luma_colour_bounds(self.LOCAL, self.LOCKED))

    def test_falls_back_to_analyze_when_locked_uninitialized(self):
        """Flags on but no roll baseline -> the per-frame analyze_fn supplies the base."""
        analyzed = LogNegativeBounds((-1.5, -1.5, -1.5), (-0.4, -0.4, -0.4))
        proc = replace(ProcessConfig(), use_luma_average=True, use_colour_average=True)  # local & locked both zero
        self._assert_close(resolve_bounds(proc, lambda: analyzed), analyzed)


if __name__ == "__main__":
    unittest.main()
