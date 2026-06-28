import unittest

import numpy as np

from dataclasses import replace

from negpy.features.exposure.normalization import (
    LogNegativeBounds,
    analyze_log_exposure_bounds,
    mix_luma_colour_bounds,
    resolve_bounds,
    resolve_bounds_detailed,
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
    LOCAL = LogNegativeBounds((-1.0, -1.1, -0.9), (-0.6, -0.5, -0.7))  # per-frame

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

    def test_mix_identity_asymmetric(self):
        """Identity must hold for asymmetric channels too (mean != median), else a
        mix(base, base) drifts and stacks when its result is persisted and re-fed."""
        # floors mean (-1.45) != median (-1.5); ceils mean (-0.4) != median (-0.5).
        asym = LogNegativeBounds((-1.5, -1.5, -1.35), (-0.5, -0.5, -0.2))
        self._assert_close(mix_luma_colour_bounds(asym, asym), asym)

    def test_no_roll_resolve_is_stable_when_fed_back(self):
        """resolve_bounds with no roll baseline must return the local base unchanged,
        so persisting it and re-resolving doesn't accumulate (the edit-stacking bug)."""
        asym = LogNegativeBounds((-1.5, -1.5, -1.35), (-0.5, -0.5, -0.2))
        proc = replace(
            ProcessConfig(),
            use_luma_average=False,
            use_colour_average=False,
            local_floors=asym.floors,
            local_ceils=asym.ceils,
        )
        first = resolve_bounds(proc, self._boom)
        proc2 = replace(proc, local_floors=first.floors, local_ceils=first.ceils)
        second = resolve_bounds(proc2, self._boom)
        self._assert_close(first, second)

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

    def test_detailed_returns_per_frame_base(self):
        """resolve_bounds_detailed exposes the per-frame base alongside the final mix —
        the base is what must be persisted (persisting the mix drifts)."""
        proc = self._proc(use_luma_average=False, use_colour_average=True)
        final, base = resolve_bounds_detailed(proc, self._boom)
        self._assert_close(base, self.LOCAL)
        self._assert_close(final, mix_luma_colour_bounds(self.LOCAL, self.LOCKED))

    def test_persisting_base_is_stable_colour_only_roll(self):
        """Colour-only roll with an asymmetric baseline: persisting the per-frame base
        and re-feeding it must not accumulate (the residual edit-stacking path)."""
        locked = LogNegativeBounds((-2.0, -2.0, -1.7), (-0.3, -0.3, -0.05))
        local = LogNegativeBounds((-1.5, -1.4, -1.2), (-0.6, -0.5, -0.4))
        proc = replace(
            ProcessConfig(),
            use_luma_average=False,
            use_colour_average=True,
            locked_floors=locked.floors,
            locked_ceils=locked.ceils,
            local_floors=local.floors,
            local_ceils=local.ceils,
        )
        _, base = resolve_bounds_detailed(proc, self._boom)
        proc2 = replace(proc, local_floors=base.floors, local_ceils=base.ceils)
        _, base2 = resolve_bounds_detailed(proc2, self._boom)
        self._assert_close(base, base2)

    def test_colour_source_does_not_move_luma_range_or_centre(self):
        """The user's bug: swapping the colour source must not change the
        luma-weighted density range (H&D slope -> gamma) or centre (-> brightness),
        which only the luma source may set."""
        from negpy.features.exposure.normalization import LUMA_R, LUMA_G, LUMA_B, luminance_density_range

        w = (LUMA_R, LUMA_G, LUMA_B)
        centre = lambda b: sum(w[c] * (b.floors[c] + b.ceils[c]) / 2.0 for c in range(3))  # noqa: E731
        for luma_src, colour_src in ((self.LOCAL, self.LOCKED), (self.LOCKED, self.LOCAL)):
            mixed = mix_luma_colour_bounds(luma_src, colour_src)
            self.assertAlmostEqual(luminance_density_range(mixed), luminance_density_range(luma_src), places=6)
            self.assertAlmostEqual(centre(mixed), centre(luma_src), places=6)
        # And the cast still differs from the luma source (colour actually applied).
        self.assertNotAlmostEqual(
            mix_luma_colour_bounds(self.LOCAL, self.LOCKED).floors[0],
            self.LOCAL.floors[0],
            places=4,
        )

    def test_luma_source_bounds_ignores_colour_average(self):
        """The metered anchor (brightness) is read off these bounds, so they must
        depend only on the luma axis — toggling colour-average never moves them."""
        from negpy.features.exposure.normalization import luma_source_bounds

        off = self._proc(use_luma_average=False, use_colour_average=False)
        colour_on = self._proc(use_luma_average=False, use_colour_average=True)
        self._assert_close(luma_source_bounds(off, self.LOCAL), self.LOCAL)
        self._assert_close(luma_source_bounds(colour_on, self.LOCAL), self.LOCAL)
        # Luma-average on -> roll baseline, still independent of the colour toggle.
        luma_both = self._proc(use_luma_average=True, use_colour_average=True)
        luma_only = self._proc(use_luma_average=True, use_colour_average=False)
        self._assert_close(luma_source_bounds(luma_both, self.LOCAL), self.LOCKED)
        self._assert_close(luma_source_bounds(luma_only, self.LOCAL), self.LOCKED)

    def test_falls_back_to_analyze_when_locked_uninitialized(self):
        """Flags on but no roll baseline -> the per-frame analyze_fn supplies the base."""
        analyzed = LogNegativeBounds((-1.5, -1.5, -1.5), (-0.4, -0.4, -0.4))
        proc = replace(ProcessConfig(), use_luma_average=True, use_colour_average=True)  # local & locked both zero
        self._assert_close(resolve_bounds(proc, lambda: analyzed), analyzed)


if __name__ == "__main__":
    unittest.main()
