import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.processor import NormalizationProcessor, PhotometricProcessor


_H = 1000
_PATCH = slice(int(0.89 * _H), int(0.99 * _H))


def _cast_negative(h: int = _H, w: int = 32, cast: float = 0.06) -> np.ndarray:
    """
    Synthetic C-41 negative in three zones: a tonal gradient, a deep-shadow
    patch carrying a blue cast (the dense-end channel misalignment), and a 1%
    thinnest-extreme anchor that is neutral — so the robust bounds stay
    channel-aligned while the p98 shadow reference lands inside the cast patch.
    """
    n_grad, n_patch = _PATCH.start, _PATCH.stop - _PATCH.start
    log_g = np.concatenate(
        [
            np.linspace(-2.83, -1.35, n_grad, dtype=np.float32),
            np.full(n_patch, -1.22, dtype=np.float32),
            np.full(h - n_grad - n_patch, -0.35, dtype=np.float32),
        ]
    )[:, None].repeat(w, axis=1)
    log_b = log_g.copy()
    log_b[_PATCH] -= cast
    return np.stack([10.0**log_g, 10.0**log_g, 10.0**log_b], axis=-1).astype(np.float32)


class TestCastRemoval(unittest.TestCase):
    """
    Cast Removal: the consolidated per-channel gray balance (two-point slope
    solve) that neutralizes a negative's residual color cast across the range.
    """

    def _render(self, img: np.ndarray, strength: float, mode: str = "C41", auto: bool = False) -> np.ndarray:
        config = WorkspaceConfig()
        # No analysis border crop — the fixture's cast fade sits near the
        # extreme and must stay inside the analyzed region.
        process = replace(config.process, analysis_buffer=0.0)
        ctx = PipelineContext(scale_factor=1.0, original_size=img.shape[:2], process_mode=mode)
        norm = NormalizationProcessor(process).process(img, ctx)
        exp = replace(config.exposure, cast_removal_strength=strength, auto_cast_removal=auto)
        return PhotometricProcessor(exp).process(norm, ctx)

    @staticmethod
    def _spread(out: np.ndarray) -> float:
        return abs(float(out[_PATCH, :, 1].mean()) - float(out[_PATCH, :, 2].mean()))

    def test_cast_shrinks_in_print_shadows(self):
        img = _cast_negative()
        off = self._render(img, strength=0.0)
        on = self._render(img, strength=1.0)
        self.assertLess(self._spread(on), self._spread(off) * 0.7)

    def test_strength_lerp_monotonic(self):
        # 0 = uncorrected, 1 = full; intermediate strengths sit strictly between.
        img = _cast_negative()
        s0 = self._spread(self._render(img, strength=0.0))
        s_half = self._spread(self._render(img, strength=0.5))
        s1 = self._spread(self._render(img, strength=1.0))
        self.assertGreater(s0, s_half)
        self.assertGreater(s_half, s1)

    def test_neutral_image_unchanged(self):
        img = _cast_negative(cast=0.0)
        off = self._render(img, strength=0.0)
        on = self._render(img, strength=1.0)
        self.assertTrue(np.allclose(on, off, atol=1e-4))

    def test_e6_mode_noop(self):
        # E6 measures no shadow refs -> cast removal falls back to the single curve.
        img = _cast_negative()
        off = self._render(img, strength=0.0, mode="E6")
        on = self._render(img, strength=1.0, mode="E6")
        self.assertTrue(np.allclose(on, off, atol=1e-6))

    def test_auto_scales_by_confidence(self):
        # A clean-neutral frame yields near-full auto correction; a casted frame's
        # near-neutral set is less tight, so auto applies a gentler correction.
        clean, casted = _cast_negative(cast=0.0), _cast_negative(cast=0.06)
        auto_clean = self._spread(self._render(clean, strength=1.0, auto=True))
        # Clean frame has no cast to begin with; confirm auto is a near-no-op there.
        self.assertLess(auto_clean, 1e-3)
        # On a casted frame, auto still reduces the spread vs. uncorrected.
        auto_cast = self._spread(self._render(casted, strength=1.0, auto=True))
        off_cast = self._spread(self._render(casted, strength=0.0))
        self.assertLess(auto_cast, off_cast)

    def test_auto_slider_still_trims(self):
        # With Auto on, the slider trims on top of the confidence ceiling.
        img = _cast_negative()
        full = self._spread(self._render(img, strength=1.0, auto=True))
        half = self._spread(self._render(img, strength=0.5, auto=True))
        self.assertGreater(half, full)

    def test_default_on(self):
        self.assertEqual(WorkspaceConfig().exposure.cast_removal_strength, 0.5)
        self.assertFalse(WorkspaceConfig().exposure.auto_cast_removal)

    def test_serialization_roundtrip(self):
        config = replace(WorkspaceConfig(), exposure=replace(WorkspaceConfig().exposure, cast_removal_strength=0.5))
        restored = WorkspaceConfig.from_flat_dict(config.to_dict())
        self.assertEqual(restored.exposure.cast_removal_strength, 0.5)

    def test_legacy_bool_migrates_to_strength(self):
        # Old saved edits stored cast_removal / auto_shadow_neutral as a bool.
        self.assertEqual(WorkspaceConfig.from_flat_dict({"cast_removal": True}).exposure.cast_removal_strength, 1.0)
        self.assertEqual(WorkspaceConfig.from_flat_dict({"cast_removal": False}).exposure.cast_removal_strength, 0.0)
        self.assertEqual(WorkspaceConfig.from_flat_dict({"auto_shadow_neutral": False}).exposure.cast_removal_strength, 0.0)


if __name__ == "__main__":
    unittest.main()
