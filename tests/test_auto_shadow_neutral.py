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

    def _render(self, img: np.ndarray, cast_removal: bool, mode: str = "C41") -> np.ndarray:
        config = WorkspaceConfig()
        # No analysis border crop — the fixture's cast fade sits near the
        # extreme and must stay inside the analyzed region.
        process = replace(config.process, analysis_buffer=0.0)
        ctx = PipelineContext(scale_factor=1.0, original_size=img.shape[:2], process_mode=mode)
        norm = NormalizationProcessor(process).process(img, ctx)
        exp = replace(config.exposure, cast_removal=cast_removal)
        return PhotometricProcessor(exp).process(norm, ctx)

    def test_cast_shrinks_in_print_shadows(self):
        img = _cast_negative()
        off = self._render(img, cast_removal=False)
        on = self._render(img, cast_removal=True)
        spread_off = abs(float(off[_PATCH, :, 1].mean()) - float(off[_PATCH, :, 2].mean()))
        spread_on = abs(float(on[_PATCH, :, 1].mean()) - float(on[_PATCH, :, 2].mean()))
        self.assertLess(spread_on, spread_off * 0.7)

    def test_neutral_image_unchanged(self):
        img = _cast_negative(cast=0.0)
        off = self._render(img, cast_removal=False)
        on = self._render(img, cast_removal=True)
        self.assertTrue(np.allclose(on, off, atol=1e-4))

    def test_e6_mode_noop(self):
        # E6 measures no shadow refs -> cast removal falls back to the single curve.
        img = _cast_negative()
        off = self._render(img, cast_removal=False, mode="E6")
        on = self._render(img, cast_removal=True, mode="E6")
        self.assertTrue(np.allclose(on, off, atol=1e-6))

    def test_default_on(self):
        self.assertTrue(WorkspaceConfig().exposure.cast_removal)

    def test_serialization_roundtrip(self):
        config = replace(WorkspaceConfig(), exposure=replace(WorkspaceConfig().exposure, cast_removal=False))
        restored = WorkspaceConfig.from_flat_dict(config.to_dict())
        self.assertFalse(restored.exposure.cast_removal)

    def test_legacy_auto_shadow_neutral_migrates(self):
        # Old saved edits used auto_shadow_neutral; it must map to cast_removal.
        self.assertFalse(WorkspaceConfig.from_flat_dict({"auto_shadow_neutral": False}).exposure.cast_removal)
        self.assertTrue(WorkspaceConfig.from_flat_dict({"auto_shadow_neutral": True}).exposure.cast_removal)


if __name__ == "__main__":
    unittest.main()
