import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.logic import shadow_neutral_offsets
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
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


class TestAutoShadowNeutral(unittest.TestCase):
    def _render(self, img: np.ndarray, auto: bool, mode: str = "C41") -> np.ndarray:
        config = WorkspaceConfig()
        # No analysis border crop — the fixture's cast fade sits near the
        # extreme and must stay inside the analyzed region.
        process = replace(config.process, analysis_buffer=0.0)
        ctx = PipelineContext(scale_factor=1.0, original_size=img.shape[:2], process_mode=mode)
        norm = NormalizationProcessor(process).process(img, ctx)
        exp = replace(config.exposure, auto_shadow_neutral=auto)
        return PhotometricProcessor(exp).process(norm, ctx)

    def test_cast_shrinks_in_print_shadows(self):
        img = _cast_negative()
        off = self._render(img, auto=False)
        on = self._render(img, auto=True)

        spread_off = abs(float(off[_PATCH, :, 1].mean()) - float(off[_PATCH, :, 2].mean()))
        spread_on = abs(float(on[_PATCH, :, 1].mean()) - float(on[_PATCH, :, 2].mean()))
        self.assertLess(spread_on, spread_off * 0.7)

    def test_neutral_image_unchanged(self):
        img = _cast_negative(cast=0.0)
        off = self._render(img, auto=False)
        on = self._render(img, auto=True)
        self.assertTrue(np.allclose(on, off, atol=1e-4))

    def test_offsets_clamped(self):
        limit = float(EXPOSURE_CONSTANTS["shadow_neutral_max_offset"])
        offsets = shadow_neutral_offsets(
            refs=(-0.30, -0.30, -2.0),
            floors=(-2.0, -2.0, -2.0),
            ceils=(-0.3, -0.3, -0.3),
        )
        self.assertEqual(offsets[1], 0.0)
        self.assertLessEqual(max(abs(o) for o in offsets), limit + 1e-9)

    def test_green_is_reference(self):
        offsets = shadow_neutral_offsets(
            refs=(-0.35, -0.32, -0.36),
            floors=(-2.0, -2.0, -2.0),
            ceils=(-0.3, -0.3, -0.3),
        )
        self.assertEqual(offsets[1], 0.0)
        self.assertGreater(offsets[0], 0.0)
        self.assertGreater(offsets[2], offsets[0])

    def test_bw_mode_noop(self):
        img = _cast_negative()
        off = self._render(img, auto=False, mode="BW")
        on = self._render(img, auto=True, mode="BW")
        self.assertTrue(np.allclose(on, off, atol=1e-6))

    def test_e6_mode_noop(self):
        img = _cast_negative()
        off = self._render(img, auto=False, mode="E6")
        on = self._render(img, auto=True, mode="E6")
        self.assertTrue(np.allclose(on, off, atol=1e-6))

    def test_serialization_roundtrip(self):
        config = WorkspaceConfig()
        self.assertTrue(config.exposure.auto_shadow_neutral)
        config = replace(config, exposure=replace(config.exposure, auto_shadow_neutral=False))
        restored = WorkspaceConfig.from_flat_dict(config.to_dict())
        self.assertFalse(restored.exposure.auto_shadow_neutral)


if __name__ == "__main__":
    unittest.main()
