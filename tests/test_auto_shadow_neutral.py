import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.logic import shadow_neutral_offsets
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.exposure.processor import NormalizationProcessor, PhotometricProcessor


def _cast_negative(h: int = 256, w: int = 64, skew: float = 1.15) -> np.ndarray:
    """
    Synthetic C-41 negative with a per-channel gamma skew: all channels share the
    same density endpoints (so min/max normalization aligns the bounds), but the
    blue channel's interior tones sit at different densities — exactly the
    film-gamma mismatch that leaves a cast in print shadows.
    """
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None].repeat(w, axis=1)
    log_rg = -2.0 + 1.7 * t
    log_b = -2.0 + 1.7 * t**skew
    return np.stack([10.0**log_rg, 10.0**log_rg, 10.0**log_b], axis=-1).astype(np.float32)


class TestAutoShadowNeutral(unittest.TestCase):
    def _render(self, img: np.ndarray, auto: bool, mode: str = "C41") -> np.ndarray:
        config = WorkspaceConfig()
        ctx = PipelineContext(scale_factor=1.0, original_size=img.shape[:2], process_mode=mode)
        norm = NormalizationProcessor(config.process).process(img, ctx)
        exp = replace(config.exposure, auto_shadow_neutral=auto)
        return PhotometricProcessor(exp).process(norm, ctx)

    def test_cast_shrinks_in_print_shadows(self):
        img = _cast_negative()
        off = self._render(img, auto=False)
        on = self._render(img, auto=True)

        # Print shadows = thin negative side = bottom rows of the gradient.
        shadows_off = off[-24:, :, :]
        shadows_on = on[-24:, :, :]
        spread_off = abs(float(shadows_off[..., 1].mean()) - float(shadows_off[..., 2].mean()))
        spread_on = abs(float(shadows_on[..., 1].mean()) - float(shadows_on[..., 2].mean()))
        self.assertLess(spread_on, spread_off * 0.5)

    def test_neutral_image_unchanged(self):
        img = _cast_negative(skew=1.0)
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
