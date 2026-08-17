"""The unsharp contrast-reduction mask.

Pins what would break silently: the sign, resolution invariance, the crop boundary and
CPU/GPU parity.
"""

import unittest
from dataclasses import replace

import cv2
import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.logic import contrast_mask_ev
from negpy.features.exposure.normalization import LogNegativeBounds, contrast_mask_plane
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.services.rendering.engine import DarkroomEngine


def _wide_range_negative(h: int = 240, w: int = 360) -> np.ndarray:
    """A negative whose subject range runs far past the paper scale, with fine texture
    at every density so micro-contrast is measurable in each third."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    key = np.exp(-3.2 * (yy / h))
    texture = 1.0 + 0.16 * np.sin(xx / 3.0) * np.sin(yy / 2.5)
    scene = np.clip(key * texture, 1e-5, None)
    scene /= scene.max()
    neg = np.clip(0.03 + 0.85 * (1.0 - scene**0.35), 1e-4, 1.0)
    return np.ascontiguousarray(np.dstack([neg, neg, neg]).astype(np.float32))


def _flat_negative(h: int = 300, w: int = 450) -> np.ndarray:
    """A negative too flat for Grade: its log range puts the straight-line slope on the
    k floor at every ISO-R, so the grade slider has no travel left."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    key = np.exp(-0.9 * (yy / h))
    texture = 1.0 + 0.14 * np.sin(xx / 3.0) * np.sin(yy / 2.5)
    scene = np.clip(key * texture, 1e-5, None)
    scene /= scene.max()
    neg = np.clip(0.25 + 0.30 * (1.0 - scene**0.35), 1e-4, 1.0)
    return np.ascontiguousarray(np.dstack([neg, neg, neg]).astype(np.float32))


def _global_spread(plane: np.ndarray) -> float:
    """Spread of the broad tonal masses, with fine detail blurred away."""
    return float(cv2.GaussianBlur(plane, (0, 0), 12.0).std())


def _bw_settings(**exposure) -> WorkspaceConfig:
    s = WorkspaceConfig()
    return replace(
        s,
        process=replace(s.process, process_mode="B&W"),
        exposure=replace(s.exposure, auto_exposure=False, auto_normalize_contrast=False, **exposure),
    )


def _micro_contrast(plane: np.ndarray) -> list[float]:
    """RMS Laplacian per horizontal third: thin, mid, dense."""
    lap = cv2.Laplacian(plane, cv2.CV_32F)
    h = plane.shape[0]
    return [float(np.sqrt((band**2).mean()) * 100) for band in (lap[: h // 3], lap[h // 3 : 2 * h // 3], lap[2 * h // 3 :])]


def _render(img: np.ndarray, settings: WorkspaceConfig, tag: str) -> np.ndarray:
    return np.asarray(DarkroomEngine().process(img.copy(), settings, tag))[:, :, 1]


class TestContrastMaskPlane(unittest.TestCase):
    def test_plane_is_zero_mean(self):
        """A sandwich is denser and the printer opens up for it, so the plane carries
        only the redistribution; otherwise the slider doubles as Print Density."""
        img = _wide_range_negative()
        plane, _ = contrast_mask_plane(img, LogNegativeBounds(floors=(-1.4, -1.4, -1.4), ceils=(-0.05, -0.05, -0.05)), None)
        self.assertAlmostEqual(float(plane.mean()), 0.0, places=5)

    def test_surround_outside_the_crop_stays_out_of_the_mask(self):
        """A rebate or surround blurred into the mask prints as a vignette the negative
        does not have."""
        h, w = 600, 900
        img = np.full((h, w, 3), 0.35, dtype=np.float32)
        border = 60
        img[:border] = img[-border:] = img[:, :border] = img[:, -border:] = 0.98

        bounds = LogNegativeBounds(floors=(-1.4, -1.4, -1.4), ceils=(-0.05, -0.05, -0.05))
        roi_norm = (border / h, (h - border) / h, border / w, (w - border) / w)

        cropped, _ = contrast_mask_plane(img, bounds, None, roi_norm=roi_norm)
        whole, _ = contrast_mask_plane(img, bounds, None)

        # A uniform picture area has no low frequencies, so its mask must be flat.
        self.assertLess(float(cropped.std()), 0.002, f"crop-respecting plane std {cropped.std():.5f}")
        # Without the crop the border bleeds in and manufactures a gradient.
        self.assertGreater(float(whole.std()), 10.0 * float(cropped.std()) + 0.01)

    def test_ev_places_the_plane_back_at_the_crop(self):
        plane = np.full((4, 6), 0.25, dtype=np.float32)
        ev = contrast_mask_ev(plane, 0.5, 1.4, (20, 30), roi=(5, 15, 6, 24))
        assert ev is not None
        self.assertEqual(ev.shape, (20, 30))
        # Edge-replicated outside the crop, so the crop tool's full-frame view has no seam.
        np.testing.assert_allclose(ev[0, 0], ev[5, 6], rtol=1e-5)

    def test_ev_is_off_when_gamma_is_zero(self):
        plane = np.zeros((8, 8), dtype=np.float32)
        self.assertIsNone(contrast_mask_ev(plane, 0.0, 1.4, (8, 8)))
        self.assertIsNone(contrast_mask_ev(None, 0.5, 1.4, (8, 8)))

    def test_ev_opposes_the_plane(self):
        """The sandwich subtracts the blurred positive: dense areas come back as
        negative exposure, thin ones as positive."""
        plane = np.array([[-0.2, 0.2]], dtype=np.float32)
        ev = contrast_mask_ev(plane, 0.5, 1.4, (1, 2))
        assert ev is not None
        self.assertGreater(ev[0, 0], 0.0)
        self.assertLess(ev[0, 1], 0.0)

    def test_negative_gamma_reverses_the_ev(self):
        """A blurred negative adds the low frequencies where a blurred positive removes
        them, so the two directions are one axis through zero."""
        plane = np.array([[0.2]], dtype=np.float32)
        reduce_ = contrast_mask_ev(plane, 0.3, 1.4, (1, 1))
        increase = contrast_mask_ev(plane, -0.3, 1.4, (1, 1))
        assert reduce_ is not None and increase is not None
        self.assertAlmostEqual(float(reduce_[0, 0]), -float(increase[0, 0]), places=6)

    def test_ev_scales_with_gamma(self):
        plane = np.array([[0.25]], dtype=np.float32)
        half = contrast_mask_ev(plane, 0.25, 1.4, (1, 1))
        full = contrast_mask_ev(plane, 0.5, 1.4, (1, 1))
        assert half is not None and full is not None
        self.assertAlmostEqual(float(full[0, 0]), 2.0 * float(half[0, 0]), places=5)


class TestContrastMaskRender(unittest.TestCase):
    def test_mask_raises_micro_contrast_at_every_density(self):
        """The range compresses, so a hard grade's local contrast survives at both ends
        instead of being traded away."""
        img = _wide_range_negative()
        plain = _render(img, _bw_settings(grade=60.0), "cm-plain")
        masked = _render(img, _bw_settings(grade=60.0, contrast_mask=0.5), "cm-masked")

        for band, (before, after) in enumerate(zip(_micro_contrast(plain), _micro_contrast(masked))):
            self.assertGreater(after, before, f"band {band}: {before:.2f} -> {after:.2f}")

    def test_mask_compresses_the_global_range(self):
        img = _wide_range_negative()
        plain = _render(img, _bw_settings(grade=60.0), "cm-plain")
        masked = _render(img, _bw_settings(grade=60.0, contrast_mask=0.5), "cm-masked")
        self.assertLess(masked.std(), plain.std())

    def test_negative_gamma_expands_the_range_where_grade_cannot(self):
        """Contrast increase, the other half of Ctein's masking chapter. On a flat
        negative the straight-line slope sits on its clamp, so Grade is inert there;
        the mask still has travel because it works on the low frequencies."""
        img = _flat_negative()
        flat = _render(img, _bw_settings(grade=115.0), "cm-flat")
        harder = _render(img, _bw_settings(grade=60.0), "cm-flat-hard")
        masked = _render(img, _bw_settings(grade=115.0, contrast_mask=-0.3), "cm-flat-mask")

        np.testing.assert_allclose(flat, harder, atol=1e-6, err_msg="grade should be clamped inert here")
        self.assertGreater(_global_spread(masked), _global_spread(flat) * 1.1)

    def test_the_two_directions_move_micro_contrast_oppositely(self):
        """At a matched global gain Grade raises micro-contrast and the mask lowers it,
        which is what makes the increasing mask more than a Grade preset."""
        img = _wide_range_negative()
        base = _render(img, _bw_settings(grade=115.0), "cm-dir-base")
        masked = _render(img, _bw_settings(grade=115.0, contrast_mask=-0.3), "cm-dir-mask")
        graded = _render(img, _bw_settings(grade=90.0), "cm-dir-grade")

        for other in (masked, graded):
            self.assertGreater(_global_spread(other), _global_spread(base))
        self.assertLess(_micro_contrast(masked)[1], _micro_contrast(base)[1])
        self.assertGreater(_micro_contrast(graded)[1], _micro_contrast(base)[1])

    def test_zero_gamma_changes_nothing(self):
        img = _wide_range_negative()
        off = _render(img, _bw_settings(grade=60.0), "cm-off")
        explicit = _render(img, _bw_settings(grade=60.0, contrast_mask=0.0), "cm-off")
        np.testing.assert_allclose(off, explicit, atol=1e-6)

    def test_mask_survives_a_resolution_change(self):
        """Sigma is a fraction of the analysis grid, not of the render."""
        img = _wide_range_negative()
        h, w = img.shape[:2]
        big = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)

        def drift(**kw):
            small = _render(img, _bw_settings(grade=60.0, **kw), "cm-res-s" + str(kw))
            large = _render(big, _bw_settings(grade=60.0, **kw), "cm-res-l" + str(kw))
            return float(np.abs(small - cv2.resize(large, (w, h), interpolation=cv2.INTER_AREA)).mean())

        self.assertLess(drift(contrast_mask=0.5), 2.0 * drift() + 0.002)


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestContrastMaskParity(unittest.TestCase):
    def setUp(self):
        from negpy.services.rendering.image_processor import ImageProcessor

        self.processor = ImageProcessor()
        if self.processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")
        self.img = _wide_range_negative(96, 144)

    def _render(self, settings: WorkspaceConfig, tag: str, prefer_gpu: bool) -> np.ndarray:
        result, _ = self.processor.run_pipeline(
            self.img.copy(),
            settings,
            tag,
            render_size_ref=float(max(self.img.shape[:2])),
            prefer_gpu=prefer_gpu,
            readback_metrics=False,
        )
        arr = np.asarray(result.readback()) if hasattr(result, "readback") else np.asarray(result)
        return arr[:, :, :3].astype(np.float64)

    def _assert_match(self, settings: WorkspaceConfig, tag: str):
        cpu = self._render(settings, tag, prefer_gpu=False)
        gpu = self._render(settings, tag, prefer_gpu=True)
        self.assertEqual(cpu.shape, gpu.shape)
        self.assertLess(float(np.mean(np.abs(cpu - gpu))), 0.01)
        self.assertLess(float(np.max(np.abs(cpu - gpu))), 0.04)

    def test_cpu_gpu_match(self):
        self._assert_match(_bw_settings(grade=60.0, contrast_mask=0.5), "contrast-mask-parity")

    def test_cpu_gpu_match_negative_gamma(self):
        self._assert_match(_bw_settings(grade=60.0, contrast_mask=-0.35), "contrast-mask-parity-neg")

    def test_slider_uploads_no_texture(self):
        base = _bw_settings(grade=60.0, contrast_mask=0.5)
        self._render(base, "contrast-mask-drag", prefer_gpu=True)
        engine = self.processor.engine_gpu
        mask_key, ev_key = engine._mask_tex_key, engine._local_ev_key
        self.assertIsNotNone(mask_key)

        self._render(replace(base, exposure=replace(base.exposure, contrast_mask=0.2)), "contrast-mask-drag", prefer_gpu=True)
        self.assertEqual(engine._mask_tex_key, mask_key)
        self.assertEqual(engine._local_ev_key, ev_key)


if __name__ == "__main__":
    unittest.main()
