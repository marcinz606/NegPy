"""The unsharp contrast-reduction mask.

The mask reaches the print through the dodge/burn EV map, so both engines build the
plane with the same helper on the same pre-geometry array. These pin the three things
that would silently break it: the sign, the resolution invariance, and CPU/GPU parity.
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
        """A sandwiched mask is denser and the printer opens up for it; the plane must
        carry only the redistribution, or the slider would double as Print Density."""
        img = _wide_range_negative()
        plane = contrast_mask_plane(img, LogNegativeBounds(floors=(-1.4, -1.4, -1.4), ceils=(-0.05, -0.05, -0.05)), None)
        self.assertAlmostEqual(float(plane.mean()), 0.0, places=5)

    def test_surround_outside_the_crop_stays_out_of_the_mask(self):
        """The enlarger projects the frame you print. Blurred over, a bright rebate or a
        black scanner surround prints as a vignette the negative does not have, so the
        plane must be built on the crop, not on the whole scan."""
        h, w = 600, 900
        img = np.full((h, w, 3), 0.35, dtype=np.float32)
        border = 60
        img[:border] = img[-border:] = img[:, :border] = img[:, -border:] = 0.98

        bounds = LogNegativeBounds(floors=(-1.4, -1.4, -1.4), ceils=(-0.05, -0.05, -0.05))
        roi_norm = (border / h, (h - border) / h, border / w, (w - border) / w)

        cropped = contrast_mask_plane(img, bounds, None, roi_norm=roi_norm)
        whole = contrast_mask_plane(img, bounds, None)

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
        """The sandwich subtracts the blurred positive, so a dense low-frequency area
        must come back as negative exposure and a thin one as positive."""
        plane = np.array([[-0.2, 0.2]], dtype=np.float32)
        ev = contrast_mask_ev(plane, 0.5, 1.4, (1, 2))
        assert ev is not None
        self.assertGreater(ev[0, 0], 0.0)
        self.assertLess(ev[0, 1], 0.0)

    def test_ev_scales_with_gamma(self):
        plane = np.array([[0.25]], dtype=np.float32)
        half = contrast_mask_ev(plane, 0.25, 1.4, (1, 1))
        full = contrast_mask_ev(plane, 0.5, 1.4, (1, 1))
        assert half is not None and full is not None
        self.assertAlmostEqual(float(full[0, 0]), 2.0 * float(half[0, 0]), places=5)


class TestContrastMaskRender(unittest.TestCase):
    def test_mask_raises_micro_contrast_at_every_density(self):
        """The point of the mask: the global range is compressed, so a hard grade's
        local contrast survives at both ends instead of being traded away."""
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

    def test_zero_gamma_changes_nothing(self):
        img = _wide_range_negative()
        off = _render(img, _bw_settings(grade=60.0), "cm-off")
        explicit = _render(img, _bw_settings(grade=60.0, contrast_mask=0.0), "cm-off")
        np.testing.assert_allclose(off, explicit, atol=1e-6)

    def test_mask_survives_a_resolution_change(self):
        """Sigma is a fraction of the analysis grid, not of the render, so the mask
        must not add a resolution dependence of its own on top of the pipeline's."""
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
    def test_cpu_gpu_match(self):
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

        img = _wide_range_negative(96, 144)
        settings = _bw_settings(grade=60.0, contrast_mask=0.5)

        def render(prefer_gpu: bool) -> np.ndarray:
            result, _ = processor.run_pipeline(
                img.copy(),
                settings,
                "contrast-mask-parity",
                render_size_ref=float(max(img.shape[:2])),
                prefer_gpu=prefer_gpu,
                readback_metrics=False,
            )
            arr = np.asarray(result.readback()) if hasattr(result, "readback") else np.asarray(result)
            return arr[:, :, :3].astype(np.float64)

        cpu = render(False)
        gpu = render(True)
        self.assertEqual(cpu.shape, gpu.shape)
        self.assertLess(float(np.mean(np.abs(cpu - gpu))), 0.01)
        self.assertLess(float(np.max(np.abs(cpu - gpu))), 0.04)


if __name__ == "__main__":
    unittest.main()
