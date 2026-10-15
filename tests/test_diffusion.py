"""Diffusion: a diffuser under the enlarger lens. The paper sees a mix of light, not
density, so flat areas print as before and the dark parts of the negative spread into the
light ones. One analysis-grid plane serves both engines."""

import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.logic import apply_characteristic_curve, channel_density_ranges
from negpy.features.exposure.normalization import (
    LogNegativeBounds,
    diffusion_grid,
    diffusion_plane,
    normalize_log_image,
    to_log_density,
    unmix_log_image,
)
from negpy.infrastructure.gpu.device import GPUDevice

SLOPE, PIVOT = 2.9, 0.21
ARGS = ((PIVOT, SLOPE), (PIVOT, SLOPE), (PIVOT, SLOPE))
BOUNDS = LogNegativeBounds((-2.0, -2.0, -2.0), (-0.2, -0.2, -0.2))


def _step_negative(h: int = 64, w: int = 96) -> np.ndarray:
    """A thin left half and a dense right half: a hard edge in the scene."""
    lin = np.full((h, w, 3), 0.5, dtype=np.float32)
    lin[:, w // 2 :] = 0.02
    return np.ascontiguousarray(lin)


def _normalized(lin: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(normalize_log_image(to_log_density(lin), BOUNDS), dtype=np.float32)


def _plane(lin: np.ndarray, panchromatic: bool = False) -> np.ndarray:
    plane = diffusion_plane(diffusion_grid(lin, None), BOUNDS, panchromatic)
    assert plane is not None
    return plane


def _print(img: np.ndarray, **kw) -> np.ndarray:
    return np.asarray(apply_characteristic_curve(img, *ARGS, frame_grade=115.0, diffusion_range=channel_density_ranges(BOUNDS), **kw))


class TestPlane(unittest.TestCase):
    def test_a_flat_frame_diffuses_to_itself(self):
        lin = np.full((40, 60, 3), 0.3, dtype=np.float32)
        plane = _plane(lin)
        np.testing.assert_allclose(plane, _normalized(lin), atol=1e-5)

    def test_the_plane_is_the_blurred_light_not_the_blurred_density(self):
        lin = _step_negative()
        plane = _plane(lin)
        h, w = lin.shape[:2]
        # Far from the edge the plane is the frame; at the edge it is the mean of the two
        # lights (0.26), which in density sits nearer the thin side than the mean density does.
        np.testing.assert_allclose(plane[h // 2, 2], _normalized(lin)[h // 2, 2], atol=1e-3)
        edge = float(plane[h // 2, w // 2 - 1, 0])
        mean_light = float(_normalized(np.full((1, 1, 3), 0.26, dtype=np.float32))[0, 0, 0])
        mean_density = float(_normalized(lin)[h // 2, :, 0].mean())
        self.assertLess(abs(edge - mean_light), abs(edge - mean_density))

    def test_a_panchromatic_plane_is_the_luma_of_a_cast_frame(self):
        lin = np.full((20, 30, 3), 0.3, dtype=np.float32)
        lin[:, :, 0] *= 1.4  # an orange-mask cast
        plane = _plane(lin, panchromatic=True)
        norm = _normalized(lin)
        lum = 0.2126 * norm[:, :, 0] + 0.7152 * norm[:, :, 1] + 0.0722 * norm[:, :, 2]
        for ch in range(3):
            np.testing.assert_allclose(plane[:, :, ch], lum, atol=1e-4)

    def test_an_unmetered_frame_has_no_plane(self):
        lin = np.full((20, 30, 3), 0.3, dtype=np.float32)
        self.assertIsNone(diffusion_plane(diffusion_grid(lin, None), LogNegativeBounds((-1.0,) * 3, (-1.0,) * 3)))

    def test_the_grid_blurs_the_unmixed_light(self):
        """The kernel mixes the light of the unmixed value, so a flat frame holds under crosstalk."""
        lin = np.full((20, 30, 3), 0.3, dtype=np.float32)
        lin[:, :, 2] = 0.1
        unmix = np.array([[1.0, -0.2, 0.0], [-0.1, 1.0, -0.1], [0.0, -0.3, 1.0]], dtype=np.float32)
        plane = diffusion_plane(diffusion_grid(lin, unmix), BOUNDS)
        expected = normalize_log_image(unmix_log_image(to_log_density(lin), unmix), BOUNDS)
        np.testing.assert_allclose(plane, expected, atol=1e-5)


class TestKernel(unittest.TestCase):
    def test_zero_diffusion_is_bit_identical(self):
        img = _normalized(_step_negative())
        plane = _plane(_step_negative())
        np.testing.assert_array_equal(_print(img), _print(img, diffusion=0.0, diffusion_plane=plane))

    def test_a_flat_frame_prints_as_before_at_full_diffusion(self):
        lin = np.full((32, 48, 3), 0.3, dtype=np.float32)
        img = _normalized(lin)
        plane = _plane(lin)
        np.testing.assert_allclose(_print(img, diffusion=1.0, diffusion_plane=plane), _print(img), atol=1e-4)

    def test_the_dense_side_spreads_into_the_thin_side_at_the_edge_only(self):
        lin = _step_negative()
        img = _normalized(lin)
        plane = _plane(lin)
        plain = _print(img)
        soft = _print(img, diffusion=0.6, diffusion_plane=plane)
        h, w = img.shape[:2]
        mid = h // 2
        np.testing.assert_allclose(soft[mid, :4], plain[mid, :4], atol=1e-4)
        np.testing.assert_allclose(soft[mid, -4:], plain[mid, -4:], atol=1e-4)
        # A thin negative prints dark. Next to the edge the dense side's darkness bleeds into
        # the thin side, so the thin side prints lighter and the dense side darker.
        self.assertGreater(float(soft[mid, w // 2 - 2, 0]), float(plain[mid, w // 2 - 2, 0]))
        self.assertLess(float(soft[mid, w // 2 + 1, 0]), float(plain[mid, w // 2 + 1, 0]))

    def test_the_full_slider_diffuses_half_the_light(self):
        lin = _step_negative()
        img = _normalized(lin)
        plane = _plane(lin)
        r = np.array(channel_density_ranges(BOUNDS), dtype=np.float64)
        half = np.log10(0.5 * 10.0 ** (img * r) + 0.5 * 10.0 ** (plane * r)) / r
        np.testing.assert_allclose(_print(img, diffusion=1.0, diffusion_plane=plane), _print(half.astype(np.float32)), atol=1e-5)

    def test_the_rect_places_the_plane_on_the_printed_frame(self):
        lin = _step_negative()
        img = _normalized(lin)
        plane = _plane(lin)
        h, w = img.shape[:2]
        whole = _print(img, diffusion=0.6, diffusion_plane=plane, diffusion_rect=(0.0, 0.0, float(w), float(h)))
        np.testing.assert_array_equal(whole, _print(img, diffusion=0.6, diffusion_plane=plane))


def _wide_range_negative(h: int = 96, w: int = 144) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    key = np.exp(-3.2 * (yy / h))
    texture = 1.0 + 0.16 * np.sin(xx / 3.0) * np.sin(yy / 2.5)
    scene = np.clip(key * texture, 1e-5, None)
    scene /= scene.max()
    neg = np.clip(0.03 + 0.85 * (1.0 - scene**0.35), 1e-4, 1.0)
    return np.ascontiguousarray(np.dstack([neg, neg, neg]).astype(np.float32))


def _settings(**exposure) -> WorkspaceConfig:
    s = WorkspaceConfig()
    return replace(s, exposure=replace(s.exposure, auto_exposure=False, auto_normalize_contrast=False, **exposure))


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestDiffusionParity(unittest.TestCase):
    def setUp(self):
        from negpy.services.rendering.image_processor import ImageProcessor

        self.processor = ImageProcessor()
        if self.processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")
        self.img = _wide_range_negative()

    def _render(self, settings: WorkspaceConfig, tag: str, prefer_gpu: bool, size_ref: float = 0.0) -> np.ndarray:
        result, _ = self.processor.run_pipeline(
            self.img.copy(),
            settings,
            tag,
            render_size_ref=size_ref or float(max(self.img.shape[:2])),
            prefer_gpu=prefer_gpu,
            readback_metrics=False,
        )
        arr = np.asarray(result.readback()) if hasattr(result, "readback") else np.asarray(result)
        return arr[:, :, :3].astype(np.float64)

    def _assert_match(self, settings: WorkspaceConfig, tag: str):
        cpu = self._render(settings, tag, prefer_gpu=False)
        gpu = self._render(settings, tag, prefer_gpu=True, size_ref=float(max(cpu.shape[:2])))
        self.assertEqual(cpu.shape, gpu.shape)
        self.assertLess(float(np.max(np.abs(cpu - gpu))), 1e-4)
        return cpu

    def test_cpu_gpu_match_and_the_diffusion_moves_the_print(self):
        soft = self._assert_match(_settings(diffusion=0.7), "diffusion-parity")
        plain = self._render(_settings(), "diffusion-parity-plain", prefer_gpu=False)
        self.assertGreater(float(np.max(np.abs(soft - plain))), 0.02)

    def test_cpu_gpu_match_with_the_contrast_mask_on_the_same_texture(self):
        self._assert_match(_settings(diffusion=0.5, contrast_mask=0.3, grade=80.0), "diffusion-mask-parity")

    def test_cpu_gpu_match_cropped(self):
        s = _settings(diffusion=0.6)
        self._assert_match(replace(s, geometry=replace(s.geometry, crop_rect=(0.2, 0.15, 0.65, 0.7))), "diffusion-parity-crop")

    def test_cpu_gpu_match_a_cast_frame_printed_as_bw(self):
        """A B&W print mixes a luma plane with its luma pixel, so a cast flat area holds."""
        self.img = np.ascontiguousarray(self.img * np.array([1.3, 1.0, 0.8], dtype=np.float32))
        s = _settings(diffusion=0.8)
        self._assert_match(replace(s, process=replace(s.process, process_mode="B&W")), "diffusion-parity-bw")

    def test_the_mix_slider_uploads_no_texture(self):
        base = _settings(diffusion=0.5)
        self._render(base, "diffusion-drag", prefer_gpu=True)
        engine = self.processor.engine_gpu
        key = engine._mask_tex_key
        self.assertIsNotNone(key)
        self._render(replace(base, exposure=replace(base.exposure, diffusion=0.8)), "diffusion-drag", prefer_gpu=True)
        self.assertEqual(engine._mask_tex_key, key)

    def test_a_trim_drag_keeps_the_blurred_grid(self):
        base = _settings(diffusion=0.5)
        self._render(base, "diffusion-trim", prefer_gpu=True)
        engine = self.processor.engine_gpu
        grid, plane = engine._diffusion_grid, engine._diffusion_plane
        self.assertIsNotNone(grid)
        trimmed = replace(base, process=replace(base.process, white_point_trim_red=0.05))
        self._render(trimmed, "diffusion-trim", prefer_gpu=True)
        self.assertIs(engine._diffusion_grid, grid)
        self.assertIsNot(engine._diffusion_plane, plane)


class TestFlatMaster(unittest.TestCase):
    def test_a_flat_master_builds_no_plane(self):
        from negpy.features.exposure.models import RenderIntent
        from negpy.services.rendering.engine import DarkroomEngine

        engine = DarkroomEngine()
        s = _settings(diffusion=0.8, render_intent=RenderIntent.FLAT)
        engine.process(_wide_range_negative(), s, "flat-master")
        self.assertIsNone(engine._diffusion_plane)


if __name__ == "__main__":
    unittest.main()
