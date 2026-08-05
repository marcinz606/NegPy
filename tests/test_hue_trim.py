"""Hue Trim: the a*b* rotation that undoes a light source's hue shift.

The rotation is written twice (hue.py and the inlined block in exposure.wgsl), so drift between
them would show up as GPU previews disagreeing with CPU exports.
"""

import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.process.hue import apply_hue_trim
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.kernel.image.logic import rgb_to_lab_working


def _hue_of(rgb: np.ndarray) -> np.ndarray:
    lab = rgb_to_lab_working(np.asarray(rgb, dtype=np.float32))
    return np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360.0


def _chroma_of(rgb: np.ndarray) -> np.ndarray:
    lab = rgb_to_lab_working(np.asarray(rgb, dtype=np.float32))
    return np.hypot(lab[..., 1], lab[..., 2])


class TestHueTrimLogic(unittest.TestCase):
    def test_zero_is_identity(self):
        img = np.random.default_rng(0).uniform(0.05, 0.95, (8, 8, 3)).astype(np.float32)
        out = apply_hue_trim(img, 0.0)
        np.testing.assert_array_equal(np.asarray(out), img)

    def test_rotates_hue_by_the_requested_angle(self):
        # Mid-chroma patches spread around the hue circle; each must rotate by the same angle.
        img = np.array(
            [[[0.6, 0.2, 0.1], [0.2, 0.5, 0.15], [0.15, 0.25, 0.6], [0.55, 0.5, 0.12]]],
            dtype=np.float32,
        )
        for angle in (-18.6, -5.0, 7.5, 18.6):
            rotated = np.asarray(apply_hue_trim(img, angle))
            delta = (_hue_of(rotated) - _hue_of(img) + 180.0) % 360.0 - 180.0
            np.testing.assert_allclose(delta, angle, atol=0.5, err_msg=f"angle {angle}")

    def test_neutrals_are_untouched(self):
        """Greys must not move, or this would fight the base-anchored cast removal upstream."""
        grey = np.stack([np.linspace(0.02, 0.98, 16, dtype=np.float32)] * 3, axis=-1)[None, :, :]
        out = np.asarray(apply_hue_trim(grey, 20.0))
        self.assertLess(float(np.max(_chroma_of(out))), 0.05)
        np.testing.assert_allclose(out, grey, atol=2e-3)

    def test_chroma_and_lightness_are_preserved(self):
        img = np.random.default_rng(3).uniform(0.05, 0.9, (16, 16, 3)).astype(np.float32)
        out = np.asarray(apply_hue_trim(img, 15.0))
        lab_in = rgb_to_lab_working(img)
        lab_out = rgb_to_lab_working(out.astype(np.float32))
        # In-gamut pixels keep L and C; the clamp to [0,1] can only pull outliers in.
        np.testing.assert_allclose(lab_out[..., 0], lab_in[..., 0], atol=1.0)
        c_in = np.hypot(lab_in[..., 1], lab_in[..., 2])
        c_out = np.hypot(lab_out[..., 1], lab_out[..., 2])
        self.assertLess(float(np.mean(np.abs(c_out - c_in))), 1.5)

    def test_opposite_angles_cancel(self):
        img = np.random.default_rng(5).uniform(0.1, 0.85, (8, 8, 3)).astype(np.float32)
        there = apply_hue_trim(img, 12.0)
        back = np.asarray(apply_hue_trim(np.asarray(there, dtype=np.float32), -12.0))
        np.testing.assert_allclose(back, img, atol=5e-3)


def _chromatic_negative(h: int = 64, w: int = 64, seed: int = 7) -> np.ndarray:
    """A synthetic linear negative whose channels oppose, so the print carries real chroma.

    A grey ramp makes the parity and has-any-effect assertions pass vacuously: a rotation barely
    moves low-chroma pixels.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.05, 0.9, w, dtype=np.float32)
    y = np.linspace(0.1, 0.8, h, dtype=np.float32)[:, None]
    r = np.repeat(x[None, :], h, axis=0) * (0.6 + 0.5 * y)
    g = np.repeat(x[::-1][None, :], h, axis=0) * (1.1 - 0.4 * y)
    b = np.repeat((0.3 + 0.5 * np.abs(x - 0.5))[None, :], h, axis=0) * (0.7 + 0.3 * y)
    img = np.stack([r, g, b], axis=-1).astype(np.float32)
    return np.ascontiguousarray(np.clip(img + rng.uniform(0, 0.01, img.shape).astype(np.float32), 0.01, 1.0))


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestHueTrimGpuParity(unittest.TestCase):
    def _render(self, processor, settings, img, prefer_gpu):
        result, _ = processor.run_pipeline(
            img, settings, "hue-parity-src", render_size_ref=float(max(img.shape[:2])), prefer_gpu=prefer_gpu, readback_metrics=False
        )
        arr = np.asarray(result.readback())[:, :, :3] if hasattr(result, "readback") else np.asarray(result)[:, :, :3]
        return arr.astype(np.float64)

    def test_cpu_gpu_match_with_hue_trim(self):
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

        img = _chromatic_negative()
        base = WorkspaceConfig()
        for angle in (18.6, -12.0):
            settings = replace(base, process=replace(base.process, hue_trim=angle))
            cpu = self._render(processor, settings, img, prefer_gpu=False)
            gpu = self._render(processor, settings, img, prefer_gpu=True)
            self.assertEqual(cpu.shape, gpu.shape)
            mad = float(np.mean(np.abs(cpu - gpu)))
            mx = float(np.max(np.abs(cpu - gpu)))
            self.assertLess(mad, 0.01, f"angle {angle}: mean abs diff {mad:.4f}")
            self.assertLess(mx, 0.04, f"angle {angle}: max abs diff {mx:.4f}")

    def test_gpu_trim_actually_changes_the_render(self):
        """Guards the uniform packing: a mis-packed lane would silently render as 0°."""
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

        img = _chromatic_negative(32, 32, seed=8)
        base = WorkspaceConfig()
        off = self._render(processor, base, img, prefer_gpu=True)
        on = self._render(processor, replace(base, process=replace(base.process, hue_trim=20.0)), img, prefer_gpu=True)
        self.assertGreater(float(np.mean(np.abs(on - off))), 0.005)
