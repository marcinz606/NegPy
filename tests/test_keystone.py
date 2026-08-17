"""Easel tilt and swing, the perspective correction.

The warp is one cv2 call. What breaks silently is everything that has to agree with it:
the shader's inverse, the meters, the point mapper behind dodge/burn masks, and
autocrop's replay.
"""

import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.geometry.logic import (
    apply_keystone,
    autocrop_detection_key,
    keystone_inverse_normalized,
    keystone_matrix,
    map_coords_to_geometry,
    map_point_keystone,
)
from negpy.infrastructure.gpu.device import GPUDevice


def _test_field(h: int = 96, w: int = 144) -> np.ndarray:
    rng = np.random.default_rng(0)
    grad = np.linspace(0.05, 0.9, w, dtype=np.float32)
    img = np.repeat(grad[None, :], h, axis=0)
    img = np.stack([img, img * 0.95, img * 0.9], axis=-1)
    return np.ascontiguousarray(img + rng.uniform(0, 0.01, img.shape).astype(np.float32))


class TestKeystoneTransform(unittest.TestCase):
    def test_zero_is_identity(self):
        img = _test_field()
        np.testing.assert_array_equal(apply_keystone(img, 0.0, 0.0), img)
        np.testing.assert_allclose(keystone_inverse_normalized(0.0, 0.0), np.eye(3), atol=1e-12)

    def test_output_keeps_the_canvas_size(self):
        """The GPU derives its intermediate dimensions from rotation alone."""
        img = _test_field()
        self.assertEqual(apply_keystone(img, 12.0, -8.0).shape, img.shape)

    def test_positive_converge_v_stretches_the_top(self):
        """A bar at the top must come out wider than the same bar at the bottom."""
        h, w = 120, 200
        img = np.zeros((h, w), np.float32)
        img[10, 60:140] = 1.0
        img[h - 11, 60:140] = 1.0
        out = apply_keystone(img, 10.0, 0.0)
        top = float((out[: h // 2] > 0.5).sum())
        bottom = float((out[h // 2 :] > 0.5).sum())
        self.assertGreater(top, bottom, f"top {top} px, bottom {bottom} px")

    def test_point_mapper_follows_the_resample(self):
        """A feature point lands where the warp put its pixels, or masks drift off what
        the canvas draws."""
        h, w = 96, 144
        for cv_, ch_ in ((8.0, 0.0), (0.0, -6.0), (12.0, 5.0)):
            img = np.zeros((h, w), np.float32)
            img[40, 30] = 1.0
            out = apply_keystone(img, cv_, ch_)
            ys, xs = np.nonzero(out > 0.25)
            got = (float(ys.mean()), float(xs.mean()))
            want = map_point_keystone(30.0, 40.0, cv_, ch_, w, h)
            self.assertAlmostEqual(got[0], want[1], delta=1.0, msg=f"y at cv={cv_} ch={ch_}")
            self.assertAlmostEqual(got[1], want[0], delta=1.0, msg=f"x at cv={cv_} ch={ch_}")

    def test_shader_inverse_undoes_the_forward_matrix(self):
        """The GPU consumes this matrix directly, so it must invert the CPU's own quad."""
        h, w = 96, 144
        fwd = keystone_matrix(9.0, -4.0, w, h)
        inv_norm = keystone_inverse_normalized(9.0, -4.0)
        to_index = np.array([[w, 0.0, -0.5], [0.0, h, -0.5], [0.0, 0.0, 1.0]], dtype=np.float64)
        inv_index = to_index @ inv_norm @ np.linalg.inv(to_index)
        np.testing.assert_allclose(inv_index @ fwd / (inv_index @ fwd)[2, 2], np.eye(3), atol=1e-8)

    def test_mask_vertices_follow_the_keystone(self):
        mapped = map_coords_to_geometry(0.5, 0.1, (96, 144), converge_v=12.0)
        plain = map_coords_to_geometry(0.5, 0.1, (96, 144))
        self.assertNotAlmostEqual(mapped[1], plain[1], places=3)

    def test_detection_key_tracks_the_correction(self):
        """Autocrop replays the keystone, so a resolved rect must not survive a change
        to it."""
        g = WorkspaceConfig().geometry
        self.assertNotEqual(autocrop_detection_key(g), autocrop_detection_key(replace(g, converge_v=6.0)))
        self.assertNotEqual(autocrop_detection_key(g), autocrop_detection_key(replace(g, converge_h=6.0)))


class TestKeystoneCoordinateMapping(unittest.TestCase):
    def test_uv_grid_carries_the_correction(self):
        from negpy.services.view.coordinate_mapping import CoordinateMapping

        plain = CoordinateMapping.create_uv_grid(96, 144, 0, 0.0)
        warped = CoordinateMapping.create_uv_grid(96, 144, 0, 0.0, converge_v=10.0)
        self.assertFalse(np.allclose(plain, warped))

    def test_off_frame_points_round_trip(self):
        """Card-edge handles sit outside the picture, where the projective model answers
        instead of the grid. It must invert cleanly."""
        from negpy.services.view.coordinate_mapping import CoordinateMapping

        grid = CoordinateMapping.create_uv_grid(96, 144, 0, 0.0, converge_v=10.0, converge_h=-6.0)
        for nx, ny in ((-0.4, 0.3), (1.35, 0.8), (0.5, -0.25)):
            rx, ry = CoordinateMapping.map_click_to_raw(nx, ny, grid)
            back = CoordinateMapping.map_raw_to_viewport(rx, ry, grid)
            self.assertAlmostEqual(back[0], nx, delta=0.02, msg=f"x for ({nx},{ny})")
            self.assertAlmostEqual(back[1], ny, delta=0.02, msg=f"y for ({nx},{ny})")


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestKeystoneParity(unittest.TestCase):
    def test_cpu_gpu_match(self):
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")

        img = _test_field()
        base = WorkspaceConfig()

        def render(prefer_gpu: bool, **geo) -> np.ndarray:
            settings = replace(base, geometry=replace(base.geometry, **geo))
            result, _ = processor.run_pipeline(
                img.copy(),
                settings,
                f"keystone-parity{geo}{prefer_gpu}",
                render_size_ref=float(max(img.shape[:2])),
                prefer_gpu=prefer_gpu,
                readback_metrics=False,
            )
            arr = np.asarray(result.readback()) if hasattr(result, "readback") else np.asarray(result)
            return arr[:, :, :3].astype(np.float64)

        # Both axes, both signs, and composed with the ops that run before it. This is
        # also the only guard on the GPU's own geometry replay for the meters: skipping
        # the keystone there makes the engines normalize different pixels, and it shows
        # up here and nowhere else.
        for geo in (
            dict(converge_v=8.0),
            dict(converge_v=-12.0),
            dict(converge_h=10.0),
            dict(converge_v=9.0, converge_h=-5.0, fine_rotation=1.5, rotation=1),
        ):
            cpu, gpu = render(False, **geo), render(True, **geo)
            self.assertEqual(cpu.shape, gpu.shape, str(geo))
            self.assertLess(float(np.mean(np.abs(cpu - gpu))), 0.01, str(geo))
            self.assertLess(float(np.max(np.abs(cpu - gpu))), 0.04, str(geo))


if __name__ == "__main__":
    unittest.main()
