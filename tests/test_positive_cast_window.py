"""Cast Removal on a Positive frame places its neutral-axis bands in the frame's own luma range,
so the axis does not depend on the file's encoding or exposure, and a black border does not empty
the shadow band. Every other transfer frame keeps the fixed window."""

import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.logic import cast_solve_inputs, neutral_axis_affine
from negpy.features.exposure.normalization import LogNegativeBounds, measure_neutral_axis_from_log, prefilter_log_grid
from negpy.features.process.models import ProcessMode
from negpy.features.transparency.logic import measure_transfer_neutral_axis, transfer_axis_bounds, transfer_bounds
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.services.rendering.engine import base_processor

FIXED = LogNegativeBounds(*transfer_bounds())


def _wedge(lo: float, hi: float, seed: int = 3, cast=(1.0, 0.86, 0.74)) -> np.ndarray:
    """A neutral wedge from `lo` to `hi` of full scale under a cast."""
    rng = np.random.default_rng(seed)
    v = np.geomspace(lo, hi, 96 * 96).astype(np.float32).reshape(96, 96)
    img = np.stack([v * cast[0], v * cast[1], v * cast[2]], axis=-1)
    return np.ascontiguousarray(img * rng.uniform(0.98, 1.02, img.shape).astype(np.float32))


def _dark_scan() -> np.ndarray:
    """Less than two decades of content, as a scanner file read as linear holds it: the fixed
    window's shadow band holds no pixels."""
    return _wedge(8e-3, 0.45)


def _bordered(image: np.ndarray, frac: float = 0.15) -> np.ndarray:
    """`image` with its left `frac` of columns at a black-border density, past the analysis buffer."""
    out = image.copy()
    out[:, : int(image.shape[1] * frac)] = 5e-4
    return out


def _slide(positive: bool) -> WorkspaceConfig:
    s = WorkspaceConfig()
    s = replace(s, process=replace(s.process, process_mode=ProcessMode.E6, positive_source=positive))
    return replace(s, exposure=replace(s.exposure, cast_removal_strength=1.0, auto_exposure=False, auto_normalize_contrast=False))


def _cpu_refs(img: np.ndarray, positive: bool):
    cfg = _slide(positive)
    h, w = img.shape[:2]
    ctx = PipelineContext(
        original_size=(h, w), scale_factor=1.0, process_mode=ProcessMode.E6, cam_xyz=None, camera_wb=None, wants_uv_grid=False
    )
    base_processor(cfg).process(img, ctx)
    return ctx.metrics.get("neutral_axis_refs")


def _gains(refs):
    return _affine(refs)[0]


def _affine(refs):
    strength, norm = cast_solve_inputs(FIXED, refs, 1.0)
    return neutral_axis_affine(norm, strength)


class TestPositiveAxisWindow(unittest.TestCase):
    def test_other_frames_take_the_fixed_window(self):
        grid = prefilter_log_grid(_dark_scan())
        self.assertEqual(transfer_axis_bounds(grid, positive=False), transfer_bounds())

    def test_a_flat_frame_takes_the_fixed_window(self):
        grid = np.full((32, 32, 3), -1.0, dtype=np.float32)
        self.assertEqual(transfer_axis_bounds(grid, positive=True), transfer_bounds())

    def test_the_window_is_the_same_on_all_channels(self):
        floors, ceils = transfer_axis_bounds(prefilter_log_grid(_dark_scan()), positive=True)
        self.assertEqual(len(set(floors)), 1)
        self.assertEqual(len(set(ceils)), 1)
        self.assertGreater(floors[0], ceils[0])

    def test_a_dark_positive_gets_an_axis(self):
        img = _dark_scan()
        self.assertIsNone(_cpu_refs(img, positive=False))
        self.assertIsNotNone(_cpu_refs(img, positive=True))

    def test_positive_off_meters_the_fixed_window_unchanged(self):
        img = _wedge(5e-4, 0.9)
        expected = measure_neutral_axis_from_log(prefilter_log_grid(img, None, 0.05), FIXED, None, 0.0, exclude_subjects=False)
        self.assertIsNotNone(expected)
        self.assertEqual(_cpu_refs(img, positive=False), expected)

    def test_encoding_does_not_move_the_axis(self):
        """A 2.2 reading scales every log density by 2.2, and the window scales with it."""
        img = _dark_scan()
        linear = _cpu_refs(img, positive=True)
        gamma = _cpu_refs(np.ascontiguousarray(img**2.2), positive=True)
        for a, b in zip(linear[:3], gamma[:3]):
            np.testing.assert_allclose(np.asarray(b), 2.2 * np.asarray(a), rtol=1e-4)
        np.testing.assert_allclose(_gains(gamma), _gains(linear), atol=1e-3)

    def test_exposure_does_not_move_the_gains(self):
        img = _dark_scan()
        base = _cpu_refs(img, positive=True)
        brighter = _cpu_refs(np.ascontiguousarray(img * 2.5), positive=True)
        for a, b in zip(base[:3], brighter[:3]):
            np.testing.assert_allclose(np.asarray(b), np.asarray(a) + np.log10(2.5), atol=1e-4)
        np.testing.assert_allclose(_gains(brighter), _gains(base), atol=1e-3)

    def test_a_frame_with_an_axis_keeps_its_refs(self):
        grid = prefilter_log_grid(_dark_scan())
        first = measure_neutral_axis_from_log(grid, LogNegativeBounds(*transfer_axis_bounds(grid, True)), None, 0.0, exclude_subjects=False)
        self.assertIsNotNone(first)
        self.assertEqual(measure_transfer_neutral_axis(grid, True), first)

    def test_a_bordered_positive_finds_its_axis(self):
        """The border pulls the window's dark end onto itself, so the first window finds no axis."""
        img = _bordered(_dark_scan())
        grid = prefilter_log_grid(img, None, 0.05)
        first = measure_neutral_axis_from_log(grid, LogNegativeBounds(*transfer_axis_bounds(grid, True)), None, 0.0, exclude_subjects=False)
        self.assertIsNone(first)
        refs = _cpu_refs(img, positive=True)
        self.assertIsNotNone(refs)
        clean = _affine(_cpu_refs(_dark_scan(), positive=True))
        np.testing.assert_allclose(np.ravel(_affine(refs)), np.ravel(clean), atol=0.01)


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestPositiveAxisGpu(unittest.TestCase):
    def setUp(self):
        from negpy.services.rendering.gpu_engine import GPUEngine

        self.engine = GPUEngine()

    def tearDown(self):
        self.engine.destroy_all()

    def test_gpu_meters_the_same_axis_as_the_cpu(self):
        for img, positive in ((_dark_scan(), True), (_dark_scan(), False), (_bordered(_dark_scan()), True)):
            _, metrics = self.engine.process_to_texture(img, _slide(positive), scale_factor=1.0, apply_layout=False, readback_metrics=True)
            cpu, gpu = _cpu_refs(img, positive), metrics.get("neutral_axis_refs")
            if cpu is None:
                self.assertIsNone(gpu)
                continue
            for a, b in zip(cpu[:3], gpu[:3]):
                np.testing.assert_allclose(np.asarray(b), np.asarray(a), atol=1e-4)

    def test_a_tiled_export_renders_like_the_preview(self):
        settings = _slide(True)
        settings = replace(settings, export=replace(settings.export, export_resolution_mode="original"))
        for frame in (_dark_scan(), _bordered(_dark_scan())):
            img = np.ascontiguousarray(np.tile(frame, (1, 8, 1)))
            tiled, _ = self.engine._process_tiled(img, settings, scale_factor=1.0)
            tex, _ = self.engine.process_to_texture(img, settings, scale_factor=1.0, apply_layout=False, readback_metrics=True)
            direct = self.engine._readback_downsampled(tex)
            self.assertEqual(tiled.shape, direct.shape)
            self.assertLess(float(np.abs(tiled - direct).mean()), 0.001)


if __name__ == "__main__":
    unittest.main()
