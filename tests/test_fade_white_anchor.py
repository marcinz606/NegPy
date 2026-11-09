"""Remove Stain (White Anchor): Fade Restoration above each channel's own white, (D - w) / a."""

import unittest
from dataclasses import replace

import numpy as np
import pytest

from fade_crossover_scene import estimate_fade
from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.normalization import LogNegativeBounds, prefilter_log_grid, unmix_log_image
from negpy.features.process.fade import (
    FADE_WHITE_HEADROOM,
    STAIN_BLUE_MIN,
    STAIN_FULL_SCALE_TOLERANCE,
    fade_white_target,
    measure_white,
    stain_suggested,
    fade_density,
)
from negpy.features.process.models import ProcessConfig, ProcessMode
from negpy.features.transparency.logic import STAIN_PIVOT_MIN_ARM, transfer_bounds, transfer_cast_affine
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.services.rendering.engine import base_processor, exposure_processor
from tests.test_fade_fixtures import make_fixture

_RATIO_G, _RATIO_B = 0.8, 0.6
# A yellow stain on clear film: blue's white sits 0.4 D below red's.
_STAIN = (0.0, -0.05, -0.4)
_WHITE = {"fade_white_r": -0.02, "fade_white_g": -0.07, "fade_white_b": -0.4}


def _stained_slide(seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    fx = make_fixture("stained", _RATIO_G, _RATIO_B, _STAIN, seed=seed)
    return fx.clean.astype(np.float32), np.power(10.0, fx.faded_log).astype(np.float32)


def _config(**process) -> WorkspaceConfig:
    cfg = WorkspaceConfig()
    return replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, **process))


def _render(image: np.ndarray, cfg: WorkspaceConfig) -> np.ndarray:
    h, w = image.shape[:2]
    ctx = PipelineContext(
        original_size=(h, w),
        scale_factor=1.0,
        process_mode=cfg.process.process_mode,
        cam_xyz=None,
        camera_wb=None,
        wants_uv_grid=False,
    )
    norm = base_processor(cfg).process(np.ascontiguousarray(image), ctx)
    return np.asarray(exposure_processor(cfg).process(norm, ctx))[..., :3].astype(np.float64)


def _chroma_rms(out: np.ndarray, ref: np.ndarray) -> float:
    """RMS of the (R-G, B-G) log difference against the reference, on tones away from the clips."""
    log_o, log_r = (np.log10(np.clip(x, 1e-4, None)) for x in (out, ref))
    diff = (log_o - log_o[..., 1:2]) - (log_r - log_r[..., 1:2])
    ok = (ref.min(axis=-1) > 0.01) & (ref.max(axis=-1) < 0.97)
    return float(np.sqrt(np.mean(diff[ok][:, [0, 2]] ** 2)))


def _slide(**fields) -> ProcessConfig:
    return ProcessConfig(process_mode=ProcessMode.E6, **fields)


def _grid(image: np.ndarray) -> np.ndarray:
    """The meter grid Estimate reads, for a slide with no crop, unmix or fade."""
    return prefilter_log_grid(image, None, 0.0)


def test_off_keeps_the_square_matrix_whatever_white_is_stored():
    stored = _slide(fade_ratio_g=_RATIO_G, fade_ratio_b=_RATIO_B, **_WHITE)
    bare = _slide(fade_ratio_g=_RATIO_G, fade_ratio_b=_RATIO_B)
    np.testing.assert_array_equal(fade_density(stored, ProcessMode.E6).matrix, fade_density(bare, ProcessMode.E6).matrix)
    assert fade_density(_slide(**_WHITE), ProcessMode.E6).matrix is None


def test_off_renders_bit_identical_with_a_stored_white():
    _clean, faded = _stained_slide()
    a = _render(faded, _config(fade_ratio_g=_RATIO_G, fade_ratio_b=_RATIO_B))
    b = _render(faded, _config(fade_ratio_g=_RATIO_G, fade_ratio_b=_RATIO_B, **_WHITE))
    np.testing.assert_array_equal(a, b)


def test_on_maps_a_bright_white_just_under_full_scale_before_the_fade():
    process = _slide(fade_ratio_g=_RATIO_G, fade_ratio_b=_RATIO_B, fade_white_anchor=True, **_WHITE)
    matrix = fade_density(process, ProcessMode.E6).matrix
    assert matrix.shape == (3, 4)
    white = np.array([[[_WHITE["fade_white_r"], _WHITE["fade_white_g"], _WHITE["fade_white_b"]]]], dtype=np.float32)
    np.testing.assert_allclose(unmix_log_image(white, matrix), -FADE_WHITE_HEADROOM, atol=1e-6)
    # One decade above the white restores as one decade over each layer's survival.
    darker = unmix_log_image(white - 1.0, matrix).reshape(3) + FADE_WHITE_HEADROOM
    np.testing.assert_allclose(darker, [-1.0, -1.0 / _RATIO_G, -1.0 / _RATIO_B], rtol=1e-5)


def test_on_keeps_a_dim_white_at_its_cleanest_channel():
    dim = {"fade_white_r": -0.8, "fade_white_g": -1.0, "fade_white_b": -1.1}
    process = _slide(fade_ratio_g=_RATIO_G, fade_ratio_b=_RATIO_B, fade_white_anchor=True, **dim)
    white = np.array([[list(dim.values())]], dtype=np.float32)
    np.testing.assert_allclose(unmix_log_image(white, fade_density(process, ProcessMode.E6).matrix), -0.8, atol=1e-6)


def test_white_headroom_is_the_documented_one():
    assert FADE_WHITE_HEADROOM == 0.1


def test_on_with_no_white_measured_changes_nothing():
    process = _slide(fade_ratio_g=_RATIO_G, fade_white_anchor=True)
    np.testing.assert_array_equal(
        fade_density(process, ProcessMode.E6).matrix,
        fade_density(_slide(fade_ratio_g=_RATIO_G), ProcessMode.E6).matrix,
    )


def test_remove_stain_applies_to_a_positive_frame():
    process = _slide(positive_source=True, fade_white_anchor=True, **_WHITE)
    assert fade_density(process, ProcessMode.E6).matrix is not None
    assert fade_white_target(process) == fade_white_target(_slide(fade_white_anchor=True, **_WHITE))


def test_white_anchor_applies_to_slides_only():
    process = _slide(fade_white_anchor=True, **_WHITE)
    assert fade_density(replace(process, process_mode=ProcessMode.C41), ProcessMode.C41).matrix is None


def test_a_stained_slide_restores_closer_to_the_clean_one_with_the_anchor_on():
    clean, faded = _stained_slide()
    reference = _render(clean, _config())
    white = measure_white(prefilter_log_grid(faded, None, 0.0))
    fade = {"fade_ratio_g": _RATIO_G, "fade_ratio_b": _RATIO_B}
    off = _render(faded, _config(**fade))
    on = _render(faded, _config(**fade, fade_white_anchor=True, fade_white_r=white[0], fade_white_g=white[1], fade_white_b=white[2]))
    err_off, err_on = _chroma_rms(off, reference), _chroma_rms(on, reference)
    assert err_on < 0.5 * err_off, (err_on, err_off)


def test_cast_removal_meters_the_anchored_film():
    """Cast Removal's neutral axis reads the grid after the affine, so the stain is gone from it."""
    _clean, faded = _stained_slide()
    white = measure_white(prefilter_log_grid(faded, None, 0.0))
    process = _slide(
        fade_ratio_g=_RATIO_G,
        fade_ratio_b=_RATIO_B,
        fade_white_anchor=True,
        fade_white_r=white[0],
        fade_white_g=white[1],
        fade_white_b=white[2],
    )
    grid = unmix_log_image(prefilter_log_grid(faded, None, 0.0), fade_density(process, ProcessMode.E6).matrix)
    top = np.percentile(grid.reshape(-1, 3), 99.9, axis=0)
    np.testing.assert_allclose(top, -FADE_WHITE_HEADROOM, atol=0.02)


# Log refs (mid, shadow, highlight, confidence): neutral at the midtone, blue lighter in the shadows.
_AXIS_SHADOW_OFF = ((-1.5, -1.5, -1.5), (-2.3, -2.3, -2.1), None, 1.0)


def _normalized_white(bounds, target: float) -> tuple:
    return tuple((target - f) / (c - f) for f, c in zip(bounds.floors, bounds.ceils))


def test_cast_removal_keeps_the_anchored_white_neutral():
    """With Remove Stain on, a shadow ref off the midtone's line must not tilt the white."""
    bounds = LogNegativeBounds(*transfer_bounds())
    gain, offset = transfer_cast_affine(bounds, _AXIS_SHADOW_OFF, 1.0, white_target=-FADE_WHITE_HEADROOM)
    w = _normalized_white(bounds, -FADE_WHITE_HEADROOM)
    landed = [gain[ch] * w[ch] + offset[ch] for ch in range(3)]
    np.testing.assert_allclose(landed, landed[1], atol=1e-9)
    np.testing.assert_allclose(gain, 1.0, atol=1e-9)


def test_cast_removal_without_the_anchor_fits_the_shadow_ref():
    bounds = LogNegativeBounds(*transfer_bounds())
    gain, _offset = transfer_cast_affine(bounds, _AXIS_SHADOW_OFF, 1.0, white_target=None)
    assert gain[2] > 1.2


def test_cast_removal_keeps_the_shadow_line_when_the_midtone_is_near_the_white():
    """A short midtone-to-white arm would turn a small cast into a large gain on every darker tone."""
    bounds = LogNegativeBounds(*transfer_bounds())
    near = ((-0.40, -0.48, -0.52), (-1.78, -1.78, -1.78), None, 1.0)
    anchored = transfer_cast_affine(bounds, near, 1.0, white_target=-FADE_WHITE_HEADROOM)
    np.testing.assert_allclose(anchored, transfer_cast_affine(bounds, near, 1.0, white_target=None))
    assert max(abs(g - 1.0) for g in anchored[0]) < 0.1


def test_the_pivot_arm_is_the_documented_one():
    assert STAIN_PIVOT_MIN_ARM == 0.5


def test_the_white_skips_cells_clipped_in_every_channel():
    _clean, faded = _stained_slide()
    grid = prefilter_log_grid(faded, None, 0.0)
    leak = grid.copy()
    leak[: max(1, leak.shape[0] // 20)] = 0.0  # a mount gap at full scale in all three channels
    np.testing.assert_allclose(measure_white(leak), measure_white(grid), atol=0.03)
    assert measure_white(leak) != (0.0, 0.0, 0.0)


def test_the_white_stops_at_full_scale_where_the_gains_lift_a_clipped_cell():
    """Recovery Off: green clips at full scale and the white-balance gains lift red and blue past it."""
    _clean, faded = _stained_slide()
    grid = prefilter_log_grid(faded, None, 0.0)
    clipped = grid.copy()
    clipped[: max(1, clipped.shape[0] // 4)] = (0.15, 0.0, 0.10)
    clipped[: max(1, clipped.shape[0] // 4), ::2, 2] = -0.01
    white = measure_white(clipped)
    assert white is not None and max(white) <= 0.0
    np.testing.assert_allclose(white, (0.0, 0.0, 0.0), atol=0.02)


def test_no_white_without_a_film_cell():
    assert measure_white(np.zeros((8, 8, 3), np.float32)) is None
    assert measure_white(np.full((8, 8, 3), -6.0, np.float32)) is None


def test_an_anchored_estimate_with_no_film_cell_says_so():
    est = estimate_fade(np.zeros((32, 32, 3), np.float32), white_anchor=True)
    assert est.white is None and est.reason
    assert (est.green, est.blue) == (1.0, 1.0)


@pytest.mark.parametrize(
    "process, target",
    [
        ({"fade_white_anchor": True, **_WHITE}, -FADE_WHITE_HEADROOM),
        ({"fade_white_anchor": True, "fade_white_r": -0.8, "fade_white_g": -1.0, "fade_white_b": -1.1}, -0.8),
        ({"fade_white_anchor": False, **_WHITE}, None),
        ({"fade_white_anchor": True}, None),
    ],
)
def test_the_transfer_stage_knows_where_the_white_is_anchored(process, target):
    cfg = _config(**process)
    assert exposure_processor(cfg).white_target == target
    assert fade_white_target(cfg.process) == target


def test_estimate_measures_the_stain_as_the_white():
    _clean, faded = _stained_slide()
    est = estimate_fade(_grid(faded))
    np.testing.assert_allclose(est.white, _STAIN, atol=0.02)


def test_the_anchored_estimate_reads_the_true_survivals_through_a_stain():
    _clean, faded = _stained_slide()
    plain = estimate_fade(_grid(faded))
    anchored = estimate_fade(_grid(faded), white_anchor=True)
    assert anchored.reason == ""
    assert anchored.green == pytest.approx(_RATIO_G, abs=0.03)
    assert anchored.blue == pytest.approx(_RATIO_B, abs=0.03)
    assert abs(plain.blue - _RATIO_B) > 0.15  # the stain inflates blue without the anchor


@pytest.mark.parametrize(
    ("white", "expected"),
    [
        ((-0.05, -0.18, -0.73), True),  # NORAD 106-25: bright, heavy yellow stain
        ((-0.04, -0.16, -0.28), True),  # NORAD 106-5: blue below red by more than the minimum
        ((-0.14, -0.12, -0.49), True),  # a heavy stain on a scan placed under full scale
        ((-0.01, -0.04, -0.16), True),  # a mild stain on a bright white
        ((-0.76, -1.05, -1.05), False),  # SilverFast at gamma 2.2: no white near full scale
        ((-2.9, -3.4, -3.25), False),  # a scan with no white object
        ((-0.25, -0.12, -0.01), False),  # bluish whites: a snow or sky frame
        ((0.0, -0.05, -0.10), False),  # a stain under the minimum
        ((-0.09, -0.02, -0.13), False),  # a red cast: blue barely below red
        ((-0.25, -0.30, -0.60), False),  # a stain on a white too far under full scale
        ((0.0, -0.4, -0.3), False),  # green, not blue, is darkest
    ],
)
def test_stain_suggestion_rule(white, expected):
    assert stain_suggested(white) is expected


def test_stain_suggestion_thresholds_are_the_documented_ones():
    assert (STAIN_FULL_SCALE_TOLERANCE, STAIN_BLUE_MIN) == (0.2, 0.08)


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestWhiteAnchorGpuParity(unittest.TestCase):
    def _render(self, processor, settings, img, prefer_gpu):
        result, _ = processor.run_pipeline(
            img, settings, "white-anchor-parity", render_size_ref=float(max(img.shape[:2])), prefer_gpu=prefer_gpu, readback_metrics=False
        )
        arr = result.readback() if hasattr(result, "readback") else result
        return np.asarray(arr)[:, :, :3].astype(np.float64)

    def _assert_parity(self, settings):
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")
        _clean, faded = _stained_slide()
        cpu = self._render(processor, settings, faded[:128, :128].copy(), prefer_gpu=False)
        gpu = self._render(processor, settings, faded[:128, :128].copy(), prefer_gpu=True)
        self.assertEqual(cpu.shape, gpu.shape)
        mad = float(np.mean(np.abs(cpu - gpu)))
        mx = float(np.max(np.abs(cpu - gpu)))
        self.assertLess(mad, 0.005, f"mean abs diff {mad:.5f}")
        self.assertLess(mx, 0.04, f"max abs diff {mx:.5f}")

    def test_fade_off(self):
        self._assert_parity(_config(fade_ratio_g=_RATIO_G, fade_ratio_b=_RATIO_B, **_WHITE))

    def test_fade_on(self):
        self._assert_parity(_config(fade_ratio_g=_RATIO_G, fade_ratio_b=_RATIO_B, fade_white_anchor=True, **_WHITE))

    def test_fade_on_with_cast_removal(self):
        cfg = _config(fade_ratio_g=_RATIO_G, fade_ratio_b=_RATIO_B, fade_white_anchor=True, **_WHITE)
        self._assert_parity(replace(cfg, exposure=replace(cfg.exposure, cast_removal_strength=1.0)))
