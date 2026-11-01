"""A camera RAW's matrix splits at its white-balance fold around the density stage: the gains apply
before the log, the primaries after the crosstalk unmix, fade and stain offset."""

import negpy_core
import numpy as np
import pytest

from negpy.domain.interfaces import PipelineContext
from negpy.features.exposure.normalization import normalize_log_image
from negpy.features.process.capture_color import _primaries, apply_camera_matrix, camera_matrix_parts, camera_to_working_matrix
from negpy.features.process.fade import measure_white, unmix_with_fade
from negpy.features.process.models import ProcessConfig, ProcessMode
from negpy.features.transparency.logic import density_stage_matrices
from negpy.features.transparency.processor import TransparencyBaseProcessor

_CAMERA = [[0.6722, -0.0635, -0.0963], [-0.4287, 1.246, 0.2046], [-0.0908, 0.2268, 0.5762]]
_WB = [2.0, 1.0, 1.5, 1.0]
_FADE = dict(fade_ratio_r=0.4, fade_ratio_g=2.0, fade_ratio_b=1.5)
_PATHS = [pytest.param(False, id="transfer"), pytest.param(True, id="positive")]


def _slide(positive: bool, **kw) -> ProcessConfig:
    return ProcessConfig(process_mode=ProcessMode.E6, positive_source=positive, linear_raw=True, **kw)


def _render(config: ProcessConfig, img: np.ndarray, readback: bool = False):
    ctx = PipelineContext(
        original_size=img.shape[:2],
        scale_factor=1.0,
        process_mode=ProcessMode.E6,
        cam_xyz=_CAMERA,
        camera_wb=_WB,
        readback_metrics=readback,
    )
    return TransparencyBaseProcessor(config).process(img, ctx), ctx


def _restored_log(ctx) -> np.ndarray:
    floors, ceils = ctx.metrics["log_bounds"].floors, ctx.metrics["log_bounds"].ceils
    return ctx.metrics["normalized_log"] * (np.asarray(ceils) - np.asarray(floors)) + np.asarray(floors)


def _unbalanced(balanced: np.ndarray) -> np.ndarray:
    """The Linear RAW buffer whose white-balanced camera channels are *balanced*."""
    return (balanced / np.asarray(_WB[:3], dtype=np.float32)).astype(np.float32)


def test_the_parts_rebuild_the_matrix_and_leave_its_bits():
    gains, primaries = camera_matrix_parts(_CAMERA, _WB)
    np.testing.assert_array_equal(gains, np.diag(np.float32([2.0, 1.0, 1.5])))
    np.testing.assert_allclose(primaries @ gains, camera_to_working_matrix(_CAMERA, _WB), atol=1e-6)
    np.testing.assert_allclose(primaries.astype(np.float64).sum(axis=1), 1.0, atol=1e-6)
    reference = np.ascontiguousarray(_primaries(_CAMERA) @ np.diag(np.array(_WB[:3]) / _WB[1]), dtype=np.float32)
    np.testing.assert_array_equal(camera_to_working_matrix(_CAMERA, _WB), reference)
    assert camera_matrix_parts(_CAMERA, None)[0] is None
    assert camera_matrix_parts(_CAMERA, [0.0, 1.0, 1.0])[0] is None
    assert camera_matrix_parts(None, _WB) == (None, None)


@pytest.mark.parametrize("positive", _PATHS)
def test_a_camera_splits_at_the_fold_only_with_a_density_matrix(positive):
    config = _slide(positive)
    gains, primaries = camera_matrix_parts(_CAMERA, _WB)
    pre, post = density_stage_matrices(config, _CAMERA, _WB, np.eye(3))
    np.testing.assert_array_equal(pre, gains)
    np.testing.assert_array_equal(post, primaries)
    pre, post = density_stage_matrices(config, _CAMERA, _WB, None)
    np.testing.assert_array_equal(pre, camera_to_working_matrix(_CAMERA, _WB))
    assert post is None


def test_a_balanced_decode_moves_only_the_primaries():
    """Highlight reconstruction bakes the white balance into the decode, so there is no fold."""
    config = ProcessConfig(process_mode=ProcessMode.E6, linear_raw=False, highlight_reconstruction=5)
    pre, post = density_stage_matrices(config, _CAMERA, _WB, np.eye(3))
    assert pre is None
    np.testing.assert_array_equal(post, camera_matrix_parts(_CAMERA, None)[1])


def test_a_negative_keeps_the_whole_matrix_before_the_log():
    config = ProcessConfig(process_mode=ProcessMode.C41, linear_raw=True)
    pre, post = density_stage_matrices(config, _CAMERA, _WB, np.eye(3))
    np.testing.assert_array_equal(pre, camera_to_working_matrix(_CAMERA, _WB))
    assert post is None


@pytest.mark.parametrize("positive", _PATHS)
def test_remove_stain_leaves_a_white_border_neutral(positive):
    """Estimate meters the white in white-balanced camera channels, and the primaries keep it neutral."""
    img = _unbalanced(np.full((40, 60, 3), 0.8, np.float32))
    _, ctx = _render(_slide(positive), img, readback=True)
    white = measure_white(ctx.metrics["transfer_grid"])
    assert white is not None
    np.testing.assert_allclose(white, np.log10(0.8), atol=1e-4)
    config = _slide(positive, fade_white_anchor=True, fade_white_r=white[0], fade_white_g=white[1], fade_white_b=white[2], **_FADE)
    _, ctx = _render(config, img)
    np.testing.assert_allclose(_restored_log(ctx).reshape(-1, 3), -0.1, atol=1e-4)


@pytest.mark.parametrize("positive", _PATHS)
def test_a_faded_gray_ramp_restores_to_neutral(positive):
    config = _slide(positive, **_FADE)
    restore = unmix_with_fade(config, ProcessMode.E6)
    gray = np.repeat(np.linspace(-1.5, -0.05, 60, dtype=np.float64)[None, :, None], 3, axis=2)
    faded_log = gray @ np.linalg.inv(restore).T
    img = _unbalanced(np.repeat(10.0**faded_log, 4, axis=0).astype(np.float32))
    _, ctx = _render(config, img)
    out = _restored_log(ctx)
    np.testing.assert_allclose(out - out[..., 1:2], 0.0, atol=1e-4)
    np.testing.assert_allclose(out[..., 1], np.repeat(gray[..., 1], 4, axis=0), atol=1e-4)
    # Converted before the log, the primaries mix the faded channels and the ramp takes a cast.
    full = camera_to_working_matrix(_CAMERA, _WB)
    converted_first = negpy_core.log_density_open(apply_camera_matrix(img, full)) @ restore.T.astype(np.float32)
    assert float(np.abs(converted_first - converted_first[..., 1:2]).max()) > 0.05


@pytest.mark.parametrize("positive", _PATHS)
def test_no_density_matrix_renders_as_the_whole_matrix_first(positive):
    img = np.random.default_rng(4).uniform(0.02, 1.0, (40, 60, 3)).astype(np.float32)
    config = _slide(positive)
    _, ctx = _render(config, img)
    linear = apply_camera_matrix(negpy_core.nan_to_num(img, 1e-6, 1.0, 1e-6), camera_to_working_matrix(_CAMERA, _WB))
    expected = normalize_log_image(negpy_core.log_density_open(linear), ctx.metrics["log_bounds"])
    np.testing.assert_array_equal(ctx.metrics["normalized_log"], expected)


@pytest.mark.parametrize("positive", _PATHS)
def test_estimate_reads_white_balanced_camera_channels(positive):
    flat = _unbalanced(np.full((40, 60, 3), (0.5, 0.2, 0.1), np.float32))
    _, ctx = _render(_slide(positive), flat, readback=True)
    np.testing.assert_allclose(ctx.metrics["transfer_grid"].reshape(-1, 3)[0], np.log10((0.5, 0.2, 0.1)), atol=1e-4)
