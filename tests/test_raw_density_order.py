"""A camera RAW's matrix splits at its white-balance fold around the density stage: the gains apply
before the log, the primaries after the crosstalk unmix, fade and stain offset."""

import negpy_core
import numpy as np
import pytest

from negpy.domain.interfaces import PipelineContext
from negpy.features.exposure.normalization import normalize_log_image, relog_matrix_image, unmix_log_image
from negpy.features.process.capture_color import (
    _primaries,
    apply_camera_matrix,
    camera_matrix_parts,
    camera_to_working_matrix,
    wb_only_cam_xyz,
)
from negpy.features.process.fade import fade_white_space, measure_white, fade_density
from negpy.features.process.models import ProcessConfig, ProcessMode
from negpy.features.transparency.logic import density_stage
from negpy.features.transparency.processor import TransparencyBaseProcessor

_CAMERA = [[0.6722, -0.0635, -0.0963], [-0.4287, 1.246, 0.2046], [-0.0908, 0.2268, 0.5762]]
_WB = [2.0, 1.0, 1.5, 1.0]
_FADE = dict(fade_ratio_r=0.4, fade_ratio_g=2.0, fade_ratio_b=1.5)
_PATHS = [pytest.param(False, id="transfer"), pytest.param(True, id="positive")]


def _slide(positive: bool, **kw) -> ProcessConfig:
    return ProcessConfig(process_mode=ProcessMode.E6, positive_source=positive, linear_raw=True, **kw)


def _render(config: ProcessConfig, img: np.ndarray, fade_grid: bool = False, cam_xyz=_CAMERA):
    ctx = PipelineContext(
        original_size=img.shape[:2],
        scale_factor=1.0,
        process_mode=ProcessMode.E6,
        cam_xyz=cam_xyz,
        camera_wb=_WB,
        wants_fade_grid=fade_grid,
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


_LOG_GAINS = np.log10([2.0, 1.0, 1.5])


@pytest.mark.parametrize("positive", _PATHS)
def test_a_camera_splits_at_the_fold_only_with_a_density_matrix(positive):
    """The gains become a log offset in the density matrix; the primaries move after it."""
    config = _slide(positive)
    unmix = np.array([[1.0, -0.1, 0.0], [0.0, 1.0, -0.2], [-0.05, 0.0, 1.0]])
    stage = density_stage(config, _CAMERA, _WB, unmix)
    assert stage.pre is None
    np.testing.assert_allclose(stage.unmix, np.hstack([unmix, (unmix @ _LOG_GAINS)[:, None]]), atol=1e-12)
    np.testing.assert_array_equal(stage.post, camera_matrix_parts(_CAMERA, _WB)[1])
    stage = density_stage(config, _CAMERA, _WB, None)
    np.testing.assert_array_equal(stage.pre, camera_to_working_matrix(_CAMERA, _WB))
    assert stage.unmix is None and stage.post is None


def test_the_gains_add_to_a_stain_offset():
    affine = np.hstack([np.eye(3) * 2.0, np.array([[0.1], [0.2], [0.3]])])
    stage = density_stage(_slide(False), _CAMERA, _WB, affine)
    np.testing.assert_allclose(stage.unmix[:, 3], affine[:, 3] + 2.0 * _LOG_GAINS, atol=1e-12)


def test_a_balanced_decode_moves_only_the_primaries():
    """Highlight reconstruction bakes the white balance into the decode, so there is no fold."""
    config = ProcessConfig(process_mode=ProcessMode.E6, linear_raw=False, highlight_reconstruction=5)
    unmix = np.eye(3) * 2.0
    stage = density_stage(config, _CAMERA, _WB, unmix)
    assert stage.pre is None and stage.unmix is unmix
    np.testing.assert_array_equal(stage.post, camera_matrix_parts(_CAMERA, None)[1])


def test_identity_primaries_do_not_convert():
    """An Input ICC leaves the camera only its white-balance fold (`wb_only_cam_xyz`): nothing to convert after."""
    stage = density_stage(_slide(False), wb_only_cam_xyz(_CAMERA), _WB, np.eye(3))
    assert stage.pre is None and stage.post is None
    np.testing.assert_allclose(stage.unmix[:, 3], _LOG_GAINS, atol=1e-12)


def test_a_negative_keeps_the_whole_matrix_before_the_log():
    config = ProcessConfig(process_mode=ProcessMode.C41, linear_raw=True)
    unmix = np.eye(3) * 2.0
    stage = density_stage(config, _CAMERA, _WB, unmix)
    np.testing.assert_array_equal(stage.pre, camera_to_working_matrix(_CAMERA, _WB))
    assert stage.unmix is unmix and stage.post is None


@pytest.mark.parametrize("positive", _PATHS)
def test_remove_stain_leaves_a_white_border_neutral_and_converts_after(positive):
    """Estimate meters the white in white-balanced camera channels, the primaries keep the border
    neutral, and a colored patch takes the primaries after the stain offset."""
    balanced = np.full((40, 60, 3), 0.8, np.float32)
    balanced[10:30, 15:45] = (0.3, 0.15, 0.08)
    img = _unbalanced(balanced)
    _, ctx = _render(_slide(positive), img, fade_grid=True)
    white = measure_white(ctx.metrics["transfer_grid"])
    assert white is not None
    np.testing.assert_allclose(white, np.log10(0.8), atol=1e-4)
    space = fade_white_space(_CAMERA, _WB)
    stored = dict(fade_white_r=white[0], fade_white_g=white[1], fade_white_b=white[2], fade_white_space=space)
    config = _slide(positive, fade_white_anchor=True, **stored, **_FADE)
    _, ctx = _render(config, img)
    restored = _restored_log(ctx)
    np.testing.assert_allclose(restored[:5].reshape(-1, 3), -0.1, atol=1e-4)
    unmix = fade_density(config, ProcessMode.E6, space).matrix
    expected = relog_matrix_image(unmix_log_image(np.log10(balanced), unmix), camera_matrix_parts(_CAMERA, _WB)[1])
    np.testing.assert_allclose(restored, expected, atol=1e-4)
    converted_first = unmix_log_image(np.log10(img @ camera_to_working_matrix(_CAMERA, _WB).T), unmix)
    assert float(np.abs(restored - converted_first)[10:30, 15:45].max()) > 0.05


@pytest.mark.parametrize("positive", _PATHS)
def test_a_faded_gray_ramp_restores_to_neutral(positive):
    # A white measured as none: the fade scales about L = 0, as the ramp was faded.
    config = _slide(positive, fade_white_space=fade_white_space(_CAMERA, _WB), **_FADE)
    restore = fade_density(config, ProcessMode.E6).matrix
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
    _, ctx = _render(_slide(positive), flat, fade_grid=True)
    np.testing.assert_allclose(ctx.metrics["transfer_grid"].reshape(-1, 3)[0], np.log10((0.5, 0.2, 0.1)), atol=1e-4)


def test_a_render_stores_the_fade_grid_only_on_request():
    img = _unbalanced(np.full((40, 60, 3), 0.5, np.float32))
    _, ctx = _render(_slide(False), img)
    assert "transfer_grid" not in ctx.metrics


def test_a_white_read_in_working_space_reads_as_stale():
    """A tag that hashes the camera matrix alone named a white read after the matrix, in working space."""
    import hashlib

    from negpy.features.process.fade import fade_white_stale

    working = hashlib.sha1(np.round(np.asarray(camera_to_working_matrix(_CAMERA, _WB), dtype=np.float64), 4).tobytes()).hexdigest()[:12]
    config = _slide(False, fade_white_r=-0.5, fade_white_g=-0.4, fade_white_b=-0.6, fade_white_space=working, **_FADE)
    assert working != fade_white_space(_CAMERA, _WB)
    assert fade_white_stale(config, fade_white_space(_CAMERA, _WB))
