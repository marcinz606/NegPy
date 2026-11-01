"""A Positive source keeps its own primaries through the decode. The engines convert them after the
crosstalk unmix, fade and stain offset, which act on the source's channels."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from negpy.desktop.session import camera_raw_preview
from negpy.domain.interfaces import PipelineContext
from negpy.features.exposure.normalization import relog_matrix_image, unmix_log_image
from negpy.features.process.capture_color import camera_to_working_matrix, is_source_profile, source_cam_xyz
from negpy.features.process.fade import unmix_with_fade
from negpy.features.process.models import ProcessConfig, ProcessMode
from negpy.features.transparency.logic import density_stage
from negpy.features.transparency.processor import TransparencyBaseProcessor
from negpy.infrastructure.loaders.helpers import camera_xyz_matrix
from negpy.infrastructure.loaders.jpeg_loader import JpegLoader
from negpy.infrastructure.loaders.source_encoding import _SRGB_COLORANTS_D50, ProfileDecode, _to_working, apply_decode, decode_positive

_P = _to_working(_SRGB_COLORANTS_D50)
_SOURCE = source_cam_xyz(_P)
# A camera's XYZ->cam rows, as libraw gives them (3 rows after camera_xyz_matrix).
_CAMERA = [[0.6722, -0.0635, -0.0963], [-0.4287, 1.246, 0.2046], [-0.0908, 0.2268, 0.5762]]
_FADE = dict(fade_ratio_r=0.4, fade_ratio_g=2.0, fade_ratio_b=1.5)


def _slide(**kw) -> ProcessConfig:
    return ProcessConfig(process_mode=ProcessMode.E6, **kw)


def _img(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.02, 1.0, (40, 60, 3)).astype(np.float32)


def test_source_cam_xyz_gives_back_the_profile_matrix():
    np.testing.assert_allclose(camera_to_working_matrix(_SOURCE), _P, atol=2e-4)
    assert source_cam_xyz(None) is None
    assert is_source_profile(_SOURCE) and not is_source_profile(_CAMERA) and not is_source_profile(None)


@pytest.mark.parametrize(
    ("positive", "unmix", "expected"),
    [(True, True, "post"), (True, False, "pre"), (False, True, "pre")],
)
def test_the_primaries_follow_the_density_matrix_only_for_a_positive_source_profile(positive, unmix, expected):
    m = np.eye(3) * 2.0 if unmix else None
    stage = density_stage(_slide(positive_source=positive), _SOURCE, None, m)
    moved = stage.post if expected == "post" else stage.pre
    assert (stage.pre is None, stage.post is None) == ((expected == "post"), (expected == "pre"))
    np.testing.assert_array_equal(moved, camera_to_working_matrix(_SOURCE))
    assert stage.unmix is m


def test_no_source_matrix_has_nothing_to_move():
    stage = density_stage(_slide(positive_source=True), None, None, np.eye(3))
    assert stage.pre is None and stage.post is None


def _render(config: ProcessConfig, img: np.ndarray, cam_xyz=_SOURCE, camera_wb=None):
    ctx = PipelineContext(
        original_size=img.shape[:2],
        scale_factor=1.0,
        process_mode=ProcessMode.E6,
        cam_xyz=cam_xyz,
        camera_wb=camera_wb,
        wants_fade_grid=True,
    )
    return TransparencyBaseProcessor(config).process(img, ctx), ctx


def _restored_log(ctx) -> np.ndarray:
    floors, ceils = ctx.metrics["log_bounds"].floors, ctx.metrics["log_bounds"].ceils
    return ctx.metrics["normalized_log"] * (np.asarray(ceils) - np.asarray(floors)) + np.asarray(floors)


def test_a_positive_fade_acts_on_the_source_channels_then_converts():
    config = _slide(positive_source=True, **_FADE)
    img = _img()
    _, ctx = _render(config, img)
    p = camera_to_working_matrix(_SOURCE)
    expected = relog_matrix_image(unmix_log_image(np.log10(img), unmix_with_fade(config, ProcessMode.E6)), p)
    converted_first = unmix_log_image(np.log10(img @ p.T), unmix_with_fade(config, ProcessMode.E6))
    restored = _restored_log(ctx)
    np.testing.assert_allclose(restored, expected, atol=1e-4)
    assert float(np.abs(restored - converted_first).max()) > 0.05


def test_a_positive_at_neutral_renders_as_the_conversion_first():
    img = _img(1)
    res, _ = _render(_slide(positive_source=True), img)
    ctx = PipelineContext(original_size=img.shape[:2], scale_factor=1.0, process_mode=ProcessMode.E6)
    p = camera_to_working_matrix(_SOURCE)
    ref = TransparencyBaseProcessor(_slide(positive_source=True)).process(img @ p.T, ctx)
    np.testing.assert_allclose(res, ref, atol=1e-5)


def test_estimate_reads_the_source_channels():
    """Estimate renders with the fade at identity, so its grid must not take the conversion."""
    img = _img(2)
    _, ctx = _render(_slide(positive_source=True), img)
    grid = ctx.metrics["transfer_grid"]
    assert grid.min() >= np.log10(img.min()) - 1e-4
    flat = np.full((40, 60, 3), (0.5, 0.2, 0.1), np.float32)
    _, ctx = _render(_slide(positive_source=True), flat)
    np.testing.assert_allclose(ctx.metrics["transfer_grid"].reshape(-1, 3)[0], np.log10((0.5, 0.2, 0.1)), atol=1e-4)


def test_a_positive_jpeg_keeps_its_primaries_and_carries_the_matrix(tmp_path):
    path = str(tmp_path / "srgb.jpg")
    Image.new("RGB", (8, 8), (255, 0, 0)).save(path, quality=100)
    ctx, metadata = JpegLoader().load(path, positive_decode="auto")
    with ctx as raw:
        assert raw.data[0, 0, 0] > 0.95 and raw.data[0, 0, 1] < 0.02
        np.testing.assert_allclose(camera_to_working_matrix(camera_xyz_matrix(raw)), _P, atol=2e-4)
    assert metadata["source_encoding"]


def test_a_wider_source_converts_in_the_decode():
    """ProPhoto reaches past Adobe RGB, so it takes the decode's gamut pass and carries no matrix."""
    prophoto = _to_working(np.array([[0.7977, 0.1352, 0.0313], [0.2880, 0.7119, 0.0001], [0.0, 0.0, 0.8249]]))
    curves = (np.linspace(0.0, 1.0, 256, dtype=np.float32),) * 3
    img = np.random.default_rng(3).integers(0, 256, (6, 8, 3), dtype=np.uint8)
    linear, matrix = decode_positive(img, ProfileDecode(curves, prophoto))
    assert matrix is None
    np.testing.assert_array_equal(linear, apply_decode(img, ProfileDecode(curves, prophoto)))
    _, srgb = decode_positive(img, ProfileDecode(curves, _P))
    assert srgb is _P


def test_a_positive_decode_is_not_a_camera_raw():
    assert camera_raw_preview(SimpleNamespace(preview_cam_xyz=_CAMERA))
    assert not camera_raw_preview(SimpleNamespace(preview_cam_xyz=_SOURCE))
    assert not camera_raw_preview(SimpleNamespace(preview_cam_xyz=None))
