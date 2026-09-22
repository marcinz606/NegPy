"""Crop by Default: the inscribed rect that avoids the replicated-edge wedge fine
rotation and keystone (Tilt/Swing) leave behind."""

import cv2
import numpy as np
import pytest

from negpy.domain.interfaces import PipelineContext
from negpy.features.geometry.logic import compute_geometry_crop_rect, get_manual_rect_coords, keystone_matrix
from negpy.features.geometry.models import GeometryConfig
from negpy.features.geometry.processor import GeometryProcessor

_SENTINEL = -1000.0


def _void_mask(fine_rotation: float, converge_v: float, converge_h: float, w: int, h: int) -> np.ndarray:
    """Marks every output pixel that touches a border-replicated/void sample, by
    replaying the same two warps with an out-of-band fill instead of BORDER_REPLICATE."""
    img = np.ones((h, w), dtype=np.float32)
    center = (w / 2.0, h / 2.0)
    m_rot = cv2.getRotationMatrix2D(center, fine_rotation, 1.0)
    img = cv2.warpAffine(img, m_rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=_SENTINEL)
    m_key = keystone_matrix(converge_v, converge_h, w, h)
    img = cv2.warpPerspective(img, m_key, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=_SENTINEL)
    return img < (_SENTINEL / 2.0)


def test_zero_geometry_is_the_full_frame():
    assert compute_geometry_crop_rect(0.0, 0.0, 0.0, 300, 200) == (0.0, 0.0, 1.0, 1.0)


def test_pure_fine_rotation_is_symmetric():
    """A rotation about the center trims equally from opposite edges."""
    x1, y1, x2, y2 = compute_geometry_crop_rect(7.0, 0.0, 0.0, 300, 200)
    assert x1 == pytest.approx(1.0 - x2, abs=1e-3)
    assert y1 == pytest.approx(1.0 - y2, abs=1e-3)


@pytest.mark.parametrize(
    "fine_rotation,converge_v,converge_h",
    [
        (5.0, 0.0, 0.0),
        (-8.0, 0.0, 0.0),
        (0.0, 10.0, 0.0),
        (0.0, 0.0, -12.0),
        (3.0, 6.0, -4.0),
        (15.0, -10.0, 10.0),
        (-20.0, 12.0, 12.0),
    ],
)
def test_computed_rect_avoids_every_void_pixel(fine_rotation, converge_v, converge_h):
    w, h = 240, 160
    x1, y1, x2, y2 = compute_geometry_crop_rect(fine_rotation, converge_v, converge_h, w, h)
    void = _void_mask(fine_rotation, converge_v, converge_h, w, h)
    px1, px2 = int(np.ceil(x1 * w)), int(np.floor(x2 * w))
    py1, py2 = int(np.ceil(y1 * h)), int(np.floor(y2 * h))
    assert not void[py1:py2, px1:px2].any()


def test_rect_is_not_overly_conservative_for_a_small_angle():
    x1, y1, x2, y2 = compute_geometry_crop_rect(1.0, 0.0, 0.0, 1000, 1000)
    assert (x2 - x1) > 0.9
    assert (y2 - y1) > 0.9


def test_geometry_processor_crop_to_valid_computes_a_roi():
    img = np.zeros((200, 300, 3), dtype=np.float32)
    config = GeometryConfig(fine_rotation=5.0, crop_to_valid=True)
    context = PipelineContext(scale_factor=1.0, original_size=(200, 300))

    GeometryProcessor(config).process(img, context)

    expected_rect = compute_geometry_crop_rect(5.0, 0.0, 0.0, 300, 200)
    assert context.active_roi == get_manual_rect_coords((200, 300), expected_rect)
    assert context.active_roi != (0, 200, 0, 300)


def test_geometry_processor_crop_to_valid_off_leaves_the_full_frame():
    img = np.zeros((200, 300, 3), dtype=np.float32)
    config = GeometryConfig(fine_rotation=5.0, crop_to_valid=False)
    context = PipelineContext(scale_factor=1.0, original_size=(200, 300))

    GeometryProcessor(config).process(img, context)

    assert context.active_roi is None


def test_geometry_processor_crop_to_valid_yields_to_a_manual_rect():
    img = np.zeros((200, 300, 3), dtype=np.float32)
    config = GeometryConfig(fine_rotation=5.0, crop_to_valid=True, crop_rect=(0.2, 0.2, 0.8, 0.8))
    context = PipelineContext(scale_factor=1.0, original_size=(200, 300))

    GeometryProcessor(config).process(img, context)

    assert context.active_roi == get_manual_rect_coords((200, 300), (0.2, 0.2, 0.8, 0.8))


def test_geometry_processor_crop_to_valid_yields_to_an_armed_auto_crop():
    """An armed auto crop (crop_from_auto, no rect yet) is still waiting on
    ImageProcessor's detection pass; Crop by Default must not race it."""
    img = np.zeros((200, 300, 3), dtype=np.float32)
    config = GeometryConfig(fine_rotation=5.0, crop_to_valid=True, crop_from_auto=True)
    context = PipelineContext(scale_factor=1.0, original_size=(200, 300))

    GeometryProcessor(config).process(img, context)

    assert context.active_roi is None
