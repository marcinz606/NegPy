"""Auto Crop resolves once, upstream of both engines, and is then a stored rect.

Re-detecting per render runs the border walk on preview and full-resolution pixels, which
can land on different frames and export a crop the user never saw.
"""

from dataclasses import replace

import cv2
import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.geometry.logic import autocrop_detection_key, has_manual_crop, resolve_autocrop_rect
from negpy.features.geometry.models import AutocropMode, GeometryConfig
from negpy.features.geometry.processor import GeometryProcessor
from negpy.services.rendering.image_processor import _resolve_armed_autocrop


def _frame_image(h: int, w: int) -> np.ndarray:
    """A bright bed with a dark exposed frame inside it, at any resolution."""
    img = np.ones((h, w, 3), dtype=np.float32)
    img[round(0.12 * h) : round(0.88 * h), round(0.10 * w) : round(0.90 * w)] = 0.05
    return img


def _armed(**geometry) -> WorkspaceConfig:
    return replace(
        WorkspaceConfig(),
        geometry=GeometryConfig(crop_from_auto=True, autocrop_ratio="Free", **geometry),
    )


def _normalized_roi(img: np.ndarray, config: WorkspaceConfig, scale_factor: float) -> tuple:
    h, w = img.shape[:2]
    context = PipelineContext(scale_factor=scale_factor, original_size=(h, w))
    GeometryProcessor(config.geometry).process(img, context)
    y1, y2, x1, x2 = context.active_roi
    return (y1 / h, y2 / h, x1 / w, x2 / w)


def test_export_crops_exactly_where_the_preview_cropped():
    """Resolve on a preview-sized buffer, render full-res off the same edit, get the same
    frame. The two buffers detect differently in isolation; only one is ever asked to."""
    full = _frame_image(2400, 3600)
    preview = cv2.resize(full, (1600, 1067), interpolation=cv2.INTER_AREA)

    previewed, resolved = _resolve_armed_autocrop(preview, _armed())
    assert resolved is not None

    # Export re-enters with the settings the preview stored, and finds nothing to resolve.
    exported, again = _resolve_armed_autocrop(full, previewed)
    assert again is None
    assert exported.geometry.crop_rect == previewed.geometry.crop_rect

    preview_roi = _normalized_roi(preview, previewed, scale_factor=1.0)
    export_roi = _normalized_roi(full, exported, scale_factor=2400 / 1600)
    assert np.allclose(preview_roi, export_roi, atol=1e-3)


def test_resolving_is_idempotent():
    img = _frame_image(1200, 1800)
    once, first = _resolve_armed_autocrop(img, _armed())
    assert first is not None
    twice, second = _resolve_armed_autocrop(img, once)
    assert second is None
    assert twice is once


def test_resolved_rect_excludes_crop_offset():
    """Crop Offset is applied to the stored rect on every render, so baking it into the
    rect as well would count it twice."""
    img = _frame_image(1200, 1800)
    _, plain = _resolve_armed_autocrop(img, _armed())
    _, offset = _resolve_armed_autocrop(img, _armed(autocrop_offset=25))
    assert plain[0] == offset[0]


def test_offset_change_keeps_the_detected_rect():
    img = _frame_image(1200, 1800)
    resolved, _ = _resolve_armed_autocrop(img, _armed())
    moved = replace(resolved, geometry=replace(resolved.geometry, autocrop_offset=12))
    _, again = _resolve_armed_autocrop(img, moved)
    assert again is None


def test_detection_inputs_rearm_the_crop():
    img = _frame_image(1200, 1800)
    resolved, _ = _resolve_armed_autocrop(img, _armed())
    for field, value in (
        ("autocrop_ratio", "5:4"),
        ("autocrop_mode", AutocropMode.FILM),
        ("autocrop_rebate_trim", 0.5),
        ("rotation", 1),
        ("flip_horizontal", True),
        ("fine_rotation", 2.0),
    ):
        stale = replace(resolved, geometry=replace(resolved.geometry, **{field: value}))
        _, redetected = _resolve_armed_autocrop(img, stale)
        assert redetected is not None, f"{field} must re-arm the auto crop"


def test_manual_rect_is_never_resolved_over():
    img = _frame_image(1200, 1800)
    manual = replace(
        WorkspaceConfig(),
        geometry=GeometryConfig(crop_rect=(0.2, 0.2, 0.8, 0.8), crop_from_auto=False),
    )
    kept, resolved = _resolve_armed_autocrop(img, manual)
    assert resolved is None
    assert kept.geometry.crop_rect == (0.2, 0.2, 0.8, 0.8)
    assert has_manual_crop(kept.geometry)


def test_detection_key_ignores_offset_only():
    base = GeometryConfig(crop_from_auto=True)
    assert autocrop_detection_key(replace(base, autocrop_offset=40)) == autocrop_detection_key(base)
    assert autocrop_detection_key(replace(base, autocrop_ratio="5:4")) != autocrop_detection_key(base)


def test_resolve_returns_none_on_a_degenerate_buffer():
    assert resolve_autocrop_rect(np.zeros((1, 1, 3), dtype=np.float32), GeometryConfig(), 1600) is None
