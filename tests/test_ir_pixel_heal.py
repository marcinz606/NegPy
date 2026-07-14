"""Regression coverage for the IR-only pixel-mask dust repair path."""

from dataclasses import replace
import hashlib

import cv2
import numpy as np
import pytest

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.geometry.models import GeometryConfig
from negpy.features.geometry.processor import GeometryProcessor
from negpy.features.retouch.ir_heal import (
    IRSupportError,
    apply_ir_dust_removal,
    detect_ir_repair_support,
)
from negpy.features.retouch.models import RetouchConfig
from negpy.features.retouch.processor import RetouchProcessor
from negpy.services.rendering.image_processor import ImageProcessor


def _golden_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = 48, 64
    yy, xx = np.mgrid[:height, :width]
    rng = np.random.default_rng(20260713)
    clean = np.stack(
        [
            0.18 + 0.004 * xx + 0.001 * yy,
            0.24 + 0.002 * xx + 0.003 * yy,
            0.31 + 0.001 * xx + 0.002 * yy,
        ],
        axis=-1,
    ).astype(np.float32)
    clean += rng.normal(0, 0.006, clean.shape).astype(np.float32)
    ir = np.full((height, width), 0.86, dtype=np.float32)
    defect = ((xx - 29) ** 2 + (yy - 21) ** 2 <= 9) | ((xx >= 41) & (xx <= 43) & (yy >= 9) & (yy <= 32))
    ir[defect] = 0.07
    damaged = clean.copy()
    damaged[defect] = np.clip(damaged[defect] + 0.37, 0, 1)
    return damaged, ir, defect


def test_full_canvas_is_golden_equivalent_to_pre_v037() -> None:
    """Golden captured directly from commit 83357e1's public healer."""

    damaged, ir, defect = _golden_fixture()
    output, mask = apply_ir_dust_removal(damaged, ir, 0.5, 3, 1.25)

    assert int((mask > 0).sum()) == int(defect.sum()) == 101
    rounded_sha = hashlib.sha256(np.round(output, 6).tobytes()).hexdigest()
    assert rounded_sha == "2820655d6a606175a4a7cbe7831a6d996c500c623753050eedae98cd64ad7f70"


def test_opaque_scanner_rails_are_excluded_from_repair_support() -> None:
    ir = np.full((80, 120), 0.85, dtype=np.float32)
    ir[:, :10] = 0.05
    ir[:, 110:] = 0.05
    ir[30:33, 60:63] = 0.05

    support = detect_ir_repair_support(ir)

    assert support.bounds == (0, 80, 10, 110)
    assert support.detected_edges == ("left", "right")
    assert support.masked_pixels_total == 1609
    assert support.masked_pixels_inside == 9


def test_support_bounds_preserve_every_outside_pixel_exactly() -> None:
    yy, xx = np.mgrid[:80, :120]
    image = np.stack([xx / 119.0, yy / 79.0, (xx + yy) / 198.0], axis=-1).astype(np.float32)
    ir = np.full((80, 120), 0.85, dtype=np.float32)
    ir[:, :10] = 0.05
    ir[:, 110:] = 0.05
    ir[38:43, 58:63] = 0.05
    damaged = image.copy()
    damaged[38:43, 58:63] += 0.35
    support = detect_ir_repair_support(ir)

    output, mask = apply_ir_dust_removal(damaged, ir, 0.5, 3, 1.0, support_bounds=support.bounds)

    outside = np.ones(ir.shape, dtype=bool)
    y0, y1, x0, x1 = support.bounds
    outside[y0:y1, x0:x1] = False
    np.testing.assert_array_equal(output[outside], damaged[outside])
    assert not np.any(mask[outside])
    assert mask[40, 60] == 255


def test_arbitrary_support_mask_preserves_every_outside_pixel_exactly() -> None:
    damaged, ir, _ = _golden_fixture()
    support = np.zeros(ir.shape, dtype=bool)
    support[5:40, 8:56] = True
    output, mask = apply_ir_dust_removal(damaged, ir, 0.5, 3, 1.0, support_mask=support)

    np.testing.assert_array_equal(output[~support], damaged[~support])
    assert not np.any(mask[~support])


def test_curved_hair_is_repaired_as_pixels_not_collapsed_to_one_blob() -> None:
    height, width = 180, 240
    yy, xx = np.mgrid[:height, :width]
    clean = np.stack(
        [
            0.28 + 0.0010 * xx + 0.0004 * yy,
            0.34 + 0.0006 * xx + 0.0007 * yy,
            0.40 + 0.0003 * xx + 0.0005 * yy,
        ],
        axis=-1,
    ).astype(np.float32)
    hair = np.zeros((height, width), dtype=np.uint8)
    points = np.array([[35, 135], [55, 95], [90, 75], [130, 82], [155, 115], [145, 145], [112, 155]], dtype=np.int32)
    cv2.polylines(hair, [points], isClosed=False, color=1, thickness=3, lineType=cv2.LINE_8)
    defect = hair.astype(bool)
    damaged = clean.copy()
    damaged[defect] = np.clip(damaged[defect] + 0.42, 0, 1)
    ir = np.full((height, width), 0.88, dtype=np.float32)
    ir[defect] = 0.04

    output, _ = apply_ir_dust_removal(damaged, ir, 0.5, 3, 1.0)

    before_mae = float(np.abs(damaged[defect] - clean[defect]).mean())
    after_mae = float(np.abs(output[defect] - clean[defect]).mean())
    changed_fraction = float(np.any(output[defect] != damaged[defect], axis=1).mean())
    assert changed_fraction > 0.99
    assert after_mae < before_mae * 0.35


def test_globally_ir_opaque_material_fails_closed() -> None:
    with pytest.raises(IRSupportError, match="IR-opaque film"):
        detect_ir_repair_support(np.full((64, 64), 0.1, dtype=np.float32))


def test_geometry_transforms_ir_on_the_same_canvas_as_rgb() -> None:
    image = np.zeros((20, 30, 3), dtype=np.float32)
    image[5, 10] = 1.0
    ir = np.zeros((20, 30), dtype=np.float32)
    ir[5, 10] = 1.0
    context = PipelineContext(original_size=(20, 30), scale_factor=1.0, ir_buffer=ir)

    output = GeometryProcessor(GeometryConfig(rotation=1, flip_horizontal=True)).process(image, context)
    ir_output = context.metrics["ir_post_geometry"]

    assert ir_output.shape == output.shape[:2]
    assert np.unravel_index(np.argmax(ir_output), ir_output.shape) == np.unravel_index(np.argmax(output[..., 0]), output.shape[:2])


def test_retouch_processor_keeps_manual_membrane_path_after_ir() -> None:
    image = np.full((80, 100, 3), 0.4, dtype=np.float32)
    image[40, 50] = 0.95
    ir = np.full((80, 100), 0.85, dtype=np.float32)
    ir[39:42, 49:52] = 0.05
    config = RetouchConfig(
        ir_dust_remove=True,
        manual_heal_strokes=[([[0.25, 0.25]], 8.0, 0.1, 0.0)],
    )
    context = PipelineContext(original_size=(80, 100), scale_factor=1.0, ir_buffer=ir)
    GeometryProcessor(GeometryConfig()).process(image, context)

    output = RetouchProcessor(config).process(image, context)

    assert output[40, 50, 0] < 0.9
    assert "ir_dust_support" in context.metrics
    assert context.metrics["ir_dust_mask_pixels"] == 9
    assert context.metrics["ir_dust_status"] == "applied"


def test_requested_ir_repair_reports_when_no_ir_plane_is_available() -> None:
    image = np.full((40, 50, 3), 0.4, dtype=np.float32)
    context = PipelineContext(
        original_size=(40, 50),
        scale_factor=1.0,
        ir_buffer=None,
    )

    output = RetouchProcessor(
        RetouchConfig(ir_dust_remove=True)
    ).process(image, context)

    np.testing.assert_array_equal(output, image)
    assert context.metrics["ir_dust_status"] == "skipped"
    assert context.metrics["ir_dust_skipped"] == "No IR channel is available for this scan"


def test_ir_only_config_stays_explicit_instead_of_becoming_membrane_strokes() -> None:
    service = ImageProcessor()
    image = np.full((80, 100, 3), 0.4, dtype=np.float32)
    ir = np.full((80, 100), 0.85, dtype=np.float32)
    ir[40, 50] = 0.05
    config = replace(WorkspaceConfig(), retouch=RetouchConfig(ir_dust_remove=True))

    augmented = service._augment_retouch(config, image, ir, "source")

    assert augmented.retouch.ir_dust_remove is True
    assert augmented.retouch.manual_heal_strokes == []


def test_direct_ir_repair_forces_cpu_backend() -> None:
    class _GPU:
        def process_to_texture(self, *args, **kwargs):
            raise AssertionError("direct IR repair must not enter the GPU membrane path")

    class _CPU:
        def __init__(self):
            self.called = False

        def process(self, image, settings, source_hash, context):
            self.called = True
            return image

    service = ImageProcessor()
    cpu = _CPU()
    setattr(service, "engine_cpu", cpu)
    setattr(service, "engine_gpu", _GPU())
    image = np.full((32, 40, 3), 0.4, dtype=np.float32)
    ir = np.full((32, 40), 0.85, dtype=np.float32)
    config = replace(WorkspaceConfig(), retouch=RetouchConfig(ir_dust_remove=True))

    service.run_pipeline(image, config, "source", 1600.0, prefer_gpu=True, ir_buffer=ir, wants_uv_grid=False)

    assert cpu.called
