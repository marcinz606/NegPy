import numpy as np
from src.features.geometry.logic import get_manual_crop_coords, get_autocrop_coords
from src.features.geometry.processor import GeometryProcessor
from src.features.geometry.models import GeometryConfig
from src.domain.interfaces import PipelineContext


def test_get_manual_crop_coords_zero_offset():
    img = np.zeros((100, 200, 3), dtype=np.float32)
    roi = get_manual_crop_coords(img, offset_px=0)
    assert roi == (0, 100, 0, 200)


def test_get_manual_crop_coords_positive_offset():
    img = np.zeros((100, 200, 3), dtype=np.float32)
    roi = get_manual_crop_coords(img, offset_px=10)
    # 10 pixels from each side
    assert roi == (10, 90, 10, 190)


def test_get_manual_crop_coords_scale_factor():
    img = np.zeros((100, 200, 3), dtype=np.float32)
    roi = get_manual_crop_coords(img, offset_px=10, scale_factor=2.0)
    # 20 pixels from each side
    assert roi == (20, 80, 20, 180)


def test_get_manual_crop_coords_negative_offset():
    img = np.zeros((100, 200, 3), dtype=np.float32)
    # Negative offset should try to expand, but be clipped to image bounds if starting from (0, h, 0, w)
    roi = get_manual_crop_coords(img, offset_px=-10)
    assert roi == (0, 100, 0, 200)


def test_geometry_processor_manual_offset():
    img = np.zeros((100, 200, 3), dtype=np.float32)
    config = GeometryConfig(autocrop=False, autocrop_offset=10)
    processor = GeometryProcessor(config)
    context = PipelineContext(scale_factor=1.0, original_size=(100, 200))

    processor.process(img, context)

    assert context.active_roi == (10, 90, 10, 190)


def test_get_autocrop_coords_assisted():
    # Verify that providing an assist luma (film base) improves crop detection
    img = np.ones((100, 100, 3), dtype=np.float32) * 0.94
    img[20:80, 20:80] = 0.5

    roi_no_assist = get_autocrop_coords(img)
    roi_assist = get_autocrop_coords(img, assist_luma=0.94)
    assert roi_assist != roi_no_assist


def test_geometry_processor_no_autocrop_no_offset():
    img = np.zeros((100, 200, 3), dtype=np.float32)
    config = GeometryConfig(autocrop=False, autocrop_offset=0)
    processor = GeometryProcessor(config)
    context = PipelineContext(scale_factor=1.0, original_size=(100, 200))

    processor.process(img, context)

    assert context.active_roi == (0, 100, 0, 200)


def test_crop_consistency_across_resolutions():
    # Simulate a full res image and a preview image
    full_h, full_w = 3000, 4500
    prev_h, prev_w = 1000, 1500

    config = GeometryConfig(autocrop=False, autocrop_offset=10)
    processor = GeometryProcessor(config)

    ctx_full = PipelineContext(
        scale_factor=max(full_h, full_w) / float(max(full_h, full_w)),
        original_size=(full_h, full_w),
    )
    processor.process(np.zeros((full_h, full_w, 3)), ctx_full)

    ctx_prev = PipelineContext(
        scale_factor=max(prev_h, prev_w) / float(max(full_h, full_w)),
        original_size=(prev_h, prev_w),
    )
    processor.process(np.zeros((prev_h, prev_w, 3)), ctx_prev)

    y1_f, y2_f, x1_f, x2_f = ctx_full.active_roi
    y1_p, y2_p, x1_p, x2_p = ctx_prev.active_roi

    assert abs(y1_f / full_h - y1_p / prev_h) < 0.001
    assert abs(x1_f / full_w - x1_p / prev_w) < 0.001
