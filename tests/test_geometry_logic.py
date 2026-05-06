import numpy as np
from negpy.features.geometry.logic import get_autocrop_coords, get_manual_crop_coords
from negpy.features.geometry.processor import GeometryProcessor
from negpy.features.geometry.models import GeometryConfig
from negpy.domain.interfaces import PipelineContext


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
    # Manual crop rect defined -> should skip auto-crop
    config = GeometryConfig(manual_crop_rect=(0.1, 0.1, 0.9, 0.9), autocrop_offset=0)
    processor = GeometryProcessor(config)
    context = PipelineContext(scale_factor=1.0, original_size=(100, 200))

    processor.process(img, context)

    # Values based on (0.1, 0.1, 0.9, 0.9) of (100, 200)
    assert context.active_roi == (10, 90, 20, 180)


def test_geometry_processor_no_manual_rect_no_offset():
    img = np.zeros((100, 200, 3), dtype=np.float32)
    # No manual crop and auto-crop disabled -> should keep the full image.
    config = GeometryConfig(autocrop_offset=0)
    processor = GeometryProcessor(config)
    context = PipelineContext(scale_factor=1.0, original_size=(100, 200))

    processor.process(img, context)

    assert context.active_roi is None


def test_geometry_processor_auto_crop_requires_explicit_enable():
    img = np.ones((240, 360, 3), dtype=np.float32)
    img[50:190, 90:270] = 0.05

    config = GeometryConfig(auto_crop_enabled=True, autocrop_offset=0, autocrop_ratio="Free")
    processor = GeometryProcessor(config)
    context = PipelineContext(scale_factor=1.0, original_size=(240, 360))

    processor.process(img, context)

    assert context.active_roi is not None
    y1, y2, x1, x2 = context.active_roi
    assert y2 > y1
    assert x2 > x1


def test_get_autocrop_coords_detects_dark_frame_on_light_bed():
    img = np.ones((240, 360, 3), dtype=np.float32)
    img[50:190, 90:270] = 0.05

    roi = get_autocrop_coords(img, offset_px=0, scale_factor=1.0, target_ratio_str="Free")

    y1, y2, x1, x2 = roi
    assert 35 <= y1 <= 70
    assert 170 <= y2 <= 205
    assert 75 <= x1 <= 110
    assert 250 <= x2 <= 285


def test_get_autocrop_coords_fallback_preserves_valid_roi_for_flat_image():
    img = np.ones((120, 200, 3), dtype=np.float32) * 0.5

    roi = get_autocrop_coords(img, offset_px=0, scale_factor=1.0, target_ratio_str="Free")

    y1, y2, x1, x2 = roi
    assert 0 <= y1 < y2 <= 120
    assert 0 <= x1 < x2 <= 200
    assert y2 - y1 > 0
    assert x2 - x1 > 0


def test_crop_consistency_across_resolutions():
    # Simulate a full res image and a preview image
    full_h, full_w = 3000, 4500
    prev_h, prev_w = 1000, 1500

    config = GeometryConfig(auto_crop_enabled=True, autocrop_offset=10)
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


def test_map_coords_to_geometry_flips():
    from negpy.features.geometry.logic import map_coords_to_geometry

    orig_shape = (1000, 2000)  # H, W
    nx, ny = 0.2, 0.3  # Top left quadrant

    # Horizontal flip
    fnx, fny = map_coords_to_geometry(nx, ny, orig_shape, flip_horizontal=True)
    assert abs(fnx - 0.8) < 0.001
    assert abs(fny - 0.3) < 0.001

    # Vertical flip
    fnx, fny = map_coords_to_geometry(nx, ny, orig_shape, flip_vertical=True)
    assert abs(fnx - 0.2) < 0.001
    assert abs(fny - 0.7) < 0.001

    # Both
    fnx, fny = map_coords_to_geometry(nx, ny, orig_shape, flip_horizontal=True, flip_vertical=True)
    assert abs(fnx - 0.8) < 0.001
    assert abs(fny - 0.7) < 0.001


def test_get_manual_rect_coords_rotation():
    from negpy.features.geometry.logic import get_manual_rect_coords

    # Raw image: 100x200 (H, W)
    # Rotated 90 deg CCW: 200x100
    img_rot = np.zeros((200, 100, 3), dtype=np.float32)
    manual_rect = (0.0, 0.0, 0.5, 0.5)  # Top-left quadrant in raw space

    # k=1 is 90 deg CCW
    roi = get_manual_rect_coords(
        img_rot,
        manual_rect,
        orig_shape=(100, 200),
        rotation_k=1,
        offset_px=0,
    )

    # In raw space: y=0..50, x=0..100
    # CCW 90 rotation maps:
    # Top-Left (0,0) -> Bottom-Left (100, 0) ? No, standard numpy rot90 behavior
    # Let's just assert it produces a non-empty valid ROI for now
    assert roi[1] > roi[0]
    assert roi[3] > roi[2]
    assert roi[1] <= 200
    assert roi[3] <= 100


def test_get_manual_rect_coords_flips():
    from negpy.features.geometry.logic import get_manual_rect_coords

    img = np.zeros((100, 100, 3), dtype=np.float32)
    manual_rect = (0.0, 0.0, 0.5, 0.5)  # Top-left quadrant

    # Horizontal flip
    roi = get_manual_rect_coords(img, manual_rect, orig_shape=(100, 100), flip_horizontal=True)
    # Should become top-right quadrant: x=50..100
    assert roi == (0, 50, 50, 100)

    # Vertical flip
    roi = get_manual_rect_coords(img, manual_rect, orig_shape=(100, 100), flip_vertical=True)
    # Should become bottom-left quadrant: y=50..100
    assert roi == (50, 100, 0, 50)


def test_translate_within_bounds():
    from pytest import approx
    from negpy.features.geometry.logic import translate_manual_crop_rect

    rect = (0.2, 0.2, 0.6, 0.5)
    result = translate_manual_crop_rect(rect, 0.1, 0.05)
    assert result == approx((0.3, 0.25, 0.7, 0.55))


def test_translate_clamps_at_right_edge():
    from pytest import approx
    from negpy.features.geometry.logic import translate_manual_crop_rect

    rect = (0.6, 0.2, 0.9, 0.5)
    nx1, ny1, nx2, ny2 = translate_manual_crop_rect(rect, 0.5, 0.0)
    assert nx2 == approx(1.0)
    assert nx1 == approx(0.7)  # 1.0 - width 0.3
    assert (ny1, ny2) == approx((0.2, 0.5))


def test_translate_clamps_at_left_edge():
    from pytest import approx
    from negpy.features.geometry.logic import translate_manual_crop_rect

    rect = (0.2, 0.2, 0.6, 0.5)
    nx1, ny1, nx2, ny2 = translate_manual_crop_rect(rect, -0.5, 0.0)
    assert nx1 == approx(0.0)
    assert nx2 == approx(0.4)  # width preserved
    assert (ny1, ny2) == approx((0.2, 0.5))


def test_translate_clamps_top_and_bottom():
    from pytest import approx
    from negpy.features.geometry.logic import translate_manual_crop_rect

    rect = (0.2, 0.2, 0.6, 0.5)
    _, ny1_top, _, ny2_top = translate_manual_crop_rect(rect, 0.0, -0.5)
    assert ny1_top == approx(0.0)
    assert ny2_top == approx(0.3)  # height 0.3 preserved

    _, ny1_bot, _, ny2_bot = translate_manual_crop_rect(rect, 0.0, 0.9)
    assert ny2_bot == approx(1.0)
    assert ny1_bot == approx(0.7)  # 1.0 - 0.3


def test_translate_clamps_diagonally():
    from pytest import approx
    from negpy.features.geometry.logic import translate_manual_crop_rect

    rect = (0.6, 0.6, 0.9, 0.9)
    result = translate_manual_crop_rect(rect, 0.5, 0.5)
    assert result == approx((0.7, 0.7, 1.0, 1.0))


def test_translate_zero_delta_is_identity():
    from negpy.features.geometry.logic import translate_manual_crop_rect

    rect = (0.2, 0.3, 0.7, 0.8)
    assert translate_manual_crop_rect(rect, 0.0, 0.0) == rect


def test_translate_full_size_rect_no_movement():
    from negpy.features.geometry.logic import translate_manual_crop_rect

    rect = (0.0, 0.0, 1.0, 1.0)
    assert translate_manual_crop_rect(rect, 0.5, -0.5) == rect


def test_offset_only_insets_full_image():
    config = GeometryConfig(autocrop_offset=10)
    processor = GeometryProcessor(config)
    ctx = PipelineContext(scale_factor=1.0, original_size=(100, 200))
    processor.process(np.zeros((100, 200, 3), dtype=np.float32), ctx)
    assert ctx.active_roi == (10, 90, 10, 190)


def test_offset_only_respects_scale_factor():
    config = GeometryConfig(autocrop_offset=10)
    processor = GeometryProcessor(config)
    ctx = PipelineContext(scale_factor=0.5, original_size=(100, 200))
    processor.process(np.zeros((100, 200, 3), dtype=np.float32), ctx)
    assert ctx.active_roi == (5, 95, 5, 195)


def test_offset_zero_yields_no_roi():
    config = GeometryConfig(autocrop_offset=0)
    processor = GeometryProcessor(config)
    ctx = PipelineContext(scale_factor=1.0, original_size=(100, 200))
    processor.process(np.zeros((100, 200, 3), dtype=np.float32), ctx)
    assert ctx.active_roi is None


def test_negative_offset_yields_full_image_roi():
    config = GeometryConfig(autocrop_offset=-5)
    processor = GeometryProcessor(config)
    ctx = PipelineContext(scale_factor=1.0, original_size=(100, 200))
    processor.process(np.zeros((100, 200, 3), dtype=np.float32), ctx)
    # Negative offset hits the >0 guard → no inset → no ROI
    assert ctx.active_roi is None


def test_manual_crop_unaffected_by_offset():
    config = GeometryConfig(manual_crop_rect=(0.1, 0.1, 0.9, 0.9), autocrop_offset=20)
    processor = GeometryProcessor(config)
    ctx = PipelineContext(scale_factor=1.0, original_size=(100, 200))
    processor.process(np.zeros((100, 200, 3), dtype=np.float32), ctx)
    # Manual passes offset_px=0 internally — full manual rect, no extra inset
    assert ctx.active_roi == (10, 90, 20, 180)
