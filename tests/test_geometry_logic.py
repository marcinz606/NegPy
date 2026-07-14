import cv2
import numpy as np
import pytest
from negpy.features.geometry.logic import detect_closest_aspect_ratio, get_autocrop_coords, get_manual_crop_coords, get_manual_rect_coords
from negpy.features.geometry.processor import CropProcessor, GeometryProcessor
from negpy.features.geometry.models import GeometryConfig
from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import AspectRatio


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


def test_trim_opaque_border_removes_holder_bands():
    from negpy.features.geometry.logic import _trim_opaque_border

    lum = np.full((100, 120), 0.4, dtype=np.float32)
    lum[90:, :] = 0.0  # opaque holder band at the bottom (10px)
    lum[:, :6] = 0.0  # opaque sliver on the left (6px)

    y1, y2, x1, x2 = _trim_opaque_border(lum, (0, 100, 0, 120))
    assert (y1, y2, x1, x2) == (0, 90, 6, 120)


def test_trim_opaque_border_noop_without_black_band():
    from negpy.features.geometry.logic import _trim_opaque_border

    lum = np.full((100, 120), 0.4, dtype=np.float32)
    roi = (10, 90, 5, 115)
    assert _trim_opaque_border(lum, roi) == roi


def test_trim_opaque_border_keeps_dark_negative_content():
    # Dark negative content still transmits film base (lum ~ 0.08) — well above the
    # opaque threshold, so it must NOT be trimmed as if it were a holder band.
    from negpy.features.geometry.logic import _trim_opaque_border

    lum = np.full((100, 120), 0.4, dtype=np.float32)
    lum[80:, :] = 0.08
    assert _trim_opaque_border(lum, (0, 100, 0, 120)) == (0, 100, 0, 120)


def test_trim_opaque_border_capped_for_all_black_roi():
    from negpy.features.geometry.logic import _trim_opaque_border

    lum = np.zeros((100, 100), dtype=np.float32)
    y1, y2, x1, x2 = _trim_opaque_border(lum, (0, 100, 0, 100))
    assert y1 <= 20 and (100 - y2) <= 20 and x1 <= 20 and (100 - x2) <= 20
    assert y2 > y1 and x2 > x1


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


def test_get_manual_rect_coords_transformed_dims():
    from negpy.features.geometry.logic import get_manual_rect_coords

    # The rect is normalized in the already-transformed image, so it scales directly by
    # the passed (transformed) dims — no rotation/flip re-mapping, no bounding box.
    img = np.zeros((200, 100, 3), dtype=np.float32)  # transformed H=200, W=100
    roi = get_manual_rect_coords(img, (0.0, 0.0, 0.5, 0.5), offset_px=0)
    assert roi == (0, 100, 0, 50)


def test_get_manual_rect_coords_offset_inset():
    from negpy.features.geometry.logic import get_manual_rect_coords

    img = np.zeros((100, 100, 3), dtype=np.float32)
    roi = get_manual_rect_coords(img, (0.2, 0.2, 0.8, 0.8), offset_px=5)
    # 0.2..0.8 of 100 = 20..80, inset 5px each side => 25..75
    assert roi == (25, 75, 25, 75)


def test_manual_crop_no_inflation_under_fine_rotation():
    # Regression (#task6): the manual crop rect lives in transformed (display) space, so
    # fine rotation — which preserves the image dims — must resolve to the identical ROI
    # instead of inflating it via a corner-mapped bounding box.
    rect = (0.2, 0.2, 0.8, 0.8)
    img = np.zeros((400, 600, 3), dtype=np.float32)

    ctx_base = PipelineContext(scale_factor=1.0, original_size=(400, 600))
    GeometryProcessor(GeometryConfig(manual_crop_rect=rect, autocrop_offset=0)).process(img, ctx_base)

    ctx_rot = PipelineContext(scale_factor=1.0, original_size=(400, 600))
    GeometryProcessor(GeometryConfig(manual_crop_rect=rect, autocrop_offset=0, fine_rotation=4.0)).process(img, ctx_rot)

    assert ctx_base.active_roi == ctx_rot.active_roi
    # Exactly the fractional slice of the transformed frame, un-inflated.
    assert ctx_base.active_roi == (80, 320, 120, 480)


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


def test_straighten_horizontal_right_end_down_rotates_ccw():
    from pytest import approx
    from negpy.features.geometry.logic import straighten_delta_degrees

    # Horizon drawn with the right end lower: lift it by rotating CCW (stored +).
    assert straighten_delta_degrees(100.0, 5.0) == approx(2.8624, abs=1e-3)
    # Same physical line drawn from the other end gives the same correction.
    assert straighten_delta_degrees(-100.0, -5.0) == approx(2.8624, abs=1e-3)


def test_straighten_horizontal_right_end_up_rotates_cw():
    from pytest import approx
    from negpy.features.geometry.logic import straighten_delta_degrees

    assert straighten_delta_degrees(100.0, -5.0) == approx(-2.8624, abs=1e-3)


def test_straighten_vertical_intent_snaps_to_plumb():
    from pytest import approx
    from negpy.features.geometry.logic import straighten_delta_degrees

    # Building edge drawn bottom-to-top with the top leaning right (CW tilt):
    # correct by rotating CCW (stored +).
    assert straighten_delta_degrees(10.0, -100.0) == approx(5.7106, abs=1e-3)
    # Top leaning left corrects CW (stored -).
    assert straighten_delta_degrees(-10.0, -100.0) == approx(-5.7106, abs=1e-3)


def test_straighten_exact_axes_need_no_correction():
    from negpy.features.geometry.logic import straighten_delta_degrees

    assert straighten_delta_degrees(50.0, 0.0) == 0.0
    assert straighten_delta_degrees(0.0, 50.0) == 0.0


def test_straighten_result_bounded_to_quarter_turn():
    from negpy.features.geometry.logic import straighten_delta_degrees

    for dx, dy in ((1, 1), (-1, 1), (1, -3), (7, 2), (-5, -9)):
        assert -45.0 <= straighten_delta_degrees(float(dx), float(dy)) <= 45.0


def test_rotate_rect_ccw_quarter_turn():
    from pytest import approx
    from negpy.features.geometry.logic import rotate_normalized_rect

    # A rect in the top-left maps to the bottom-left after one CCW turn: content at
    # (u, v) moves to (v, 1 - u), so (0.1,0.1)->(0.1,0.9) and (0.4,0.3)->(0.3,0.6).
    rect = (0.1, 0.1, 0.4, 0.3)
    assert rotate_normalized_rect(rect, 1) == approx((0.1, 0.6, 0.3, 0.9))


def test_rotate_rect_cw_is_inverse_of_ccw():
    from pytest import approx
    from negpy.features.geometry.logic import rotate_normalized_rect

    rect = (0.15, 0.2, 0.6, 0.55)
    ccw = rotate_normalized_rect(rect, 1)
    assert rotate_normalized_rect(ccw, -1) == approx(rect)


def test_rotate_rect_180_flips_both_axes():
    from pytest import approx
    from negpy.features.geometry.logic import rotate_normalized_rect

    rect = (0.1, 0.2, 0.4, 0.5)
    assert rotate_normalized_rect(rect, 2) == approx((0.6, 0.5, 0.9, 0.8))


def test_rotate_rect_four_turns_is_identity():
    from pytest import approx
    from negpy.features.geometry.logic import rotate_normalized_rect

    rect = (0.15, 0.2, 0.6, 0.55)
    assert rotate_normalized_rect(rect, 4) == approx(rect)


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


def test_manual_crop_applies_offset():
    config = GeometryConfig(manual_crop_rect=(0.1, 0.1, 0.9, 0.9), autocrop_offset=20)
    processor = GeometryProcessor(config)
    ctx = PipelineContext(scale_factor=1.0, original_size=(100, 200))
    processor.process(np.zeros((100, 200, 3), dtype=np.float32), ctx)
    # Manual crop rect inset by autocrop_offset (20px at scale_factor=1.0)
    assert ctx.active_roi == (30, 70, 40, 160)


def test_manual_rect_coords_fractional_inset_scale_invariant():
    # Verify get_manual_rect_coords is scale-invariant:
    # calling at preview dims (sf=1.0) then upscaling == calling at full-res dims (sf=3.75).
    # This pins the invariant violated by the pre-fix GPU double-scale bug.
    PREV_H, PREV_W = 1066, 1600
    FULL_H, FULL_W = 4000, 6000
    scale_factor = FULL_W / PREV_W  # 3.75
    manual_rect = (0.1, 0.1, 0.9, 0.9)
    offset_px = 30

    roi_prev = get_manual_rect_coords(
        (PREV_H, PREV_W),
        manual_rect,
        offset_px=offset_px,
        scale_factor=1.0,
    )
    sy, sx = FULL_H / PREV_H, FULL_W / PREV_W
    roi_prev_scaled = (
        int(roi_prev[0] * sy),
        int(roi_prev[1] * sy),
        int(roi_prev[2] * sx),
        int(roi_prev[3] * sx),
    )

    roi_full = get_manual_rect_coords(
        (FULL_H, FULL_W),
        manual_rect,
        offset_px=offset_px,
        scale_factor=scale_factor,
    )

    # Integer truncation at both stages causes ≤2px divergence; bug would cause ~250px error
    for a, b in zip(roi_prev_scaled, roi_full):
        assert abs(a - b) <= 2, f"scale invariant broken: {roi_prev_scaled} vs {roi_full}"


def test_autocrop_margin_scale_invariant():
    # Verify the margin formula inside get_autocrop_coords is scale-invariant.
    # The detection step is not scale-invariant (fixed-size kernels), so we test
    # apply_margin_to_roi directly with a known pre-margin ROI at both scales.
    # This pins the arithmetic the GPU engine relies on after the double-scale fix.
    from negpy.features.geometry.logic import apply_margin_to_roi

    PREV_H, PREV_W = 1066, 1600
    FULL_H, FULL_W = 4000, 6000
    scale_factor = FULL_W / PREV_W  # 3.75
    offset_px = 20

    # Known pre-margin ROI (10%-90% of image) — represents a proportional detection result
    roi_prev_init = (int(0.1 * PREV_H), int(0.9 * PREV_H), int(0.1 * PREV_W), int(0.9 * PREV_W))
    roi_full_init = (int(0.1 * FULL_H), int(0.9 * FULL_H), int(0.1 * FULL_W), int(0.9 * FULL_W))

    # get_autocrop_coords: margin = (2 + offset_px) * scale_factor
    roi_prev = apply_margin_to_roi(roi_prev_init, PREV_H, PREV_W, (2 + offset_px) * 1.0)
    roi_full = apply_margin_to_roi(roi_full_init, FULL_H, FULL_W, (2 + offset_px) * scale_factor)

    sy, sx = FULL_H / PREV_H, FULL_W / PREV_W
    roi_prev_scaled = (
        int(roi_prev[0] * sy),
        int(roi_prev[1] * sy),
        int(roi_prev[2] * sx),
        int(roi_prev[3] * sx),
    )

    # Integer truncation causes ≤2px divergence; pre-fix bug caused ~(sf²-sf)*offset ≈ 200px error
    for a, b in zip(roi_prev_scaled, roi_full):
        assert abs(a - b) <= 2, f"margin scale invariant broken: {roi_prev_scaled} vs {roi_full}"


# ── detect_closest_aspect_ratio ──────────────────────────────────────────


def test_detect_closest_aspect_ratio_landscape_3_2():
    # 240x360 image with dark frame inset; contour detection pads ~13px in width.
    # 140h × 210w dark area yields detected ratio ~1.63 → snaps to "3:2".
    img = np.ones((240, 360, 3), dtype=np.float32)
    img[50:190, 75:285] = 0.05
    ratio = detect_closest_aspect_ratio(img)
    assert ratio == "3:2"


def test_detect_closest_aspect_ratio_landscape_4_3():
    # 300x400 image; 200h × 266w dark inset is 1.33 → snaps to "4:3".
    # (Rect was 250w=1.25 before; the old inflated contour bounds happened to read
    # it as ~1.34. Detection is accurate now, so the fixture must actually be 4:3.)
    img = np.ones((300, 400, 3), dtype=np.float32)
    img[50:250, 70:336] = 0.05
    ratio = detect_closest_aspect_ratio(img)
    assert ratio == "4:3"


def test_detect_closest_aspect_ratio_portrait_2_3():
    # Portrait 2:3 frame: 360x240 image, dark area 60:300 (h=240) x 40:200 (w=160) -> 2:3.
    # (Rect centered with 40px margins; the previous 20px right margin was inside the
    # fixed-size morphology kernels' bleed range, inflating the detected box.)
    img = np.ones((360, 240, 3), dtype=np.float32)
    img[60:300, 40:200] = 0.05
    ratio = detect_closest_aspect_ratio(img)
    assert ratio == "2:3"


def test_detect_closest_aspect_ratio_square():
    img = np.ones((300, 300, 3), dtype=np.float32)
    img[30:270, 30:270] = 0.05  # Large dark square, contour detection yields ~1.06 → snaps to "1:1"
    ratio = detect_closest_aspect_ratio(img)
    assert ratio == "1:1"


def test_detect_closest_aspect_ratio_fallback_on_flat_image():
    img = np.ones((120, 200, 3), dtype=np.float32) * 0.5
    ratio = detect_closest_aspect_ratio(img, fallback="3:2")
    # Flat image: detection may produce degenerate or threshold-based ROI.
    # Result must be a valid AspectRatio enum value (not Free or Original).
    assert ratio in {r.value for r in AspectRatio if r not in (AspectRatio.FREE, AspectRatio.ORIGINAL)}


def test_detect_closest_aspect_ratio_excludes_free_and_original():
    # Awkward ratio: function must not return "Free" or "Original".
    img = np.ones((300, 900, 3), dtype=np.float32)
    img[60:240, 100:800] = 0.05  # ~3.89:1, panoramic
    ratio = detect_closest_aspect_ratio(img)
    assert ratio not in ("Free", "Original")


def test_detect_closest_aspect_ratio_orientation_landscape_picks_landscape_set():
    img = np.ones((240, 360, 3), dtype=np.float32)
    img[60:180, 80:280] = 0.05  # landscape ~5:3
    ratio = detect_closest_aspect_ratio(img)
    w_r, h_r = map(float, ratio.split(":"))
    assert w_r >= h_r  # landscape or 1:1


def test_detect_closest_aspect_ratio_orientation_portrait_picks_portrait_set():
    img = np.ones((360, 240, 3), dtype=np.float32)
    img[80:280, 60:180] = 0.05  # portrait ~3:5
    ratio = detect_closest_aspect_ratio(img)
    w_r, h_r = map(float, ratio.split(":"))
    assert w_r <= h_r  # portrait or 1:1


def test_detect_closest_aspect_ratio_image_dims_sanity_check():
    # 3:2 image (360x240) with a wide dark inset (~2.7:1) that would normally snap to
    # 65:24. The image-dims sanity check should override and return 3:2 instead.
    img = np.ones((240, 360, 3), dtype=np.float32)
    img[90:150, 20:340] = 0.05  # 60h × 320w ≈ 5.3:1 dark band
    ratio = detect_closest_aspect_ratio(img)
    assert ratio == "3:2"


def _normalized_roi(roi, h, w):
    y1, y2, x1, x2 = roi
    return (y1 / h, y2 / h, x1 / w, x2 / w)


@pytest.mark.parametrize("rotation_k", [0, 1, 2, 3])
@pytest.mark.parametrize("flip_h", [False, True])
def test_manual_crop_roi_consistent_preview_vs_export(rotation_k, flip_h):
    # The export path reuses GeometryProcessor at full-res scale_factor while the
    # preview runs it downsampled. A manual crop (+ rotation/flip) must resolve to the
    # same fractional region in both, or export won't match the preview (#218).
    full_h, full_w = 3000, 4500
    prev_h, prev_w = 1000, 1500

    config = GeometryConfig(manual_crop_rect=(0.15, 0.2, 0.7, 0.85), rotation=rotation_k, flip_horizontal=flip_h)
    proc = GeometryProcessor(config)

    ctx_full = PipelineContext(scale_factor=1.0, original_size=(full_h, full_w))
    img_full = proc.process(np.zeros((full_h, full_w, 3), dtype=np.float32), ctx_full)

    ctx_prev = PipelineContext(scale_factor=max(prev_h, prev_w) / float(max(full_h, full_w)), original_size=(prev_h, prev_w))
    img_prev = proc.process(np.zeros((prev_h, prev_w, 3), dtype=np.float32), ctx_prev)

    # active_roi lives in post-rotation pixel space; normalize by the rotated frame dims.
    norm_full = _normalized_roi(ctx_full.active_roi, *img_full.shape[:2])
    norm_prev = _normalized_roi(ctx_prev.active_roi, *img_prev.shape[:2])
    for a, b in zip(norm_full, norm_prev):
        assert abs(a - b) < 0.005


@pytest.mark.parametrize("rotation_k", [0, 1, 2, 3])
def test_manual_crop_extracts_same_marker_at_preview_and_export(rotation_k):
    # End-to-end content parity: a marker filling the crop rect must survive the crop
    # identically at full-res (export) and downsampled (preview) resolution. The crop
    # rect is now in transformed (display) space, so a centered marker + centered rect
    # stay aligned under any 90° rotation.
    full_h, full_w = 600, 900
    full = np.zeros((full_h, full_w, 3), dtype=np.float32)
    full[210:390, 315:585] = 1.0  # centered block, normalized (0.35..0.65) in both axes
    prev = cv2.resize(full, (300, 200), interpolation=cv2.INTER_AREA)

    config = GeometryConfig(manual_crop_rect=(0.35, 0.35, 0.65, 0.65), rotation=rotation_k)
    proc = GeometryProcessor(config)
    cropper = CropProcessor(config)

    ctx_full = PipelineContext(scale_factor=1.0, original_size=(full_h, full_w))
    out_full = cropper.process(proc.process(full, ctx_full), ctx_full)

    ctx_prev = PipelineContext(scale_factor=full_w / 300.0, original_size=(200, 300))
    out_prev = cropper.process(proc.process(prev, ctx_prev), ctx_prev)

    # Crop region equals the marker → both crops are (near) fully white regardless of rotation.
    assert out_full.size > 0 and out_prev.size > 0
    assert out_full.mean() > 0.95
    assert out_prev.mean() > 0.9


def _three_tier_negative() -> np.ndarray:
    # Synthetic raw negative: light bed (1.0) >> film base/rebate (0.78) > exposed frame (0.25).
    img = np.full((480, 720, 3), 1.0, dtype=np.float32)
    img[80:400, 100:620] = 0.78  # film strip incl. rebate
    img[105:375, 135:585] = 0.25  # exposed frame, 270h x 450w
    return img


def _three_tier_negative_at(long_edge: int) -> np.ndarray:
    # Same scene as _three_tier_negative, fraction-defined, at arbitrary resolution.
    h, w = round(long_edge * 480 / 720), long_edge
    img = np.full((h, w, 3), 1.0, dtype=np.float32)
    img[round(h * 80 / 480) : round(h * 400 / 480), round(w * 100 / 720) : round(w * 620 / 720)] = 0.78
    img[round(h * 105 / 480) : round(h * 375 / 480), round(w * 135 / 720) : round(w * 585 / 720)] = 0.25
    return img


def test_autocrop_film_mode_keeps_rebate():
    img = _three_tier_negative()

    roi = get_autocrop_coords(img, offset_px=0, scale_factor=1.0, target_ratio_str="Free", mode="film")

    y1, y2, x1, x2 = roi
    # ROI reaches into the rebate (outside the exposed frame at 105:375, 135:585),
    # near film bounds 80:400, 100:620 (Free snaps the 1.6:1 film box to 3:2).
    assert y1 < 105 and y2 > 375
    assert x1 < 135 and x2 > 585
    assert 65 <= y1 <= 100
    assert 380 <= y2 <= 420
    assert 95 <= x1 <= 130
    assert 590 <= x2 <= 630


def test_autocrop_image_mode_excludes_rebate():
    img = _three_tier_negative()

    roi = get_autocrop_coords(img, offset_px=0, scale_factor=1.0, target_ratio_str="Free", mode="image")

    y1, y2, x1, x2 = roi
    # ROI hugs the exposed frame (105:375, 135:585); x is narrowed further by the
    # Free→3:2 snap (frame is 1.67:1). Crucially, the rebate band stays excluded.
    assert 88 <= y1 <= 120
    assert 360 <= y2 <= 392
    assert 137 <= x1 <= 170
    assert 550 <= x2 <= 583


def test_autocrop_modes_nest():
    img = _three_tier_negative()

    film = get_autocrop_coords(img, offset_px=0, scale_factor=1.0, target_ratio_str="Free", mode="film")
    image = get_autocrop_coords(img, offset_px=0, scale_factor=1.0, target_ratio_str="Free", mode="image")

    fy1, fy2, fx1, fx2 = film
    iy1, iy2, ix1, ix2 = image
    assert fy1 <= iy1 and fy2 >= iy2
    assert fx1 <= ix1 and fx2 >= ix2
    assert (iy2 - iy1) * (ix2 - ix1) < (fy2 - fy1) * (fx2 - fx1)


def test_autocrop_free_snaps_to_standard_ratio():
    # Exposed frame is 450w x 270h ≈ 5:3; "Free" must snap to a standard ratio (closest: 3:2).
    img = np.full((480, 720, 3), 1.0, dtype=np.float32)
    img[80:400, 100:620] = 0.78
    img[115:385, 135:585] = 0.25  # 270h x 450w ≈ 5:3 inside film

    roi = get_autocrop_coords(img, offset_px=0, scale_factor=1.0, target_ratio_str="Free", mode="image")

    y1, y2, x1, x2 = roi
    aspect = (x2 - x1) / (y2 - y1)
    standard = []
    for ratio in AspectRatio:
        if ratio in (AspectRatio.FREE, AspectRatio.ORIGINAL):
            continue
        w_r, h_r = map(float, ratio.value.split(":"))
        standard.append(w_r / h_r)
    assert any(abs(aspect - s) / s < 0.03 for s in standard)


def _with_sprocket_holes(img: np.ndarray) -> np.ndarray:
    # Bright (bed-level) sprocket holes punched in both rebate bands.
    img = img.copy()
    for x in range(110, 610, 40):
        img[84:94, x : x + 10] = 1.0
        img[386:396, x : x + 10] = 1.0
    return img


def _low_contrast_bw_negative() -> np.ndarray:
    # B&W negative with dark rebate: rebate/image separation too small for tier path.
    img = np.full((480, 720, 3), 1.0, dtype=np.float32)
    img[80:400, 100:620] = 0.40
    img[105:375, 135:585] = 0.25
    return img


def test_autocrop_image_mode_handles_tilt():
    from negpy.features.geometry.logic import apply_fine_rotation

    img = apply_fine_rotation(_three_tier_negative(), 1.5)

    roi = get_autocrop_coords(img, offset_px=0, scale_factor=1.0, target_ratio_str="Free", mode="image")

    y1, y2, x1, x2 = roi
    # Same frame as the straight fixture, bands widened for the tilt.
    assert 78 <= y1 <= 130
    assert 350 <= y2 <= 402
    assert 127 <= x1 <= 180
    assert 540 <= x2 <= 593


def test_autocrop_film_mode_survives_sprocket_holes():
    img = _with_sprocket_holes(_three_tier_negative())

    roi = get_autocrop_coords(img, offset_px=0, scale_factor=1.0, target_ratio_str="Free", mode="film")

    y1, y2, x1, x2 = roi
    assert 65 <= y1 <= 100
    assert 380 <= y2 <= 420
    assert 95 <= x1 <= 130
    assert 590 <= x2 <= 630


def test_autocrop_image_mode_survives_sprocket_holes():
    img = _with_sprocket_holes(_three_tier_negative())

    roi = get_autocrop_coords(img, offset_px=0, scale_factor=1.0, target_ratio_str="Free", mode="image")

    y1, y2, x1, x2 = roi
    assert 88 <= y1 <= 120
    assert 360 <= y2 <= 392
    assert 137 <= x1 <= 170
    assert 550 <= x2 <= 583


def test_autocrop_image_mode_bright_region_touching_edge():
    # Thin-negative band (deep shadow ~ near rebate level) touching the top frame edge:
    # must stay classified as image, not get cropped away.
    img = _three_tier_negative()
    img[105:145, 135:585] = 0.70

    roi = get_autocrop_coords(img, offset_px=0, scale_factor=1.0, target_ratio_str="Free", mode="image")

    y1, _, _, _ = roi
    assert 95 <= y1 <= 125


def test_tier_refinement_handles_low_contrast_dark_rebate():
    # B&W negative with a dark (0.40) but uniform rebate: the per-side plateau
    # detection must still find it and exclude it from the image crop.
    img = _low_contrast_bw_negative()

    roi = get_autocrop_coords(img, offset_px=0, scale_factor=1.0, target_ratio_str="Free", mode="image")
    y1, y2, x1, x2 = roi
    assert 85 <= y1 <= 125
    assert 355 <= y2 <= 395
    assert 115 <= x1 <= 175
    assert 545 <= x2 <= 605


def test_tier_refinement_rejects_when_separation_too_small():
    from negpy.features.geometry.logic import _detection_luma, _refine_film_roi_by_tiers

    # Rebate barely distinguishable from the image (0.78 vs 0.75): the separation
    # gate must reject so the Sobel path decides instead.
    img = np.full((480, 720, 3), 1.0, dtype=np.float32)
    img[80:400, 100:620] = 0.78
    img[105:375, 135:585] = 0.75

    assert _refine_film_roi_by_tiers(_detection_luma(img), (73, 406, 93, 626)) is None


def test_autocrop_resolution_stability():
    # Detection is internally normalized to AUTOCROP_DETECT_RES (1800): 2400 gets
    # downsampled, 900 does not — the normalized ROIs must agree regardless.
    img_a, img_b = _three_tier_negative_at(900), _three_tier_negative_at(2400)

    roi_a = get_autocrop_coords(img_a, offset_px=0, scale_factor=1.0, target_ratio_str="3:2", mode="image")
    roi_b = get_autocrop_coords(img_b, offset_px=0, scale_factor=1.0, target_ratio_str="3:2", mode="image")

    norm_a = _normalized_roi(roi_a, *img_a.shape[:2])
    norm_b = _normalized_roi(roi_b, *img_b.shape[:2])
    for a, b in zip(norm_a, norm_b):
        assert abs(a - b) < 0.015

    # Pairwise agreement is not sufficient: platform-specific floating-point ties
    # can move both results to the same wrong side of a smoothed edge.  The exposed
    # frame and the centered 3:2 crop both occupy 9/16 of each image dimension.
    expected = (7 / 32, 25 / 32, 7 / 32, 25 / 32)
    for normalized in (norm_a, norm_b):
        np.testing.assert_allclose(normalized, expected, rtol=0.0, atol=0.01)


def test_autocrop_parity_preview_vs_fullres():
    # GPU contract: detect on an INTER_AREA-downsampled image with margin carried by
    # scale_factor, upscale ROI — must match full-res detection (CPU path).
    full = _three_tier_negative_at(3600)
    h, w = full.shape[:2]

    roi_full = get_autocrop_coords(full, offset_px=10, scale_factor=3600 / 1600, target_ratio_str="3:2", mode="image")

    small = cv2.resize(full, (round(w * 0.5), round(h * 0.5)), interpolation=cv2.INTER_AREA)
    roi_small = get_autocrop_coords(small, offset_px=10, scale_factor=1800 / 1600, target_ratio_str="3:2", mode="image")
    roi_up = tuple(v * 2 for v in roi_small)

    norm_full = _normalized_roi(roi_full, h, w)
    norm_up = _normalized_roi(roi_up, h, w)
    for a, b in zip(norm_full, norm_up):
        assert abs(a - b) < 0.01


def test_autocrop_borderless_scan_returns_full_frame():
    # Full-frame scan with no light bed or holder visible: mid-tone textured content
    # with a dark region. Detection must not latch onto the dark blob — both modes
    # should return (near) the full frame.
    h, w = 480, 720
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 0.45 + 0.25 * np.sin(xx / 17.0) * np.cos(yy / 23.0)  # textured mid-tones
    img = np.repeat(base[..., None], 3, axis=2).astype(np.float32)
    img[40:240, 180:560] = 0.12  # dark sky-like blob

    for mode in ("film", "image"):
        y1, y2, x1, x2 = get_autocrop_coords(img, offset_px=0, scale_factor=1.0, target_ratio_str="Free", mode=mode)
        area = (y2 - y1) * (x2 - x1)
        assert area >= 0.90 * h * w, f"mode={mode}: cropped to {area / (h * w):.2f} of frame"


def test_autocrop_image_mode_full_bleed_frame_keeps_film_box():
    # Image content fills the entire film area (no rebate). Image mode must return
    # the film box, not carve an arbitrary inner crop out of the picture.
    h, w = 480, 720
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.full((h, w, 3), 1.0, dtype=np.float32)  # light bed
    texture = 0.30 + 0.20 * np.sin(xx / 13.0) * np.cos(yy / 19.0)
    img[80:400, 100:620] = np.repeat(texture[80:400, 100:620, None], 3, axis=2)

    y1, y2, x1, x2 = get_autocrop_coords(img, offset_px=0, scale_factor=1.0, target_ratio_str="Free", mode="image")

    # Film box is 80:400, 100:620 — crop must cover (almost) all of it.
    assert (y2 - y1) * (x2 - x1) >= 0.90 * 320 * 520
    assert y1 >= 60 and y2 <= 420 and x1 >= 80 and x2 <= 640  # and stay near the film


def test_find_rebate_level_requires_opposite_pair():
    from negpy.features.geometry.logic import _find_rebate_level

    lum = np.full((200, 200), 0.2, dtype=np.float32)
    lum[:8, :] = 1.0  # a few bed-level pixels so the global P99 anchor sits high
    roi = (0, 200, 0, 200)

    one_side = lum.copy()
    one_side[:, -16:] = 0.8  # lone bright strip (e.g. a sunlit window edge)
    assert _find_rebate_level(one_side, roi) is None

    pair = lum.copy()
    pair[:, :16] = 0.8
    pair[:, -16:] = 0.8  # rebate border on both left and right
    res = _find_rebate_level(pair, roi)
    assert res is not None and abs(res[0] - 0.8) < 0.05


def test_autocrop_image_mode_single_bright_side_not_rebate():
    # High-key full-bleed frame with a uniform bright strip on ONE side only (a
    # sunlit window edge) plus a dark subject. The bright strip must NOT be taken
    # for film rebate — image mode must keep the frame, not crop to the subject.
    h, w = 480, 720
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    texture = 0.45 + 0.20 * np.sin(xx / 13.0) * np.cos(yy / 19.0)
    img = np.repeat(texture[..., None], 3, axis=2).astype(np.float32)
    img[:, -40:] = 0.80  # uniform bright strip on the right edge only
    img[120:300, 200:480] = 0.15  # dark subject

    for mode in ("film", "image"):
        y1, y2, x1, x2 = get_autocrop_coords(img, offset_px=0, scale_factor=1.0, target_ratio_str="Free", mode=mode)
        area = (y2 - y1) * (x2 - x1)
        assert area >= 0.85 * h * w, f"mode={mode}: cropped to {area / (h * w):.2f} of frame"


def test_place_window_by_occupancy_maximizes_coverage():
    from negpy.features.geometry.logic import _place_window_by_occupancy

    occ = np.zeros(14, dtype=np.float32)
    occ[4:10] = 1.0
    assert _place_window_by_occupancy(0, 14, 6, occ, 1.0) == 4


def test_place_window_by_occupancy_ties_center():
    from negpy.features.geometry.logic import _place_window_by_occupancy

    occ = np.ones(20, dtype=np.float32)
    assert _place_window_by_occupancy(0, 20, 10, occ, 1.0) == 5


def test_place_window_by_occupancy_clamps_to_bounds():
    from negpy.features.geometry.logic import _place_window_by_occupancy

    occ = np.zeros(20, dtype=np.float32)
    occ[16:] = 1.0
    assert _place_window_by_occupancy(0, 20, 8, occ, 1.0) == 12


def test_aspect_placement_uniform_occupancy_matches_centering():
    # Symmetric fixture: occupancy is uniform inside the frame, so occupancy
    # placement must reproduce the centered crop of the pre-placement behavior.
    img = _three_tier_negative()

    roi = get_autocrop_coords(img, offset_px=0, scale_factor=1.0, target_ratio_str="3:2", mode="image")

    y1, y2, x1, x2 = roi
    assert abs((x1 - 135) - (585 - x2)) <= 6  # crop centered within the frame
    assert abs((x2 - x1) / (y2 - y1) - 1.5) < 0.02


def test_enforce_roi_aspect_ratio_centering_unchanged():
    from negpy.features.geometry.logic import enforce_roi_aspect_ratio

    # Pins the manual-crop path: plain centering, no occupancy involved.
    assert enforce_roi_aspect_ratio((100, 400, 50, 650), 480, 720, "3:2") == (100, 400, 125, 575)
    assert enforce_roi_aspect_ratio((100, 400, 50, 650), 480, 720, "Free") == (100, 400, 50, 650)


def test_autocrop_specific_ratio_holds_in_both_modes():
    img = _three_tier_negative()

    for mode in ("film", "image"):
        roi = get_autocrop_coords(img, offset_px=0, scale_factor=1.0, target_ratio_str="3:2", mode=mode)
        y1, y2, x1, x2 = roi
        aspect = (x2 - x1) / (y2 - y1)
        assert abs(aspect - 1.5) < 0.05, f"mode={mode}: aspect {aspect} not ~3:2"


# ── flip toggle (mirror parity under fine rotation) ──────────────────────


def test_toggle_flip_toggles_flag_and_negates_fine_rotation():
    from negpy.features.geometry.logic import toggle_flip

    geo = GeometryConfig(fine_rotation=5.0)
    flipped = toggle_flip(geo, horizontal=True)
    assert flipped.flip_horizontal is True
    assert flipped.fine_rotation == -5.0

    flipped_v = toggle_flip(geo, horizontal=False)
    assert flipped_v.flip_vertical is True
    assert flipped_v.fine_rotation == -5.0

    # Toggling the same axis again restores the original geometry exactly.
    assert toggle_flip(flipped, horizontal=True) == geo


def test_toggle_flip_zero_fine_rotation_stays_zero():
    from negpy.features.geometry.logic import toggle_flip

    assert toggle_flip(GeometryConfig(), horizontal=True).fine_rotation == 0.0


def test_mirror_normalized_rect():
    from pytest import approx
    from negpy.features.geometry.logic import mirror_normalized_rect

    assert mirror_normalized_rect((0.1, 0.2, 0.5, 0.7), horizontal=True) == approx((0.5, 0.2, 0.9, 0.7))
    assert mirror_normalized_rect((0.1, 0.2, 0.5, 0.7), horizontal=False) == approx((0.1, 0.3, 0.5, 0.8))


def test_toggle_flip_mirrors_manual_crop_rect():
    from pytest import approx
    from negpy.features.geometry.logic import toggle_flip

    geo = GeometryConfig(fine_rotation=3.0, manual_crop_rect=(0.1, 0.2, 0.5, 0.7))
    flipped_h = toggle_flip(geo, horizontal=True)
    assert flipped_h.manual_crop_rect == approx((0.5, 0.2, 0.9, 0.7))
    flipped_v = toggle_flip(geo, horizontal=False)
    assert flipped_v.manual_crop_rect == approx((0.1, 0.3, 0.5, 0.8))
    # Round trip restores the rect (corner order preserved).
    assert toggle_flip(flipped_h, horizontal=True).manual_crop_rect == approx(geo.manual_crop_rect)


@pytest.mark.parametrize("horizontal", [True, False])
def test_flip_with_negated_angle_is_exact_display_mirror_in_mapper(horizontal):
    # The reported bug: flipping while fine rotation is set must mirror the DISPLAYED
    # (rotated) image. The pipeline applies flip BEFORE fine rotation, and
    # mirror(rotate(+a, x)) == rotate(-a, mirror(x)) — so a raw-space point mapped
    # through (flip, -a) must land at the mirror of its (+a, no flip) position.
    from negpy.features.geometry.logic import map_coords_to_geometry

    orig_shape = (1000, 1500)
    for nx, ny in [(0.5, 0.5), (0.35, 0.4), (0.62, 0.58), (0.45, 0.3)]:
        bx, by = map_coords_to_geometry(nx, ny, orig_shape, fine_rotation=5.0)
        fx, fy = map_coords_to_geometry(
            nx,
            ny,
            orig_shape,
            fine_rotation=-5.0,
            flip_horizontal=horizontal,
            flip_vertical=not horizontal,
        )
        if horizontal:
            assert abs(fx - (1.0 - bx)) < 1e-6 and abs(fy - by) < 1e-6
        else:
            assert abs(fx - bx) < 1e-6 and abs(fy - (1.0 - by)) < 1e-6


# ── crop-tool rotation handles ───────────────────────────────────────────


def test_rotation_drag_angle_follows_cursor_like_a_wheel():
    from pytest import approx
    from negpy.features.geometry.logic import rotation_drag_angle

    # Right-side handle dragged upward (screen y decreases): the wheel turns
    # counter-clockwise, which is positive fine rotation.
    center, press = (0.0, 0.0), (100.0, 0.0)
    assert rotation_drag_angle(0.0, center, press, (100.0, -100.0)) == approx(45.0)
    # Same magnitude downward turns clockwise (negative).
    assert rotation_drag_angle(0.0, center, press, (100.0, 100.0)) == approx(-45.0)
    # Adds to the drag-start angle.
    assert rotation_drag_angle(2.0, center, press, (100.0, -100.0)) == approx(45.0)  # clamped
    assert rotation_drag_angle(2.0, center, press, (100.0, -10.0)) == approx(2.0 + np.degrees(np.arctan2(10.0, 100.0)))


def test_rotation_drag_angle_clamps_to_limit():
    from negpy.features.geometry.logic import rotation_drag_angle
    from negpy.features.geometry.models import FINE_ROTATION_LIMIT

    center, press = (0.0, 0.0), (100.0, 0.0)
    assert rotation_drag_angle(0.0, center, press, (-10.0, -100.0)) == FINE_ROTATION_LIMIT
    assert rotation_drag_angle(0.0, center, press, (-10.0, 100.0)) == -FINE_ROTATION_LIMIT


def test_rotation_drag_angle_robust_across_atan2_seam():
    from pytest import approx
    from negpy.features.geometry.logic import rotation_drag_angle

    # Left-side handle crossing the ±180° atan2 seam: a tiny upward move must
    # produce a tiny clockwise (negative) delta, not a ±360° jump.
    center = (0.0, 0.0)
    angle = rotation_drag_angle(0.0, center, (-100.0, 1.0), (-100.0, -1.0))
    assert angle == approx(-np.degrees(np.arctan2(1.0, 100.0)) * 2.0, abs=1e-6)
    assert abs(angle) < 2.0


def test_rotation_drag_angle_sensitivity():
    from pytest import approx
    from negpy.features.geometry.logic import rotation_drag_angle

    center, press = (0.0, 0.0), (100.0, 0.0)
    full = rotation_drag_angle(0.0, center, press, (100.0, -20.0))
    fine = rotation_drag_angle(0.0, center, press, (100.0, -20.0), sensitivity=0.2)
    assert fine == approx(full * 0.2)


def test_flip_with_negated_angle_mirrors_rendered_image():
    # Pixel-level check on GeometryProcessor: toggling a horizontal flip on a
    # fine-rotated image must produce the mirror of the previous render.
    from negpy.features.geometry.logic import toggle_flip

    # Smooth asymmetric image so sub-pixel interpolation differences stay small.
    h, w = 240, 360
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 0.2 + 0.6 * (xx / w) * (yy / h) + 0.15 * np.sin(xx / 40.0)
    img = np.repeat(base[..., None], 3, axis=2).astype(np.float32)

    geo = GeometryConfig(fine_rotation=4.0)
    ctx_a = PipelineContext(scale_factor=1.0, original_size=(h, w))
    out_a = GeometryProcessor(geo).process(img, ctx_a)

    ctx_b = PipelineContext(scale_factor=1.0, original_size=(h, w))
    out_b = GeometryProcessor(toggle_flip(geo, horizontal=True)).process(img, ctx_b)

    # Interior comparison: the rotation's border-replicate wedges differ at the edges.
    m = 40
    np.testing.assert_allclose(out_b[m:-m, m:-m], np.fliplr(out_a)[m:-m, m:-m], atol=0.01)
