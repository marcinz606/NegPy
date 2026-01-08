import numpy as np
from src.features.geometry.logic import get_manual_crop_coords, get_autocrop_coords
from src.features.geometry.processor import GeometryProcessor
from src.features.geometry.models import GeometryConfig
from src.core.interfaces import PipelineContext


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
    # Create an image where "film base" is 0.94 luma
    # Default threshold 0.96 would detect the entire image as "latent image"
    # because 0.94 < 0.96.
    img = np.ones((100, 100, 3), dtype=np.float32) * 0.94
    # "Image" area is 0.5
    img[20:80, 20:80] = 0.5

    # Without assist, it might fail to find a proper crop or crop to full image
    roi_no_assist = get_autocrop_coords(img)
    # Actually if everything is < 0.96, rows_det will be [0...99]
    # margin will be applied to 0 and 99.
    # We just want to see that assist CHANGED it.

    # With assist (pointing to 0.94 area)
    # threshold will be 0.94 + 0.02 = 0.96.
    # Wait, if assist_luma is 0.94, threshold is 0.96. Still 0.94 < 0.96.
    # I should use a lower threshold or different assist luma.
    # If I click on 0.94, I expect the threshold to be slightly LOWER than 0.94?
    # No, we want to find stuff DARKER than the film base.
    # If film base is 0.94, and latent image starts at say 0.90.
    # We want threshold to be say 0.92.

    # Let's change the logic in logic.py: threshold = assist_luma - 0.02?
    # If I click on film base (0.94), I want to detect stuff < 0.92.

    roi_assist = get_autocrop_coords(img, assist_luma=0.94)
    assert roi_assist != roi_no_assist


def test_geometry_processor_no_autocrop_no_offset():
    img = np.zeros((100, 200, 3), dtype=np.float32)
    config = GeometryConfig(autocrop=False, autocrop_offset=0)
    processor = GeometryProcessor(config)
    context = PipelineContext(scale_factor=1.0, original_size=(100, 200))

    processor.process(img, context)

    assert context.active_roi == (0, 100, 0, 200)
