"""Local Flash: a mask's own preflash, pre-summed into plane 2 and added to the frame's flash
at the pixel, so a full-frame flash is the frame's Preflash and a masked one stops at the mask."""

import numpy as np

from negpy.features.exposure.logic import apply_characteristic_curve
from negpy.features.local.logic import compute_local_maps
from negpy.features.local.models import LocalAdjustmentsConfig, LocalMask

SLOPE, PIVOT = 2.9, 0.21
ARGS = ((PIVOT, SLOPE), (PIVOT, SLOPE), (PIVOT, SLOPE))


def _image(h: int = 40, w: int = 40) -> np.ndarray:
    x = np.linspace(-0.3, 1.2, w, dtype=np.float32)
    return np.ascontiguousarray(np.repeat(np.repeat(x[None, :, None], h, axis=0), 3, axis=2))


def test_a_full_frame_flash_is_the_frames_preflash():
    img = _image()
    global_flash = apply_characteristic_curve(img, *ARGS, frame_grade=115.0, preflash=0.5)
    local = apply_characteristic_curve(img, *ARGS, frame_grade=115.0, flash_map=np.full(img.shape[:2], 0.5, np.float32))
    np.testing.assert_allclose(local, global_flash, atol=1e-6)


def test_the_frames_flash_and_a_masks_add():
    img = _image()
    both = apply_characteristic_curve(img, *ARGS, frame_grade=115.0, preflash=0.2, flash_map=np.full(img.shape[:2], 0.3, np.float32))
    summed = apply_characteristic_curve(img, *ARGS, frame_grade=115.0, preflash=0.5)
    np.testing.assert_allclose(both, summed, atol=1e-6)


def test_a_masked_flash_stops_at_the_mask():
    img = _image(100, 100)
    mask = LocalMask(vertices=((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)), flash=0.8, feather=0.0)
    maps = compute_local_maps(LocalAdjustmentsConfig(masks=(mask,)), 100, 100, (100, 100))
    plain = apply_characteristic_curve(img, *ARGS, frame_grade=115.0)
    flashed = apply_characteristic_curve(img, *ARGS, frame_grade=115.0, flash_map=np.ascontiguousarray(maps[:, :, 2]))
    np.testing.assert_array_equal(flashed[:, 60:], plain[:, 60:])
    assert float(np.max(np.abs(flashed[:, :40] - plain[:, :40]))) > 0.01
