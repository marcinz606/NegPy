import numpy as np

from negpy.desktop.view.canvas.overlay import feathered_mask_image
from negpy.features.local.logic import _rasterise_mask
from PyQt6.QtGui import QColor, QImage

DODGE = QColor(232, 200, 74)
SQUARE = [(20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)]
W = H = 100


def _to_array(img: QImage) -> np.ndarray:
    bits = img.bits()
    bits.setsize(img.sizeInBytes())
    return np.frombuffer(bits, np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)[:, : img.width()]


def test_interior_fully_tinted():
    img = feathered_mask_image(SQUARE, W, H, sigma_px=6.0, color=DODGE, max_alpha=70)
    arr = _to_array(img)
    center = arr[50, 50]
    assert center[3] == 70
    expected = [int(c * 70 / 255) for c in (DODGE.red(), DODGE.green(), DODGE.blue())]
    assert list(center[:3]) == expected


def test_edge_is_feathered():
    img = feathered_mask_image(SQUARE, W, H, sigma_px=6.0, color=DODGE, max_alpha=70)
    alpha = _to_array(img)[..., 3]
    inside, edge, outside = int(alpha[50, 26]), int(alpha[50, 20]), int(alpha[50, 14])
    assert inside > edge > outside
    assert abs(edge - 35) <= 10


def test_zero_sigma_hard_edge():
    img = feathered_mask_image(SQUARE, W, H, sigma_px=0.0, color=DODGE, max_alpha=70)
    alpha = _to_array(img)[..., 3]
    assert alpha[50, 50] == 70
    assert alpha[50, 17] == 0


def test_parity_with_pipeline_rasteriser():
    sigma = 4.0
    img = feathered_mask_image(SQUARE, W, H, sigma_px=sigma, color=DODGE, max_alpha=70)
    alpha = _to_array(img)[..., 3]
    norm = [(x / W, y / H) for x, y in SQUARE]
    expected = (_rasterise_mask(norm, H, W, sigma) * 70).astype(np.uint8)
    assert np.array_equal(alpha, expected)
