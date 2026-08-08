import numpy as np

from negpy.desktop.view.canvas.overlay import feathered_mask_image
from negpy.features.local.logic import rasterise
from negpy.features.local.models import MaskShape
from PyQt6.QtGui import QColor, QImage

DODGE = QColor(232, 200, 74)
SQUARE = [(20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)]
W = H = 100


def _to_array(img: QImage) -> np.ndarray:
    bits = img.bits()
    bits.setsize(img.sizeInBytes())
    # Copy the data. The view points to Qt memory that the QImage releases.
    return np.frombuffer(bits, np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)[:, : img.width()].copy()


def _tint(shape: MaskShape = MaskShape.POLYGON, pts=None, sigma: float = 6.0, invert: bool = False) -> np.ndarray:
    img = feathered_mask_image(shape, pts or SQUARE, W, H, sigma_px=sigma, color=DODGE, max_alpha=70, invert=invert)
    return _to_array(img)


def test_interior_fully_tinted():
    center = _tint()[50, 50]
    assert center[3] == 70
    expected = [int(c * 70 / 255) for c in (DODGE.red(), DODGE.green(), DODGE.blue())]
    assert list(center[:3]) == expected


def test_edge_is_feathered():
    alpha = _tint()[..., 3]
    # The smooth outline goes out past its control points. The edge is near x=12.
    inside, edge, outside = int(alpha[50, 30]), int(alpha[50, 12]), int(alpha[50, 0])
    assert inside > edge > outside
    assert abs(edge - 35) <= 10


def test_zero_sigma_hard_edge():
    alpha = _tint(sigma=0.0)[..., 3]
    assert alpha[50, 50] == 70
    assert alpha[50, 5] == 0


def test_invert_tints_outside_instead():
    alpha = _tint(sigma=0.0, invert=True)[..., 3]
    assert alpha[50, 50] == 0
    assert alpha[50, 5] == 70


def test_oval_tint_is_round():
    # Centre and both axis ends. The area in the axes is tinted, the box corner is not.
    alpha = _tint(MaskShape.OVAL, [(50.0, 50.0), (90.0, 50.0), (50.0, 90.0)], sigma=0.0)[..., 3]
    assert alpha[50, 85] == 70
    assert alpha[85, 85] == 0


def test_gradient_tint_ramps_along_its_axis():
    alpha = _tint(MaskShape.GRADIENT, [(20.0, 50.0), (80.0, 50.0)], sigma=0.0)[..., 3]
    assert alpha[50, 10] == 70  # behind the full-exposure edge
    assert alpha[50, 90] == 0  # past the fade-out edge
    assert 20 < int(alpha[50, 50]) < 50


def test_parity_with_pipeline_rasteriser():
    sigma = 4.0
    alpha = _tint(sigma=sigma)[..., 3]
    norm = [(x / W, y / H) for x, y in SQUARE]
    expected = (rasterise(MaskShape.POLYGON, norm, H, W, sigma) * 70).astype(np.uint8)
    assert np.array_equal(alpha, expected)
