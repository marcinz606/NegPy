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


# Three masks in a 100x100 view: the one being edited, one crossing it, and one clear of
# both. Each is quoted by the centre pixel that only its own tint reaches.
_SELECTED = [(10.0, 10.0), (45.0, 10.0), (45.0, 45.0), (10.0, 45.0)]
_CROSSING = [(35.0, 35.0), (75.0, 35.0), (75.0, 75.0), (35.0, 75.0)]
_CLEAR = [(75.0, 5.0), (95.0, 5.0), (95.0, 25.0), (75.0, 25.0)]
_CENTRES = {"selected": (20, 20), "crossing": (65, 65), "clear": (85, 15)}


def _overlay_with_three_masks():
    """A 100x100 view with mask 0 selected, ready for _draw_local_masks."""
    from dataclasses import replace
    from PyQt6.QtCore import QRectF
    from negpy.desktop.session import AppState
    from negpy.desktop.view.canvas.overlay import CanvasOverlay
    from negpy.features.local.models import LocalAdjustmentsConfig, LocalMask
    from negpy.services.view.coordinate_mapping import CoordinateMapping

    overlay = CanvasOverlay(AppState())
    overlay._view_rect = QRectF(0, 0, W, H)
    overlay.state.last_metrics["uv_grid"] = CoordinateMapping.create_uv_grid(W, H, 0, 0.0)
    overlay.state.local_selected_mask = 0
    masks = tuple(LocalMask(vertices=tuple((x / W, y / H) for x, y in pts), stops=-1.0) for pts in (_SELECTED, _CROSSING, _CLEAR))
    overlay.state.config = replace(overlay.state.config, local=LocalAdjustmentsConfig(masks=masks))
    return overlay


def _tint_alphas(overlay) -> dict:
    """The alpha each mask's tint puts at its own centre, clear of every outline."""
    from PyQt6.QtGui import QPainter

    img = QImage(W, H, QImage.Format.Format_ARGB32)
    img.fill(0)
    painter = QPainter(img)
    overlay._draw_local_masks(painter)
    painter.end()
    painted = _to_array(img)
    return {name: int(painted[y, x][3]) for name, (x, y) in _CENTRES.items()}


def test_every_mask_is_tinted_when_idle(qapp):
    assert all(a > 0 for a in _tint_alphas(_overlay_with_three_masks()).values())


def test_slider_drag_holds_off_the_selected_tint_and_the_ones_it_crosses(qapp):
    overlay = _overlay_with_three_masks()
    overlay.set_local_slider_drag(True)
    during = _tint_alphas(overlay)

    assert during["selected"] == 0
    assert during["crossing"] == 0
    assert during["clear"] > 0  # nowhere near the area being judged

    overlay.set_local_slider_drag(False)
    assert all(a > 0 for a in _tint_alphas(overlay).values())


def test_a_vertex_drag_holds_off_the_crossing_tint_too(qapp):
    """The selected mask's own fill already sits out a vertex drag; its neighbour joins it."""
    from PyQt6.QtCore import QPointF

    overlay = _overlay_with_three_masks()
    overlay._local_mask_screen_ctrl = [[QPointF(x, y) for x, y in _SELECTED]]
    assert overlay._try_start_vertex_edit(QPointF(10, 10)) is True

    during = _tint_alphas(overlay)
    assert during["selected"] == 0 and during["crossing"] == 0
    assert during["clear"] > 0

    overlay._end_local_edit()
    assert all(a > 0 for a in _tint_alphas(overlay).values())
