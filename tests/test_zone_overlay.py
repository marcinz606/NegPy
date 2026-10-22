"""Zone Overlay: the zone field it is drawn from, its looks and its hover."""

from unittest.mock import MagicMock

import numpy as np
from conftest import FakeRepo
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QPainter, QPixmap

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState
from negpy.desktop.view.canvas.overlay import CanvasOverlay
from negpy.desktop.view.styles.color_vision import PALETTES
from negpy.features.exposure.analysis import ZONE_FIELD_MAX_EDGE, zone_anchors, zone_field, zone_shares
from negpy.kernel.image.logic import working_oetf_encode


def _flat(value: float, h: int = 180, w: int = 240) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.float32)


def _mid() -> float:
    return float(working_oetf_encode(np.asarray([0.18], dtype=np.float32))[0])


def test_mid_gray_reads_zone_five_everywhere():
    assert np.all(zone_field(_flat(_mid())) == 5)


def test_field_is_capped_on_the_long_edge():
    zones = zone_field(_flat(0.5, 1067, 1600))
    assert max(zones.shape) <= ZONE_FIELD_MAX_EDGE


def test_a_ramp_climbs_zone_by_zone():
    ramp = np.repeat(np.linspace(0.0, 1.0, 600, dtype=np.float32)[None, :, None], 3, axis=2).repeat(40, axis=0)
    row = zone_field(ramp)[20]
    assert row[0] == 0 and row[-1] == 10
    assert np.all(np.diff(row.astype(int)) >= 0)


def test_grain_does_not_shatter_a_flat_field():
    rng = np.random.default_rng(0)
    grainy = np.clip(_flat(_mid(), 1067, 1600) + rng.normal(0, 0.05, (1067, 1600, 3)), 0.0, 1.0).astype(np.float32)
    assert len(zone_anchors(zone_field(grainy))) <= 3


def test_black_and_white_halves_get_one_numeral_each_inside_their_region():
    img = _flat(0.0)
    img[:, 120:] = 1.0
    zones = zone_field(img)
    anchors = zone_anchors(zones)
    assert {zone for _x, _y, zone, _inset in anchors} == {0, 10}
    for x, y, zone, _inset in anchors:
        assert zones[y, x] == zone


def test_shares_sum_to_the_frame():
    img = _flat(0.0)
    img[:, 120:] = 1.0
    shares = zone_shares(zone_field(img))
    assert shares.sum() == 1.0
    # The grain blur leaves a thin band of middle zones along the hard edge.
    assert shares[0] == shares[10] > 0.45


def test_degenerate_input_has_no_field():
    assert zone_field(np.zeros((1, 240, 3), dtype=np.float32)) is None
    assert zone_field(np.zeros((180, 240), dtype=np.float32)) is None


def test_every_color_vision_has_a_ramp_with_v_neutral():
    for palette in PALETTES:
        assert len(palette.zone_ramp) == 9
        assert palette.zone_ramp[4] == "#8C8C8C"


def _overlay(style: str) -> CanvasOverlay:
    state = AppState()
    state.zones_overlay = True
    state.zones_style = style
    overlay = CanvasOverlay(state)
    img = _flat(0.0, 300, 400)
    img[:, 200:] = 1.0
    overlay.update_buffer(img, "Adobe RGB")
    overlay.resize(400, 300)
    overlay._view_rect = QRectF(0, 0, 400, 300)
    return overlay


def _paint(overlay: CanvasOverlay) -> None:
    pixmap = QPixmap(400, 300)
    painter = QPainter(pixmap)
    overlay._draw_zone_map(painter)
    painter.end()


def test_both_map_looks_mark_paper_black_and_white_in_the_clipping_pair():
    from negpy.desktop.view.canvas.overlay import zone_colors

    for palette in PALETTES:
        for style in ("grid", "zone_map"):
            colors = zone_colors(style, palette.key)
            assert (colors[0][0].name(), colors[-1][0].name()) == tuple(c.lower() for c in palette.pair), (style, palette.key)


def test_both_looks_paint_a_map_and_report_the_shares(qapp):
    for style in ("false_color", "zone_map"):
        overlay = _overlay(style)
        shares = []
        overlay.zone_shares_changed.connect(shares.append)
        _paint(overlay)
        assert overlay._zone_image is not None
        assert len(shares) == 1 and shares[0][0] == shares[0][10] > 0.45


def test_hovering_the_frame_reports_its_zone_until_the_pointer_leaves(qapp):
    overlay = _overlay("false_color")
    _paint(overlay)
    seen = []
    overlay.zone_pointer_changed.connect(seen.append)
    overlay._update_zone_hover(QPointF(300, 200))
    overlay._update_zone_hover(QPointF(301, 200))
    overlay._update_zone_hover(QPointF(50, 50))
    overlay.leaveEvent(None)
    assert seen == [10, 0, None]


def test_a_highlighted_zone_is_spotlit_with_the_overlay_on_or_off(qapp):
    for overlay_on in (True, False):
        overlay = _overlay("false_color")
        overlay.state.zones_overlay = overlay_on
        overlay.state.zone_highlight = 0
        pixmap = QPixmap(400, 300)
        painter = QPainter(pixmap)
        overlay._draw_ui(painter)
        painter.end()
        assert list(overlay._frame_washes) == [("zone_spot", 0)]
        assert overlay._zone_image is None  # the spotlight replaces the map


def test_legend_rings_the_pointer_zone_and_reports_hover(qapp):
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QMouseEvent

    from negpy.desktop.view.canvas.overlay import zone_colors
    from negpy.desktop.view.canvas.zone_legend import ZoneLegend

    legend = ZoneLegend()
    legend.set_colors(zone_colors("false_color", "standard"))
    legend.set_shares(np.full(11, 1 / 11))
    assert legend.readout() == ("Zone –", "")
    legend.set_ringed(4)
    assert legend.readout() == ("Zone IV", "9%")
    hovered = []
    legend.zone_hovered.connect(hovered.append)
    x = legend._cell(6).center().x()
    legend.mouseMoveEvent(
        QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(x, 5),
            QPointF(x, 5),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    assert legend.readout() == ("Zone VI", "9%")
    legend.leaveEvent(None)
    assert legend._ringed == 4
    assert hovered == [6, None]
    assert legend.grab().width() == legend.width()


def test_esc_closes_the_zone_overlay_only_once_no_tool_is_armed():
    from negpy.desktop.session import ToolMode
    from negpy.desktop.view.keyboard_shortcuts import _context_cancel

    window = MagicMock()
    window.light_table_active.return_value = False
    window.canvas.overlay.cancel_in_progress.return_value = False
    for tool, closes in ((ToolMode.NONE, True), (ToolMode.CROP_MANUAL, False)):
        ctrl = MagicMock()
        ctrl.state = AppState(zones_overlay=True, active_tool=tool)
        _context_cancel(ctrl, window)
        assert ctrl.toggle_zones_overlay.called is closes
        assert ctrl.cancel_active_tool.called is not closes


def test_zone_style_is_a_saved_preference():
    ctrl = MagicMock()
    ctrl.state = AppState()
    ctrl.session.repo = FakeRepo()
    AppController.set_zones_style(ctrl, "zone_map")
    assert ctrl.state.zones_style == "zone_map"
    assert ctrl.session.repo.data["zones_style"] == "zone_map"


def test_highlight_zone_ignores_the_strip_hover_setting():
    ctrl = MagicMock()
    ctrl.state = AppState(zone_hover_highlight=False)
    AppController.highlight_zone(ctrl, 4)
    assert ctrl.state.zone_highlight == 4
    ctrl.zone_highlight_changed.emit.assert_called_once()


def test_the_grid_look_draws_the_grid_not_the_map(qapp):
    overlay = _overlay("grid")
    _paint(overlay)
    assert overlay._zone_cells is not None
    assert overlay._zone_image is None
