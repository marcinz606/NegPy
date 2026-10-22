"""Canvas chrome: fit margin, navigator map, cursor probe chip and the persisted view preferences."""

from unittest.mock import MagicMock

import pytest
from conftest import FakeRepo
from PyQt6.QtCore import QPoint, QRectF, Qt
from PyQt6.QtTest import QTest

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState
from negpy.desktop.view.canvas.navigator import CanvasNavigator, map_size, visible_part
from negpy.desktop.view.canvas.overlay import CanvasOverlay, probe_lines
from negpy.features.exposure.densitometer import DensitometerReading


def test_fit_margin_keeps_the_frame_clear_of_the_edges(qapp):
    overlay = CanvasOverlay(AppState())
    overlay.resize(1000, 600)
    overlay.fit_margin = 12.0
    overlay.update_buffer(None, "sRGB", gpu_size=(1500, 1000))
    r = overlay._view_rect
    assert r.top() == pytest.approx(12.0)
    assert r.bottom() == pytest.approx(588.0)
    assert r.center().x() == pytest.approx(500.0)


def test_visible_part_is_frame_normalized():
    frame = QRectF(-500.0, -250.0, 2000.0, 1000.0)
    seen = visible_part(frame, QRectF(0.0, 0.0, 1000.0, 500.0))
    assert (seen.x(), seen.y(), seen.width(), seen.height()) == pytest.approx((0.25, 0.25, 0.5, 0.5))


def test_visible_part_of_a_frame_off_screen_is_empty():
    assert visible_part(QRectF(2000.0, 0.0, 100.0, 100.0), QRectF(0.0, 0.0, 1000.0, 500.0)).isEmpty()


def test_map_keeps_the_frame_aspect_on_a_fixed_long_edge():
    assert (map_size(2.0).width(), map_size(2.0).height()) == (160, 80)
    assert (map_size(0.5).width(), map_size(0.5).height()) == (80, 160)


def test_navigator_press_asks_to_center_on_that_spot(qapp):
    nav = CanvasNavigator()
    nav.set_view(None, 2.0, QRectF(0.25, 0.25, 0.5, 0.5))
    nav.resize(nav.sizeHint())
    nav.show()
    got = []
    nav.center_requested.connect(lambda x, y: got.append((x, y)))
    left = nav.width() - 160
    QTest.mouseClick(nav, Qt.MouseButton.LeftButton, pos=QPoint(left + 40, 60))
    assert got == [pytest.approx((0.25, 0.75))]


def test_probe_chip_reads_zone_density_and_channel_densities():
    reading = DensitometerReading(dd_rgb=(0.1, 0.17, 0.2), val_luma=0.5, print_density=0.62, zone=6.33)
    assert probe_lines(reading) == ("VI⅓ · D 0.62", "ΔD 0.10·0.17·0.20")


def _controller() -> MagicMock:
    ctrl = MagicMock()
    ctrl.state = AppState()
    ctrl.session.repo = FakeRepo()
    ctrl.set_canvas_info_level = lambda level: AppController.set_canvas_info_level(ctrl, level)
    return ctrl


def test_cursor_probe_is_a_saved_preference():
    ctrl = _controller()
    AppController.toggle_cursor_probe(ctrl)
    assert ctrl.state.cursor_probe is True
    assert ctrl.session.repo.data["cursor_probe"] is True
    ctrl.cursor_probe_changed.emit.assert_called_with(True)


def test_frame_info_cycles_off_basic_full_and_is_saved():
    ctrl = _controller()
    seen = []
    for _ in range(3):
        AppController.cycle_canvas_info(ctrl)
        seen.append(ctrl.state.canvas_info_level)
    assert seen == [2, 0, 1]
    assert ctrl.session.repo.data["canvas_info_level"] == 1


def test_navigator_press_beside_the_map_falls_through(qapp):
    nav = CanvasNavigator()
    nav.set_view(None, 0.5, QRectF(0.0, 0.0, 0.5, 0.5))
    nav.resize(nav.sizeHint())
    nav.show()
    got = []
    nav.center_requested.connect(lambda x, y: got.append((x, y)))
    QTest.mouseClick(nav, Qt.MouseButton.LeftButton, pos=QPoint(5, 40))
    assert got == []


@pytest.mark.parametrize(("thumb", "frame", "picture"), [((120, 80), (3000, 2000), (120, 80)), ((120, 120), (2000, 3000), (80, 120))])
def test_the_map_cuts_the_frame_out_of_its_thumbnail(qapp, thumb, frame, picture):
    from types import SimpleNamespace

    from PyQt6.QtGui import QColor, QIcon, QPixmap

    from negpy.desktop.view.canvas.widget import ImageCanvas
    from negpy.services.assets.thumbnails import asset_thumbnail_key

    pixmap = QPixmap(*thumb)
    pixmap.fill(QColor(200, 0, 0))
    asset = {"hash": "h", "path": "/a.tif"}
    state = SimpleNamespace(selected_file_idx=0, uploaded_files=[asset], thumbnails={asset_thumbnail_key(asset): QIcon(pixmap)})
    canvas = SimpleNamespace(overlay=SimpleNamespace(_qimage=None), state=state, display_size=lambda: frame)

    image = ImageCanvas._navigator_image(canvas)

    assert (image.width(), image.height()) == picture
    assert image.pixelColor(0, 0) == image.pixelColor(picture[0] - 1, picture[1] - 1) == QColor(200, 0, 0)
