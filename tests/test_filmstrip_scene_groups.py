"""Scene sort lays each scene's frames out as their own block on a tinted band, starting a
new row; every other order keeps Qt's own flow."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QListView

from negpy.desktop.session import AppState, AssetListModel
from negpy.desktop.view.sidebar.files import ThumbnailGridView, _ThumbnailDelegate
from negpy.desktop.view.styles.theme import scene_color
from negpy.services.assets.thumbnails import asset_thumbnail_key


def _letterboxed_thumbnail() -> QIcon:
    """Square, like a cached thumbnail: a 3:2 picture on transparent bars."""
    pix = QPixmap(150, 150)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.fillRect(0, 25, 150, 100, QColor(160, 150, 130))
    painter.end()
    return QIcon(pix)


def _grid(qapp, order: str, width: int = 240, thumbnails: bool = False):
    state = AppState()
    scenes = [(1, "s1", "Beach")] * 3 + [(2, "s2", "Night")] * 2 + [None] * 2
    state.uploaded_files = [
        {"name": f"f{i}.nef", "path": f"/r/f{i}.nef", "hash": f"h{i}", **({"scene": scene} if scene else {})}
        for i, scene in enumerate(scenes)
    ]
    if thumbnails:
        for f in state.uploaded_files:
            state.thumbnails[asset_thumbnail_key(f)] = _letterboxed_thumbnail()
    model = AssetListModel(state)
    model.set_sort_order(order)
    view = ThumbnailGridView(target_cell=100)
    view.setModel(model)
    view.setItemDelegate(_ThumbnailDelegate(view, state=state))
    view.setViewMode(QListView.ViewMode.IconMode)
    view.setResizeMode(QListView.ResizeMode.Adjust)
    view.resize(width, 700)
    view.show()
    QTest.qWait(150)
    return view, model


def _rects(view, model):
    return [view.visualRect(model.index(row, 0)) for row in range(model.rowCount())]


def test_each_scene_starts_a_new_row_at_the_first_column(qapp):
    view, model = _grid(qapp, "scene")
    rects = _rects(view, model)
    first_column = rects[0].x()

    for _ordinal, first, _last in model.scene_runs():
        assert rects[first].x() == first_column


def test_no_row_holds_two_scenes(qapp):
    view, model = _grid(qapp, "scene", width=460)  # four columns: every block fits one row
    rects = _rects(view, model)
    runs = model.scene_runs()
    tops = [{rects[row].y() for row in range(first, last + 1)} for _o, first, last in runs]

    for i, a in enumerate(tops):
        for b in tops[i + 1 :]:
            assert not a & b


def test_a_band_behind_each_scene_and_none_behind_frames_in_no_scene(qapp):
    view, model = _grid(qapp, "scene")
    rects = _rects(view, model)

    assert [band[0] for band in view._scene_bands] == [1, 2]
    for (ordinal, first, last, band), run in zip(view._scene_bands, model.scene_runs()):
        assert (ordinal, first, last) == run
        assert all(band.contains(rects[row]) for row in range(first, last + 1))


def test_the_band_is_tinted_with_the_scene_color(qapp):
    view, _model = _grid(qapp, "scene")
    ordinal, _first, _last, band = view._scene_bands[0]
    image = view.viewport().grab().toImage()
    background = QColor(image.pixel(view.viewport().width() - 2, view.viewport().height() - 2))
    tint = QColor(image.pixel(band.center().x(), band.top() + 1))
    scene = QColor(scene_color(ordinal))

    def distance(a: QColor, b: QColor) -> int:
        return abs(a.red() - b.red()) + abs(a.green() - b.green()) + abs(a.blue() - b.blue())

    assert distance(tint, scene) < distance(background, scene)


def test_name_order_keeps_qts_own_flow(qapp):
    view, model = _grid(qapp, "name")
    rects = _rects(view, model)
    columns = view.columns_for_width(view.viewport().width())

    assert view._scene_bands == []
    assert rects[columns].x() == rects[0].x()
    assert rects[columns].y() == rects[0].y() + view.gridSize().height()


def _viewport_image(view) -> QImage:
    view.viewport().repaint()
    return view.viewport().grab().toImage()


def test_the_band_leaves_every_picture_untinted(qapp):
    """A dimmed frame and its transparent letterbox show the plain strip, not the tint."""
    view, model = _grid(qapp, "scene", thumbnails=True)
    index = model.index(0, 0)
    picture = view.itemDelegate().picture_rect(view.visualRect(index), index)
    band = view._scene_bands[0][3]
    inside = [(picture.center().x(), picture.center().y()), (picture.center().x(), picture.top() + 4)]  # picture, letterbox
    outside = (band.center().x(), band.top() + 1)

    tinted = _viewport_image(view)
    view.SCENE_BAND_ALPHA = 0.0
    plain = _viewport_image(view)

    for point in inside:
        assert tinted.pixel(*point) == plain.pixel(*point)
    assert tinted.pixel(*outside) != plain.pixel(*outside)
