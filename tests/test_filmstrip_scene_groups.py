"""Scene sort lays each scene's frames out as their own block on a tinted band, starting a
new row; every other order keeps Qt's own flow."""

from PyQt6.QtGui import QColor
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QListView

from negpy.desktop.session import AppState, AssetListModel
from negpy.desktop.view.sidebar.files import ThumbnailGridView, _ThumbnailDelegate
from negpy.desktop.view.styles.theme import scene_color


def _grid(qapp, order: str, width: int = 240):
    state = AppState()
    scenes = [(1, "s1", "Beach")] * 3 + [(2, "s2", "Night")] * 2 + [None] * 2
    state.uploaded_files = [
        {"name": f"f{i}.nef", "path": f"/r/f{i}.nef", "hash": f"h{i}", **({"scene": scene} if scene else {})}
        for i, scene in enumerate(scenes)
    ]
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

    assert [ordinal for ordinal, _rect in view._scene_bands] == [1, 2]
    for (ordinal, band), (run_ordinal, first, last) in zip(view._scene_bands, model.scene_runs()):
        assert ordinal == run_ordinal
        assert all(band.contains(rects[row]) for row in range(first, last + 1))


def test_the_band_is_tinted_with_the_scene_color(qapp):
    view, _model = _grid(qapp, "scene")
    ordinal, band = view._scene_bands[0]
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
