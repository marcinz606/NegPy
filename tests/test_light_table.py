from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image
from PyQt6.QtCore import QItemSelectionModel, QPoint, QPointF, QRect, Qt
from PyQt6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap, QStandardItem, QStandardItemModel, QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QListView, QStackedWidget, QStyle, QStyleOptionViewItem, QToolButton, QWidget

from negpy.desktop.session import DesktopSessionManager
from negpy.desktop.view.keyboard_shortcuts import _context_cancel
from negpy.desktop.view.main_window import MainWindow
from negpy.desktop.view.sidebar.files import (
    THUMB_CELL_MIN,
    FileBrowser,
    _FilmDelegate,
    justified_rows,
    print_dmax,
    _RESHAPE_MS,
    picture_in_thumbnail,
    roll_film,
    table_color,
)
from negpy.desktop.view.styles.theme import THEME
from negpy.domain.models import WorkspaceConfig
from negpy.features.metadata.gear_models import FilmStock, GearLibrary
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.kernel.image.logic import prepare_thumbnail
from negpy.services.assets.gear import GearProfiles
from negpy.services.film_edge.geometry import SheetFormat


WHITE = THEME.canvas_bg_white


class _Dock:
    def __init__(self, visible=True, floating=False):
        self.visible = visible
        self.floating = floating

    def isVisible(self):
        return self.visible

    def isFloating(self):
        return self.floating

    def setVisible(self, visible):
        self.visible = visible


def _window():
    stack = QStackedWidget()
    stack.addWidget(QWidget())
    grid = QListView()
    stack.addWidget(grid)
    button = QToolButton()
    button.setCheckable(True)
    win = SimpleNamespace(
        central_stack=stack,
        drawer=_Dock(),
        toolbar=SimpleNamespace(btn_light_table=button),
        file_browser=SimpleNamespace(light_table_view=grid),
        strip_open=True,
        _strip_open_before_light_table=True,
        controller=MagicMock(),
        state=SimpleNamespace(selected_file_idx=0),
    )
    win.toolbar.set_light_table = lambda on: setattr(win.toolbar, "light_table", on)
    win.film_strip_expanded = lambda: win.strip_open
    win.set_film_strip_expanded = lambda on, persist=True: setattr(win, "strip_open", on)
    win.controller.session.asset_model.actual_to_display.return_value = -1
    win.light_table_active = lambda: MainWindow.light_table_active(win)
    return win


def test_the_light_table_takes_the_canvas_and_strip_place_and_gives_it_back(qapp):
    win = _window()

    MainWindow.set_light_table(win, True)
    assert win.central_stack.currentIndex() == 1
    assert win.toolbar.btn_light_table.isChecked()
    assert not win.strip_open, "the strip folds under the grid; the toolbar stays"
    assert win.drawer.visible, "an open controls panel stays open"
    assert win.toolbar.light_table, "the bar shows the Light Table row"

    MainWindow.set_light_table(win, False)
    assert win.central_stack.currentIndex() == 0
    assert not win.toolbar.btn_light_table.isChecked()
    assert not win.toolbar.light_table
    assert win.strip_open and win.drawer.visible


def test_leaving_the_light_table_keeps_a_folded_strip_folded(qapp):
    win = _window()
    win.strip_open = False

    MainWindow.set_light_table(win, True)
    MainWindow.set_light_table(win, False)

    assert not win.strip_open


def test_the_light_table_will_not_open_on_an_empty_grid(qapp):
    win = _window()
    win.controller.session.asset_model.rowCount.return_value = 0
    win.state.uploaded_files = [{"name": "a.tif"}]
    win.toolbar.btn_light_table.setChecked(True)

    MainWindow.set_light_table(win, True)

    assert win.central_stack.currentIndex() == 0
    assert win.strip_open
    assert not win.toolbar.btn_light_table.isChecked()
    win.controller.set_status.assert_called_once_with("No frames match the filter", 3000)


def test_the_light_table_closes_when_its_grid_empties(qapp):
    win = _window()
    MainWindow.set_light_table(win, True)
    win.set_light_table = lambda on: MainWindow.set_light_table(win, on)
    win.controller.session.asset_model.rowCount.return_value = 0
    win.state.uploaded_files = [{"name": "a.tif"}]

    MainWindow._leave_empty_light_table(win)

    assert win.central_stack.currentIndex() == 0
    assert win.strip_open


def test_the_light_table_leaves_a_hidden_controls_panel_hidden(qapp):
    win = _window()
    win.drawer.visible = False

    MainWindow.set_light_table(win, True)
    MainWindow.set_light_table(win, False)

    assert not win.drawer.visible


def test_escape_closes_the_light_table_before_anything_else(qapp):
    win = _window()
    MainWindow.set_light_table(win, True)
    win.set_light_table = lambda on: MainWindow.set_light_table(win, on)
    controller = MagicMock()

    _context_cancel(controller, win)

    assert not win.light_table_active()
    controller.cancel_active_tool.assert_not_called()


def test_one_key_hides_the_panel_and_collapses_the_strip_then_brings_both_back(qapp):
    win = _window()
    expanded = [True]
    win.film_strip_expanded = lambda: expanded[0]
    win.set_film_strip_expanded = lambda on: expanded.__setitem__(0, on)

    MainWindow.toggle_side_panels(win)
    assert not win.drawer.visible and not expanded[0]
    MainWindow.toggle_side_panels(win)
    assert win.drawer.visible and expanded[0]
    win.controller.session.repo.save_global_setting.assert_called_with("panel_right_visible", True)


@pytest.fixture
def browser(qapp):
    repo = MagicMock(spec=StorageRepository)
    repo.get_global_setting.return_value = None
    repo.load_file_settings.return_value = None
    repo.load_file_settings_by_path.return_value = None
    repo.load_file_settings_many.return_value = {}
    repo.get_max_history_index.return_value = 0
    session = DesktopSessionManager(repo)
    session.state.uploaded_files = [{"name": f"{i}.tif", "path": f"/r/{i}.tif", "hash": f"h{i}"} for i in range(3)]
    session.asset_model.refresh()
    controller = MagicMock()
    controller.session = session
    controller.thumbnail_refresh_running = False
    controller.half_frame_mode_for_roll.return_value = False
    return FileBrowser(controller)


def test_the_strip_and_the_light_table_lie_on_the_canvas_background(browser):
    from negpy.desktop.view.canvas.toolbar import CANVAS_COLORS

    gray = CANVAS_COLORS[2][0]
    browser.session.set_canvas_bg(2)

    for view in (browser.list_view, browser.light_table_view):
        assert f"background-color: {gray}" in view.styleSheet()


def test_the_light_table_shares_the_strip_selection(browser):
    assert browser.light_table_view.model() is browser.list_view.model()
    assert browser.light_table_view.selectionModel() is browser.list_view.selectionModel()


def test_a_click_on_the_light_table_selects_a_frame_without_opening_it(browser, qapp):
    from PyQt6.QtTest import QTest

    opened = []
    browser.session.file_selected.connect(opened.append)
    view = browser.light_table_view
    view.resize(1200, 700)
    view.show()
    qapp.processEvents()

    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=view.visualRect(view.model().index(1, 0)).center())
    qapp.processEvents()

    assert [index.row() for index in view.selectionModel().selectedIndexes()] == [1]
    assert opened == [] and browser.session.state.current_file_path is None
    view.hide()


def test_opening_a_frame_from_the_light_table_asks_for_the_canvas(browser, monkeypatch):
    opened, left = [], []
    monkeypatch.setattr(browser, "_activate_file", opened.append)
    browser.light_table_opened.connect(lambda: left.append(True))
    index = browser.light_table_view.model().index(1, 0)

    browser._open_from_light_table(index)

    assert opened == [index] and left == [True]


def _qimage(img: Image.Image) -> QImage:
    arr = np.ascontiguousarray(np.asarray(img.convert("RGB")))
    return QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format.Format_RGB888).copy()


def _padded(picture: Image.Image) -> Image.Image:
    """A thumbnail as an older cache holds it: the picture padded to a 120 px square."""
    from negpy.kernel.image.logic import THUMBNAIL_PAD

    picture = prepare_thumbnail(picture, 120)
    square = Image.new("RGB", (120, 120), THUMBNAIL_PAD)
    square.paste(picture, ((120 - picture.width) // 2, (120 - picture.height) // 2))
    return square


@pytest.mark.parametrize("size", [(300, 200), (200, 300)])
def test_a_padded_thumbnail_gives_back_its_picture(qapp, size):
    rect = picture_in_thumbnail(_qimage(_padded(Image.new("RGB", size, (200, 120, 40)))))

    assert (rect.width(), rect.height()) == ((120, 80) if size[0] > size[1] else (80, 120))


def test_a_thumbnail_that_is_not_square_is_all_picture(qapp):
    from negpy.kernel.image.logic import THUMBNAIL_PAD

    night = Image.new("RGB", (300, 200), THUMBNAIL_PAD)
    night.paste((200, 120, 40), (0, 60, 300, 140))

    rect = picture_in_thumbnail(_qimage(prepare_thumbnail(night, 120)))

    assert (rect.width(), rect.height()) == (120, 80)


def _print(black: tuple[int, int, int]) -> Image.Image:
    """A pale print with one patch at its D-max."""
    img = Image.new("RGB", (300, 200), (230, 225, 215))
    img.paste(black, (0, 0, 60, 40))
    return img


@pytest.mark.parametrize("thumbnail", [lambda img: prepare_thumbnail(img, 120), _padded], ids=["thumbnail", "padded"])
def test_a_prints_dmax_is_its_darkest_tone_not_the_thumbnail_padding(qapp, thumbnail):
    tone = print_dmax(_qimage(thumbnail(_print((30, 22, 12)))))

    assert tone == pytest.approx((30, 22, 12), abs=3)


def test_the_rebate_takes_the_rolls_mean_print_dmax(browser):
    from negpy.services.assets.thumbnails import asset_thumbnail_key

    state = browser.session.state
    for info, black in zip(state.uploaded_files, ((30, 20, 10), (50, 40, 30), (40, 30, 20)), strict=True):
        state.thumbnails[asset_thumbnail_key(info)] = QIcon(QPixmap.fromImage(_qimage(prepare_thumbnail(_print(black), 120))))
    browser.controller.load_gear_library.return_value = _hp5()[1]

    browser._measure_rebate()

    assert browser._rebate_tone == pytest.approx((40, 30, 20), abs=3)
    assert browser._light_table_delegate._film[1] == browser._rebate_tone


def _hp5(**meta) -> tuple[WorkspaceConfig, GearLibrary]:
    config = WorkspaceConfig()
    config = replace(config, metadata=replace(config.metadata, film_stock_id="film-hp5-400", **meta))
    return config, GearLibrary(film_stocks=GearProfiles._read_bundled_list("film_stocks.json", FilmStock))


def test_the_roll_film_is_the_rolls_format_and_rebate(qapp):
    config, library = _hp5()

    geometry, rebate = roll_film({}, config, library, half_frame=False)

    assert geometry.format == SheetFormat.FULL_FRAME
    assert max(rebate) < 40
    assert roll_film({}, config, library, half_frame=False, rebate=(30, 20, 10))[1] == (30, 20, 10)
    assert roll_film({}, config, library, half_frame=True)[0].format == SheetFormat.HALF_FRAME
    assert roll_film({}, _hp5(format="120")[0], library, half_frame=False)[0].format == SheetFormat.MEDIUM


def test_the_gear_library_is_read_again_only_when_the_film_changes(browser):
    loads = browser.controller.load_gear_library
    loads.reset_mock()
    loads.return_value = _hp5()[1]
    browser._film_key = None

    browser.sync_ui()
    browser.sync_ui()
    assert loads.call_count == 1

    state = browser.session.state
    state.config = replace(state.config, metadata=replace(state.config.metadata, film_stock_id="film-hp5-400"))
    browser.sync_ui()
    assert loads.call_count == 2


def test_justified_rows_fill_the_width_and_the_last_row_keeps_its_height():
    rows = justified_rows([1.5, 1.5, 0.75, 1.5, 0.75], width=400, height=100, gap=4)

    assert [len(frames) for _height, frames in rows] == [4, 1]
    height, frames = rows[0]
    assert height == 73 and sum(width for _x, width in frames) == 400
    assert all(x + width == next_x for (x, width), (next_x, _w) in zip(frames, frames[1:]))
    assert rows[1] == (100, [(0, 79)])
    assert justified_rows([], 400, 100, 4) == []


def _show_strip(browser) -> None:
    """A wide, low strip. A fixed size, because the offscreen platform can give a shown window another size."""
    browser.setFixedSize(1400, 200)
    browser.show()


def _thumbs(browser, sizes) -> None:
    from negpy.services.assets.thumbnails import asset_thumbnail_key

    state = browser.session.state
    for info, size in zip(state.uploaded_files, sizes, strict=True):
        picture = prepare_thumbnail(Image.new("RGB", size, (40, 160, 220)), 120)
        state.thumbnails[asset_thumbnail_key(info)] = QIcon(QPixmap.fromImage(_qimage(picture)))
    browser.session.asset_model.refresh_thumbnails({asset_thumbnail_key(info) for info in state.uploaded_files})


def test_each_strip_frame_is_as_wide_as_its_upright_picture(browser, qapp):
    _thumbs(browser, [(300, 200), (200, 300), (300, 200)])
    strip = browser.list_view
    _show_strip(browser)
    qapp.processEvents()

    landscape, portrait = (strip.visualRect(strip.model().index(row, 0)) for row in (0, 1))
    picture = landscape.height() - strip.STRIP_GAP - 2 * _FilmDelegate._BORDER
    assert landscape.height() == strip.viewport().height()
    assert landscape.width() == round(picture * 1.5) + _FilmDelegate._BORDER
    assert portrait.width() == round(picture / 1.5) + _FilmDelegate._BORDER
    assert portrait.left() == landscape.right() + 1
    assert strip.STRIP_GAP == 2 * THEME.space_xl and strip.viewport().geometry().top() == 0
    browser.hide()


def test_the_light_table_rows_fill_its_width(browser, qapp):
    state = browser.session.state
    state.uploaded_files = [{"name": f"{i}.tif", "path": f"/r/{i}.tif", "hash": f"h{i}"} for i in range(12)]
    browser.session.asset_model.refresh()
    _thumbs(browser, [(300, 200), (200, 300)] * 6)
    view, vw = _shown_table(browser, qapp)
    qapp.processEvents()

    rows: dict[int, list[QRect]] = {}
    for row in range(12):
        cell = view.visualRect(view.model().index(row, 0))
        rows.setdefault(cell.top(), []).append(cell)
    *full, last = (rows[top] for top in sorted(rows))
    assert full, "more than one row"
    for cells in full:
        assert cells[0].left() == 0 and cells[-1].right() == vw - 1
        assert len({cell.height() for cell in cells}) == 1
    assert last[0].left() == 0 and last[-1].right() < vw - 1
    assert last[0].height() == view._slim_height() + 2 * _FilmDelegate._BORDER + view.STRIP_GAP
    view.hide()


def _near(c: QColor, color: QColor) -> bool:
    return abs(c.red() - color.red()) + abs(c.green() - color.green()) + abs(c.blue() - color.blue()) < 60


def test_a_scene_draws_a_line_over_its_frames_that_starts_with_its_name(browser, qapp):
    from PyQt6.QtCore import QItemSelectionModel

    from negpy.desktop.view.styles.theme import scene_color

    _scene_roll(browser, [None, NIGHT, NIGHT, None])
    _thumbs(browser, [(300, 200)] * 4)
    strip = browser.list_view
    _show_strip(browser)
    strip.selectionModel().clearSelection()
    strip.selectionModel().select(strip.model().index(2, 0), QItemSelectionModel.SelectionFlag.Select)
    qapp.processEvents()
    bands = [strip.itemDelegate().band(strip.visualRect(strip.model().index(row, 0))) for row in range(4)]
    image = strip.viewport().grab().toImage()
    color = QColor(scene_color(2))

    assert [_near(image.pixelColor(band.center().x(), band.top() + 1), color) for band in bands] == [False, True, True, False]
    assert image.pixelColor(bands[2].left() + 1, bands[2].center().y()) == QColor(THEME.border_checked), "the mat still frames the sides"
    assert not _near(image.pixelColor(bands[1].center().x(), bands[1].bottom() - 1), color), "only the top border"
    assert _near(image.pixelColor(bands[1].left() + 3, bands[1].top() - 3), color), "the name sits on the line at the run's start"
    assert not _near(image.pixelColor(bands[2].center().x(), bands[2].top() - 3), color)
    browser.hide()


def test_each_row_of_a_scene_starts_with_its_name(browser, qapp):
    from negpy.desktop.view.styles.theme import scene_color

    _scene_roll(browser, [NIGHT] * 8)
    _thumbs(browser, [(300, 200)] * 8)
    view, vw = _shown_table(browser, qapp)
    view.selectionModel().clearSelection()
    qapp.processEvents()
    bands = [view.itemDelegate().band(view.visualRect(view.model().index(row, 0))) for row in range(8)]
    image = view.viewport().grab().toImage()
    color = QColor(scene_color(2))
    starts = [band for band in bands if band.left() == 0]

    assert len(starts) > 1, "the scene wraps"
    assert all(_near(image.pixelColor(band.left() + 3, band.top() - 3), color) for band in starts)
    view.hide()


def _shown_table(browser, qapp):
    view = browser.light_table_view
    # A fixed size, because the offscreen platform can give a shown window another size.
    view.setFixedSize(1200, 700)
    view.show()
    qapp.processEvents()
    return view, view.viewport().width()


def test_zoom_takes_one_frame_a_row_away_and_saves_the_size(browser, qapp):
    view, vw = _shown_table(browser, qapp)
    columns = view.columns_for_width(vw)

    view.zoom(1)
    assert view.columns_for_width(vw) == columns - 1
    browser.session.repo.save_global_setting.assert_called_with("light_table_cell", view._target_cell)

    view.zoom(-100)
    assert view._target_cell >= THUMB_CELL_MIN
    view.zoom(100)
    assert view._row_step() <= view.viewport().height()
    assert view.columns_for_width(vw) > 1
    view.hide()


def _ctrl_wheel(angle_y: int, modifiers=Qt.KeyboardModifier.ControlModifier) -> QWheelEvent:
    at = QPointF(10.0, 10.0)
    return QWheelEvent(at, at, QPoint(), QPoint(0, angle_y), Qt.MouseButton.NoButton, modifiers, Qt.ScrollPhase.NoScrollPhase, False)


def test_set_columns_keeps_to_the_range_and_reports_each_layout(browser, qapp):
    view, _vw = _shown_table(browser, qapp)
    reports = []
    view.columns_changed.connect(lambda: reports.append(view.columns()))
    fewest, default, most = view.column_range()
    assert fewest < default < most

    view.set_columns(fewest + 1)
    assert view.columns() == reports[-1] == fewest + 1
    browser.session.repo.save_global_setting.assert_called_with("light_table_cell", view._target_cell)
    view.set_columns(most + 5)
    assert view.columns() == most
    view.set_columns(0)
    assert view.columns() == fewest
    view.hide()


def test_the_bar_slider_follows_a_zoom_of_the_table(browser, qapp):
    from test_canvas_toolbar import _make_toolbar

    view, _vw = _shown_table(browser, qapp)
    win = SimpleNamespace(file_browser=browser, toolbar=_make_toolbar())
    view.zoom(1)
    MainWindow._sync_frames_per_row(win)
    fewest, default, most = view.column_range()
    slider = win.toolbar.frames_per_row
    assert (slider.value(), slider._min, slider.default_value(), slider._max) == (view.columns(), fewest, default, most)
    view.hide()


def test_ctrl_and_the_wheel_zoom_the_light_table(browser, qapp):
    view, vw = _shown_table(browser, qapp)
    columns = view.columns_for_width(vw)

    view.wheelEvent(_ctrl_wheel(120))
    assert view.columns_for_width(vw) == columns - 1
    view.wheelEvent(_ctrl_wheel(-60))
    assert view.columns_for_width(vw) == columns - 1
    view.wheelEvent(_ctrl_wheel(-60))
    assert view.columns_for_width(vw) == columns
    view.wheelEvent(_ctrl_wheel(-120, Qt.KeyboardModifier.NoModifier))
    assert view.columns_for_width(vw) == columns
    view.hide()


def test_the_light_table_opens_at_the_saved_frame_size(browser):
    browser.session.repo.get_global_setting.side_effect = lambda key, default=None: 300 if key == "light_table_cell" else default

    assert FileBrowser(browser.controller).light_table_view._target_cell == 300


def _scene_roll(browser, scenes):
    state = browser.session.state
    state.uploaded_files = [
        {"name": f"{i}.tif", "path": f"/r/{i}.tif", "hash": f"h{i}", **({"scene": scene} if scene else {})}
        for i, scene in enumerate(scenes)
    ]
    browser.session.asset_model.refresh()
    browser.controller.load_gear_library.return_value = _hp5()[1]
    browser._film_key = None
    browser.sync_ui()


BEACH, NIGHT = (1, "s1", "Beach"), (2, "s2", "Night")


def test_a_loop_rings_each_run_of_a_scenes_frames_on_one_row(browser, qapp):
    _scene_roll(browser, [None, BEACH, BEACH, NIGHT, None, BEACH])
    strip, table = browser.list_view, browser.light_table_view
    _show_strip(browser)
    qapp.processEvents()

    runs = [(scene, index.row(), first, last) for scene, index, first, last in strip.scene_runs_on_rows()]
    assert [(scene, row) for scene, row, _first, _last in runs] == [((*BEACH, False), 1), ((*NIGHT, False), 3), ((*BEACH, False), 5)]
    assert runs[0][3] == strip.visualRect(strip.model().index(2, 0))

    table.resize(table.gridSize().width() * 2 + 2 * THEME.space_2xl + 30, 600)  # two frames a row
    table._relayout()
    table.show()
    qapp.processEvents()
    assert [index.row() for _scene, index, _first, _last in table.scene_runs_on_rows()] == [1, 2, 3, 5]
    browser.hide()


def test_a_found_scene_runs_as_a_found_mark(browser, qapp):
    _scene_roll(browser, [BEACH, None, None])
    for f in browser.session.state.uploaded_files[1:]:
        f["found_scene"] = (2, 0, "Scene 2")
    browser.session.asset_model.refresh()
    _show_strip(browser)
    qapp.processEvents()

    runs = [(scene, index.row()) for scene, index, _first, _last in browser.list_view.scene_runs_on_rows()]
    assert runs == [((*BEACH, False), 0), ((2, 0, "Scene 2", True), 1)]
    browser.hide()


def _found_roll(browser, qapp, marks):
    """Frames carrying found marks (None for no scene), on a Light Table six frames wide."""
    state = browser.session.state
    state.uploaded_files = [
        {
            "name": f"{i}.tif",
            "path": f"/r/{i}.tif",
            "hash": f"h{i}",
            **({"found_scene": (1 + k, k, f"Scene {1 + k}")} if k is not None else {}),
        }
        for i, k in enumerate(marks)
    ]
    browser.session.asset_model.refresh()
    browser.controller.load_gear_library.return_value = _hp5()[1]
    browser._film_key = None
    browser.sync_ui()
    table = browser.light_table_view
    table.resize(table.gridSize().width() * 6 + 2 * THEME.space_2xl + 30, 600)
    table._relayout()
    table.show()
    qapp.processEvents()
    return table


def _select(view, rows):
    sel = view.selectionModel()
    sel.clearSelection()
    for row in rows:
        sel.select(view.model().index(row, 0), QItemSelectionModel.SelectionFlag.Select)


def _centre(view, row):
    return view.visualRect(view.model().index(row, 0)).center()


def _selected(view):
    return sorted(i.row() for i in view.selectionModel().selectedIndexes())


def _drag(view, start, end):
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _centre(view, start))
    QTest.mouseMove(view.viewport(), _centre(view, end))
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _centre(view, end))


def test_dragging_selected_frames_onto_another_found_scene_drops_them_there(browser, qapp):
    table = _found_roll(browser, qapp, [0, 0, 1, 1, None])
    _select(table, [0, 1])
    drops = []
    table.found_scene_drop.connect(drops.append)

    _drag(table, 0, 2)

    assert drops == [1]
    assert _selected(table) == [0, 1]
    browser.hide()


def test_a_drop_right_after_selecting_moves_the_new_selection(browser, qapp):
    """The browser commits a selection after a short debounce; a quick drag must not act on the old one."""
    table = _found_roll(browser, qapp, [0, 0, 1, 1, None])
    seen = []
    browser.controller.add_to_found_scene.side_effect = lambda k: seen.append((k, sorted(browser.session.state.selected_indices)))
    _select(table, [0, 1])

    _drag(table, 0, 2)

    assert seen == [(1, [0, 1])]
    browser.hide()


def test_a_frame_that_is_not_selected_drags_at_once(browser, qapp):
    table = _found_roll(browser, qapp, [0, 0, 1, 1, None])
    seen = []
    browser.controller.add_to_found_scene.side_effect = lambda k: seen.append((k, sorted(browser.session.state.selected_indices)))
    _select(table, [3])

    _drag(table, 0, 2)

    assert seen == [(1, [0])]
    assert _selected(table) == [0]
    browser.hide()


def test_the_dragged_frame_follows_the_pointer_and_leaves_a_faint_slot(browser, qapp):
    table = _found_roll(browser, qapp, [0, 0, 1, 1, None])
    _thumbs(browser, [(300, 200)] * 5)
    QTest.qWait(3 * _RESHAPE_MS)
    empty = QPoint(_centre(table, 2).x(), table.visualRect(table.model().index(4, 0)).bottom() + 80)
    near, slot = empty - QPoint(4, 4), _centre(table, 0)
    before = table.viewport().grab().toImage()

    QTest.mousePress(table.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, slot)
    QTest.mouseMove(table.viewport(), empty)
    qapp.processEvents()
    during = table.viewport().grab().toImage()
    QTest.mouseRelease(table.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, empty)

    bg = before.pixelColor(near)
    gap = lambda c: abs(c.red() - bg.red()) + abs(c.green() - bg.green()) + abs(c.blue() - bg.blue())  # noqa: E731
    assert gap(during.pixelColor(near)) > 60, "the frame is drawn under the pointer"
    assert gap(during.pixelColor(slot)) < gap(before.pixelColor(slot)) / 2, "its own place fades to the table"
    browser.hide()


def test_dragging_onto_a_frame_with_no_scene_leaves_the_frames_out(browser, qapp):
    table = _found_roll(browser, qapp, [0, 0, 1, 1, None])
    _select(table, [2, 3])
    drops = []
    table.found_scene_drop.connect(drops.append)

    _drag(table, 3, 4)

    assert drops == [-1]
    browser.hide()


def test_a_still_press_on_a_selected_frame_is_a_plain_click(browser, qapp):
    table = _found_roll(browser, qapp, [0, 0, 1, 1, None])
    _select(table, [0, 1])
    drops = []
    table.found_scene_drop.connect(drops.append)

    _drag(table, 1, 1)

    assert drops == []
    assert _selected(table) == [1]
    browser.hide()


def test_a_double_click_on_a_selected_frame_still_opens_it_while_reviewing(browser, qapp):
    table = _found_roll(browser, qapp, [0, 0, 1, 1, None])
    _select(table, [0, 1])
    opened = []
    table.doubleClicked.connect(lambda index: opened.append(index.row()))

    # The platform sends a click, then the double click; QTest.mouseDClick alone sends only the second.
    QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _centre(table, 1))
    QTest.mouseDClick(table.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _centre(table, 1))

    assert opened == [1]
    browser.hide()


def test_a_drag_on_the_light_table_selects_only_the_pressed_frame(browser, qapp):
    """The Light Table's drag moves found scenes' frames, so it never sweeps a selection."""
    table = _found_roll(browser, qapp, [None] * 5)
    _select(table, [0, 1])

    _drag(table, 1, 3)
    assert _selected(table) == [1]

    _drag(table, 2, 4)
    assert _selected(table) == [2]
    browser.hide()


def test_a_drag_from_the_light_tables_empty_space_selects_nothing(browser, qapp):
    table = _found_roll(browser, qapp, [None] * 5)
    empty = QPoint(_centre(table, 0).x(), table.visualRect(table.model().index(4, 0)).bottom() + 40)
    assert not table.indexAt(empty).isValid()

    QTest.mousePress(table.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, empty)
    QTest.mouseMove(table.viewport(), _centre(table, 2))
    QTest.mouseRelease(table.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _centre(table, 2))

    assert _selected(table) == []
    browser.hide()


def test_the_film_strip_keeps_its_drag_selection_while_reviewing(browser, qapp):
    _found_roll(browser, qapp, [0, 0, 1, 1, None])
    strip = browser.list_view
    _show_strip(browser)
    qapp.processEvents()
    _select(strip, [0, 1])
    drops = []
    strip.found_scene_drop.connect(drops.append)

    _drag(strip, 1, 3)

    assert drops == []
    assert _selected(strip) == [1, 2, 3]
    browser.hide()


def test_a_portrait_frame_stays_upright(qapp):
    icon = QIcon(QPixmap.fromImage(_qimage(prepare_thumbnail(Image.new("RGB", (200, 300), (0, 0, 200)), 120))))
    delegate = _FilmDelegate()
    delegate.set_film(*roll_film({}, *_hp5(), half_frame=False))
    model = QStandardItemModel()
    model.appendRow(QStandardItem())

    picture = delegate._thumbnail(icon, model.index(0, 0))

    assert picture.height() > picture.width()


def _paint_cell(delegate: _FilmDelegate, hover: bool) -> tuple[QImage, QRect]:
    delegate.set_film(*roll_film({}, *_hp5(), half_frame=False))
    model = QStandardItemModel()
    item = QStandardItem()
    item.setData(
        QIcon(QPixmap.fromImage(_qimage(prepare_thumbnail(Image.new("RGB", (300, 200), (40, 160, 220)), 120)))),
        Qt.ItemDataRole.DecorationRole,
    )
    model.appendRow(item)
    # A cell is the picture's shape inside its border.
    cell = QRect(0, 0, 300 + delegate._BORDER, 200 + 2 * delegate._BORDER)
    canvas = QImage(cell.width(), cell.height(), QImage.Format.Format_RGB888)
    canvas.fill(QColor(WHITE))
    option = QStyleOptionViewItem()
    option.rect = cell
    if hover:
        option.state |= QStyle.StateFlag.State_MouseOver
    painter = QPainter(canvas)
    delegate.paint(painter, option, model.index(0, 0))
    painter.end()
    return canvas, delegate._window(option.rect)


@pytest.mark.parametrize("bg", [2, 3])
def test_a_selected_frame_takes_a_red_mat_and_the_open_frame_the_lighter_red(browser, qapp, bg):
    from PyQt6.QtCore import QItemSelectionModel

    browser.session.set_canvas_bg(bg)
    _thumbs(browser, [(300, 200)] * 3)
    browser.session.state.current_file_path, browser.session.state.current_file_hash = "/r/1.tif", "h1"
    strip = browser.list_view
    _show_strip(browser)
    qapp.processEvents()
    model, select = strip.model(), QItemSelectionModel.SelectionFlag.Select
    strip.selectionModel().select(model.index(1, 0), select)
    strip.selectionModel().select(model.index(2, 0), select)
    qapp.processEvents()
    bands = [strip.itemDelegate().band(strip.visualRect(model.index(row, 0))) for row in range(3)]
    image = strip.viewport().grab().toImage()
    border = _FilmDelegate._BORDER

    def at(row: int, below_top: int) -> QColor:
        return image.pixelColor(bands[row].center().x(), bands[row].top() + below_top)

    table, open_red = QColor(table_color(bg)), QColor(THEME.border_checked_hover)
    assert max(at(0, border - 1).getRgb()[:3]) < 40, "an unselected frame keeps its rebate"
    assert at(1, border - 1) == open_red, "the mat fills the border"
    assert at(2, border - 1) == QColor(THEME.border_checked)
    window = strip.itemDelegate()._window(strip.visualRect(model.index(1, 0)))
    picture = image.pixelColor(window.center().x(), window.top())
    assert picture.red() < picture.green(), "the picture stays clear"
    for below_top, alpha in ((-1, 0.9), (-2, 0.8)):
        faded = [round(alpha * a + (1 - alpha) * b) for a, b in zip(open_red.getRgb()[:3], table.getRgb()[:3])]
        assert at(1, below_top).getRgb()[:3] == pytest.approx(faded, abs=2), "two pixels past the border fade"
    assert at(1, -3) == table, "nothing past the mat"
    x, y = window.center().x(), window.center().y()
    sides = [(window.left() - border, y), (window.right() + border, y), (x, window.top() - border), (x, window.bottom() + border)]
    assert [image.pixelColor(px, py) for px, py in sides] == [open_red] * 4, "as wide on every side, over the selected neighbor's"
    browser.hide()


def test_a_selected_frame_whose_thumbnail_is_still_loading_takes_its_mat(browser, qapp):
    from PyQt6.QtCore import QItemSelectionModel

    strip = browser.list_view
    _show_strip(browser)
    strip.selectionModel().select(strip.model().index(1, 0), QItemSelectionModel.SelectionFlag.ClearAndSelect)
    qapp.processEvents()
    band = strip.itemDelegate().band(strip.visualRect(strip.model().index(1, 0)))

    assert strip.viewport().grab().toImage().pixelColor(band.center().x(), band.top() + 1) == QColor(THEME.border_checked)
    browser.hide()


def test_scene_sort_starts_each_scene_on_a_new_row_of_the_light_table(browser, qapp):
    _scene_roll(browser, [BEACH] * 3 + [NIGHT] * 2 + [None] * 2)
    _thumbs(browser, [(300, 200)] * 7)
    browser.session.asset_model.set_sort_order("scene")
    view, _vw = _shown_table(browser, qapp)
    QTest.qWait(3 * _RESHAPE_MS)
    model = view.model()
    cells = [view.visualRect(model.index(row, 0)) for row in range(model.rowCount())]
    tops = [{cells[row].top() for row in range(first, last + 1)} for _ordinal, first, last in model.scene_runs()]

    assert all(cells[first].left() == cells[0].left() for _ordinal, first, _last in model.scene_runs())
    assert all(not a & b for i, a in enumerate(tops) for b in tops[i + 1 :]), "no row holds two scenes"
    view.hide()


def test_the_frame_under_the_pointer_gets_no_ring(browser, qapp):
    from PyQt6.QtCore import QEvent
    from PyQt6.QtGui import QHoverEvent

    _thumbs(browser, [(300, 200)] * 3)
    strip = browser.list_view
    _show_strip(browser)
    strip.selectionModel().clearSelection()
    qapp.processEvents()
    band = strip.itemDelegate().band(strip.visualRect(strip.model().index(1, 0)))
    at = QPointF(band.center())
    qapp.sendEvent(strip.viewport(), QHoverEvent(QEvent.Type.HoverMove, at, at, QPointF(-1, -1)))
    qapp.processEvents()
    image = strip.viewport().grab().toImage()

    assert image.pixelColor(band.center().x(), band.top() - 1) == QColor(table_color(browser.session.state.canvas_bg_index))
    browser.hide()


def test_no_gray_border_edges_a_picture_on_film(qapp):
    canvas, window = _paint_cell(_FilmDelegate(), hover=True)

    assert canvas.pixelColor(window.center().x(), window.top()) == QColor(40, 160, 220)


@pytest.mark.parametrize("hover", [False, True])
def test_an_idle_picture_dims_but_not_its_film(qapp, hover):
    canvas, window = _paint_cell(_FilmDelegate(), hover)

    picture = canvas.pixelColor(window.center())
    assert (picture == QColor(40, 160, 220)) == hover
    assert picture.lightness() <= QColor(40, 160, 220).lightness()
    film = canvas.pixelColor(window.center().x(), window.top() - 2)
    assert max(film.red(), film.green(), film.blue()) < 40, "the rebate is not dimmed"


def test_the_light_table_does_not_dim_an_idle_picture(browser, qapp):
    canvas, window = _paint_cell(browser._light_table_delegate, hover=False)
    strip, _ = _paint_cell(browser._thumbnail_delegate, hover=False)

    assert canvas.pixelColor(window.center()) == QColor(40, 160, 220)
    assert strip.pixelColor(window.center()) != QColor(40, 160, 220), "the Film Strip still dims it"


def _long_roll(browser, count: int) -> None:
    state = browser.session.state
    state.uploaded_files = [{"name": f"{i}.tif", "path": f"/r/{i}.tif", "hash": f"h{i}"} for i in range(count)]
    browser.session.asset_model.refresh()
    _thumbs(browser, [(300, 200), (200, 300)] * (count // 2))


def test_the_roll_dmax_reads_a_few_frames_spread_over_the_roll(browser, monkeypatch):
    from negpy.desktop.view.sidebar import files

    _long_roll(browser, 20)
    measured = []
    monkeypatch.setattr(files, "print_dmax", lambda image: measured.append(image) or (20, 20, 20))
    browser._dmax_by_pixmap.clear()

    browser._measure_rebate()

    assert len(measured) == files._DMAX_SAMPLES
    assert browser._rebate_tone == (20, 20, 20)


def test_the_roll_dmax_reads_the_same_frames_whatever_the_sort_or_filter(browser, monkeypatch):
    from negpy.desktop.view.sidebar import files
    from negpy.services.assets.thumbnails import asset_thumbnail_key

    _long_roll(browser, 20)
    state = browser.session.state
    for i, info in enumerate(state.uploaded_files):
        state.thumbnails[asset_thumbnail_key(info)] = QIcon(QPixmap.fromImage(_qimage(Image.new("RGB", (60, 40), (10 * i, 0, 0)))))
    monkeypatch.setattr(files, "print_dmax", lambda image: image.pixelColor(30, 20).getRgb()[:3])
    browser._measure_rebate()
    tone = browser._rebate_tone

    browser.session.asset_model.set_sort_descending(True)
    browser.session.asset_model.set_filter("1", regex=False)
    browser._measure_rebate()

    assert browser._rebate_tone == tone


def test_a_new_rebate_tone_repaints_without_a_layout(browser, qapp):
    _long_roll(browser, 8)
    view, _vw = _shown_table(browser, qapp)
    cells = view._slim_layout()

    browser._rebate_tone = (30, 20, 10)
    browser._sync_film()

    assert view._slim_cells is cells
    assert browser._light_table_delegate._film[1] == (30, 20, 10)
    view.hide()


def test_a_long_roll_measures_each_picture_once(browser, qapp, monkeypatch):
    from negpy.desktop.view.sidebar import files

    monkeypatch.setattr(_FilmDelegate, "_CACHE_LIMIT", 2)
    _long_roll(browser, 8)
    view, _vw = _shown_table(browser, qapp)
    view._slim_cells = None
    view._slim_layout()
    calls = []
    measure = files.picture_in_thumbnail
    monkeypatch.setattr(files, "picture_in_thumbnail", lambda image: calls.append(1) or measure(image))

    browser._rebate_tone = (30, 20, 10)
    browser._sync_film()
    view._slim_cells = None
    view._slim_layout()

    assert calls == [], "a new rebate tone and a layout keep each picture's bounds"
    view.hide()


def test_the_same_table_color_leaves_the_stylesheet(browser, monkeypatch):
    view = browser.light_table_view
    styled = []
    monkeypatch.setattr(view, "setStyleSheet", styled.append)

    view.set_table_color(table_color(browser.session.state.canvas_bg_index))

    assert styled == []


def test_a_thumbnail_lays_the_frames_out_again_only_when_it_changes_a_frame_shape(browser, qapp):
    from negpy.services.assets.thumbnails import asset_thumbnail_key

    _thumbs(browser, [(300, 200)] * 3)
    view, _vw = _shown_table(browser, qapp)
    cells = view._slim_layout()
    state = browser.session.state
    key = asset_thumbnail_key(state.uploaded_files[1])

    def arrive(size) -> None:
        picture = prepare_thumbnail(Image.new("RGB", size, (40, 160, 220)), 120)
        state.thumbnails[key] = QIcon(QPixmap.fromImage(_qimage(picture)))
        browser.session.asset_model.refresh_thumbnails({key})

    arrive((300, 200))
    QTest.qWait(3 * _RESHAPE_MS)
    assert view._slim_cells is cells, "the same shape keeps the layout"
    arrive((200, 300))
    arrive((300, 300))
    assert view._slim_cells is cells and view._reshape_timer.isActive(), "a burst of new shapes waits for one layout"
    QTest.qWait(3 * _RESHAPE_MS)
    assert view._slim_layout()[1].width() < cells[1].width(), "a new shape lays the rows out again"
    view.hide()


def test_a_new_shape_after_a_reindex_lays_the_frames_out_again(browser, qapp):
    from negpy.services.assets.thumbnails import asset_thumbnail_key

    _thumbs(browser, [(200, 300)] * 3)
    view, _vw = _shown_table(browser, qapp)
    browser.session.asset_model.set_sort_order("name")
    QTest.qWait(3 * _RESHAPE_MS)
    portrait = view._slim_layout()[1].width()
    state = browser.session.state
    key = asset_thumbnail_key(state.uploaded_files[1])

    state.thumbnails[key] = QIcon(QPixmap.fromImage(_qimage(prepare_thumbnail(Image.new("RGB", (300, 200), (40, 160, 220)), 120))))
    browser.session.asset_model.refresh_thumbnails({key})
    QTest.qWait(3 * _RESHAPE_MS)

    assert view._slim_layout()[1].width() > 2 * portrait
    view.hide()


def test_a_portrait_thumbnail_moves_the_strip_frames_after_it(browser, qapp):
    from negpy.services.assets.thumbnails import asset_thumbnail_key

    _thumbs(browser, [(300, 200)] * 3)
    strip = browser.list_view
    _show_strip(browser)
    qapp.processEvents()
    state = browser.session.state
    key = asset_thumbnail_key(state.uploaded_files[1])
    landscape = strip.visualRect(strip.model().index(1, 0)).width()

    picture = prepare_thumbnail(Image.new("RGB", (200, 300), (40, 160, 220)), 120)
    state.thumbnails[key] = QIcon(QPixmap.fromImage(_qimage(picture)))
    browser.session.asset_model.refresh_thumbnails({key})
    QTest.qWait(3 * _RESHAPE_MS)
    portrait, after = (strip.visualRect(strip.model().index(row, 0)) for row in (1, 2))

    assert portrait.width() < landscape / 2
    assert after.left() == portrait.right() + 1
    browser.hide()


def test_a_roll_without_scenes_finds_no_runs_without_reading_its_frames(browser, monkeypatch):
    view = browser.light_table_view
    monkeypatch.setattr(view, "rectForIndex", lambda index: pytest.fail("read a frame's rect"))

    assert view.scene_runs_on_rows() == []


def test_a_paint_reads_the_scene_runs_once_until_the_next_layout(browser, qapp, monkeypatch):
    _scene_roll(browser, [NIGHT] * 40)
    _thumbs(browser, [(300, 200)] * 40)
    view, _vw = _shown_table(browser, qapp)
    runs = view.scene_runs_on_rows()
    monkeypatch.setattr(view, "rectForIndex", lambda index: pytest.fail("read a frame's rect"))

    view.verticalScrollBar().setValue(50)
    scrolled = view.scene_runs_on_rows()

    assert [first.top() for _s, _i, first, _l in scrolled] == [first.top() - 50 for _s, _i, first, _l in runs]
    monkeypatch.undo()
    view.set_film(view._film)
    view.updateGeometries()
    assert view._scene_runs is None
    view.hide()


def test_a_paint_builds_marks_only_for_the_frames_it_repaints(browser, qapp, monkeypatch):
    _long_roll(browser, 40)
    view, _vw = _shown_table(browser, qapp)
    view.selectAll()
    qapp.processEvents()
    delegate = view.itemDelegate()
    first = view.visualRect(view.model().index(0, 0))
    windows = []
    window = delegate._window
    monkeypatch.setattr(delegate, "_window", lambda cell: windows.append(cell) or window(cell))
    image = QImage(view.viewport().size(), QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    try:
        view._paint_marks(painter, delegate, first.adjusted(8, 8, -8, -8))
    finally:
        painter.end()

    assert windows == [first]
    view.hide()


def test_the_strip_and_the_light_table_measure_a_picture_once(qapp, monkeypatch):
    from negpy.desktop.view.sidebar import files

    calls = []
    measure = files.picture_in_thumbnail
    monkeypatch.setattr(files, "picture_in_thumbnail", lambda image: calls.append(1) or measure(image))
    pixmap = QPixmap.fromImage(_qimage(prepare_thumbnail(Image.new("RGB", (300, 200), (40, 160, 220)), 120)))

    assert _FilmDelegate()._picture_bounds(pixmap) == _FilmDelegate()._picture_bounds(pixmap)
    assert len(calls) == 1
