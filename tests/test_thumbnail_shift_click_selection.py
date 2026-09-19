"""Thumbnail shift-click / ctrl-click selection invariants.

A click's mode — plain replace, Shift-range or Ctrl-toggle — is decided once, from the
modifiers held at press; no later stage of the same gesture, and no unrelated selection sync,
may change that decision. The Shift-range anchor tracks Qt's current index rather than a
field this view owns across gestures, so it follows a right-click or a programmatic resync.
`AssetListModel` remaps persistent indexes (Qt's selection, current index, and this view's
own anchor) by each row's cached identity, not its list position, across a sort, a filter, or
a file added to or removed from the list.
"""

from PyQt6.QtCore import Qt, QEvent, QItemSelectionModel, QPointF
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from negpy.desktop.session import AssetListModel
from negpy.desktop.view.sidebar.session_panel import SessionPanel

from conftest import FakeController as _Controller, FakeRepo as _Repo


def _files(n):
    # mtime descending as name ascends, so name-sort and date-sort disagree.
    return [{"name": f"f{i}.dng", "path": f"/tmp/f{i}.dng", "hash": f"h{i}", "mtime": 100 - i} for i in range(n)]


def _browser(qapp):
    controller = _Controller(_Repo())
    controller.state.uploaded_files = _files(8)
    controller.session.asset_model = AssetListModel(controller.state)
    panel = SessionPanel(controller)
    panel.resize(400, 800)
    panel.show()
    qapp.processEvents()
    return panel.file_browser


def _click(browser, display_row, modifier=Qt.KeyboardModifier.NoModifier):
    model = browser.session.asset_model
    idx = model.index(display_row, 0)
    rect = browser.list_view.visualRect(idx)
    QTest.mouseClick(browser.list_view.viewport(), Qt.MouseButton.LeftButton, modifier, rect.center())


def _click_with_modifier_released_before_button(browser, display_row, press_modifier):
    """Presses with `press_modifier` held, releases with none — Shift/Ctrl coming up a
    moment before the mouse button, the timing this repo's own trackpad/mouse testing hit."""
    model = browser.session.asset_model
    idx = model.index(display_row, 0)
    pos = browser.list_view.visualRect(idx).center()
    QTest.mousePress(browser.list_view.viewport(), Qt.MouseButton.LeftButton, press_modifier, pos)
    QTest.mouseRelease(browser.list_view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)


def _selected_names(browser):
    model = browser.session.asset_model
    sel = browser.list_view.selectionModel().selectedIndexes()
    actuals = sorted(model.display_to_actual(i.row()) for i in sel)
    return [browser.session.state.uploaded_files[a]["name"] for a in actuals]


def test_shift_click_range_survives_a_sort_change(qapp):
    browser = _browser(qapp)

    _click(browser, 0)
    _click(browser, 3, Qt.KeyboardModifier.ShiftModifier)
    assert _selected_names(browser) == ["f0.dng", "f1.dng", "f2.dng", "f3.dng"]
    # `session` is a MagicMock here (FakeController), so update_selection() from the real
    # debounced commit is a no-op — set what it would have written once settled.
    browser.session.state.selected_indices = [0, 1, 2, 3]
    browser.session.state.selected_file_idx = 3

    browser._apply_sort_order("date", save=False)

    # Same four files stay highlighted, not "whatever rows 0-3 are now".
    assert sorted(_selected_names(browser)) == ["f0.dng", "f1.dng", "f2.dng", "f3.dng"]

    current = browser.list_view.selectionModel().currentIndex()
    current_actual = browser.session.asset_model.display_to_actual(current.row())
    assert browser.session.state.uploaded_files[current_actual]["name"] == "f3.dng"


def test_fresh_shift_click_after_sort_change_selects_the_right_range(qapp):
    browser = _browser(qapp)

    browser._apply_sort_order("date", save=False)  # newest first, by display row: f7..f0

    _click(browser, 1)  # f6
    _click(browser, 5, Qt.KeyboardModifier.ShiftModifier)  # f2

    assert sorted(_selected_names(browser)) == sorted(["f6.dng", "f5.dng", "f4.dng", "f3.dng", "f2.dng"])


def test_sync_ui_does_not_clobber_a_pending_shift_click(qapp):
    """A stray state_changed (select_file, background thumbnail work, ...) landing
    before the 200ms selection debounce commits must not revert a shift-click."""
    browser = _browser(qapp)
    # A stale prior selection sitting in session state, as if some earlier action
    # had committed it.
    browser.session.state.selected_indices = [7]
    browser.session.state.selected_file_idx = 7

    _click(browser, 0)
    _click(browser, 3, Qt.KeyboardModifier.ShiftModifier)
    assert browser.selection_timer.isActive()
    assert _selected_names(browser) == ["f0.dng", "f1.dng", "f2.dng", "f3.dng"]

    # Simulates a stray state_changed arriving mid-debounce (session is a MagicMock here,
    # so nothing actually emits it — call the connected slot directly).
    browser.sync_ui()

    assert _selected_names(browser) == ["f0.dng", "f1.dng", "f2.dng", "f3.dng"]


def test_click_leaving_multiple_items_selected_does_not_activate(qapp):
    """A click that (per Qt's own selection result) leaves more than one item selected must not
    collapse it by activating just the clicked file — regardless of what any modifier-key query
    would have reported."""
    browser = _browser(qapp)
    model = browser.session.asset_model
    sel_model = browser.list_view.selectionModel()
    sel_model.select(model.index(1, 0), QItemSelectionModel.SelectionFlag.Select)
    sel_model.select(model.index(3, 0), QItemSelectionModel.SelectionFlag.Select)

    browser._on_item_clicked(model.index(3, 0))

    browser.session.select_file.assert_not_called()


def test_click_leaving_exactly_the_clicked_item_selected_activates_it(qapp):
    browser = _browser(qapp)
    model = browser.session.asset_model
    sel_model = browser.list_view.selectionModel()
    sel_model.select(model.index(3, 0), QItemSelectionModel.SelectionFlag.Select)

    browser._on_item_clicked(model.index(3, 0))

    browser.session.select_file.assert_called_once_with(3)


def test_shift_release_before_mouse_button_still_extends_the_range(qapp):
    """Shift held for the press, released by the time the mouse button comes up — the press
    still decides the gesture as a Shift-range regardless of what the release event reports."""
    browser = _browser(qapp)

    _click(browser, 0)
    _click_with_modifier_released_before_button(browser, 3, Qt.KeyboardModifier.ShiftModifier)

    assert _selected_names(browser) == ["f0.dng", "f1.dng", "f2.dng", "f3.dng"]


def test_ctrl_release_before_mouse_button_still_toggles_the_item(qapp):
    browser = _browser(qapp)

    _click(browser, 0)
    _click_with_modifier_released_before_button(browser, 3, Qt.KeyboardModifier.ControlModifier)

    assert sorted(_selected_names(browser)) == ["f0.dng", "f3.dng"]


def test_release_at_a_different_row_than_the_last_move_still_applies(qapp):
    """The release-time reapply is otherwise uncovered: a drag's last move event and its
    release don't always land at the same row (event coalescing, a final jump), so release
    must independently apply the range up to wherever it actually ends, not just leave
    whatever the last move set standing."""
    browser = _browser(qapp)
    browser.window().resize(400, 3000)
    model = browser.session.asset_model
    viewport = browser.list_view.viewport()

    def pos(row):
        return browser.list_view.visualRect(model.index(row, 0)).center()

    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos(2))
    QTest.mouseMove(viewport, pos(4))
    assert _selected_names(browser) == ["f2.dng", "f3.dng", "f4.dng"]

    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos(6))
    assert _selected_names(browser) == ["f2.dng", "f3.dng", "f4.dng", "f5.dng", "f6.dng"]


def test_stray_move_event_with_different_modifiers_does_not_corrupt_the_range(qapp):
    """Reproduces the exact mechanism a live-app diagnostic log caught: a press correctly
    extends the range, but a mouseMoveEvent delivered while the button is still held — read
    with different (here, no) modifiers, as Qt's own drag-tracking does independently of the
    press — must not be allowed to recompute and collapse it."""
    browser = _browser(qapp)
    model = browser.session.asset_model
    viewport = browser.list_view.viewport()

    _click(browser, 0)

    pos = browser.list_view.visualRect(model.index(3, 0)).center()
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, pos)
    assert _selected_names(browser) == ["f0.dng", "f1.dng", "f2.dng", "f3.dng"]

    stray_move = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(pos),
        QPointF(viewport.mapToGlobal(pos)),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(viewport, stray_move)
    assert _selected_names(browser) == ["f0.dng", "f1.dng", "f2.dng", "f3.dng"]

    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)
    assert _selected_names(browser) == ["f0.dng", "f1.dng", "f2.dng", "f3.dng"]


def test_click_and_hold_drag_still_selects_the_swept_range(qapp):
    """The fix for the modifier race must not cost the click-and-hold-drag range select some
    users rely on instead of shift-click: press with no modifier, drag across several cells,
    release — the whole swept range ends up selected."""
    browser = _browser(qapp)
    model = browser.session.asset_model
    viewport = browser.list_view.viewport()

    def pos(row):
        return browser.list_view.visualRect(model.index(row, 0)).center()

    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos(2))
    assert _selected_names(browser) == ["f2.dng"]

    for row in (3, 4, 5):
        QTest.mouseMove(viewport, pos(row))
    assert _selected_names(browser) == ["f2.dng", "f3.dng", "f4.dng", "f5.dng"]

    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos(5))
    assert _selected_names(browser) == ["f2.dng", "f3.dng", "f4.dng", "f5.dng"]


def test_shift_drag_extends_from_the_original_anchor_not_the_press_point(qapp):
    """A forward drag alone can't tell this view's anchor-preserving range apart from Qt's own
    native rectangle-from-press-point drag select — they agree there. A drag that reverses
    direction can: shift-pressing row 6 (anchor 4) then dragging back to row 2 gives [2,3,4]
    anchored on 4, where a press-point rectangle would instead give [2..6]."""
    browser = _browser(qapp)
    browser.window().resize(400, 3000)  # tall enough that every row is on-screen without a
    # mid-drag autoscroll, which would shift a row's screen position out from under a
    # position computed before the drag reached it
    model = browser.session.asset_model
    viewport = browser.list_view.viewport()

    def pos(row):
        return browser.list_view.visualRect(model.index(row, 0)).center()

    _click(browser, 4)
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, pos(6))
    assert _selected_names(browser) == ["f4.dng", "f5.dng", "f6.dng"]

    QTest.mouseMove(viewport, pos(2))
    assert _selected_names(browser) == ["f2.dng", "f3.dng", "f4.dng"]

    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos(2))
    assert _selected_names(browser) == ["f2.dng", "f3.dng", "f4.dng"]


def test_ctrl_click_deselecting_the_clicked_item_does_not_activate_it(qapp):
    """A Ctrl+click that toggles the clicked item OFF (leaving it out of the resulting
    selection) must not still activate it."""
    browser = _browser(qapp)
    model = browser.session.asset_model
    sel_model = browser.list_view.selectionModel()
    sel_model.select(model.index(1, 0), QItemSelectionModel.SelectionFlag.Select)
    # item 3 itself is not selected here, simulating a click that just toggled it off.

    browser._on_item_clicked(model.index(3, 0))

    browser.session.select_file.assert_not_called()


def test_ctrl_shift_click_extends_the_selection_additively(qapp):
    """Ctrl+Shift unions the anchor-to-row range into the existing selection; a plain
    Shift-range instead replaces it."""
    browser = _browser(qapp)

    _click(browser, 0)
    _click(browser, 5, Qt.KeyboardModifier.ControlModifier)
    assert sorted(_selected_names(browser)) == sorted(["f0.dng", "f5.dng"])

    _click(browser, 7, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
    assert sorted(_selected_names(browser)) == sorted(["f0.dng", "f5.dng", "f6.dng", "f7.dng"])


def test_right_click_then_shift_click_anchors_at_the_right_clicked_row(qapp):
    """A right-click must not leave the shift-range anchor stale at whatever the last left
    click was — reading Qt's current index fresh at every press picks it up for free."""
    browser = _browser(qapp)
    model = browser.session.asset_model

    _click(browser, 0)
    pos = browser.list_view.visualRect(model.index(6, 0)).center()
    QTest.mouseClick(browser.list_view.viewport(), Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, pos)
    _click(browser, 7, Qt.KeyboardModifier.ShiftModifier)

    assert sorted(_selected_names(browser)) == sorted(["f6.dng", "f7.dng"])


def test_shift_click_with_no_prior_selection_selects_just_the_clicked_item(qapp):
    """The first click of a session — or any Shift-click with no valid current index to
    extend from — falls back to a plain click instead of selecting nothing."""
    browser = _browser(qapp)

    _click(browser, 4, Qt.KeyboardModifier.ShiftModifier)

    assert _selected_names(browser) == ["f4.dng"]


def test_refresh_after_removing_a_file_keeps_selection_on_the_surviving_file(qapp):
    """`AssetListModel.refresh()` runs after session.py pops a file out of `uploaded_files`
    (an unload). The remap must follow by each row's identity, not its old list position —
    a position already means something else once a row is gone from that list."""
    browser = _browser(qapp)

    _click(browser, 4)
    assert _selected_names(browser) == ["f4.dng"]

    del browser.session.state.uploaded_files[1]
    browser.session.asset_model.refresh()

    assert _selected_names(browser) == ["f4.dng"]


def test_shift_click_anchor_survives_a_sort_direction_change(qapp):
    browser = _browser(qapp)
    model = browser.session.asset_model

    _click(browser, 0)  # f0.dng
    browser._apply_sort_direction(True, save=False)

    target_display = model.actual_to_display(4)  # f4.dng
    pos = browser.list_view.visualRect(model.index(target_display, 0)).center()
    QTest.mouseClick(browser.list_view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, pos)

    assert sorted(_selected_names(browser)) == sorted(["f0.dng", "f1.dng", "f2.dng", "f3.dng", "f4.dng"])


def test_shift_click_anchor_survives_a_sheet_filter_change(qapp):
    browser = _browser(qapp)
    model = browser.session.asset_model

    _click(browser, 2)  # f2.dng
    browser.session.state.uploaded_files[1]["excluded"] = True  # f1.dng drops out
    browser._apply_sheet_filter("unrejected", save=False)

    target_display = model.actual_to_display(5)  # f5.dng
    pos = browser.list_view.visualRect(model.index(target_display, 0)).center()
    QTest.mouseClick(browser.list_view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, pos)

    assert sorted(_selected_names(browser)) == sorted(["f2.dng", "f3.dng", "f4.dng", "f5.dng"])
