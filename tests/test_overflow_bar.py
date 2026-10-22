import pytest
from PyQt6.QtWidgets import QPushButton

from negpy.desktop.view.widgets.overflow_bar import OverflowBar


@pytest.fixture
def tabs(qapp):
    bar = OverflowBar(height=38, min_item=36)
    for i in range(10):
        bar.add_button(QPushButton(), f"Tab {i}")
    bar.show()
    return bar


def _resize(qapp, bar, width):
    bar.resize(width, bar._height)
    qapp.processEvents()
    return [i for i, btn in enumerate(bar.buttons) if not btn.isHidden()]


def test_minimum_width_does_not_grow_with_the_button_count(qapp):
    """The whole point: the floor used to be one button per item, so every button added to a
    panel widened that panel permanently."""
    small, large = OverflowBar(), OverflowBar()
    for i in range(3):
        small.add_button(QPushButton(), f"B{i}")
    for i in range(30):
        large.add_button(QPushButton(), f"B{i}")
    assert small.minimumSizeHint().width() == large.minimumSizeHint().width()


def test_tiled_bar_shows_everything_when_wide_enough(qapp, tabs):
    assert len(_resize(qapp, tabs, 10 * 36)) == 10
    assert not tabs.overflow_btn.isVisible()


def test_tiled_bar_spills_as_it_narrows(qapp, tabs):
    wide = _resize(qapp, tabs, 360)
    mid = _resize(qapp, tabs, 240)
    narrow = _resize(qapp, tabs, 160)
    assert len(wide) == 10 > len(mid) > len(narrow) >= 1
    assert tabs.overflow_btn.isVisible()


def test_tiled_bar_keeps_at_least_one_button_under_extreme_squeeze(qapp, tabs):
    assert len(_resize(qapp, tabs, 1)) == 1


def test_pinned_button_is_never_spilled(qapp, tabs):
    """A tab strip that hides where you are is worse than one that truncates — the active tab
    takes the last visible slot instead of falling into the menu."""
    tabs.set_pinned(9)
    visible = _resize(qapp, tabs, 120)
    assert 9 in visible and visible[-1] == 9


def test_a_fixed_width_tile_keeps_its_width_and_the_rest_share_the_room(qapp):
    bar = OverflowBar(height=38, min_item=80)
    for i in range(4):
        bar.add_button(QPushButton(), f"Tab {i}")
    for i in range(2):
        icon = QPushButton()
        icon.setFixedWidth(38)
        bar.add_button(icon, f"Icon {i}")
    bar.show()

    assert len(_resize(qapp, bar, 4 * 100 + 2 * 38)) == 6
    assert [b.width() for b in bar.buttons] == [100, 100, 100, 100, 38, 38]
    assert _resize(qapp, bar, 4 * 80 + 2 * 38 - 1) == [0, 1, 2, 3, 4] and bar.overflow_btn.isVisible()


def test_tiled_buttons_fill_the_strip_without_gaps(qapp, tabs):
    visible = _resize(qapp, tabs, 240)
    edges = [(tabs.buttons[i].geometry().x(), tabs.buttons[i].geometry().right() + 1) for i in visible]
    assert edges[0][0] == 0
    assert all(left[1] == right[0] for left, right in zip(edges, edges[1:]))
    assert edges[-1][1] + OverflowBar.OVERFLOW_W == tabs.width()


def test_full_tiled_strip_uses_the_whole_width(qapp, tabs):
    _resize(qapp, tabs, 400)
    edges = [(b.geometry().x(), b.geometry().right() + 1) for b in tabs.buttons]
    assert edges[0][0] == 0 and edges[-1][1] == tabs.width()


def test_menu_entry_clicks_the_real_button(qapp, tabs):
    visible = _resize(qapp, tabs, 160)
    hidden_btn = tabs.buttons[len(visible)]
    fired = []
    hidden_btn.clicked.connect(lambda *_: fired.append(1))

    tabs.build_overflow_menu().actions()[0].trigger()
    assert fired == [1]


def test_menu_mirrors_toggle_state(qapp, tabs):
    tabs.buttons[9].setCheckable(True)
    tabs.buttons[9].setChecked(True)
    _resize(qapp, tabs, 160)

    action = [a for a in tabs.build_overflow_menu().actions() if a.text() == "Tab 9"][0]
    assert action.isCheckable() and action.isChecked()
