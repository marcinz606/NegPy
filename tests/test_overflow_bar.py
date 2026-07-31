import pytest
from PyQt6.QtWidgets import QFrame, QPushButton, QToolButton

from negpy.desktop.view.widgets.overflow_bar import OverflowBar


@pytest.fixture
def tabs(qapp):
    bar = OverflowBar(tile=True, height=38, min_item=36)
    for i in range(10):
        bar.add_button(QPushButton(), f"Tab {i}")
    bar.show()
    return bar


@pytest.fixture
def toolbar(qapp):
    bar = OverflowBar(height=28, spacing=4)
    for i in range(6):
        btn = QToolButton()
        btn.setFixedWidth(24)
        bar.add_button(btn, f"Action {i}")
        if i == 2:
            sep = QFrame()
            sep.setFixedWidth(1)
            bar.add_separator(sep)
    bar.show()
    return bar


def _resize(qapp, bar, width):
    bar.resize(width, bar._height)
    qapp.processEvents()
    return [i for i, btn in enumerate(bar.buttons) if not btn.isHidden()]


def test_minimum_width_does_not_grow_with_the_button_count(qapp):
    """The whole point: the floor used to be one button per item, so every button added to a
    panel widened that panel permanently."""
    small, large = OverflowBar(tile=True), OverflowBar(tile=True)
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


def test_natural_bar_keeps_button_widths_and_packs_left(qapp, toolbar):
    _resize(qapp, toolbar, 400)
    assert not toolbar.overflow_btn.isVisible()
    assert toolbar.buttons[0].geometry().x() == 0
    assert all(btn.width() == 24 for btn in toolbar.buttons)


def test_natural_bar_spills_the_tail_into_the_menu(qapp, toolbar):
    visible = _resize(qapp, toolbar, 90)
    assert len(visible) < 6
    assert toolbar.overflow_btn.isVisible()
    labels = [action.text() for action in toolbar.build_overflow_menu().actions()]
    assert labels == [f"Action {i}" for i in range(len(visible), 6)]


def test_separator_is_never_offered_in_the_menu(qapp, toolbar):
    _resize(qapp, toolbar, 90)
    assert all(action.text().startswith("Action") for action in toolbar.build_overflow_menu().actions())


def test_a_trailing_separator_is_dropped_rather_than_left_dangling(qapp, toolbar):
    """Cutting the row right after the separator would leave a hairline floating at the end."""
    for width in range(60, 200, 2):
        _resize(qapp, toolbar, width)
        shown = [item for item, _label in toolbar._items if not item.isHidden()]
        if shown:
            assert not isinstance(shown[-1], QFrame), width


def test_menu_entry_clicks_the_real_button(qapp, toolbar):
    visible = _resize(qapp, toolbar, 90)
    hidden_btn = toolbar.buttons[len(visible)]
    fired = []
    hidden_btn.clicked.connect(lambda *_: fired.append(1))

    toolbar.build_overflow_menu().actions()[0].trigger()
    assert fired == [1]


def test_menu_mirrors_toggle_state(qapp, toolbar):
    toolbar.buttons[5].setCheckable(True)
    toolbar.buttons[5].setChecked(True)
    _resize(qapp, toolbar, 90)

    action = [a for a in toolbar.build_overflow_menu().actions() if a.text() == "Action 5"][0]
    assert action.isCheckable() and action.isChecked()


def test_overflow_button_stays_hidden_when_only_a_separator_would_spill(qapp):
    bar = OverflowBar(height=28, spacing=0)
    btn = QToolButton()
    btn.setFixedWidth(24)
    bar.add_button(btn, "Only")
    sep = QFrame()
    sep.setFixedWidth(1)
    bar.add_separator(sep)
    bar.show()
    _resize(qapp, bar, 24)
    assert not bar.overflow_btn.isVisible()
