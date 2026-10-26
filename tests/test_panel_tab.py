from PyQt6.QtGui import QColor

from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.panel_tab import PanelTab


def test_an_icon_only_tab_keeps_its_name_for_the_menu_and_find(qapp):
    tab = PanelTab("mdi6.star-outline", "Favorites", icon_only=True)
    assert tab.text() == "Favorites" and not tab.icon().isNull()
    assert tab.minimumWidth() == tab.maximumWidth() == PanelTab.ICON_ONLY_WIDTH


def test_a_page_tab_is_wide_enough_for_its_icon_and_name(qapp):
    assert PanelTab.min_width(["Geometry"], stacked=False) > PanelTab.min_width(["Geometry"], stacked=True)


def test_a_mark_shows_edits_on_the_tab(qapp):
    tab = PanelTab("mdi6.crop", "Geometry")
    assert not tab.is_marked()
    tab.set_marked(True)
    assert tab.is_marked()


def _image(tab, width: int, height: int = 38):
    tab.resize(width, height)
    return tab.grab().toImage()


def test_a_checked_tab_takes_the_warm_chip(qapp):
    tab = PanelTab("mdi6.crop", "Geometry")
    probe = (THEME.space_xs + THEME.space_sm, 19)
    rest = _image(tab, 105).pixelColor(*probe).name()
    tab.setChecked(True)

    assert rest != QColor(THEME.bg_checked).name()
    assert _image(tab, 105).pixelColor(*probe).name() == QColor(THEME.bg_checked).name()


def test_the_edit_dot_stays_inside_the_chip_of_a_narrow_tab(qapp):
    for tab, width in (
        (PanelTab("mdi6.crop", "Geometry"), PanelTab.min_width(["Geometry"], stacked=False)),
        (PanelTab("mdi6.history", "History", icon_only=True), PanelTab.ICON_ONLY_WIDTH),
    ):
        tab.setChecked(True)
        tab.set_marked(True)
        image = _image(tab, width)
        dot = QColor(THEME.mark_edited)
        xs = [
            x
            for x in range(width)
            for y in range(38)
            if max(abs((c := image.pixelColor(x, y)).red() - dot.red()), abs(c.green() - dot.green()), abs(c.blue() - dot.blue())) < 40
        ]

        # The chip's edge runs half a pixel inside space_xs; the dot keeps a pixel clear of it.
        assert xs and max(xs) < width - THEME.space_xs - 2, tab.text()
