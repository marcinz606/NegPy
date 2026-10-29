"""Shortcut hints: a mouse edit on a shortcut-bearing control shows its keys once per session."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextDocument
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QPushButton, QWidget

from negpy.desktop.view import shortcut_registry
from negpy.desktop.view.canvas.hud import CanvasHud
from negpy.desktop.view.shortcut_hints import ShortcutHints, _hint_html
from negpy.desktop.view.shortcut_registry import default_bindings, display_key, shortcut_ids_in, tooltip_with_shortcut
from negpy.desktop.view.widgets.sliders import CompactSlider

if not QApplication.instance():
    _app = QApplication(sys.argv)


@pytest.fixture(autouse=True)
def _default_keys(monkeypatch):
    monkeypatch.setattr(shortcut_registry, "current_bindings", default_bindings)


@pytest.fixture
def hints():
    state = SimpleNamespace(shortcut_hints=True)
    shown: list[str] = []
    filt = ShortcutHints(state, shown.append)
    QApplication.instance().installEventFilter(filt)
    yield state, shown
    QApplication.instance().removeEventFilter(filt)


def _button(label: str, action_id: str) -> QPushButton:
    btn = QPushButton(label)
    btn.setToolTip(tooltip_with_shortcut("Pick a gray spot.", action_id))
    btn.show()
    return btn


def test_a_tooltip_carries_its_bound_ids_and_renders_without_them() -> None:
    tip = tooltip_with_shortcut("Warmer or cooler.", ["temp_warm", "temp_cool", "toggle_auto_density"])
    assert shortcut_ids_in(tip) == ["temp_warm", "temp_cool"]
    doc = QTextDocument()
    doc.setHtml(tip)
    assert "shortcut" not in doc.toPlainText()
    assert shortcut_ids_in(tooltip_with_shortcut("No key.", "toggle_auto_density")) == []


def test_a_mouse_click_shows_the_keys_once_per_session(hints) -> None:
    _state, shown = hints
    btn = _button("WB Picker", "pick_wb")
    QTest.mouseClick(btn, Qt.MouseButton.LeftButton)
    QTest.mouseClick(btn, Qt.MouseButton.LeftButton)
    assert len(shown) == 1
    assert "WB Picker" in shown[0] and display_key("Shift+W") in shown[0]


def test_the_keyboard_and_the_off_setting_show_no_hint(hints) -> None:
    state, shown = hints
    btn = _button("WB Picker", "pick_wb")
    btn.click()
    QTest.keyClick(btn, Qt.Key.Key_Space)
    state.shortcut_hints = False
    QTest.mouseClick(btn, Qt.MouseButton.LeftButton)
    assert shown == []


def test_a_slider_hint_names_the_slider_and_both_keys(hints) -> None:
    _state, shown = hints
    slider = CompactSlider("Temperature", 2000.0, 12000.0, 5500.0, step=50.0)
    slider.setToolTip(tooltip_with_shortcut("Warmer or cooler.", ["temp_warm", "temp_cool"]))
    slider.show()
    QTest.mouseClick(slider.slider, Qt.MouseButton.LeftButton)
    assert len(shown) == 1
    assert "Temperature" in shown[0] and ">T<" in shown[0] and ">G<" in shown[0]


def test_a_long_hint_stays_on_one_line_in_the_strip() -> None:
    parent = QWidget()
    parent.resize(1200, 400)
    hud = CanvasHud(parent)
    hud.resize(1200, hud.minimumHeight())
    heights = []
    for name in ("Toe", "Highlights Density"):
        hud.showMessage(_hint_html(name, ["Alt+Shift+T", "Alt+T"]), 1000)
        hud.layout().activate()
        heights.append(hud.message.heightForWidth(hud.message.width()))
    assert heights[0] == heights[1]
