from types import SimpleNamespace
from unittest.mock import MagicMock

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from conftest import FakeController, FakeRepo
from negpy.desktop.view.sidebar.controls_panel import ControlsPanel
from negpy.desktop.view.widgets import command_palette
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.desktop.view.widgets.command_palette import SYNONYMS, CommandPalette, build_index, make_entry, rank
from negpy.desktop.view.widgets.sliders import CompactSlider


def _names(entries):
    return [e.name for e in entries]


def test_a_synonym_ranks_its_darkroom_names_first():
    entries = [make_entry("slider", n, "Exposure › Tone", None) for n in ("Contrast Mask", "ISO-R Grade", "Toe")]
    assert _names(rank(entries, "contrast")) == ["ISO-R Grade", "Contrast Mask"]


def test_exact_name_beats_prefix_beats_substring():
    entries = [make_entry("slider", n, "Exposure › Tone", None) for n in ("Shoulder Width", "Toe Width", "Toe")]
    assert _names(rank(entries, "toe")) == ["Toe", "Toe Width"]
    assert _names(rank(entries, "width")) == ["Shoulder Width", "Toe Width"]


def test_every_query_word_must_match_and_the_card_counts():
    entries = [make_entry("slider", "Width", "Finish › Finishing", None), make_entry("slider", "Width", "Roll › Crop", None)]
    assert [e.where for e in rank(entries, "width finishing")] == ["Finish › Finishing"]
    assert rank(entries, "   ") == []


def test_a_partial_synonym_already_matches():
    entries = [make_entry("card", "Filtration", "Exposure", None)]
    assert _names(rank(entries, "white bal")) == ["Filtration"]


def test_every_synonym_names_a_real_control_or_card(qapp):
    panel = ControlsPanel(FakeController(FakeRepo()))
    sections = [v for v in vars(panel).values() if isinstance(v, CollapsibleSection)]
    names = {s.title_label.text() for s in sections}
    names |= {slider.label.text() for s in sections for slider in s.findChildren(CompactSlider)}
    missing = sorted({n for targets in SYNONYMS.values() for n in targets} - names)
    assert missing == []


def _fake_window():
    root = QWidget()
    card = CollapsibleSection("Tone")
    body = QWidget()
    body_layout = QVBoxLayout(body)
    toe = CompactSlider("Toe", -1.0, 1.0, 0.0)
    retired = CompactSlider("Toe Width", 0.1, 5.0, 2.5)
    body_layout.addWidget(toe)
    body_layout.addWidget(retired)
    card.set_content(body)
    retired.setVisible(False)
    favourites = QWidget()
    QVBoxLayout(favourites).addWidget(CompactSlider("Toe", -1.0, 1.0, 0.0))
    layout = QVBoxLayout(root)
    layout.addWidget(card)
    layout.addWidget(favourites)
    panel = SimpleNamespace(
        findChildren=root.findChildren,
        tab_path=lambda _w: ["Frame", "Exposure"],
        favourites_sidebar=favourites,
        reveal_widget=MagicMock(),
    )
    manager = SimpleNamespace(action_for=lambda action_id: (lambda: None) if action_id in ("toggle_compare", "density_up") else None)
    window = QWidget()
    window.right_panel = panel
    window.shortcut_manager = manager
    window.drawer = SimpleNamespace(isVisible=lambda: True)
    window._keep = root
    return window, root, toe


def test_index_holds_live_controls_and_actions_only(qapp):
    window, _root, toe = _fake_window()
    entries = build_index(window)
    sliders = [e for e in entries if e.kind == "slider"]
    assert [(e.name, e.where, e.target) for e in sliders] == [("Toe", "Exposure › Tone", toe)]
    assert [(e.kind, e.name) for e in entries if e.kind == "card"] == [("card", "Tone")]
    assert [e.target for e in entries if e.kind == "action"] == ["toggle_compare"]


def test_a_slider_row_drives_the_real_control(qapp):
    window, _root, toe = _fake_window()
    palette = CommandPalette(window)
    palette.query.setText("toe")
    mirror = palette.results.itemWidget(palette.results.item(0)).findChild(CompactSlider)
    mirror.mirror_value(0.4, commit=True)
    assert abs(toe.value() - 0.4) < 1e-9


def test_enter_opens_the_first_hit(qapp, monkeypatch):
    window, _root, toe = _fake_window()
    opened = []
    monkeypatch.setattr(command_palette, "open_entry", lambda _w, entry: opened.append(entry.target))
    monkeypatch.setattr(command_palette.QTimer, "singleShot", staticmethod(lambda _ms, fn: fn()))
    palette = CommandPalette(window)
    palette.query.setText("toe")
    palette.eventFilter(palette.query, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier))
    assert opened == [toe]


def test_opening_a_slider_reveals_and_focuses_it(qapp, monkeypatch):
    window, _root, toe = _fake_window()
    monkeypatch.setattr(command_palette.QTimer, "singleShot", staticmethod(lambda _ms, fn: fn()))
    command_palette.open_entry(window, make_entry("slider", "Toe", "Exposure › Tone", toe))
    window.right_panel.reveal_widget.assert_called_once_with(toe)


def test_opening_an_action_runs_it(qapp):
    ran = []
    window = SimpleNamespace(shortcut_manager=SimpleNamespace(action_for=lambda action_id: lambda: ran.append(action_id)))
    command_palette.open_entry(window, make_entry("action", "Before/after split", "\\", "toggle_compare"))
    assert ran == ["toggle_compare"]
