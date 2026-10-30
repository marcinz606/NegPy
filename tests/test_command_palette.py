from types import SimpleNamespace
from unittest.mock import MagicMock

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QListWidget, QVBoxLayout, QWidget

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
    presets = QListWidget()
    presets.addItem("Portra Warm")
    window = QWidget()
    window.right_panel = panel
    window.controls_panel = SimpleNamespace(presets_sidebar=SimpleNamespace(preset_list=presets, apply_btn=MagicMock()))
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
    assert [(e.name, e.where) for e in entries if e.kind == "preset"] == [("Portra Warm", "Preset")]


def test_opening_a_preset_selects_it_and_starts_its_apply(qapp):
    window, _root, _toe = _fake_window()
    command_palette.open_entry(window, make_entry("preset", "Portra Warm", "Preset", "Portra Warm"))
    presets = window.controls_panel.presets_sidebar
    assert presets.preset_list.currentItem().text() == "Portra Warm"
    presets.apply_btn.click.assert_called_once_with()


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


def _real_window():
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = RightPanel(FakeController(FakeRepo()))
    window = QWidget()
    window.right_panel = panel
    window.controls_panel = panel.controls_panel
    window.shortcut_manager = SimpleNamespace(action_for=lambda _id: None)
    window.drawer = SimpleNamespace(isVisible=lambda: True)
    return window, panel


def test_a_channel_card_indexes_every_page_whichever_it_shows(qapp):
    window, panel = _real_window()
    chooser = panel.controls_panel.process_sidebar.ch_btn
    for index in (0, 2):
        chooser.setCurrentIndex(index)
        names = [e.name for e in build_index(window) if e.kind == "slider" and e.name.startswith("Black Point")]
        assert sorted(names) == ["Black Point (Blue)", "Black Point (Global)", "Black Point (Green)", "Black Point (Red)"]
    swapped = {e.name for e in build_index(window) if e.kind == "slider" and "Toe Width" in e.name}
    assert swapped == {"Toe Width (Global)", "Toe Width (Red)", "Toe Width (Green)", "Toe Width (Blue)"}


def test_a_channel_row_shows_its_page_and_writes_its_channel(qapp, monkeypatch):
    window, panel = _real_window()
    sidebar = panel.controls_panel.process_sidebar
    written = []
    sidebar.controller.set_roll_default = lambda _section, **kw: written.append(kw)
    palette = CommandPalette(window)
    palette.query.setText("black point")
    labels = [palette.results.itemWidget(palette.results.item(i)).findChild(CompactSlider).label.text() for i in range(4)]
    assert sorted(labels) == ["Black Point (Blue)", "Black Point (Global)", "Black Point (Green)", "Black Point (Red)"]
    palette.query.setText("black point green")
    entry = palette.results.item(0).data(Qt.ItemDataRole.UserRole)
    assert entry.name == "Black Point (Green)" and sidebar.ch_btn.currentIndex() == 0
    mirror = palette.results.itemWidget(palette.results.item(0)).findChild(CompactSlider)
    mirror.mirror_value(0.1, commit=True)
    assert sidebar.ch_btn.currentIndex() == 2
    assert set(written[-1]) == {"persist", "readback_metrics", "black_point_trim_green"}
    assert abs(written[-1]["black_point_trim_green"] - 0.1) < 1e-9
    monkeypatch.setattr(command_palette.QTimer, "singleShot", staticmethod(lambda _ms, fn: None))
    command_palette.open_entry(window, make_entry("slider", "Black Point", "", sidebar.black_point_slider, (sidebar.ch_btn, 0)))
    assert sidebar.ch_btn.currentIndex() == 0


def test_alt_process_sliders_are_found_on_bw_and_pick_their_process(qapp):
    from dataclasses import replace

    from negpy.features.altprocess.models import AltProcess
    from negpy.features.process.models import ProcessMode

    window, panel = _real_window()
    sidebar = panel.controls_panel.altproc_sidebar
    state = sidebar.state
    sidebar.sync_ui()
    assert not [e for e in build_index(window) if e.name.startswith("Snatch Point")]
    state.config = replace(state.config, process=replace(state.config.process, process_mode=ProcessMode.BW))
    sidebar.sync_ui()
    hits = [e for e in build_index(window) if e.name in ("Snatch Point (Lith)", "Exposure (Lith)", "Exposure (Cyanotype)")]
    assert len(hits) == 3
    written = []
    sidebar.update_config_section = lambda section, **kw: written.append(kw)
    snatch = next(e for e in hits if e.name == "Snatch Point (Lith)")
    command_palette.show_page(snatch)
    assert written[-1]["alt_process"] == AltProcess.LITH


def test_color_sliders_are_found_in_every_region(qapp):
    window, panel = _real_window()
    panel.controls_panel.color_sidebar.sync_ui()
    temps = [e.name for e in build_index(window) if e.name.startswith("Temperature")]
    assert sorted(temps) == ["Temperature (Global)", "Temperature (Highlights)", "Temperature (Shadows)"]


def test_a_paged_slider_keeps_its_synonyms_and_lists_its_pages_in_order(qapp):
    window, panel = _real_window()
    panel.controls_panel.process_sidebar.sync_ui()
    hits = [e.name for e in rank(build_index(window), "blacks") if e.name.startswith("Black Point")]
    assert hits == ["Black Point (Global)", "Black Point (Red)", "Black Point (Green)", "Black Point (Blue)"]


def test_leaving_the_app_closes_find(qapp):
    window, _root, _toe = _fake_window()
    palette = CommandPalette(window)
    palette.show()
    palette._on_app_state(Qt.ApplicationState.ApplicationInactive)
    assert not palette.isVisible()


def test_unpaired_show_and_hide_events_keep_one_app_state_connection(qapp):
    from PyQt6.QtGui import QHideEvent, QShowEvent

    window, _root, _toe = _fake_window()
    palette = CommandPalette(window)
    palette.hideEvent(QHideEvent())
    palette.show()
    palette.showEvent(QShowEvent())
    palette.hide()
    palette.hideEvent(QHideEvent())
    assert not palette._watching_app
    palette.show()
    palette._on_app_state(Qt.ApplicationState.ApplicationInactive)
    assert not palette.isVisible()


def test_a_paged_slider_typed_in_full_ranks_as_exact():
    paged = make_entry("slider", "Black Point (Global)", "Roll › Metering", None, None, None, "Black Point")
    card = make_entry("card", "Black Point", "Roll", None)
    assert rank([card, paged], "black point")[0] is paged


def test_the_tour_sees_find_open(qapp):
    from negpy.desktop.view.widgets import tutorial_steps

    # A popup another test left open would be the active one.
    while (stale := qapp.activePopupWidget()) is not None:
        stale.close()
    window, _root, _toe = _fake_window()
    palette = CommandPalette(window)
    # No event pass: an app-state change queued by an earlier test would close Find.
    palette.show()
    assert tutorial_steps._palette_open(window)
    palette.reject()
    assert not tutorial_steps._palette_open(window)
