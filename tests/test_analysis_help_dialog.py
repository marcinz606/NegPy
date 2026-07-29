"""The ⓘ in the Analysis header and the guide it opens."""

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest

from negpy.desktop.view.shortcut_registry import REGISTRY
from negpy.desktop.view.widgets.analysis_help_dialog import AnalysisHelpDialog
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.desktop.view.widgets.stats import PROBE_TOOLTIP, STAT_TOOLTIPS


def test_the_guide_covers_every_read_out_in_the_panel() -> None:
    """One heading per widget stacked in the Analysis section; a dropped topic leaves that
    part of the panel unexplained with nothing else in the UI to explain it."""
    dlg = AnalysisHelpDialog()
    text = dlg.body.toPlainText()

    for topic in ("Photometric curve", "histograms", "LIN / LOG", "Clipping", "Step wedge", "Zone strip", "Probe", "Negative stats"):
        assert topic in text, f"the guide never mentions {topic!r}"


def test_the_probe_and_stats_prose_is_the_tooltip_prose() -> None:
    """Imported, not retyped: the tooltip and the guide must not drift apart."""
    dlg = AnalysisHelpDialog()
    text = dlg.body.toPlainText()

    assert PROBE_TOOLTIP in text
    for name, tip in STAT_TOOLTIPS.items():
        assert name in text
        assert tip in text


def test_the_info_button_is_opt_in() -> None:
    """CollapsibleSection backs ~12 sections; only Analysis asks for the button."""
    assert CollapsibleSection("Plain").info_btn is None
    assert CollapsibleSection("Analysis", info=True).info_btn is not None


def test_clicking_info_asks_for_help_without_collapsing_the_section() -> None:
    """The button is nested inside the header's toggle button, so a click that leaked
    through would fold the panel shut behind the dialog."""
    section = CollapsibleSection("Analysis", expanded=True, info=True)
    section.show()
    requests: list[int] = []
    section.info_requested.connect(lambda: requests.append(1))

    assert section.info_btn is not None
    QTest.mouseClick(section.info_btn, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(10, 10))

    assert requests == [1]
    assert section.toggle_button.isChecked()


def test_the_guide_is_a_bindable_action() -> None:
    entry = REGISTRY["show_analysis_help"]
    assert entry.category == "Help"
    assert entry.default_key == ""  # no obvious free key; assignable in the editor
