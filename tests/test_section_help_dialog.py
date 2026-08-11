"""The ⓘ in a section header and the guide it renders out of docs/USER_GUIDE.md."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QWidget

from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.desktop.view.widgets.section_help_dialog import SectionHelpDialog, _guides, guide_markdown, has_guide

# The Analysis read-out plus every section ControlsPanel builds. A key here with no
# marker in the doc silently drops that panel's ⓘ, which nothing else would catch.
GUIDED_KEYS = (
    "analysis",
    "presets",
    "sensor",
    "process",
    "roll",
    "geometry",
    "flatfield",
    "colour",
    "tone",
    "local",
    "lab",
    "altproc",
    "toning",
    "retouch",
    "finish",
)


def test_every_panel_key_resolves_to_a_guide() -> None:
    assert set(_guides()) == set(GUIDED_KEYS)
    for key in GUIDED_KEYS:
        assert has_guide(key), f"no <!-- panel:{key} --> marker above a heading in USER_GUIDE.md"


def test_a_slice_stops_at_the_next_section() -> None:
    """Slices end at the next same-or-higher heading, so a panel can't show its neighbour's
    controls; the #### topics inside §3 are deeper and must stay in."""
    assert "Roll Analysis" not in guide_markdown("process")
    assert "Setup tab" not in guide_markdown("analysis")
    assert "Step wedge" in guide_markdown("analysis")


def test_cross_doc_links_are_flattened_to_their_text() -> None:
    """Qt paints anchors in the app's accent red, unreadable at body size, and the modal has
    nowhere to navigate to anyway."""
    # The CROSSTALK.md links live in the Calibration section, which owns the crosstalk
    # matrix; Process only cross-references it in prose.
    sensor = guide_markdown("sensor")

    assert "CROSSTALK.md" in sensor
    for key in ("sensor", "process"):
        assert "](" not in guide_markdown(key), f"unflattened link in the {key} guide"


def test_the_analysis_guide_still_covers_every_read_out() -> None:
    """One topic per widget stacked in the Analysis section; a dropped topic leaves that
    part of the panel unexplained with nothing else in the UI to explain it."""
    text = guide_markdown("analysis")

    for topic in ("Photometric curve", "histograms", "LIN / LOG", "Clipping", "Step wedge", "Zone strip", "Probe", "Negative stats"):
        assert topic in text, f"the guide never mentions {topic!r}"


@pytest.mark.parametrize("key", GUIDED_KEYS)
def test_the_dialog_renders_markdown_for_every_panel(key: str) -> None:
    dlg = SectionHelpDialog(key, key.title())
    text = dlg.body.toPlainText()

    assert text.strip()
    assert "###" not in text, "markdown syntax leaked into the rendered body"


def test_the_info_button_is_opt_in() -> None:
    """CollapsibleSection also backs the Export, Metadata and Scan sections, which have no guide."""
    assert CollapsibleSection("Plain").info_btn is None
    assert CollapsibleSection("Analysis", info=True).info_btn is not None


def test_the_guide_is_parented_to_the_section_not_the_panel() -> None:
    """Qt centres a dialog on parent.window(). ControlsPanel is never added to a layout —
    only its pages are — so as the parent it centres the guide on a phantom window at 0,0
    and the guide opens in the screen corner. The section is in the tree."""
    from negpy.desktop.view.sidebar import controls_panel as cp

    panel = SimpleNamespace(controller=SimpleNamespace(session=SimpleNamespace(repo=SimpleNamespace(get_global_setting=lambda _k: None))))
    parents: list[object] = []
    with patch.object(cp, "SectionHelpDialog", lambda k, t, parent: parents.append(parent) or MagicMock()):
        section = cp.ControlsPanel._make_section(panel, "Analysis", "analysis", QWidget())
        section.info_requested.emit()

    assert parents == [section]


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
