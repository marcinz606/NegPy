from PyQt6.QtWidgets import QHBoxLayout

from conftest import FakeController, FakeRepo
from negpy.desktop.view.sidebar.controls_panel import ControlsPanel
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.desktop.view.widgets.sliders import CompactSlider


def test_no_row_holds_two_sliders(qapp):
    panel = ControlsPanel(FakeController(FakeRepo()))
    # Cards sit in RightPanel's pages, not under the panel.
    sections = [v for v in vars(panel).values() if isinstance(v, CollapsibleSection)]
    assert len(sections) > 10
    crowded = []
    for section in sections:
        for row in section.findChildren(QHBoxLayout):
            items = [row.itemAt(i).widget() for i in range(row.count())]
            sliders = [w for w in items if isinstance(w, CompactSlider)]
            if len(sliders) > 1:
                crowded.append([s.label.text() for s in sliders])
    assert crowded == []
