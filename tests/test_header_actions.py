import pytest
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from negpy.desktop.view.styles.templates import header_actions, labeled_action, section_subheader


@pytest.mark.parametrize("count", [1, 2])
def test_actions_take_a_quarter_of_the_row_each(qapp, count):
    host = QWidget()
    col = QVBoxLayout(host)
    col.setContentsMargins(0, 0, 0, 0)
    buttons = [labeled_action("fa5s.redo", f"Action {i}", "tip") for i in range(count)]
    col.addLayout(header_actions(section_subheader("ANALYSIS"), *buttons))
    host.resize(400, 80)
    host.show()
    qapp.processEvents()
    for btn in buttons:
        assert btn.width() == pytest.approx(100, abs=8)
        assert btn.geometry().right() <= 400
    assert buttons[-1].geometry().right() == pytest.approx(399, abs=2)


def test_a_narrow_row_keeps_every_label(qapp):
    host = QWidget()
    col = QVBoxLayout(host)
    col.setContentsMargins(0, 0, 0, 0)
    buttons = [labeled_action("fa5s.redo", "Reanalyze", "tip"), labeled_action("fa5s.crosshairs", "Use Frame", "tip")]
    col.addLayout(header_actions(section_subheader("ROLLS"), *buttons))
    host.resize(300, 80)
    host.show()
    qapp.processEvents()
    for btn in buttons:
        assert btn.width() >= btn.sizeHint().width()
        assert not btn._compact()
