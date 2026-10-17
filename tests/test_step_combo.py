from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
import pytest

from negpy.desktop.view.widgets.step_combo import StepCombo


def _combo(*labels: str) -> StepCombo:
    combo = StepCombo()
    combo.addItems(labels)
    combo.resize(240, combo.height())
    return combo


def _click(combo: StepCombo, where: str) -> None:
    x = {"prev": 4, "face": combo.width() // 2, "next": combo.width() - 4}[where]
    QTest.mouseClick(combo, Qt.MouseButton.LeftButton, pos=QPoint(x, combo.height() // 2))


def _record(combo: StepCombo) -> tuple[list, list]:
    changed, activated = [], []
    combo.currentIndexChanged.connect(changed.append)
    combo.activated.connect(activated.append)
    return changed, activated


def test_the_arrows_step_to_the_neighbor_choice_and_emit_like_a_pick(qapp):
    combo = _combo("Free", "1:1", "3:2")
    combo.setCurrentIndex(1)
    changed, activated = _record(combo)

    _click(combo, "next")
    assert combo.currentText() == "3:2"
    _click(combo, "prev")
    _click(combo, "prev")
    assert combo.currentText() == "Free"
    assert changed == [2, 1, 0]
    assert activated == [2, 1, 0]


def test_the_arrows_skip_headers_separators_and_grayed_choices(qapp):
    combo = StepCombo()
    combo.resize(240, combo.height())
    combo.addItem("Digital (Adobe RGB)")
    combo.insertSeparator(1)
    combo.add_header("Paper")
    combo.addItem("Kodak Endura Premier")
    combo.model().item(3).setEnabled(False)
    combo.addItem("Kodak Vision 2383")
    combo.setCurrentIndex(0)

    _click(combo, "next")
    assert combo.currentText() == "Kodak Vision 2383"
    _click(combo, "prev")
    assert combo.currentText() == "Digital (Adobe RGB)"


def test_the_arrows_stop_at_the_ends(qapp):
    combo = _combo("Free", "1:1")
    changed, activated = _record(combo)

    _click(combo, "prev")
    assert combo.currentIndex() == 0
    combo.setCurrentIndex(1)
    _click(combo, "next")
    assert combo.currentIndex() == 1
    assert changed == [1]
    assert activated == []


def test_a_click_on_the_face_opens_the_list(qapp):
    combo = _combo("Free", "1:1")
    combo.show()

    _click(combo, "face")

    assert combo.view().isVisible()
    assert combo.currentIndex() == 0
    combo.hidePopup()


def test_a_header_never_becomes_current(qapp):
    combo = StepCombo()
    changed, _ = _record(combo)

    combo.add_header("Digital")
    combo.addItem("Digital (Adobe RGB)")

    assert combo.currentIndex() == -1
    assert changed == []


@pytest.mark.parametrize("locale", ["zh_CN", "ja", "ko"])
def test_export_ratio_arrows_keep_stored_values_in_translated_ui(qapp, locale: str) -> None:
    from negpy.desktop.view.widgets.export_settings_form import ExportSettingsForm
    from negpy.kernel.system import i18n

    previous = i18n.language()
    try:
        i18n.configure(locale)
        form = ExportSettingsForm()
        combo = form.ratio_combo
        assert form.values()["paper_aspect_ratio"] == "Original"

        _click(combo, "next")
        assert combo.currentText() != "Free"
        assert form.values()["paper_aspect_ratio"] == "Free"

        _click(combo, "next")
        assert form.values()["paper_aspect_ratio"] == "1:1"

        _click(combo, "prev")
        _click(combo, "prev")
        assert combo.currentText() != "Original"
        assert form.values()["paper_aspect_ratio"] == "Original"
    finally:
        i18n.configure(previous)
