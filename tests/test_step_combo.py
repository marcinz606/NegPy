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


def test_the_arrows_wrap_past_the_ends(qapp):
    combo = StepCombo()
    combo.resize(240, combo.height())
    combo.add_header("Ratio")
    combo.addItems(["Free", "1:1", "3:2"])
    combo.setCurrentIndex(1)
    changed, activated = _record(combo)

    _click(combo, "prev")
    assert combo.currentText() == "3:2"
    _click(combo, "next")
    assert combo.currentText() == "Free"
    assert changed == [3, 1]
    assert activated == [3, 1]


def test_the_arrows_do_nothing_with_one_choice(qapp):
    combo = _combo("Free")
    changed, activated = _record(combo)

    _click(combo, "prev")
    _click(combo, "next")

    assert combo.currentIndex() == 0
    assert changed == activated == []


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


@pytest.mark.parametrize("locale", ["zh_CN", "ja", "ko", "de"])
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


def test_the_face_sits_on_the_track_surface_with_short_dividers(qapp):
    """The dropdown takes a segmented control's look: bg_input inside a border_primary edge."""
    from PyQt6.QtGui import QColor

    from negpy.desktop.view.styles.theme import THEME

    combo = _combo("A", "B", "C")
    combo.setCurrentIndex(1)
    image = combo.grab().toImage()
    cells = combo._cells()
    x, h = int(cells["prev"].right()), combo.height()
    track = QColor(THEME.bg_input)

    assert image.pixelColor(int(cells["prev"].center().x()), 3) == track
    assert image.pixelColor(x, 3) == track and image.pixelColor(x - 1, 3) == track, "the divider stops short of the edge"
    assert track not in (image.pixelColor(x, h // 2), image.pixelColor(x - 1, h // 2)), "the divider shows at mid height"
    assert image.pixelColor(int(cells["prev"].center().x()), 0) == QColor(THEME.border_primary)


def test_a_setting_stepper_stops_at_the_ends_and_opens_no_list(qapp):
    from negpy.desktop.view.sidebar.live_view_window import SettingStepper

    stepper = SettingStepper()
    stepper.resize(240, stepper.height())
    stepper.addItem("1/60", "1/60")
    stepper.addItem("1/125", "1/125")
    _changed, activated = _record(stepper)

    assert stepper.currentIndex() == 0
    _click(stepper, "prev")
    assert (stepper.currentIndex(), activated) == (0, [])
    _click(stepper, "next")
    _click(stepper, "next")
    assert (stepper.currentIndex(), activated) == (1, [1])
    assert stepper.stepping()

    _click(stepper, "face")
    assert not stepper.view().isVisible()


def test_set_choices_replaces_the_list_quietly_and_a_grayed_choice_is_stepped_past(qapp):
    combo = StepCombo.of(["8-bit"], "Bit depth")
    combo.resize(240, combo.height())
    changed, activated = _record(combo)
    combo.set_choices(["150", "300", "600"], [150, 300, 600])
    assert (changed, combo.currentIndex(), combo.currentData()) == ([], 0, 150)

    combo.set_item_enabled(1, False)
    combo.set_item_tooltip(1, "Over the pixel budget")
    assert not combo.is_item_enabled(1)
    assert combo.itemData(1, Qt.ItemDataRole.ToolTipRole) == "Over the pixel budget"
    _click(combo, "next")
    assert combo.currentIndex() == 2 and activated == [2]
