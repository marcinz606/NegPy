"""The step arrows after a slider's value: one step per press, ten with Shift, one commit per release."""

from unittest.mock import patch

from PyQt6.QtCore import Qt

from negpy.desktop.view.widgets.sliders import CompactSlider, KelvinSlider


def _slider(**kwargs) -> tuple[CompactSlider, list[float]]:
    slider = CompactSlider("Grade", 0.0, 5.0, 2.0, step=0.05, **kwargs)
    committed: list[float] = []
    slider.valueCommitted.connect(committed.append)
    return slider, committed


def test_a_press_steps_once_and_the_release_commits(qapp):
    slider, committed = _slider()
    up = slider._step_arrows.layout().itemAt(0).widget()
    up.pressed.emit()
    assert slider.value() == 2.05
    assert committed == []
    up.released.emit()
    assert committed == [2.05]


def test_shift_steps_by_ten_and_the_range_clamps(qapp):
    slider, _ = _slider()
    with patch("negpy.desktop.view.widgets.sliders.QApplication.keyboardModifiers", return_value=Qt.KeyboardModifier.ShiftModifier):
        slider.step(-1)
        assert slider.value() == 1.5
        for _ in range(10):
            slider.step(-1)
    assert slider.value() == 0.0


def test_a_step_marks_the_row_edited(qapp):
    slider, _ = _slider()
    slider.step(1)
    assert not slider._edited_dot.isHidden()


def test_arrows_show_only_under_the_pointer_and_never_when_disabled(qapp):
    slider, _ = _slider()
    assert slider._step_arrows.isHidden()
    slider.enterEvent(None)
    assert not slider._step_arrows.isHidden()
    slider.leaveEvent(None)
    assert slider._step_arrows.isHidden()
    slider.setEnabled(False)
    slider.enterEvent(None)
    assert slider._step_arrows.isHidden()


def test_a_kelvin_step_moves_by_its_own_step(qapp):
    slider = KelvinSlider("Temperature")
    slider.step(1)
    assert slider.value() == 5550.0


def test_the_value_field_is_as_wide_as_its_widest_value(qapp):
    for slider in (
        KelvinSlider("Temperature"),
        CompactSlider("Cyan", -1.0, 1.0, 0.0),
        CompactSlider("Grade", 0, 200, 110, step=1.0, unit=" R"),
    ):
        spin = slider.spin
        fm = spin.fontMetrics()
        widest = max(
            fm.horizontalAdvance(f"{spin.prefix()}{spin.textFromValue(v)}{spin.suffix()}") for v in (spin.minimum(), spin.maximum())
        )
        assert widest < spin.sizeHint().width() <= widest + 12


def test_a_step_finer_than_the_shown_decimals_still_moves(qapp):
    slider = CompactSlider("Feather", 0.0, 0.15, 0.04, step=0.005, precision=1000)
    slider.step(1)
    assert slider.value() == 0.05


def test_a_log_slider_steps_over_the_gap_below_its_floor(qapp):
    from negpy.desktop.view.widgets.sliders import LogSlider

    slider = LogSlider("Capture Delay", 0, 10, 0, log_floor=0.1, unit=" s")
    slider.step(1)
    assert slider.value() == 0.1
    slider.step(-1)
    assert slider.value() == 0.0


def test_a_held_arrow_stays_shown_when_the_pointer_leaves(qapp):
    slider, _ = _slider()
    slider.enterEvent(None)
    up = slider._step_arrows.layout().itemAt(0).widget()
    up.setDown(True)
    slider.leaveEvent(None)
    assert not slider._step_arrows.isHidden()
    up.setDown(False)
    up.released.emit()
    assert slider._step_arrows.isHidden()
