from unittest.mock import MagicMock

from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QMouseEvent

from negpy.desktop.view.widgets.sliders import CompactSlider, KelvinSlider


def _label_event(event_type, x: float, button=Qt.MouseButton.LeftButton, buttons=Qt.MouseButton.LeftButton):
    pos = QPointF(x, 5.0)
    return QMouseEvent(event_type, pos, pos, button, buttons, Qt.KeyboardModifier.NoModifier)


def test_adjust_by_emits_change_and_commit(qapp):
    slider = CompactSlider("Density", 0.0, 2.0, 1.0)
    changed = MagicMock()
    committed = MagicMock()
    slider.valueChanged.connect(changed)
    slider.valueCommitted.connect(committed)

    slider.adjust_by(0.1)

    assert slider.value() == 1.1
    changed.assert_called_once_with(1.1)
    committed.assert_called_once_with(1.1)


def test_adjust_by_clamps_to_range(qapp):
    slider = CompactSlider("Density", 0.0, 2.0, 1.0)

    slider.adjust_by(99.0)
    assert slider.value() == 2.0

    slider.adjust_by(-99.0)
    assert slider.value() == 0.0


def test_label_scrub_debounces_value_changes(qapp):
    slider = CompactSlider("Density", 0.0, 2.0, 1.0)
    changed = MagicMock()
    committed = MagicMock()
    slider.valueChanged.connect(changed)
    slider.valueCommitted.connect(committed)

    press = _label_event(QEvent.Type.MouseButtonPress, 0.0)
    move = _label_event(QEvent.Type.MouseMove, 40.0, button=Qt.MouseButton.NoButton)
    release = _label_event(QEvent.Type.MouseButtonRelease, 40.0, buttons=Qt.MouseButton.NoButton)

    assert slider.eventFilter(slider.label, press)
    assert slider.eventFilter(slider.label, move)

    # Scrub moves are coalesced through the debounce timer, not emitted per move.
    changed.assert_not_called()
    assert slider.timer.isActive()
    assert slider.value() == 1.2  # dx=40 * span/400 sensitivity

    assert slider.eventFilter(slider.label, release)
    committed.assert_called_once_with(1.2)
    slider.timer.stop()


def test_kelvin_slider_mired_travel(qapp):
    s = KelvinSlider("Temperature")
    # Slider ints are mired*10: 12000K left, 3000K right (warm on the right).
    assert (s.slider.minimum(), s.slider.maximum()) == (833, 3333)
    assert s.value() == 5500

    s.slider.setValue(3333)
    assert s.value() == 3000
    s.slider.setValue(833)
    assert s.value() == 12000

    s.setValue(12000)
    assert s.slider.value() == 833
    s.setValue(3000)
    assert s.slider.value() == 3333
    s.timer.stop()


def test_kelvin_slider_handle_tracks_temperature(qapp):
    from negpy.desktop.view.widgets.sliders import _kelvin_handle_color

    warm, cool = _kelvin_handle_color(3000.0), _kelvin_handle_color(12000.0)
    assert warm.red() > warm.blue()
    assert cool.blue() > cool.red()

    s = KelvinSlider("Temperature")
    s.setValue(3000)
    warm_qss = s.slider.styleSheet()
    s.setValue(12000)
    assert s.slider.styleSheet() != warm_qss
    s.timer.stop()


def test_kelvin_slider_default_roundtrips_exactly(qapp):
    s = KelvinSlider("Temperature")
    s.setValue(4300)
    s.setValue(5500)
    # The 10K snap keeps the default exact so the edited-state check stays clean.
    assert abs(s.spin.value() - 5500.0) < 1e-6
    assert s.slider.value() == 1818
    s.timer.stop()
