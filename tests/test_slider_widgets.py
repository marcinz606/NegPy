from unittest.mock import MagicMock

from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QMouseEvent

from negpy.desktop.view.styles.theme import THEME
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


def test_slider_keyboard_step_matches_the_declared_step(qapp):
    """The slider's own arrow-key step lives in its internal precision-scaled int
    space. Passing step=0.1 at precision=100 must move the handle by 0.1, not by
    Qt's raw 1-unit default (1/precision = 0.01), which read as no movement at all."""
    slider = CompactSlider("Fine Rotation", -45.0, 45.0, 0.0, step=0.1)
    assert slider.slider.singleStep() == 10  # 0.1 * precision(100)


def test_slider_keyboard_step_scales_with_a_nondefault_precision(qapp):
    slider = CompactSlider("Hue Trim", -30.0, 30.0, 0.0, step=0.5, precision=10)
    assert slider.slider.singleStep() == 5  # 0.5 * precision(10)


def test_integer_slider_keyboard_step_is_unaffected(qapp):
    """precision=1 sliders (ISO, grade points) already worked; the fix must not move them."""
    slider = CompactSlider("Grade", -40.0, 40.0, 0.0, step=5.0, precision=1)
    assert slider.slider.singleStep() == 5


def test_label_scrub_debounces_value_changes(qapp):
    """The handle must never wait on a render.

    Emitting mid-drag pulls the whole per-render UI fan-out onto the handle's own
    thread, so moves are coalesced until the gesture pauses.
    """
    slider = CompactSlider("Density", 0.0, 2.0, 1.0)
    changed = MagicMock()
    committed = MagicMock()
    slider.valueChanged.connect(changed)
    slider.valueCommitted.connect(committed)

    press = _label_event(QEvent.Type.MouseButtonPress, 0.0)
    move = _label_event(QEvent.Type.MouseMove, 40.0, button=Qt.MouseButton.NoButton)
    move_again = _label_event(QEvent.Type.MouseMove, 60.0, button=Qt.MouseButton.NoButton)
    release = _label_event(QEvent.Type.MouseButtonRelease, 60.0, buttons=Qt.MouseButton.NoButton)

    assert slider.eventFilter(slider.label, press)
    assert slider.eventFilter(slider.label, move)
    assert slider.eventFilter(slider.label, move_again)

    changed.assert_not_called()
    assert slider.timer.isActive()
    assert slider.value() == 1.3  # dx=60 * span/400 sensitivity

    assert slider.eventFilter(slider.label, release)
    committed.assert_called_once_with(1.3)
    # The commit renders that value, so the pending frame must not fire again.
    assert not slider.timer.isActive()


def test_commit_flushes_a_pending_frame_when_the_value_returned(qapp):
    """Dragging away and back emits no commit, so the trailing frame is the only
    thing that would restore the preview."""
    slider = CompactSlider("Density", 0.0, 2.0, 1.0)
    changed = MagicMock()
    committed = MagicMock()
    slider.valueChanged.connect(changed)
    slider.valueCommitted.connect(committed)

    slider.slider.setValue(slider._to_int(1.5))
    assert slider.timer.isActive()
    changed.assert_not_called()

    slider.slider.setValue(slider._to_int(1.0))  # back to the committed value
    slider._on_committed()

    committed.assert_not_called()
    changed.assert_called_once_with(1.0)
    assert not slider.timer.isActive()


def test_setvalue_rebases_commit_baseline_across_reuse(qapp):
    """Regression for #crosstalk-separation-not-saved: a widget reused across images
    (setValue() on file switch, e.g. Process sidebar sync_ui()) must rebase its commit
    baseline, or dragging a later image to the same value the widget last committed for
    a *different* image silently no-ops the commit (valueCommitted never fires)."""
    slider = CompactSlider("Separation", 0.0, 1.0, 0.0)
    committed = MagicMock()
    slider.valueCommitted.connect(committed)

    slider.adjust_by(1.0)
    assert slider.value() == 1.0
    committed.assert_called_once_with(1.0)

    # Simulate switching to another image whose saved value happens to also be 0.0.
    slider.setValue(0.0)
    committed.reset_mock()

    # Dragging the new image's slider back up to 1.0 must commit again.
    slider.adjust_by(1.0)
    assert slider.value() == 1.0
    committed.assert_called_once_with(1.0)


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


def test_compact_slider_edited_dot_and_locked_style(qapp):
    slider = CompactSlider("Density", 0.0, 2.0, 1.0)
    assert not slider._edited_dot.isVisibleTo(slider)

    slider.setValue(1.5)  # edited (differs from default)
    assert slider._edited_dot.isVisibleTo(slider)

    slider.setEnabled(False)
    assert THEME.text_muted in slider.label.styleSheet()
    assert not slider._edited_dot.isVisibleTo(slider)

    slider.setEnabled(True)
    assert slider._edited_dot.isVisibleTo(slider)

    slider.setValue(1.0)
    assert not slider._edited_dot.isVisibleTo(slider)


def test_kelvin_slider_locked_handle_is_muted(qapp):
    s = KelvinSlider("Temperature")
    live_qss = s.slider.styleSheet()

    s.setEnabled(False)
    assert THEME.text_muted in s.slider.styleSheet()
    assert s.slider.styleSheet() != live_qss

    s.setEnabled(True)
    assert s.slider.styleSheet() == live_qss
    s.timer.stop()


def test_kelvin_slider_default_roundtrips_exactly(qapp):
    s = KelvinSlider("Temperature")
    s.setValue(4300)
    s.setValue(5500)
    # The 10K snap keeps the default exact so the edited-state check stays clean.
    assert abs(s.spin.value() - 5500.0) < 1e-6
    assert s.slider.value() == 1818
    s.timer.stop()


def test_value_box_prints_a_point_and_reads_a_comma(qapp):
    slider = CompactSlider("Print Density", -2.0, 2.0, 0.5)
    assert slider.spin.text() == "0.50"
    assert slider.spin.valueFromText("1,25") == 1.25


def test_align_slider_columns_lines_up_label_and_value(qapp):
    from PyQt6.QtWidgets import QVBoxLayout, QWidget

    from negpy.desktop.view.widgets.sliders import align_slider_columns

    root = QWidget()
    layout = QVBoxLayout(root)
    short = CompactSlider("Toe", 0.0, 1.0, 0.5)
    long = CompactSlider("Highlights Density", -1.0, 1.0, 0.0, unit=" st")
    layout.addWidget(short)
    layout.addWidget(long)

    align_slider_columns(root)

    assert short.label.minimumWidth() == long.label.minimumWidth() >= long.label.sizeHint().width()
    assert short.spin.minimumWidth() == long.spin.minimumWidth() >= long.spin.sizeHint().width()
