from types import SimpleNamespace
from unittest.mock import MagicMock

from negpy.desktop.view.keyboard_shortcuts import ShortcutManager
from negpy.desktop.view.widgets.sliders import CompactSlider, apply_slider_value_visibility
from PyQt6.QtWidgets import QVBoxLayout, QWidget


def test_keyboard_adjust_reports_the_new_value(qapp):
    """The slider may be on a hidden tab and its value box only opens on hover, so an
    inc/dec shortcut used to change the image with no readable feedback at all."""
    slider = CompactSlider("Print Density", -2.0, 2.0, 0.0, unit=" EV")
    window = SimpleNamespace(controller=MagicMock())

    manager = SimpleNamespace(window=window, slider_steps={})
    manager._announce = lambda s, g: ShortcutManager._announce(manager, s, g)
    ShortcutManager._slider_adjuster(manager, lambda: slider, "density_up")()

    assert slider.value() > 0.0
    message = window.controller.set_status.call_args.args[0]
    assert message.startswith("Print Density")
    assert "EV" in message


def test_pinning_opens_every_value_box(qapp):
    root = QWidget()
    layout = QVBoxLayout(root)
    sliders = [CompactSlider("A", 0.0, 1.0, 0.5), CompactSlider("B", 0.0, 1.0, 0.5)]
    for s in sliders:
        layout.addWidget(s)

    assert all(s.spin.maximumWidth() == 0 for s in sliders)

    apply_slider_value_visibility(root, True)
    assert all(s.spin.maximumWidth() > 0 for s in sliders)

    apply_slider_value_visibility(root, False)
    assert all(s.spin.maximumWidth() == 0 for s in sliders)


def test_pinned_slider_keeps_its_value_box_after_the_pointer_leaves(qapp):
    slider = CompactSlider("A", 0.0, 1.0, 0.5)
    slider.set_value_pinned(True)

    slider.leaveEvent(None)

    assert slider.spin.maximumWidth() > 0
