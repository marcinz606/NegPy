from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.geometry import GeometrySidebar
from negpy.features.geometry.models import AutocropMode


def _sidebar():
    controller = MagicMock()
    controller.state = AppState()
    return GeometrySidebar(controller), controller


def test_rebate_trim_slider_spans_its_whole_range(qapp):
    # CompactSlider's `precision` is an int multiplier, not decimal places. Passing 0
    # collapses the QSlider to [0, 0]: a locked handle under a correct-looking readout.
    sidebar, _ = _sidebar()

    assert sidebar.rebate_trim_slider.slider.minimum() == 0
    assert sidebar.rebate_trim_slider.slider.maximum() == 150

    sidebar.rebate_trim_slider.setValue(100.0)
    assert sidebar.rebate_trim_slider.slider.value() == 100
    assert sidebar.rebate_trim_slider.value() == 100.0


def test_rebate_trim_slider_reflects_the_stored_fraction(qapp):
    sidebar, controller = _sidebar()
    cfg = controller.state.config
    controller.state.config = replace(cfg, geometry=replace(cfg.geometry, autocrop_rebate_trim=1.25))

    sidebar.sync_ui()

    assert sidebar.rebate_trim_slider.value() == 125.0


def test_rebate_trim_commit_stores_a_fraction_not_a_percentage(qapp):
    sidebar, controller = _sidebar()

    sidebar._on_rebate_trim_committed(150.0)

    saved = controller.session.update_config.call_args[0][0]
    assert saved.geometry.autocrop_rebate_trim == 1.5


def test_rebate_trim_slider_follows_the_crop_mode(qapp):
    sidebar, controller = _sidebar()
    cfg = controller.state.config

    controller.state.config = replace(cfg, geometry=replace(cfg.geometry, autocrop_mode=AutocropMode.FILM))
    sidebar.sync_ui()
    assert sidebar.rebate_trim_slider.isEnabled() is False

    controller.state.config = replace(cfg, geometry=replace(cfg.geometry, autocrop_mode=AutocropMode.IMAGE))
    sidebar.sync_ui()
    assert sidebar.rebate_trim_slider.isEnabled() is True
