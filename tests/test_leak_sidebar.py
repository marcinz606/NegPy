from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from negpy.desktop.session import AppState, ToolMode
from negpy.desktop.view.sidebar.leak import LeakSidebar
from negpy.features.leak.models import LeakConfig
from negpy.features.leak.shapes import new_band
from negpy.features.process.models import ProcessConfig, ProcessMode


def _sidebar():
    controller = MagicMock()
    controller.state = AppState()
    return controller, LeakSidebar(controller)


def test_frame_sliders_drag_into_the_config(qapp):
    controller, sb = _sidebar()
    controller.state.config = replace(controller.state.config, leak=LeakConfig(leak_remove=True))
    sb.sync_ui()
    for slider, value, field, expected in (
        (sb.dmax_slider, 1.5, "leak_dmax", 1.5),
        (sb.grain_slider, 20.0, "leak_grain", 0.2),
        (sb.contrast_slider, 60.0, "leak_contrast", 0.6),
    ):
        assert slider.isEnabled()
        slider.slider.setSliderDown(True)
        slider.slider.setValue(slider._to_int(value))
        slider.slider.setSliderDown(False)
        config = controller.apply_config.call_args.args[0]
        assert getattr(config.leak, field) == pytest.approx(expected)


def test_the_selected_leak_takes_its_own_strength_falloff_and_color(qapp):
    from negpy.features.leak.models import LeakColor
    from negpy.features.leak.shapes import new_edge, new_spot

    controller, sb = _sidebar()
    shapes = (new_band((0.7, 0.1), (0.7, 0.9)), new_edge((0.1, 0.1), (0.9, 0.1)), new_spot((0.5, 0.5), (0.6, 0.5), (0.5, 0.6)))
    controller.state.config = replace(controller.state.config, leak=LeakConfig(leak_remove=True, leak_shapes=shapes))
    controller.state.leak_selected = -1
    sb.sync_ui()
    assert sb.leak_list.count() == 3
    assert not sb.strength_slider.isEnabled() and not sb.color_btn.isEnabled()
    controller.state.leak_selected = 1
    sb.sync_ui()
    assert sb.strength_slider.isEnabled() and not sb.falloff_slider.isEnabled(), "an edge's fade is the film's"
    controller.state.leak_selected = 2
    sb.sync_ui()
    assert sb.falloff_slider.isEnabled()
    sb.strength_slider.slider.setSliderDown(True)
    sb.strength_slider.slider.setValue(sb.strength_slider._to_int(250.0))
    sb.strength_slider.slider.setSliderDown(False)
    assert controller.update_selected_leak.call_args.kwargs == {"strength": 2.5}
    sb.color_btn.setCurrentIndex(1)
    controller.update_selected_leak.assert_called_with(color=LeakColor.BACK)
    sb.trim_slider.slider.setSliderDown(True)
    sb.trim_slider.slider.setValue(sb.trim_slider._to_int(20.0))
    sb.trim_slider.slider.setSliderDown(False)
    assert controller.update_selected_leak.call_args.kwargs == {"trim": pytest.approx(0.2)}


def test_black_and_white_film_hides_the_color_choice(qapp):
    from negpy.features.leak.models import LeakColor

    controller, sb = _sidebar()
    band = replace(new_band((0.7, 0.1), (0.7, 0.9)), color=LeakColor.BACK)
    controller.state.config = replace(controller.state.config, leak=LeakConfig(leak_remove=True, leak_shapes=(band,)))
    controller.state.leak_selected = 0
    sb.sync_ui()
    assert not sb.color_row.isHidden()
    assert not sb.trim_group.isHidden()
    controller.state.config = replace(controller.state.config, process=ProcessConfig(process_mode=ProcessMode.BW))
    sb.sync_ui()
    assert sb.color_row.isHidden() and sb.trim_group.isHidden()


def test_header_counts_leaks_and_the_buttons_reach_the_controller(qapp):
    controller, sb = _sidebar()
    controller.state.config = replace(controller.state.config, leak=LeakConfig(leak_shapes=(new_band((0.7, 0.1), (0.7, 0.9)),)))
    sb.sync_ui()
    assert sb.leaks_subheader.text() == "LEAKS · 1"
    assert sb.undo_btn.isEnabled() and sb.clear_btn.isEnabled()
    sb.band_btn.click()
    controller.set_active_tool.assert_called_with(ToolMode.LEAK_BAND)
    sb.edge_btn.click()
    controller.set_active_tool.assert_called_with(ToolMode.LEAK_EDGE)
    sb.spot_btn.click()
    controller.set_active_tool.assert_called_with(ToolMode.LEAK_SPOT)
    sb.paint_btn.click()
    controller.set_active_tool.assert_called_with(ToolMode.LEAK_PAINT)
    sb.auto_btn.click()
    controller.auto_leaks.assert_called_once()
    sb.roll_btn.click()
    controller.leaks_from_roll.assert_called_once()
    sb.undo_btn.click()
    controller.undo_last_leak.assert_called_once()


def test_the_burned_hint_follows_the_last_render(qapp):
    controller, sb = _sidebar()
    controller.state.config = replace(controller.state.config, leak=LeakConfig(leak_remove=True))
    controller.state.leak_burned = True
    sb.sync_ui()
    assert sb.burned_hint.isVisibleTo(sb)
    controller.state.leak_burned = False
    sb.sync_ui()
    assert not sb.burned_hint.isVisibleTo(sb)
    sb.burned_btn.click()
    controller.set_leak_show_burned.assert_called_with(True)


def test_a_slide_takes_it_and_a_positive_grays_the_card_out_under_a_hint(qapp):
    controller, sb = _sidebar()
    controller.state.config = replace(controller.state.config, process=ProcessConfig(process_mode=ProcessMode.E6))
    sb.sync_ui()
    assert sb.leak_btn.isEnabled() and not sb.unsupported_hint.isVisibleTo(sb)
    controller.state.config = replace(controller.state.config, process=ProcessConfig(process_mode=ProcessMode.E6, positive_source=True))
    sb.sync_ui()
    assert not sb.leak_btn.isEnabled() and not sb.band_btn.isEnabled()
    assert sb.unsupported_hint.isVisibleTo(sb)


def test_recover_layers_reaches_the_config(qapp):
    controller, sb = _sidebar()
    controller.state.config = replace(controller.state.config, leak=LeakConfig(leak_remove=True))
    sb.sync_ui()
    assert sb.recover_btn.isChecked() and sb.recover_btn.isEnabled()
    sb.recover_btn.click()
    config = controller.apply_config.call_args.args[0]
    assert config.leak.leak_recover is False
