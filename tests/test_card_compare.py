"""A held card eye shows the frame without that card, then gives the compare view back."""

from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG


def _controller(**state) -> MagicMock:
    ctrl = MagicMock()
    ctrl.state = AppState(preview_raw=np.zeros((2, 2, 3), np.float32), **state)
    ctrl._card_compare_saved = None
    ctrl._compare_before_key.return_value = "k"
    for name in ("begin_compare_hold", "end_compare_hold", "_restore_card_compare"):
        setattr(ctrl, name, getattr(AppController, name).__get__(ctrl))
    return ctrl


def test_holding_a_card_borrows_the_reference_and_gives_it_back():
    step = replace(DEFAULT_WORKSPACE_CONFIG)
    ctrl = _controller(compare_reference=step, compare_reference_label="STEP 2")
    card_off = replace(DEFAULT_WORKSPACE_CONFIG)

    AppController.begin_card_compare(ctrl, card_off, "WITHOUT TONE")
    assert ctrl.state.compare_hold is True
    assert ctrl.state.compare_reference is card_off
    assert ctrl.state.compare_reference_label == "WITHOUT TONE"
    ctrl._request_compare_baseline.assert_called_once()

    ctrl.state.compare_before = np.zeros((2, 2, 3), np.float32)
    AppController.end_card_compare(ctrl)
    assert ctrl.state.compare_hold is False
    assert ctrl.state.compare_reference is step
    assert ctrl.state.compare_reference_label == "STEP 2"
    assert ctrl.state.compare_before is None


def test_a_refused_hold_leaves_the_reference_alone():
    ctrl = _controller(negative_peek=True)
    AppController.begin_card_compare(ctrl, replace(DEFAULT_WORKSPACE_CONFIG), "WITHOUT TONE")
    assert ctrl.state.compare_hold is False
    assert ctrl.state.compare_reference is None
    assert ctrl._card_compare_saved is None
    AppController.end_card_compare(ctrl)
    assert ctrl.state.compare_reference is None


def test_ending_a_card_hold_in_the_split_renders_its_baseline_again():
    ctrl = _controller(compare_mode=True)
    AppController.begin_card_compare(ctrl, replace(DEFAULT_WORKSPACE_CONFIG), "WITHOUT TONE")
    ctrl._request_compare_baseline.reset_mock()
    AppController.end_card_compare(ctrl)
    ctrl._request_compare_baseline.assert_called_once()


def test_a_card_eye_turns_off_its_own_card_alone(qapp):
    from negpy.desktop.view.sidebar.controls_panel import ControlsPanel

    controller = MagicMock()
    controller.state = AppState()
    controller.active_diptych.return_value = None
    panel = ControlsPanel(controller)
    cfg = DEFAULT_WORKSPACE_CONFIG
    controller.state.config = replace(
        cfg,
        exposure=replace(cfg.exposure, density=cfg.exposure.density + 0.3, grade=cfg.exposure.grade + 1.0),
        lab=replace(cfg.lab, sharpen=cfg.lab.sharpen + 0.5),
    )
    edit = controller.state.config

    tone_off = panel._card_off_config("tone")
    assert tone_off.lab == edit.lab
    assert (tone_off.exposure.density, tone_off.exposure.grade) == (cfg.exposure.density, cfg.exposure.grade)

    lab_off = panel._card_off_config("lab")
    assert lab_off.lab == cfg.lab
    assert lab_off.exposure == edit.exposure

    eye = panel.lab_section.compare_btn
    eye.pressed.emit()
    controller.begin_card_compare.assert_called_once()
    assert controller.begin_card_compare.call_args.args[0].lab == cfg.lab
    eye.released.emit()
    controller.end_card_compare.assert_called_once()
