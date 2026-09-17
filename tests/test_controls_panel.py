"""_sync_roll_locks: each Roll-tab card's lock button, shown only while a roll gives
it something to lock away from. _reset_process_fields: a card's reset scoped to only
the fields it shows.

Stub-on-unbound-method, like test_right_panel_wiring.py: ControlsPanel pulls in every
sidebar in the app, so no test here constructs a real one.
"""

from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.controls_panel import ControlsPanel
from negpy.features.exposure.models import ExposureConfig
from negpy.features.process.models import ProcessConfig


def _panel_stub(*, active_roll_id="roll1", locked_cards=()) -> MagicMock:
    panel = MagicMock()
    panel.film_section = MagicMock()
    panel.sensor_section = MagicMock()
    panel.demosaic_section = MagicMock()
    panel.process_section = MagicMock()
    panel.controller.state.active_roll_id = active_roll_id
    panel.controller.roll_card_locked.side_effect = lambda key: key in locked_cards
    return panel


def test_sync_roll_locks_hides_every_lock_without_an_active_roll():
    panel = _panel_stub(active_roll_id=None)

    ControlsPanel._sync_roll_locks(panel)

    panel.film_section.set_lock_button.assert_called_once_with(False, False)
    panel.sensor_section.set_lock_button.assert_called_once_with(False, False)
    panel.demosaic_section.set_lock_button.assert_called_once_with(False, False)
    panel.process_section.set_lock_button.assert_called_once_with(False, False)


def test_sync_roll_locks_covers_the_film_card():
    """Film Mode locks and shows the same as any other Roll-tab card."""
    panel = _panel_stub(locked_cards={"film"})

    ControlsPanel._sync_roll_locks(panel)

    panel.film_section.set_lock_button.assert_called_once_with(True, True)


def test_sync_roll_locks_sets_each_sections_lock_button():
    panel = _panel_stub(locked_cards={"demosaic"})

    ControlsPanel._sync_roll_locks(panel)

    panel.film_section.set_lock_button.assert_called_once_with(True, False)
    panel.sensor_section.set_lock_button.assert_called_once_with(True, False)
    panel.demosaic_section.set_lock_button.assert_called_once_with(True, True)
    panel.process_section.set_lock_button.assert_called_once_with(True, False)


def test_reset_process_fields_only_touches_the_given_fields():
    """Regression: Normalization's own reset must not also reset Positive (moved
    beside Film Mode) or a Calibration/Demosaic field -- all three cards, and Film
    Mode, live on the same ProcessConfig, but each reset is scoped to its own card."""
    panel = MagicMock()
    panel.controller.state = AppState()
    cfg = panel.controller.state.config
    panel.controller.state.config = replace(
        cfg,
        process=replace(cfg.process, analysis_buffer=0.2, positive_source=True, sensor_profile="Custom"),
    )

    ControlsPanel._reset_process_fields(panel, ("analysis_buffer",))

    new_cfg = panel.controller.apply_config.call_args[0][0]
    assert new_cfg.process.analysis_buffer == ProcessConfig().analysis_buffer
    assert new_cfg.process.positive_source is True
    assert new_cfg.process.sensor_profile == "Custom"


def test_reset_exposure_fields_turns_auto_off_for_a_positive_frame():
    """Regression: Tone's reset used to restore ExposureConfig's own flat default
    (on) regardless of Positive, so resetting a Positive frame turned Auto Density/
    Auto Grade back on instead of to the value auto_meter_for_positive_source gives it."""
    panel = MagicMock()
    panel.controller.state = AppState()
    cfg = panel.controller.state.config
    panel.controller.state.config = replace(
        cfg,
        process=replace(cfg.process, positive_source=True),
        exposure=replace(cfg.exposure, auto_exposure=True, auto_normalize_contrast=True, density=1.4),
    )

    ControlsPanel._reset_exposure_fields(panel, ("auto_exposure", "auto_normalize_contrast", "density"))

    new_cfg = panel.controller.session.update_config.call_args[0][0]
    assert new_cfg.exposure.auto_exposure is False
    assert new_cfg.exposure.auto_normalize_contrast is False
    assert new_cfg.exposure.density == ExposureConfig().density


def test_reset_exposure_fields_turns_auto_on_for_a_negative_frame():
    panel = MagicMock()
    panel.controller.state = AppState()
    cfg = panel.controller.state.config
    panel.controller.state.config = replace(
        cfg,
        process=replace(cfg.process, positive_source=False),
        exposure=replace(cfg.exposure, auto_exposure=False, auto_normalize_contrast=False),
    )

    ControlsPanel._reset_exposure_fields(panel, ("auto_exposure", "auto_normalize_contrast"))

    new_cfg = panel.controller.session.update_config.call_args[0][0]
    assert new_cfg.exposure.auto_exposure is True
    assert new_cfg.exposure.auto_normalize_contrast is True


def test_sync_modified_dots_does_not_flag_a_positive_frames_own_auto_default():
    """A Positive frame with Auto Density/Grade correctly off is at its own default,
    not "modified" -- the Tone header's dot must not count it."""
    panel = MagicMock()
    panel.controller.state = AppState()
    cfg = panel.controller.state.config
    panel.controller.state.config = replace(
        cfg,
        process=replace(cfg.process, positive_source=True),
        exposure=replace(cfg.exposure, auto_exposure=False, auto_normalize_contrast=False),
    )
    panel.tone_section = MagicMock()

    ControlsPanel._sync_modified_dots(panel)

    panel.tone_section.set_modified.assert_called_once_with(0)
