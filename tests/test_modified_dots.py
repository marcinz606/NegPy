from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.controls_panel import ControlsPanel
from negpy.features.process.models import ProcessMode
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG


def _panel():
    controller = MagicMock()
    controller.state = AppState()
    # An untouched file's config, not AppState()'s own bare WorkspaceConfig() placeholder --
    # crosstalk_strength (and friends) differ between the two (see DEFAULT_WORKSPACE_CONFIG).
    controller.state.config = DEFAULT_WORKSPACE_CONFIG
    return controller, ControlsPanel(controller)


def test_calibration_edits_light_their_section(qapp):
    """Calibration owns fields on ProcessConfig, so it was never counted — its header
    showed no count and its reset button stayed hidden forever."""
    controller, panel = _panel()
    panel._sync_modified_dots()
    assert panel.sensor_section.modified_count == 0
    assert not panel.sensor_section.reset_btn.isVisible()

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, crosstalk_strength=0.4, hue_trim=3.0))
    panel._sync_modified_dots()

    assert panel.sensor_section.modified_count == 2


def test_flat_field_and_lens_edits_light_the_optics_section(qapp):
    controller, panel = _panel()
    panel._sync_modified_dots()
    assert panel.optics_section.modified_count == 0

    cfg = controller.state.config
    controller.state.config = replace(
        cfg,
        flatfield=replace(cfg.flatfield, apply=True, profile_id="rig-1"),
        geometry=replace(cfg.geometry, distortion_k1=0.05),
    )
    panel._sync_modified_dots()

    assert panel.optics_section.modified_count == 3


def test_calibration_fields_are_not_double_counted_in_process(qapp):
    """The two sections share ProcessConfig; a Calibration edit must not also inflate
    the Process count."""
    controller, panel = _panel()
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, crosstalk_strength=0.4))
    panel._sync_modified_dots()

    assert panel.sensor_section.modified_count == 1
    assert panel.process_section.modified_count == 0


def test_calibration_reset_button_restores_its_fields(qapp):
    """The header's reset button was shown by the count above but never connected, so it
    was a control that did nothing."""
    controller, panel = _panel()
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, crosstalk_strength=0.4, hue_trim=3.0))

    panel.sensor_section.reset_requested.emit()

    applied = controller.apply_config.call_args[0][0]
    assert applied.process.crosstalk_strength == cfg.process.crosstalk_strength
    assert applied.process.hue_trim == cfg.process.hue_trim


def test_optics_reset_restores_both_of_its_roll_cards(qapp):
    """Lens and Flat Field are roll cards, so the reset goes out the same door an edit
    does and each card's lock follows it."""
    controller, panel = _panel()
    cfg = controller.state.config
    controller.state.config = replace(
        cfg,
        flatfield=replace(cfg.flatfield, apply=True, profile_id="rig-1"),
        geometry=replace(cfg.geometry, distortion_k1=0.05, fine_rotation=1.5),
    )

    panel.optics_section.reset_requested.emit()

    calls = {c.args[0]: c.kwargs for c in controller.set_roll_default.call_args_list}
    assert calls["flatfield"] == {"apply": False, "profile_id": ""}
    assert calls["lens"]["distortion_k1"] == cfg.geometry.distortion_k1
    assert "fine_rotation" not in calls["lens"]


def test_crop_reset_restores_its_own_card(qapp):
    controller, panel = _panel()
    cfg = controller.state.config
    controller.state.config = replace(cfg, geometry=replace(cfg.geometry, autocrop_offset=9, distortion_k1=0.05, fine_rotation=1.5))

    panel.autocrop_section.reset_requested.emit()
    card, changes = controller.set_roll_default.call_args[0][0], controller.set_roll_default.call_args[1]
    assert card == "autocrop"
    assert changes["autocrop_offset"] == cfg.geometry.autocrop_offset
    assert "distortion_k1" not in changes


def test_geometry_reset_leaves_the_roll_scoped_cards_alone(qapp):
    controller, panel = _panel()
    cfg = controller.state.config
    controller.state.config = replace(cfg, geometry=replace(cfg.geometry, fine_rotation=1.5, autocrop_offset=9, distortion_k1=0.05))

    panel.geometry_section.reset_requested.emit()

    applied = controller.apply_config.call_args[0][0]
    assert applied.geometry.fine_rotation == cfg.geometry.fine_rotation
    assert applied.geometry.autocrop_offset == 9
    assert applied.geometry.distortion_k1 == 0.05


def test_transparency_at_true_default_shows_color_unmodified(qapp):
    """cast_removal_strength's real default on a slide is 0 (cast_removal_for_mode), not
    the bare ExposureConfig 0.5 — an untouched transparency must not show as modified."""
    controller, panel = _panel()
    cfg = controller.state.config
    controller.state.config = replace(
        cfg,
        process=replace(cfg.process, process_mode=ProcessMode.E6),
        exposure=replace(cfg.exposure, cast_removal_strength=0.0),
    )
    panel._sync_modified_dots()

    assert panel.color_section.modified_count == 0


def test_color_reset_zeroes_cast_removal_on_transparency(qapp):
    """Resetting Color on a slide must land on cast_removal_for_mode's default (0), not
    the flat ExposureConfig default (0.5), or the reset reintroduces a gray-balance the
    live render never wants on a transparency."""
    controller, panel = _panel()
    cfg = controller.state.config
    controller.state.config = replace(
        cfg,
        process=replace(cfg.process, process_mode=ProcessMode.E6),
        exposure=replace(cfg.exposure, cast_removal_strength=0.0, wb_cyan=0.3),
    )

    panel.color_section.reset_requested.emit()

    new_config = controller.session.update_config.call_args[0][0]
    assert new_config.exposure.cast_removal_strength == 0.0
    assert new_config.exposure.wb_cyan == 0.0


def test_raw_decode_counts_and_resets_highlight_recovery(qapp):
    controller, panel = _panel()
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, highlight_reconstruction=2))
    panel._sync_modified_dots()
    assert panel.demosaic_section.modified_count == 1

    panel.demosaic_section.reset_requested.emit()

    applied = controller.apply_config.call_args[0][0]
    assert applied.process.highlight_reconstruction == cfg.process.highlight_reconstruction


def test_roll_analysis_counts_the_averages_and_the_picked_roll_not_metering(qapp):
    controller, panel = _panel()
    cfg = controller.state.config
    controller.state.config = replace(
        cfg, process=replace(cfg.process, use_luma_average=not cfg.process.use_luma_average, roll_name="Tri-X")
    )
    panel._sync_modified_dots()

    assert panel.baseline_section.modified_count == 2
    assert panel.process_section.modified_count == 0

    panel.baseline_section.reset_requested.emit()
    applied = controller.apply_config.call_args[0][0]
    assert applied.process.use_luma_average == cfg.process.use_luma_average
