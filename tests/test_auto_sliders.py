from dataclasses import replace

import numpy as np
import pytest

from negpy.features.exposure.auto_sliders import (
    NEUTRAL,
    enable_auto,
    freeze_auto,
    print_shown_values,
    print_stored_value,
)
from negpy.features.exposure.logic import (
    auto_highlight_from_metrics,
    curve_params_from_metrics,
    local_grade_factor_map,
    split_grade_deltas,
)
from negpy.features.exposure.models import ExposureConfig
from negpy.features.exposure.normalization import LogNegativeBounds
from negpy.features.local.models import LocalAdjustmentsConfig, LocalMask
from negpy.features.process.models import ProcessMode
from negpy.features.transparency.logic import (
    transfer_auto_terms,
    transfer_curve_params,
    transfer_shown_values,
    transfer_stored_value,
)

AD, AG = "auto_exposure", "auto_normalize_contrast"
MODE = ProcessMode.C41


def _metrics(**over):
    m = {
        "metered_anchor": 0.52,
        "textural_range": 0.8,
        "norm_density_range": 1.3,
        "shadow_point": 0.93,
        "highlight_point": 0.04,
        "final_bounds": LogNegativeBounds((0.2, 0.5, 0.8), (1.6, 1.9, 2.1)),
    }
    m.update(over)
    return m


def _edited(**over):
    """A frame with trims on top of both autos, and grade couplings a freeze must keep."""
    fields = dict(
        density=1.15,
        grade=105.0,
        highlight_density=0.08,
        shadow_grade=-12.0,
        highlight_grade=6.0,
        shadow_grade_trim_red=4.0,
        grade_trim_blue=-5.0,
    )
    fields.update(over)
    return ExposureConfig(**fields)


def _printed(exposure, metrics):
    """What the print stage applies: curve params, total highlight density, split gains."""
    slopes, pivots, curvs = curve_params_from_metrics(exposure, MODE, metrics)
    hl = exposure.highlight_density + auto_highlight_from_metrics(exposure, MODE, metrics)
    split = split_grade_deltas(
        exposure.grade,
        exposure.shadow_grade,
        exposure.highlight_grade,
        (exposure.shadow_grade_trim_red, exposure.shadow_grade_trim_green, exposure.shadow_grade_trim_blue),
        (exposure.highlight_grade_trim_red, exposure.highlight_grade_trim_green, exposure.highlight_grade_trim_blue),
    )
    return np.array([*slopes, *pivots, *curvs, hl, *split[0], *split[1]])


@pytest.mark.parametrize("shadow_point", [None, 0.93, 0.99])
def test_freezing_both_autos_keeps_the_print(shadow_point):
    metrics = _metrics(shadow_point=shadow_point)
    exposure = _edited()
    local = LocalAdjustmentsConfig()
    before = _printed(exposure, metrics)

    frozen, local = freeze_auto(exposure, local, AG, print_shown_values(exposure, MODE, metrics))
    frozen, local = freeze_auto(frozen, local, AD, print_shown_values(frozen, MODE, metrics))

    assert not frozen.auto_exposure and not frozen.auto_normalize_contrast
    np.testing.assert_allclose(_printed(frozen, metrics), before, atol=1e-9)


def test_shown_values_are_what_prints():
    metrics = _metrics()
    exposure = _edited()
    shown = print_shown_values(exposure, MODE, metrics)

    manual = replace(exposure, auto_exposure=False, auto_normalize_contrast=False, **shown)
    # Split grades and trims follow the stored grade, so compare the base curve only.
    manual = replace(manual, shadow_grade=0.0, highlight_grade=0.0, shadow_grade_trim_red=0.0, grade_trim_blue=0.0)
    auto = replace(exposure, shadow_grade=0.0, highlight_grade=0.0, shadow_grade_trim_red=0.0, grade_trim_blue=0.0)
    np.testing.assert_allclose(_printed(manual, metrics), _printed(auto, metrics), atol=1e-9)
    assert set(shown) == {"density", "grade", "highlight_density"}


def test_shown_values_follow_the_meter():
    exposure = ExposureConfig()
    dense = print_shown_values(exposure, MODE, _metrics(metered_anchor=0.56))
    thin = print_shown_values(exposure, MODE, _metrics(metered_anchor=0.40))
    assert dense["density"] != thin["density"]
    assert print_shown_values(exposure, MODE, _metrics(textural_range=1.2))["grade"] > dense["grade"]


@pytest.mark.parametrize("field", ["density", "grade", "highlight_density"])
def test_stored_value_inverts_shown(field):
    metrics = _metrics()
    exposure = _edited()
    shown = print_shown_values(exposure, MODE, metrics)[field]
    assert print_stored_value(exposure, MODE, metrics, field, shown) == pytest.approx(getattr(exposure, field), abs=1e-6)


def test_off_or_unmetered_autos_show_nothing():
    exposure = ExposureConfig(auto_exposure=False, auto_normalize_contrast=False)
    assert print_shown_values(exposure, MODE, _metrics()) == {}
    assert print_shown_values(ExposureConfig(), MODE, {}) == {}
    # A GPU render with Auto Grade just switched on publishes None for its meters.
    partial = print_shown_values(ExposureConfig(), MODE, _metrics(textural_range=None, highlight_point=None))
    assert set(partial) == {"density"}


def test_stored_value_passes_through_without_a_meter():
    exposure = ExposureConfig(auto_exposure=False)
    assert print_stored_value(exposure, MODE, _metrics(), "density", 1.4) == 1.4


def test_freeze_rescales_mask_grades():
    metrics = _metrics()
    exposure = _edited()
    local = LocalAdjustmentsConfig(masks=(LocalMask(grade=-20.0), LocalMask(grade=10.0)))
    gains = local_grade_factor_map(np.array([m.grade for m in local.masks]), exposure.grade)

    frozen, new_local = freeze_auto(exposure, local, AG, print_shown_values(exposure, MODE, metrics))

    assert frozen.grade != exposure.grade
    np.testing.assert_allclose(local_grade_factor_map(np.array([m.grade for m in new_local.masks]), frozen.grade), gains, atol=1e-6)


def test_freeze_without_meters_only_turns_the_auto_off():
    exposure = _edited()
    frozen, _ = freeze_auto(exposure, LocalAdjustmentsConfig(), AD, {})
    assert frozen == replace(exposure, auto_exposure=False)


def test_enable_resets_driven_fields_to_neutral():
    exposure = _edited(auto_exposure=False, auto_normalize_contrast=False, grade=140.0)
    local = LocalAdjustmentsConfig(masks=(LocalMask(grade=-20.0),))

    on, _ = enable_auto(exposure, local, AD)
    assert on.auto_exposure and on.density == NEUTRAL["density"]
    assert on.grade == exposure.grade

    on, new_local = enable_auto(exposure, local, AG)
    assert on.auto_normalize_contrast
    assert on.grade == NEUTRAL["grade"] and on.highlight_density == NEUTRAL["highlight_density"]
    # The split grades keep their contrast ratio against the new base grade.
    assert split_grade_deltas(on.grade, on.shadow_grade, 0.0)[0][1] == pytest.approx(
        split_grade_deltas(exposure.grade, exposure.shadow_grade, 0.0)[0][1]
    )
    assert new_local.masks[0].grade == pytest.approx(-20.0 * NEUTRAL["grade"] / 140.0)


def _transfer_printed(exposure, metrics):
    offset, contrast, _, _ = transfer_curve_params(exposure)
    offset, contrast, hl_auto = transfer_auto_terms(
        exposure,
        offset,
        contrast,
        metrics.get("textural_range"),
        metrics.get("metered_anchor"),
        metrics.get("shadow_point"),
        metrics.get("highlight_point"),
    )
    return np.array([offset, contrast, exposure.highlight_density + hl_auto])


@pytest.mark.parametrize("textural", [0.6, 1.4, 2.2])
def test_transfer_shown_values_are_what_prints(textural):
    metrics = _metrics(metered_anchor=0.3, textural_range=textural, shadow_point=0.7, highlight_point=0.05)
    exposure = ExposureConfig(density=0.9, grade=120.0, highlight_density=0.05)
    shown = transfer_shown_values(exposure, metrics)

    manual = replace(exposure, auto_exposure=False, auto_normalize_contrast=False, **shown)
    np.testing.assert_allclose(_transfer_printed(manual, metrics), _transfer_printed(exposure, metrics), atol=1e-9)


@pytest.mark.parametrize("field", ["density", "grade", "highlight_density"])
def test_transfer_stored_value_inverts_shown(field):
    metrics = _metrics(metered_anchor=0.3, textural_range=0.9, shadow_point=None, highlight_point=0.05)
    exposure = ExposureConfig(density=0.9, grade=120.0, highlight_density=0.05)
    shown = transfer_shown_values(exposure, metrics)[field]
    assert transfer_stored_value(exposure, metrics, field, shown) == pytest.approx(getattr(exposure, field), abs=1e-6)


def _state_with_meters(metrics):
    from negpy.desktop.session import AppState

    state = AppState()
    state.current_file_hash = "frame"
    state.auto_meters["frame"] = metrics
    return state


def test_only_a_plain_render_of_the_open_frame_records_meters():
    from negpy.desktop.auto_sliders import record_meters

    store: dict = {}
    plain = {**_metrics(), "memo_key": "k", "source_hash": "frame"}
    record_meters(store, "frame", {**plain, "memo_key": ""})  # proxy, override or tool frame
    record_meters(store, "frame", {**plain, "source_hash": "other"})
    record_meters(store, "frame", {**plain, "diptych": True})
    assert store == {}
    record_meters(store, "frame", plain)
    assert store["frame"]["metered_anchor"] == plain["metered_anchor"]
    assert "memo_key" not in store["frame"]


def test_controller_toggle_off_keeps_the_print_and_on_moves_to_the_meter():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from negpy.desktop.controller import AppController

    metrics = _metrics()
    state = _state_with_meters(metrics)
    state.config = replace(state.config, exposure=_edited())
    ctrl = SimpleNamespace(state=state, apply_config=MagicMock())

    AppController.set_auto(ctrl, AD, False)
    off = ctrl.apply_config.call_args.args[0].exposure
    assert not off.auto_exposure
    assert off.density == pytest.approx(print_shown_values(_edited(), MODE, metrics)["density"])

    AppController.set_auto(ctrl, AD, True)
    on = ctrl.apply_config.call_args.args[0].exposure
    assert on.auto_exposure and on.density == NEUTRAL["density"]
