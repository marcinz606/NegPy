from dataclasses import replace

import numpy as np
import pytest

from negpy.features.exposure.auto_sliders import print_shown_values, print_stored_value
from negpy.features.exposure.logic import (
    auto_highlight_from_metrics,
    auto_shadow_from_metrics,
    curve_params_from_metrics,
)
from negpy.features.exposure.models import ExposureConfig
from negpy.features.exposure.normalization import LogNegativeBounds
from negpy.features.process.models import ProcessMode
from negpy.features.transparency.logic import (
    transfer_auto_terms,
    transfer_curve_params,
    transfer_shown_values,
    transfer_stored_value,
)

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
    """A frame with trims on top of both autos."""
    fields = dict(density=1.15, grade=105.0, shadow_density=-0.04, highlight_density=0.08)
    fields.update(over)
    return ExposureConfig(**fields)


def _printed(exposure, metrics):
    """What the print stage applies: curve params and the total shadow and highlight densities."""
    slopes, pivots, curvs = curve_params_from_metrics(exposure, MODE, metrics)
    hl = exposure.highlight_density + auto_highlight_from_metrics(exposure, MODE, metrics)
    sh = exposure.shadow_density + auto_shadow_from_metrics(exposure, MODE, metrics)
    return np.array([*slopes, *pivots, *curvs, hl, sh])


@pytest.mark.parametrize("shadow_point", [None, 0.93, 0.99])
def test_shown_values_are_what_prints(shadow_point):
    metrics = _metrics(shadow_point=shadow_point)
    exposure = _edited()
    shown = print_shown_values(exposure, MODE, metrics)

    manual = replace(exposure, auto_exposure=False, auto_normalize_contrast=False, **shown)
    np.testing.assert_allclose(_printed(manual, metrics), _printed(exposure, metrics), atol=1e-9)
    expected = {"density", "grade", "highlight_density"} | ({"shadow_density"} if shadow_point is not None else set())
    assert set(shown) == expected


def test_shown_values_follow_the_meter():
    exposure = ExposureConfig()
    dense = print_shown_values(exposure, MODE, _metrics(metered_anchor=0.56))
    thin = print_shown_values(exposure, MODE, _metrics(metered_anchor=0.40))
    assert dense["density"] != thin["density"]
    assert print_shown_values(exposure, MODE, _metrics(textural_range=1.2))["grade"] > dense["grade"]


@pytest.mark.parametrize("field", ["density", "grade", "shadow_density", "highlight_density"])
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
    partial = print_shown_values(ExposureConfig(), MODE, _metrics(textural_range=None, shadow_point=None, highlight_point=None))
    assert set(partial) == {"density"}


def test_stored_value_passes_through_without_a_meter():
    exposure = ExposureConfig(auto_exposure=False)
    assert print_stored_value(exposure, MODE, _metrics(), "density", 1.4) == 1.4


def _transfer_printed(exposure, metrics):
    offset, contrast, _, _ = transfer_curve_params(exposure)
    offset, contrast, hl_auto, sh_auto = transfer_auto_terms(
        exposure,
        offset,
        contrast,
        metrics.get("textural_range"),
        metrics.get("metered_anchor"),
        metrics.get("shadow_point"),
        metrics.get("highlight_point"),
    )
    return np.array([offset, contrast, exposure.highlight_density + hl_auto, exposure.shadow_density + sh_auto])


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
