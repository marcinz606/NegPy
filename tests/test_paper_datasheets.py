"""The RA4 profiles against the datasheet curves they are fitted to (scripts/print_curves/)."""

import dataclasses
import pathlib

import numpy as np
import pytest

from negpy.features.exposure.logic import default_grade_range, grade_coupled_shape, grade_to_slope, print_curve
from negpy.features.exposure.models import ExposureConfig
from negpy.features.exposure.papers import PAPER_PROFILES
from negpy.features.process.models import ProcessMode

CURVES = pathlib.Path(__file__).parents[1] / "scripts" / "print_curves"
STOCKS = sorted(p.stem for p in CURVES.glob("*.csv"))
# Above this the layers part and a Status A reading is no longer one layer's dye amount.
NEUTRAL_TOP = 1.8


def _sheet(key):
    a = np.genfromtxt(CURVES / f"{key}.csv", delimiter=",", names=True)
    return a["log_e"], np.stack([a[c] for c in "rgb"], 1)


def test_every_curve_file_names_a_profile():
    assert STOCKS and set(STOCKS) <= set(PAPER_PROFILES)


@pytest.mark.parametrize("key", STOCKS)
def test_neutral_curve_follows_the_datasheet(key):
    x, rgb = _sheet(key)
    d = rgb.mean(1)
    rng = default_grade_range()
    grade = 100.0 * (np.interp(0.9 * d.max(), d, x) - np.interp(d.min() + 0.04, d, x))
    exposure = dataclasses.replace(ExposureConfig(), paper_profile=key, paper_dmin=True, paper_black=True, grade=grade)
    slope = grade_to_slope(grade, rng)
    curve = print_curve(exposure, slope, 0.0, ProcessMode.C41, toe=grade_coupled_shape(slope, 0.0, 0.0)[0])
    sel = d < NEUTRAL_TOP

    def rms(x0):
        xn = ((x - x0) / rng).astype(np.float32).reshape(1, -1, 1).repeat(3, 2)
        return float(np.sqrt(np.mean((np.asarray(curve(xn), dtype=np.float64)[0, sel, 0] - d[sel]) ** 2)))

    assert min(rms(x0) for x0 in np.linspace(x.min() - 1.0, x.max() + 1.0, 801)) < 0.05


@pytest.mark.parametrize("key", STOCKS)
def test_maximum_density_follows_the_datasheet(key):
    _, rgb = _sheet(key)
    profile = PAPER_PROFILES[key]
    assert abs(profile.d_max - rgb.max()) < 0.1
    assert min(profile.d_max_drop) == 0.0 and profile.d_max_drop[2] == max(profile.d_max_drop) > 0.0


@pytest.mark.parametrize("key", STOCKS)
def test_paper_white_follows_the_datasheet(key):
    feet = _sheet(key)[1].min(0)
    profile = PAPER_PROFILES[key]
    assert abs(profile.d_min - feet.mean()) < 0.02
    np.testing.assert_allclose(profile.base_tint_cmy, feet - feet.mean(), atol=0.02)
