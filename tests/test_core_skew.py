import math

import negpy_core
import numpy as np
import pytest

from core_pin import digest, linux_pin


@pytest.mark.parametrize(
    "horizontal, nearest, pin",
    [
        (True, False, "33b68dc6247e76e4b75e42f22526794f3e0963dea0ee9461955595f62df66ae0"),
        (False, False, "79ecf0e77eceec1f434524d7282b3a5334af7e2296befb09327cf218681023e7"),
        (True, True, "33b68dc6247e76e4b75e42f22526794f3e0963dea0ee9461955595f62df66ae0"),
        (False, True, "3295e712cb659f034ef6005f55ab36eab5feb00b4283129cdf357e7992952888"),
    ],
)
@linux_pin
def test_skew_sample_pin(horizontal, nearest, pin):
    rng = np.random.default_rng(31)
    lum = rng.random((180, 260), dtype=np.float32) * np.float32(0.05)
    lum[60:, :] += np.float32(0.8)
    lum[:, 200:] += np.float32(0.3)
    lum[:, 7] = 1.0
    stations = np.linspace(4.0, (260 if horizontal else 180) - 5.0, 37)
    center = np.full(stations.size, 61.0) + np.linspace(-3.0, 3.0, stations.size)
    xs, ys = negpy_core.skew_sample(lum, horizontal, stations, center, 14.0, nearest)
    assert xs.size > 0
    assert digest(xs, ys) == pin


def _frame(angle=1.2, size=(420, 600), seed=5):
    """A bright film box on a dark bed, turned by `angle` degrees, with grain."""
    h, w = size
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    t = math.radians(angle)
    u = (x - w / 2) * math.cos(t) + (y - h / 2) * math.sin(t)
    v = -(x - w / 2) * math.sin(t) + (y - h / 2) * math.cos(t)
    lum = np.where((np.abs(u) < 0.38 * w) & (np.abs(v) < 0.36 * h), 0.85, 0.08)
    return (lum + np.random.default_rng(seed).normal(0, 0.01, lum.shape)).astype(np.float32)


@pytest.mark.parametrize(
    "angle, horizontal, pin",
    [
        (-3.0, True, "8cea7bcfccf642f0d92977c018f46f7933bc70a3cc00817b5e1b9371090d3ba0"), (-0.4, True, "3f1cbfdf2ddaa5e5af32ff5b3f8e4786d884a9e168085732eab487ddcf26283e"), (0.0, True, "dfe8b695eb4791bb1c10edaf7fd07265d17428eb838918d6707047088cef79ad"), (0.9, True, "38735c44b64473d6eed3600d3412237904187b9b0f7916a0bd6c21216a5a868e"), (2.5, True, "fccc16cfe0b19b107ffc4decb8e1fd4c4ffc871e1aa5bc8457b24fb222d9dc3d"),
        (-3.0, False, "faee2bea08193b19690cab0931609a5173faa2d4fb939e73e443288292bc455e"), (-0.4, False, "bb96a39748cdab2a4c03b035242999e3ba4ee09e6bd44d867ced3d710bc1a9ed"), (0.0, False, "564180147499d0fd04c1d964188a1ff94b86f011d27216e2952fe8d2d2f14ed1"), (0.9, False, "baf5b8a1c350a50e1fc6967512031f6d406efc9bb19dfaa82cf35e4faf83e6b7"), (2.5, False, "c60ed5b7e5663f2f7280f9429162b526f3b894c9273a9f1970a71157619ef309"),
    ],
)  # fmt: skip
@linux_pin
def test_fit_side_pin(angle, horizontal, pin):
    lum = _frame(angle, seed=int(angle * 10) + 50)
    h, w = lum.shape
    sides = (
        [(0.14 * h, 0.12 * w, 0.88 * w, -1), (0.86 * h, 0.12 * w, 0.88 * w, 1)]
        if horizontal
        else [(0.12 * w, 0.14 * h, 0.86 * h, -1), (0.88 * w, 0.14 * h, 0.86 * h, 1)]
    )
    got = [negpy_core.skew_fit_side(lum, horizontal, at, start, stop, out) for at, start, stop, outward in sides for out in (outward, 0)]
    assert any(g is not None for g in got)
    assert digest(got) == pin
