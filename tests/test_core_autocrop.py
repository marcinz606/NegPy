"""Hash pins of the batch autocrop profile reads and the occupancy window on fixed inputs."""

import numpy as np
import pytest

from core_pin import digest, linux_pin
from negpy.features.geometry.batch_autocrop import CropEvidence, RollCropTemplate, _rect_from_edge_profile, _top_edge_slope
from negpy.features.geometry.logic import _place_window_by_occupancy


def _tilted_box(tilt, seed, size=(400, 600)):
    h, w = size
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    top = 100 + tilt * (np.pi / 180) * (x - w / 2)
    lum = np.where(y >= top, 0.8, 0.1) + np.random.default_rng(seed).normal(0, 0.02, (h, w))
    return lum.astype(np.float32)


@pytest.mark.parametrize(
    "tilt, pin",
    [
        (-2.0, "6f6b1f71e2687aa404eae12bd9c885b8de724eb1d843fee446c13a7374718208"),
        (-0.7, "d5855848f5dfcac4cd91e34965c8017d46bd4ea2fe64e3d96acbbf021d6b0950"),
        (0.0, "97814420f219096dce9b7bccdb78ba1ed641971d9bea590bbde014a9b132e4b7"),
        (0.3, "2eed232be67b65aaad7bc6ba4340df423b1ae271366f61fb6a65dd6da01fd06b"),
        (1.6, "7308f002e5bdcc9b81fcd2d68d000b5009c90d5337513d5493f39b843f90ae36"),
    ],
)
@linux_pin
def test_top_edge_slope_pin(tilt, pin):
    got = []
    for seed in range(4):
        lum = _tilted_box(tilt, seed)
        for roi in ((100, 300, 50, 550), (8, 300, 0, 600), (100, 300, 300, 330)):
            got.append(_top_edge_slope(lum, roi))
    assert any(g is not None for g in got)
    assert digest(got) == pin


@linux_pin
def test_rect_from_edge_profile_pin():
    rng = np.random.default_rng(7)
    got = []
    for _ in range(3000):
        n = int(rng.integers(8, 400))
        profile = (rng.random(n) * rng.uniform(0.05, 0.6)).astype(np.float32)
        t = RollCropTemplate(
            width=float(rng.uniform(0.5, 0.95)), fallback_width=float(rng.uniform(0.5, 1.0)), height=0.8, center_x=float(rng.uniform(0.4, 0.6)),
            top=0.1, correction_angle=0.0, width_mad=float(rng.uniform(0, 0.02)), center_mad=float(rng.uniform(0, 0.02)), top_mad=0.0,
            angle_mad=0.0, confidence=0.7, sample_count=5,
        )  # fmt: skip
        for edge in (t.center_x - t.width / 2, t.center_x + t.width / 2):
            if rng.random() < 0.8:
                profile[int(np.clip(round((edge + rng.normal(0, 0.01)) * (n - 1)), 0, n - 1))] = rng.uniform(0.5, 2.0)
        if rng.random() < 0.05:
            profile[int(rng.integers(n))] = np.nan
        item = CropEvidence(
            "k", (100, 150), None, 0.0, 0.0, vertical_edge_contrast=float(rng.uniform(0, 0.3)), vertical_edge_profile=profile
        )
        got.append(_rect_from_edge_profile(item, t))
    assert sum(g is not None for g in got) > 100
    assert digest(got) == "731ad91f420d1c8e37fd0f7829d18414be15a306a6c8e610489795318aa52451"


@linux_pin
def test_place_window_by_occupancy_pin():
    rng = np.random.default_rng(3)
    got = []
    for _ in range(5000):
        n = int(rng.integers(5, 300))
        occ = (rng.random(n) > rng.random()).astype(np.float32) * rng.choice([1.0, 0.5], n).astype(np.float32)
        scale = float(rng.choice([1.0, 0.5, 0.37, 0.25]))
        end = int(rng.integers(2, int(n / scale) + 2))
        start = int(rng.integers(0, end))
        target = float(rng.uniform(1, end - start + 1)) if rng.random() < 0.5 else float(rng.integers(1, end - start + 1))
        got.append(_place_window_by_occupancy(start, end, target, occ, scale))
    assert digest(got) == "75f380746aba9a674a2c09abdd17af2f602c49165fda7acf891fd48e5f74ad22"
