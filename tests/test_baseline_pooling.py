import numpy as np

from negpy.domain.types import LUMA_B, LUMA_G, LUMA_R
from negpy.features.exposure.normalization import pool_frame_bounds

W = np.array([LUMA_R, LUMA_G, LUMA_B])
BASE_F = np.array([-1.3, -1.6, -1.4])
BASE_C = np.array([-0.2, -0.15, -0.06])


def _roll(n: int, jitter: float = 0.02, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return BASE_F + rng.uniform(-jitter, jitter, (n, 3)), BASE_C + rng.uniform(-jitter, jitter, (n, 3))


def _chroma(floors, ceils) -> np.ndarray:
    floors, ceils = np.asarray(floors), np.asarray(ceils)
    return np.concatenate([floors - W @ floors, ceils - W @ ceils])


def test_uniform_roll_has_no_outliers_and_pools_to_its_centre():
    floors, ceils = _roll(7)
    pooled, outliers = pool_frame_bounds(floors, ceils)
    assert not outliers.any()
    np.testing.assert_allclose(pooled.floors, BASE_F, atol=0.03)
    np.testing.assert_allclose(pooled.ceils, BASE_C, atol=0.03)


def test_frame_under_a_different_light_is_flagged_and_left_out_of_the_color():
    floors, ceils = _roll(7)
    floors[3] += (0.05, -0.4, -0.6)
    ceils[3] += (-0.15, -0.7, -0.9)
    pooled, outliers = pool_frame_bounds(floors, ceils)
    assert outliers.tolist() == [False, False, False, True, False, False, False]
    inliers = np.delete(np.arange(7), 3)
    expected = np.mean([_chroma(floors[i], ceils[i]) for i in inliers], axis=0)
    np.testing.assert_allclose(_chroma(pooled.floors, pooled.ceils), expected, atol=1e-9)
    # The cast shifts luma through G, so the outlier's luma stays out of the pool too.
    assert np.isclose(W @ np.array(pooled.floors), np.median(floors[inliers] @ W))


def test_denser_frame_with_the_same_color_is_not_an_outlier_and_luma_is_the_median():
    floors, ceils = _roll(5, jitter=0.0)
    floors[0] -= 0.8
    ceils[0] -= 0.8
    pooled, outliers = pool_frame_bounds(floors, ceils)
    assert not outliers.any()
    assert np.isclose(W @ np.array(pooled.floors), np.median(floors @ W))
    assert np.isclose(W @ np.array(pooled.ceils), np.median(ceils @ W))


def test_pooled_chroma_comes_from_one_set_of_frames_on_every_channel():
    # Two color clusters: a per-channel trim would take red from one and green/blue from
    # the other. Whole-frame pooling keeps the majority cluster on all channels.
    floors, ceils = _roll(7, jitter=0.0)
    floors[:2] += (0.2, -0.5, -0.6)
    ceils[:2] += (-0.2, -0.7, -0.8)
    pooled, outliers = pool_frame_bounds(floors, ceils)
    assert outliers.tolist() == [True, True, False, False, False, False, False]
    np.testing.assert_allclose(_chroma(pooled.floors, pooled.ceils), _chroma(floors[2], ceils[2]), atol=1e-9)


def test_small_pool_never_flags_outliers():
    floors, ceils = _roll(2, jitter=0.0)
    floors[1] += (0.0, -0.8, -0.9)
    _, outliers = pool_frame_bounds(floors, ceils)
    assert not outliers.any()


def test_frames_that_share_no_light_all_flag_and_all_pool():
    floors, ceils = _roll(3, jitter=0.0)
    for i in range(3):
        floors[i, i] -= 0.8
    pooled, outliers = pool_frame_bounds(floors, ceils)
    assert outliers.all()
    np.testing.assert_allclose(_chroma(pooled.floors, pooled.ceils), np.mean([_chroma(f, c) for f, c in zip(floors, ceils)], axis=0))
