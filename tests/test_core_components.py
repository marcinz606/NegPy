import cv2
import negpy_core
import numpy as np
import pytest
from core_oracle import cv2_oracle

pytestmark = cv2_oracle

_RNG = np.random.default_rng(4)


def _mask(h, w, density):
    return ((_RNG.random((h, w)) < density) * 255).astype(np.uint8)


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("shape", [(1, 1), (1, 57), (57, 1), (2, 2), (31, 47), (128, 200), (257, 129)])
@pytest.mark.parametrize("density", [0.0, 0.1, 0.4, 0.7, 1.0])
def test_connected_components_with_stats_match_cv2(connectivity, shape, density):
    mask = _mask(*shape, density)
    want = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
    count, labels, stats, centroids = negpy_core.connected_components_with_stats(mask, connectivity)
    assert count == want[0]
    assert np.array_equal(labels, want[1])
    assert np.array_equal(stats, want[2])
    assert centroids.tobytes() == want[3].tobytes()


def _sparse(h, w, density):
    return (_RNG.random((h, w)) >= density).astype(np.uint8)


@pytest.mark.parametrize("shape", [(1, 1), (1, 40), (40, 1), (37, 53), (150, 90)])
@pytest.mark.parametrize("density", [0.0, 0.005, 0.05, 0.5])
@pytest.mark.parametrize("dist_type, mask_size", [(cv2.DIST_C, 3), (cv2.DIST_L2, 5)])
def test_distance_transform_matches_cv2(shape, density, dist_type, mask_size):
    mask = _sparse(*shape, density)
    want = cv2.distanceTransform(mask, dist_type, mask_size)
    assert negpy_core.distance_transform(mask, dist_type, mask_size).tobytes() == want.tobytes()


@pytest.mark.parametrize("shape", [(1, 1), (37, 53), (150, 90)])
@pytest.mark.parametrize("density", [0.0, 0.005, 0.05, 0.5])
def test_distance_transform_l2_mask3_within_one_ulp(shape, density):
    mask = _sparse(*shape, density)
    want = cv2.distanceTransform(mask, cv2.DIST_L2, 3).view(np.int32).astype(np.int64)
    got = negpy_core.distance_transform(mask, cv2.DIST_L2, 3).view(np.int32).astype(np.int64)
    assert np.abs(got - want).max(initial=0) <= 1


@pytest.mark.parametrize("shape", [(1, 1), (1, 40), (37, 53), (150, 90)])
@pytest.mark.parametrize("density", [0.005, 0.05, 0.5])
@pytest.mark.parametrize("mask_size", [3, 5])
def test_distance_transform_with_labels_matches_cv2(shape, density, mask_size):
    mask = _sparse(*shape, density)
    want_d, want_l = cv2.distanceTransformWithLabels(mask, cv2.DIST_L2, mask_size, labelType=cv2.DIST_LABEL_PIXEL)
    got_d, got_l = negpy_core.distance_transform_labels(mask)
    assert got_d.tobytes() == want_d.tobytes()
    assert np.array_equal(got_l, want_l)
