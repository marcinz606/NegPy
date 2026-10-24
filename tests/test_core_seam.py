"""dp_seams and distance_transform against cv2 on synthetic overlaps, and hash pins of the seam and
feather blends."""

import functools

import cv2
import negpy_core
import numpy as np
import pytest
from core_oracle import cv2_oracle
from core_pin import digest

from negpy.features.stitch import logic

pytestmark = cv2_oracle

_RNG = np.random.default_rng(31)


def _parts(h, w, n, shift, angle, rng=_RNG):
    """`n` noisy, slightly rotated views of one scene, each covering a shifted window."""
    wide = w + (n - 1) * shift
    scene = cv2.GaussianBlur(rng.random((h, wide, 3), dtype=np.float32), (0, 0), 4)
    imgs, masks = [], []
    for i in range(n):
        img = np.zeros((h, wide, 3), np.float32)
        x0 = i * shift
        img[:, x0 : x0 + w] = scene[:, x0 : x0 + w] + rng.normal(0, 0.01, (h, w, 3)).astype(np.float32)
        m = np.zeros((h, wide), np.uint8)
        m[:, x0 : x0 + w] = 1
        rot = cv2.getRotationMatrix2D((wide / 2, h / 2), float(rng.normal(0, angle)), 1.0)
        imgs.append(np.ascontiguousarray(cv2.warpAffine(img, rot, (wide, h))))
        masks.append(np.ascontiguousarray(cv2.warpAffine(m, rot, (wide, h), flags=cv2.INTER_NEAREST)))
    return imgs, masks


def _cut_cost(imgs, cut):
    """DpSeamFinder's color cost summed over every step where the owning part changes."""
    owner = np.full(cut[0].shape, -1)
    for i, c in enumerate(cut):
        owner[(owner < 0) & (c > 0)] = i
    stack = np.stack(imgs)
    total = 0.0
    for a, b in (
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
    ):
        oa, ob = owner[a], owner[b]
        ys, xs = np.nonzero((oa >= 0) & (ob >= 0) & (oa != ob))
        ia, ib = oa[ys, xs], ob[ys, xs]
        pa = stack[:, a[0], a[1]][:, ys, xs]
        pb = stack[:, b[0], b[1]][:, ys, xs]
        n = np.arange(len(ys))
        d1 = ((pa[ia, n] - pb[ib, n]) ** 2).sum(-1)
        d2 = ((pb[ia, n] - pa[ib, n]) ** 2).sum(-1)
        total += float(((d1 + d2) / 2).sum())
    return total


def _check_seams(imgs, masks):
    want = [cv2.UMat.get(m) for m in cv2.detail_DpSeamFinder("COLOR").find(imgs, [(0, 0)] * len(imgs), [cv2.UMat(m.copy()) for m in masks])]
    got = negpy_core.dp_seams(imgs, masks)
    overlap = np.sum([m > 0 for m in masks], 0) >= 2
    for w, g in zip(want, got):
        assert np.array_equal(w[~overlap] > 0, g[~overlap] > 0)
    assert _cut_cost(imgs, got) <= 1.01 * _cut_cost(imgs, want) + 1e-6


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("angle", [0.0, 3.0])
def test_dp_seams_match_cv2_on_synthetic_overlaps(n, angle):
    for h, w, shift in ((40, 60, 25), (128, 128, 60), (300, 400, 250)):
        _check_seams(*_parts(h, w, n, shift, angle))


def test_dp_seams_match_cv2_on_small_random_cuts():
    for _ in range(100):
        h, w = (int(v) for v in _RNG.integers(8, 40, 2))
        _check_seams(*_parts(h, w, 2, int(_RNG.integers(2, w)), 5.0))


def test_find_seams_partitions_the_coverage():
    imgs, masks = _parts(90, 120, 3, 50, 2.0)
    seams = logic.find_seams(imgs, [m > 0 for m in masks], (imgs[0].shape[1], imgs[0].shape[0]))
    assert seams is not None
    covered = np.any([m > 0 for m in masks], 0)
    assert np.array_equal(np.sum(seams, 0) >= 1, covered)


@functools.cache
def _blend_cases():
    rng = np.random.default_rng(31)
    cases = []
    for n, size in ((2, (120, 160, 70)), (3, (200, 260, 150)), (4, (64, 90, 40))):
        imgs, masks = _parts(*size, n, 3.0, rng)
        seams = [c > 0 for c in negpy_core.dp_seams(imgs, masks)]
        cases.append((imgs, [m > 0 for m in masks], seams, 12.0))
    return cases


def test_distance_transform_stays_near_cv2():
    masks = [m for _, ms, ss, _ in _blend_cases() for m in (*ms, *ss)]
    masks += [_RNG.random((1500, 1600)) > 0.0005, np.pad(np.ones((1400, 1500), bool), 50)]
    for m in masks:
        for src in (m.astype(np.uint8), (~m).astype(np.uint8)):
            want = cv2.distanceTransform(src, cv2.DIST_L2, 3)
            assert np.abs(negpy_core.distance_transform(src, cv2.DIST_L2, 3) - want).max() <= 0.005


def test_seam_blend_pin():
    got = [logic.seam_blend(warped, masks, seams, feather) for warped, masks, seams, feather in _blend_cases()]
    assert digest(got) == "f6ecc6b24049f1bd51302ecba5b1094f84f4bdc643c4fa530a0c7c47e846d2a9"


def test_feather_blend_pin():
    got = [logic.feather_blend(warped, masks) for warped, masks, _, _ in _blend_cases()]
    assert digest(got) == "2704490969b2167d5fe478617608c152081ed2b9f0519d556809e281dfa61a66"
