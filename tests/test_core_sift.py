"""The quality gate of the SIFT registration port: negpy_core.register_pair against the true
transform of synthetic pairs and against cv2's SIFT + BFMatcher + estimateAffinePartial2D."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from core_oracle import cv2_oracle

import negpy_core
from negpy.features.stitch import logic as L

pytestmark = cv2_oracle

ROOT = Path(__file__).resolve().parents[1]
_TIF = ROOT / "samples" / "panorama" / "Img428 - NX Studio Tiff.TIF"
_CASES = [(1.5, 1.01, 0.2), (-2.8, 0.975, 0.25), (3.0, 1.03, 0.15), (-1.0, 0.99, 0.3)]


def _synthetic():
    rng = np.random.default_rng(5)
    base = cv2.GaussianBlur(rng.random((1100, 1600), dtype=np.float32), (0, 0), 2.0)
    grad = np.linspace(0.0, 0.3, 1600, dtype=np.float32)[None, :]
    mono = np.clip(0.1 + 0.6 * base + grad, 0.0, 1.0)
    return np.stack([mono, mono * 0.9, mono * 1.1], axis=-1).clip(0.0, 1.0)


def _source(name):
    if name == "synthetic":
        return _synthetic()
    if not _TIF.exists():
        pytest.skip("panorama samples not present")
    import tifffile

    return tifffile.imread(_TIF).astype(np.float32) / 65535.0


def _similarity(angle, scale, tx, ty):
    a = np.deg2rad(angle)
    c, s = scale * np.cos(a), scale * np.sin(a)
    return np.array([[c, -s, tx], [s, c, ty]])


def _cut(src, angle, scale, shift):
    """Part A at the top left of ``src``; part B moved along the long side by a known similarity.
    Returns (A, B, t) with t mapping B coordinates to A coordinates."""
    big_h, big_w = src.shape[:2]
    if big_w >= big_h:
        h, w = int(big_h * 0.9), int(big_w * 0.62)
        t = _similarity(angle, scale, shift * big_w, 0.02 * big_h)
    else:
        h, w = int(big_h * 0.62), int(big_w * 0.9)
        t = _similarity(angle, scale, 0.02 * big_w, shift * big_h)
    b = cv2.warpAffine(src, t, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    return np.ascontiguousarray(src[:h, :w]), b, t


def _corner_error(m, t, w, h):
    box = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    return float(np.max(np.hypot(*((box @ m[:, :2].T + m[:, 2]) - (box @ t[:, :2].T + t[:, 2])).T)))


def _cv2_register(ref, mov):
    sift = cv2.SIFT_create(nfeatures=6000)
    ref_kp, ref_desc = sift.detectAndCompute(ref, None)
    mov_kp, mov_desc = sift.detectAndCompute(mov, None)
    pairs = cv2.BFMatcher().knnMatch(mov_desc, ref_desc, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    mov_pts = np.float32([mov_kp[m.queryIdx].pt for m in good])
    ref_pts = np.float32([ref_kp[m.trainIdx].pt for m in good])
    matrix, mask = cv2.estimateAffinePartial2D(mov_pts, ref_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return matrix, int(mask.sum())


@pytest.mark.parametrize("name", ["synthetic", "full"])
def test_register_pair_recovers_a_known_similarity(name):
    src = _source(name)
    for angle, scale, shift in _CASES:
        a, b, t = _cut(src, angle, scale, shift)
        (pa, sa), (pb, sb) = L.build_proxy(a), L.build_proxy(b)
        matrix, inliers = L.register_pair(pa, pb)
        ref_matrix, ref_inliers = _cv2_register(pa, pb)
        h, w = b.shape[:2]
        err = _corner_error(L.scale_affine(matrix, sa, sb), t, w, h)
        ref_err = _corner_error(L.scale_affine(ref_matrix, sa, sb), t, w, h)
        print(f"{name} {angle:+.1f} deg x{scale}: corner error {err:.4f} px ({inliers} inliers), cv2 {ref_err:.4f} px ({ref_inliers})")
        assert err < 0.1, (angle, scale, err, ref_err)


@pytest.mark.parametrize("name", ["synthetic", "full"])
def test_keypoint_count_matches_cv2(name):
    proxy, _ = L.build_proxy(_source(name))
    for nfeatures in (0, 6000):
        want = len(cv2.SIFT_create(nfeatures=nfeatures).detect(proxy, None))
        got = negpy_core.sift_detect(proxy, nfeatures)[0].shape[0]
        print(f"{name} nfeatures={nfeatures}: {got} keypoints, cv2 {want}")
        assert abs(got - want) <= 0.1 * want


def test_descriptors_match_cv2():
    proxy, _ = L.build_proxy(_synthetic())
    kp, want = cv2.SIFT_create().detectAndCompute(proxy, None)
    pts, got = negpy_core.sift_detect(proxy, 0)
    by_pos = {}
    for k, d in zip(kp, want.astype(np.int32)):
        by_pos.setdefault((round(k.pt[0], 2), round(k.pt[1], 2)), []).append(d)
    far = [np.full(128, 999, np.int32)]
    diffs = np.array(
        [
            min(np.abs(d - w).max() for w in by_pos.get((round(float(x), 2), round(float(y), 2)), far))
            for (x, y), d in zip(pts, got.astype(np.int32))
        ]
    )
    print(f"descriptors within 2 of cv2's: {np.mean(diffs <= 2):.4f}, identical: {np.mean(diffs == 0):.4f}")
    assert np.mean(diffs <= 2) >= 0.95
