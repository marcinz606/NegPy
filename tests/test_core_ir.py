"""Hash pins of the IR dust kernels on fixed inputs, and checks of their behavior."""

from pathlib import Path

import numpy as np
from core_oracle import cv2_oracle
from core_pin import digest, linux_pin

import negpy_core
from negpy.features.retouch import logic as L


def _scene(seed=0, shape=(120, 90)):
    rng = np.random.default_rng(seed)
    plane = (0.8 + 0.05 * rng.standard_normal(shape)).astype(np.float32)
    plane[30:34, 20:60] = 0.3  # a hair
    plane[:, :6] = 0.01  # the holder margin
    plane[70:72, 40:42] = 2.0  # glare
    img = rng.random(shape + (3,), dtype=np.float32)
    return plane, img


@linux_pin
def test_ir_live_pin():
    plane, _ = _scene()
    assert digest(L._ir_live(plane)) == "7e4f95df2b3c21efc28db00c31e9976e55368208e9c7429d5642e3dddf98031f"


@linux_pin
def test_ir_glare_pin():
    plane, _ = _scene(1)
    assert digest(L._ir_glare(plane, L._ir_live(plane))) == "75bd6aef0b97b9289f5377bf663efe145146a9fe7a65d984e8c70e9bfe7d9f03"


@linux_pin
def test_normalize_ir_pin():
    plane, _ = _scene(2)
    assert digest(L.normalize_ir(plane, plane > 1.5)) == "da76788fcbdde9f1d3ea1b8c292ca1639c2ec9fea5925167cde03ad230bf130a"


@linux_pin
def test_ir_clean_base_pin():
    plane, img = _scene(3)
    assert digest(L._ir_clean_base(img, L.normalize_ir(plane))) == "44d67c21fef954bf342bd1962ff87ae74fff0d10f05aaae0515050039ac3e24e"


@linux_pin
def test_ir_normalize_ratio_pin():
    plane, _ = _scene(4)
    assert (
        digest(L._ir_normalize_ratio(L.normalize_ir(plane), L._ir_live(plane)))
        == "1d7741f2d74be534ef2170f312bd2991dd7c54c9a2c29b4b17d033a90dea3dd7"
    )


@cv2_oracle
def test_ghost_divide_matches_numpy_float32_exp():
    rng = np.random.default_rng(5)
    ratio = rng.random((64, 48), dtype=np.float32)
    vis_log = -3.0 * rng.random((64, 48, 3), dtype=np.float32)
    b = np.array([0.31, 0.07, 0.52], dtype=np.float32)
    want = np.clip(ratio / np.exp((vis_log * b).sum(-1)), 0.0, 1.5).astype(np.float32)
    assert negpy_core.ir_ghost_divide(ratio, vis_log, b.tolist()).tobytes() == want.tobytes()


def test_every_binding_enters_a_pool_through_detach():
    """A thread that ran small work stays in its one-thread pool, so a binding that called
    py.detach directly would run its large work on one core."""
    src = Path(__file__).parents[1] / "negpy_core" / "src" / "py"
    offenders = [p.name for p in src.glob("*.rs") if p.name != "mod.rs" and ".detach(" in p.read_text()]
    assert offenders == []


@linux_pin
def test_route_wide_defects_pin():
    score = np.ones((200, 160), dtype=np.float32)
    score[20:40, 30:60] = L._IR_SCORE_FLOOR  # wide: routed
    score[100:103, 10:150] = L._IR_SCORE_FLOOR  # a thin line: too narrow
    score[150:170, 100:104] = L._IR_SCORE_FLOOR
    assert digest(L.route_wide_defects(score, budget=None)) == "e5c1f1dc519734413bed46f94a1186dce2807521aedffa7e999da4aa8b9b9350"
    assert L.route_wide_defects(np.ones((50, 50), dtype=np.float32)) is None


@linux_pin
def test_ir_defect_score_pin():
    plane, _ = _scene(6)
    assert digest(L.ir_defect_score(L.normalize_ir(plane), 0.6)) == "4bb42b3bab8ba0b50d3f586974b0ab712c37da029ad2ce136b3712fca6ca0f7a"


@linux_pin
def test_fit_refraction_gammas_pin():
    rng = np.random.default_rng(7)
    ratio = rng.uniform(0.6, 1.0, (160, 120)).astype(np.float32)
    img = rng.random((160, 120, 3), dtype=np.float32)
    vis_log = ((ratio - np.float32(1.0))[..., None] * np.float32([1.1, 1.3, 1.6])).astype(np.float32)
    vis_log += 0.01 * rng.standard_normal(vis_log.shape).astype(np.float32)
    assert digest(L._fit_refraction_gammas(ratio, vis_log, img)) == "7ab6eeee1906753359a717030cf88b16131c69f7021bb18be9d53b581590ed1d"


@linux_pin
def test_compute_dust_stats_pin():
    rng = np.random.default_rng(8)
    img = (0.3 + 0.2 * rng.random((140, 180, 3))).astype(np.float32)
    img[40:43, 50:53] *= 0.2  # a speck
    assert digest(L.compute_dust_stats(img, 2)) == "1e3e0ae2a35da752b5c5e36fa2286c8e59d1f00d9e6aa26e504468e89c85ecdc"


def test_split_hairs_matches_components():
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[10:12, 5:70] = 1  # a hair
    mask[40:45, 40:45] = 1  # a speck
    compact, hairs = L.split_hairs(mask)
    want_hair = np.zeros_like(mask)
    want_hair[10:12, 5:70] = 1
    np.testing.assert_array_equal(hairs, want_hair)
    np.testing.assert_array_equal(compact, mask - want_hair)
    assert L.split_hairs(mask - want_hair)[1] is None
