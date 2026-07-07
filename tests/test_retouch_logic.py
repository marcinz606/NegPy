import json

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.retouch.logic import (
    _capsule_boundary,
    apply_dust_removal,
    apply_manual_heals,
    build_heal_regions,
    select_source_offset,
)
from negpy.features.retouch.models import RetouchConfig


def _regions_for_spot(nx, ny, size, shape, scale=1.0):
    h, w = shape
    return build_heal_regions([([[nx, ny]], size, 0.15, 0.0)], [], (h, w), 0, 0.0, False, False, 0.0, scale, (w, h))


def test_manual_dust_removal_effect():
    # Use grey background and white dust (inverted film scan scenario)
    img = np.full((100, 100, 3), 0.5, dtype=np.float32)
    img[48:53, 48:53] = 1.0

    orig_mean = np.mean(img)

    res = apply_dust_removal(
        img.copy(),
        dust_remove=False,
        dust_threshold=0.75,
        dust_size=2,
        heal_regions=_regions_for_spot(0.5, 0.5, 10, (100, 100)),
        scale_factor=1.0,
    )

    res_mean = np.mean(res)
    # The healing should make the white spot darker (closer to 0.5 background)
    assert res_mean < orig_mean

    spot_area = res[48:53, 48:53]
    assert np.mean(spot_area) < 0.9


def test_manual_dust_removal_no_spots():
    img = np.ones((100, 100, 3), dtype=np.float32)
    res = apply_dust_removal(
        img.copy(),
        dust_remove=False,
        dust_threshold=0.75,
        dust_size=2,
        heal_regions=None,
        scale_factor=1.0,
    )
    assert np.array_equal(img, res)


def test_auto_dust_removal_low_res():
    # Simple isolated white pixel on dark background
    img = np.zeros((100, 100, 3), dtype=np.float32)
    img[50, 50] = 1.0

    res = apply_dust_removal(
        img.copy(),
        dust_remove=True,
        dust_threshold=0.5,
        dust_size=2,
        heal_regions=None,
        scale_factor=1.0,
    )

    # The bright pixel should be gone
    assert res[50, 50, 0] < 0.5


def test_auto_dust_removal_high_res():
    # Larger spot at high scale
    img = np.zeros((200, 200, 3), dtype=np.float32)
    img[98:103, 98:103] = 1.0

    res = apply_dust_removal(
        img.copy(),
        dust_remove=True,
        dust_threshold=0.5,
        dust_size=4,
        heal_regions=None,
        scale_factor=2.0,
    )

    # The bright spot should be healed
    assert np.mean(res[98:103, 98:103]) < 0.5


def test_auto_detection_uses_perceptual_luma():
    """Retouch receives scene-linear data and detects on a display-encoded copy.
    A dim speck whose LINEAR luma sits below the 0.15 bright-region floor but
    whose ENCODED luma clears it must still be caught (heal runs in linear)."""
    img = np.zeros((100, 100, 3), dtype=np.float32)
    img[50, 50] = 0.10  # linear 0.10 < 0.15 floor; encoded ~0.28 > 0.15

    res = apply_dust_removal(
        img.copy(),
        dust_remove=True,
        dust_threshold=0.5,
        dust_size=2,
        heal_regions=None,
        scale_factor=1.0,
    )

    assert res[50, 50, 0] < 0.04


def test_auto_dust_removal_cloud_protection():
    # Soft gradients should NOT be treated as dust
    y, x = np.mgrid[0:100, 0:100]
    img_gray = (np.sin(x / 10.0) * np.cos(y / 10.0) * 0.1) + 0.5
    img = np.stack([img_gray] * 3, axis=-1).astype(np.float32)

    res = apply_dust_removal(
        img.copy(),
        dust_remove=True,
        dust_threshold=0.5,
        dust_size=2,
        heal_regions=None,
        scale_factor=1.0,
    )

    # Soft gradients should remain identical or very close
    np.testing.assert_allclose(img, res, atol=0.01)


def test_auto_heal_avoids_other_defects():
    """P2 guard: the reflection-copy source must skip masked pixels — a second
    speck one heal-radius away must not be copied into the healed area."""
    img = np.zeros((100, 100, 3), dtype=np.float32)
    img[50, 50] = 1.0
    img[50, 55] = 1.0  # decoy defect near the reflection source distance

    res = apply_dust_removal(
        img.copy(),
        dust_remove=True,
        dust_threshold=0.5,
        dust_size=2,
        heal_regions=None,
        scale_factor=1.0,
    )
    assert res[50, 50, 0] < 0.5
    assert res[50, 55, 0] < 0.5


def test_membrane_recovers_gradient():
    """The MVC membrane clone must reconstruct a linear gradient under a speck —
    diffusion-style fills can't; this is the quality bar for the new heal."""
    h, w = 80, 120
    grad = np.linspace(0.2, 0.6, w, dtype=np.float32)[None, :, None].repeat(h, axis=0)
    img = np.repeat(grad, 3, axis=2)
    clean = img.copy()
    img[36:44, 56:64] = 0.95

    regions = _regions_for_spot(60.0 / w, 40.0 / h, 16.0, (h, w))
    out = apply_manual_heals(img, *regions)

    err = np.abs(out[36:44, 56:64] - clean[36:44, 56:64]).mean()
    assert err < 0.02


def test_stroke_heals_scratch():
    """A polyline stroke heals a diagonal scratch line."""
    rng = np.random.default_rng(7)
    h, w = 120, 160
    grad = np.linspace(0.2, 0.6, w, dtype=np.float32)[None, :, None].repeat(h, axis=0)
    img = (np.repeat(grad, 3, axis=2) + rng.normal(0, 0.01, (h, w, 3))).astype(np.float32)
    clean = img.copy()
    mask = np.zeros((h, w), bool)
    for t in np.linspace(0, 1, 200):
        x, y = int(30 + t * 90), int(30 + t * 50)
        img[y : y + 2, x : x + 2] = 0.9
        mask[y : y + 2, x : x + 2] = True

    pts = [[30.0 / w, 30.0 / h], [75.0 / w, 55.0 / h], [120.0 / w, 80.0 / h]]
    off = select_source_offset(img, pts, 5.0, 0)
    regions = build_heal_regions([(pts, 10.0, off[0], off[1])], [], (h, w), 0, 0.0, False, False, 0.0, 1.0, (w, h))
    out = apply_manual_heals(img, *regions)

    err_before = np.abs(img[mask] - clean[mask]).mean()
    err_after = np.abs(out[mask] - clean[mask]).mean()
    assert err_after < err_before * 0.2


def test_clone_source_dust_not_recloned():
    """Dust sitting in the clone-source patch must not be copied into the heal —
    the sample guard replaces bright outliers with their 3×3 luma-median pixel."""
    rng = np.random.default_rng(11)
    h, w = 100, 100
    img = (np.full((h, w, 3), 0.5) + rng.normal(0, 0.01, (h, w, 3))).astype(np.float32)
    img[47:53, 47:53] = 0.95  # defect being healed
    img[49:51, 69:71] = 0.95  # dust inside the source patch (offset +20px)

    strokes = [([[0.5, 0.5]], 12.0, 20.0 / w, 0.0)]
    regions = build_heal_regions(strokes, [], (h, w), 0, 0.0, False, False, 0.0, 1.0, (w, h))
    out = apply_manual_heals(img, *regions)

    healed = out[44:56, 44:56]
    assert healed.max() < 0.7, "dust from the source patch was recloned into the heal"


def test_heal_gate_leaves_clean_pixels_untouched():
    """The brush marks a search area: only bright dust inside it is replaced,
    clean pixels within the brush stay byte-identical (modulo OETF round-trip)."""
    rng = np.random.default_rng(21)
    h, w = 100, 100
    img = (np.full((h, w, 3), 0.5) + rng.normal(0, 0.01, (h, w, 3))).astype(np.float32)
    img[49:52, 49:52] = 0.95  # small speck, large brush around it

    strokes = [([[0.5, 0.5]], 15.0, 25.0 / w, 0.0)]
    regions = build_heal_regions(strokes, [], (h, w), 0, 0.0, False, False, 0.0, 1.0, (w, h))
    out = apply_manual_heals(img, *regions)

    assert out[49:52, 49:52].max() < 0.7, "dust inside the brush was not healed"

    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(xx - 50, yy - 50)
    clean_in_brush = (dist < 13) & (dist > 4)
    np.testing.assert_allclose(
        out[clean_in_brush],
        img[clean_in_brush],
        atol=2e-3,
        err_msg="clean pixels inside the brush were altered",
    )


def test_source_scoring_penalizes_dusty_patch():
    """select_source_offset must prefer a clean patch over one with a speck inside
    (rim-band SSD alone can't see interior dust)."""
    rng = np.random.default_rng(5)
    h, w = 120, 120
    img = (np.full((h, w, 3), 0.5) + rng.normal(0, 0.005, (h, w, 3))).astype(np.float32)
    img[56:64, 56:64] = 0.95  # defect at center
    # Dust inside the +x candidate patch interior (ring candidate at 2.6r ≈ 10px)
    img[59:61, 69:71] = 0.95

    off = select_source_offset(img, [[0.5, 0.5]], 4.0, 0)
    sx, sy = 60 + off[0] * w, 60 + off[1] * h
    patch = img[int(sy) - 4 : int(sy) + 4, int(sx) - 4 : int(sx) + 4]
    assert patch.max() < 0.7, "scoring picked a source patch containing dust"


def test_capsule_boundary_is_closed_ordered_loop():
    pts = np.array([[20.0, 20.0], [60.0, 40.0]], dtype=np.float64)
    loop = _capsule_boundary(pts, 5.0, 32)
    assert loop.shape[1] == 2
    assert len(loop) >= 16
    # Every sample sits on the capsule outline (distance ~radius from the chain).
    from negpy.features.retouch.logic import _dist_to_chain

    for bx, by in loop:
        assert abs(_dist_to_chain(float(bx), float(by), pts) - 5.0) < 0.5
    # Ordered loop: consecutive samples are close relative to the perimeter.
    seg = np.diff(np.vstack([loop, loop[:1]]), axis=0)
    step = np.hypot(seg[:, 0], seg[:, 1])
    assert step.max() < 5.0 * step.mean()


def test_select_source_offset_avoids_defect():
    """Scoring must reject candidates whose band lands on a second defect."""
    rng = np.random.default_rng(3)
    h, w = 100, 100
    img = (np.full((h, w, 3), 0.5) + rng.normal(0, 0.005, (h, w, 3))).astype(np.float32)
    img[46:54, 46:54] = 0.95  # the defect being healed
    img[46:54, 20:36] = 0.05  # strong anomaly left of it

    off = select_source_offset(img, [[0.5, 0.5]], 4.0, 0)
    sx, sy = 50 + off[0] * w, 50 + off[1] * h
    val = img[int(np.clip(sy, 0, h - 1)), int(np.clip(sx, 0, w - 1))]
    assert abs(float(val.mean()) - 0.5) < 0.1


def test_legacy_spot_conversion():
    regions = build_heal_regions([], [(0.5, 0.5, 8.0)], (100, 100), 0, 0.0, False, False, 0.0, 1.0, (100, 100))
    reg_i, reg_f, pts = regions
    assert len(reg_i) == 1
    assert reg_i[0, 1] == 1  # single-point chain
    assert reg_i[0, 3] >= 16  # boundary loop present
    assert reg_f[0, 0] == 4.0  # radius px = size/2 (brush size is a diameter)
    assert np.hypot(reg_f[0, 1], reg_f[0, 2]) > 4.0  # fallback offset clears the spot


def test_heal_footprint_stays_within_brush():
    """Nothing outside the brush circle may change — the healed footprint must
    not exceed the on-screen cursor. A bright strip crossing the brush is healed
    only inside it."""
    rng = np.random.default_rng(31)
    h, w = 100, 100
    img = (np.full((h, w, 3), 0.4) + rng.normal(0, 0.01, (h, w, 3))).astype(np.float32)
    img[48:52, :] = 0.95  # dust strip across the whole frame

    strokes = [([[0.5, 0.5]], 16.0, 0.0, 25.0 / h)]  # radius 8 at scale 1
    regions = build_heal_regions(strokes, [], (h, w), 0, 0.0, False, False, 0.0, 1.0, (w, h))
    out = apply_manual_heals(img, *regions)

    changed = np.abs(out.astype(np.float64) - img).max(axis=2) > 5e-3
    ys, xs = np.where(changed)
    assert len(ys) > 0, "strip inside the brush was not healed"
    dist = np.hypot(xs + 0.5 - 50.0, ys + 0.5 - 50.0)
    assert dist.max() <= 8.0, f"heal leaked {dist.max():.2f}px from center, brush radius is 8"
    assert out[48:52, 80:].min() > 0.9, "strip outside the brush must stay untouched"


def test_heal_radius_matches_cursor_fraction():
    """Pipeline heal radius must equal the overlay cursor circle: the cursor
    (overlay._brush_screen_radius) draws size/(2·preview_render_size) of the
    view; the pipeline radius normalized by the render long edge is the same."""
    from negpy.kernel.system.config import APP_CONFIG

    size = 12.0
    full_dims = (1600, 1067)
    scale_factor = max(full_dims) / float(APP_CONFIG.preview_render_size)
    _, reg_f, _ = build_heal_regions([([[0.5, 0.5]], size, 0.1, 0.0)], [], (2000, 3000), 0, 0.0, False, False, 0.0, scale_factor, full_dims)
    pipeline_fraction = reg_f[0, 0] / max(full_dims)
    cursor_fraction = size / (2.0 * APP_CONFIG.preview_render_size)
    assert abs(pipeline_fraction - cursor_fraction) < 1e-9


def test_heal_strokes_serialization_roundtrip():
    cfg = WorkspaceConfig(
        retouch=RetouchConfig(
            manual_dust_spots=[(0.1, 0.2, 6.0)],
            manual_heal_strokes=[([[0.3, 0.4], [0.5, 0.6]], 5.0, 0.02, -0.01)],
        )
    )
    data = json.loads(json.dumps(cfg.to_dict()))
    restored = WorkspaceConfig.from_flat_dict(data)
    strokes = restored.retouch.manual_heal_strokes
    assert len(strokes) == 1
    pts, size, dx, dy = strokes[0]
    assert pts == [[0.3, 0.4], [0.5, 0.6]]
    assert (size, dx, dy) == (5.0, 0.02, -0.01)
    assert list(map(list, restored.retouch.manual_dust_spots))[0] == [0.1, 0.2, 6.0]


def test_old_config_without_strokes_loads_default():
    cfg = WorkspaceConfig(retouch=RetouchConfig(manual_dust_spots=[(0.1, 0.2, 6.0)]))
    data = cfg.to_dict()
    data.pop("manual_heal_strokes")
    restored = WorkspaceConfig.from_flat_dict(data)
    assert restored.retouch.manual_heal_strokes == []
