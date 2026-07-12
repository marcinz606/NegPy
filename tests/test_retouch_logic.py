import json

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.retouch.logic import (
    _capsule_boundary,
    _mask_to_strokes,
    _pick_source_offsets,
    apply_manual_heals,
    build_heal_regions,
    detect_ir_regions,
    detect_luma_regions,
    select_source_offset,
)
from negpy.features.retouch.models import HEAL_SIZE_REF, RetouchConfig


def _size_at_ref(diameter_px, shape):
    """Convert a brush diameter in image px to the stored HEAL_SIZE_REF-scale size."""
    return diameter_px * HEAL_SIZE_REF / max(shape)


def _regions_for_spot(nx, ny, size_px, shape):
    h, w = shape
    size = _size_at_ref(size_px, shape)
    return build_heal_regions([([[nx, ny]], size, 0.15, 0.0)], [], (h, w), 0, 0.0, False, False, 0.0, (w, h))


def test_polyline_stroke_is_smoothed():
    """A >=3-waypoint scratch heals along a densified Catmull-Rom chain."""
    h, w = 120, 160
    pts = [[0.2, 0.2], [0.5, 0.35], [0.8, 0.2], [0.6, 0.6]]
    reg_i, _reg_f, _pts = build_heal_regions(
        [(pts, _size_at_ref(10.0, (h, w)), 0.1, 0.0)], [], (h, w), 0, 0.0, False, False, 0.0, (w, h)
    )
    chain_len = int(reg_i[0][1])
    assert chain_len > len(pts), "polyline heal chain was not smoothed/densified"


def test_two_point_stroke_not_densified():
    """A straight 2-point stroke stays exactly 2 chain points (nothing to smooth)."""
    h, w = 120, 160
    pts = [[0.3, 0.3], [0.7, 0.6]]
    reg_i, _reg_f, _pts = build_heal_regions(
        [(pts, _size_at_ref(10.0, (h, w)), 0.1, 0.0)], [], (h, w), 0, 0.0, False, False, 0.0, (w, h)
    )
    assert int(reg_i[0][1]) == 2


def test_manual_dust_removal_effect():
    # Use grey background and white dust (inverted film scan scenario)
    img = np.full((100, 100, 3), 0.5, dtype=np.float32)
    img[48:53, 48:53] = 1.0

    orig_mean = np.mean(img)

    res = apply_manual_heals(img.copy(), *_regions_for_spot(0.5, 0.5, 10, (100, 100)))

    res_mean = np.mean(res)
    # The healing should make the white spot darker (closer to 0.5 background)
    assert res_mean < orig_mean

    spot_area = res[48:53, 48:53]
    assert np.mean(spot_area) < 0.9


def test_manual_dust_removal_no_regions_is_noop():
    img = np.ones((100, 100, 3), dtype=np.float32)
    empty = build_heal_regions([], [], (100, 100), 0, 0.0, False, False, 0.0, (100, 100))
    res = apply_manual_heals(img.copy(), *empty)
    assert np.array_equal(img, res)


def test_detect_luma_regions_cloud_protection():
    """Soft gradients in the source must NOT be detected as dust (the wide-window
    texture penalty and z-score guard, ported to the density proxy)."""
    y, x = np.mgrid[0:160, 0:160]
    trans = 0.2 + 0.1 * np.sin(x / 10.0) * np.cos(y / 10.0)  # smooth transmission field
    img = np.stack([trans] * 3, axis=-1).astype(np.float32)
    assert detect_luma_regions(img, 0.66, 4) == []


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
    regions = build_heal_regions([(pts, _size_at_ref(10.0, (h, w)), off[0], off[1])], [], (h, w), 0, 0.0, False, False, 0.0, (w, h))
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

    strokes = [([[0.5, 0.5]], _size_at_ref(12.0, (h, w)), 20.0 / w, 0.0)]
    regions = build_heal_regions(strokes, [], (h, w), 0, 0.0, False, False, 0.0, (w, h))
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

    strokes = [([[0.5, 0.5]], _size_at_ref(15.0, (h, w)), 25.0 / w, 0.0)]
    regions = build_heal_regions(strokes, [], (h, w), 0, 0.0, False, False, 0.0, (w, h))
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


def test_gate_zero_heals_dark_defects():
    """gate=0 regions (5-tuple strokes: synthesized IR/dark-dust) clone
    unconditionally — a DARK defect the bright-only gate vetoes is healed."""
    rng = np.random.default_rng(13)
    h, w = 100, 100
    img = (np.full((h, w, 3), 0.5) + rng.normal(0, 0.01, (h, w, 3))).astype(np.float32)
    img[48:53, 48:53] = 0.05  # dark defect

    size = _size_at_ref(12.0, (h, w))
    gated = build_heal_regions([([[0.5, 0.5]], size, 20.0 / w, 0.0)], [], (h, w), 0, 0.0, False, False, 0.0, (w, h))
    out_gated = apply_manual_heals(img, *gated)
    assert out_gated[48:53, 48:53].mean() < 0.15, "bright-only gate must veto dark defects"

    ungated = build_heal_regions([([[0.5, 0.5]], size, 20.0 / w, 0.0, 0.0)], [], (h, w), 0, 0.0, False, False, 0.0, (w, h))
    out = apply_manual_heals(img, *ungated)
    assert out[48:53, 48:53].mean() > 0.4, "gate=0 region must clone over the dark defect"


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


def test_source_scoring_penalizes_dark_detail_patch():
    """Interior penalty must be symmetric: dark detail inside a candidate patch
    (clean rim) would be cloned onto the plain background just like a speck."""
    rng = np.random.default_rng(5)
    h, w = 120, 120
    img = (np.full((h, w, 3), 0.5) + rng.normal(0, 0.005, (h, w, 3))).astype(np.float32)
    img[56:64, 56:64] = 0.95  # defect at center
    img[59:61, 69:71] = 0.05  # dark detail inside the +x candidate patch interior

    off = select_source_offset(img, [[0.5, 0.5]], 4.0, 0)
    sx, sy = 60 + off[0] * w, 60 + off[1] * h
    patch = img[int(sy) - 4 : int(sy) + 4, int(sx) - 4 : int(sx) + 4]
    assert patch.min() > 0.3, "scoring picked a source patch containing dark detail"


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
    size = _size_at_ref(8.0, (100, 100))
    regions = build_heal_regions([], [(0.5, 0.5, size)], (100, 100), 0, 0.0, False, False, 0.0, (100, 100))
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

    strokes = [([[0.5, 0.5]], _size_at_ref(16.0, (h, w)), 0.0, 25.0 / h)]  # radius 8 in image px
    regions = build_heal_regions(strokes, [], (h, w), 0, 0.0, False, False, 0.0, (w, h))
    out = apply_manual_heals(img, *regions)

    changed = np.abs(out.astype(np.float64) - img).max(axis=2) > 5e-3
    ys, xs = np.where(changed)
    assert len(ys) > 0, "strip inside the brush was not healed"
    dist = np.hypot(xs + 0.5 - 50.0, ys + 0.5 - 50.0)
    assert dist.max() <= 8.0, f"heal leaked {dist.max():.2f}px from center, brush radius is 8"
    assert out[48:52, 80:].min() > 0.9, "strip outside the brush must stay untouched"


def test_heal_radius_matches_cursor_fraction():
    """Pipeline heal radius must equal the overlay cursor circle: the cursor
    (overlay._brush_screen_radius) draws size/(2·HEAL_SIZE_REF) of the view;
    the pipeline radius normalized by the render long edge is the same,
    independent of the preview render resolution."""
    size = 12.0
    full_dims = (2400, 1600)
    _, reg_f, _ = build_heal_regions([([[0.5, 0.5]], size, 0.1, 0.0)], [], (2000, 3000), 0, 0.0, False, False, 0.0, full_dims)
    pipeline_fraction = reg_f[0, 0] / max(full_dims)
    cursor_fraction = size / (2.0 * HEAL_SIZE_REF)
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


def test_preset_save_excludes_frame_specific_heals(tmp_path, monkeypatch):
    from dataclasses import replace as dc_replace

    import negpy.services.assets.presets as presets_mod
    from negpy.kernel.system.config import APP_CONFIG

    monkeypatch.setattr(presets_mod, "APP_CONFIG", dc_replace(APP_CONFIG, presets_dir=str(tmp_path)))
    cfg = WorkspaceConfig(
        retouch=RetouchConfig(
            dust_remove=True,
            manual_dust_spots=[(0.1, 0.2, 6.0)],
            manual_heal_strokes=[([[0.3, 0.4]], 5.0, 0.02, -0.01)],
        )
    )
    presets_mod.Presets.save_preset("t", cfg)
    data = json.loads((tmp_path / "t.json").read_text())
    assert "manual_heal_strokes" not in data
    assert "manual_dust_spots" not in data
    assert data["dust_remove"] is True


def _dusty_source(h=160, w=160, seed=42):
    rng = np.random.default_rng(seed)
    img = (np.full((h, w, 3), 0.18) * (1.0 + rng.normal(0, 0.02, (h, w, 3)))).astype(np.float32)
    img[80:83, 80:83] = 0.005  # dust blocks scanner light: dark on the scan
    return img


def test_detect_luma_regions_finds_dark_speck():
    img = _dusty_source()
    strokes = detect_luma_regions(img, dust_threshold=0.66, dust_size=4)
    assert len(strokes) >= 1
    pts, size, sdx, sdy, gate = strokes[0]
    assert abs(pts[0][0] - 81.5 / 160) < 0.03
    assert abs(pts[0][1] - 81.5 / 160) < 0.03
    assert gate == 1.0
    # Radius covers the 3px speck plus pad: size/2 · (160/HEAL_SIZE_REF) ∈ [2, 8] px.
    radius_px = size * 160 / (2.0 * HEAL_SIZE_REF)
    assert 2.0 < radius_px < 8.0


def test_detect_luma_regions_exposure_invariant():
    """The density proxy is self-normalized: a 2-stop exposure shift must yield
    the identical stroke set (no detection flicker while grading)."""
    img = _dusty_source()
    assert detect_luma_regions(img, 0.66, 4) == detect_luma_regions(img * 4.0, 0.66, 4)


def test_detect_luma_regions_strokes_are_plain_floats():
    strokes = detect_luma_regions(_dusty_source(), 0.66, 4)
    json.dumps(strokes)  # numpy scalars would raise / hash repr-dependently
    for pts, size, sdx, sdy, gate in strokes:
        assert all(type(c) is float for p in pts for c in p)
        assert all(type(v) is float for v in (size, sdx, sdy, gate))


def test_detect_luma_regions_clean_frame_is_empty():
    rng = np.random.default_rng(9)
    img = (np.full((160, 160, 3), 0.18) * (1.0 + rng.normal(0, 0.02, (160, 160, 3)))).astype(np.float32)
    assert detect_luma_regions(img, 0.66, 4) == []


def test_detect_ir_regions_speck_ungated():
    ir = np.full((120, 120), 0.9, dtype=np.float32)
    ir[60:63, 60:63] = 0.1
    strokes = detect_ir_regions(ir, threshold=0.5)
    assert len(strokes) == 1
    pts, size, sdx, sdy, gate = strokes[0]
    assert gate == 0.0
    assert abs(pts[0][0] - 61.5 / 120) < 0.02
    json.dumps(strokes)


def test_detect_ir_elongated_scratch_becomes_capsule():
    ir = np.full((200, 200), 0.9, dtype=np.float32)
    for t in range(80):
        x, y = 40 + t, 60 + t // 2
        ir[y : y + 2, x : x + 2] = 0.1
    strokes = detect_ir_regions(ir, threshold=0.5)
    assert len(strokes) == 1
    pts, size = strokes[0][0], strokes[0][1]
    assert len(pts) >= 3, "a hair must become a polyline capsule, not a circle"
    radius_px = size * 200 / (2.0 * HEAL_SIZE_REF)
    assert radius_px < 8.0, "capsule radius must hug the hair, not its bounding circle"


def test_pick_source_offsets_footprint_is_mask_free():
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[95:105, 95:105] = 1  # defect
    mask[95:105, 115:135] = 1  # dirty area right of it
    comps = _mask_to_strokes(mask, 1.5, 8)
    offsets = _pick_source_offsets(mask, comps, np.full((200, 200), 0.5, dtype=np.float32))
    assert len(offsets) == len(comps) == 2
    for (chain, radius, _area), (ox, oy) in zip(comps, offsets):
        b = int(1.2 * radius)
        for px, py in chain:
            sx, sy = int(px + ox), int(py + oy)
            assert mask[sy - b : sy + b + 1, sx - b : sx + b + 1].sum() == 0, "clone source overlaps a defect"


def test_pick_source_offsets_avoids_detail():
    """A mask-free patch full of dark detail must lose to a background-matched
    one — cloning detail onto a plain background is the visible failure mode."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[99:101, 60:140] = 1  # horizontal scratch → dirs = perpendicular / along axis
    guide = np.full((200, 200), 0.5, dtype=np.float32)
    guide[:90, :] = 0.1  # dark textured band above; below stays flat and matched
    guide[:90, ::2] = 0.3
    comps = _mask_to_strokes(mask, 4.0, 8)
    ox, oy = _pick_source_offsets(mask, comps, guide)[0]
    assert oy > 0, "picker chose the dark detail band over the matched background"


def test_detected_regions_heal_end_to_end():
    """detect → build_heal_regions → membrane heal removes the speck (ungated,
    healing the source frame directly)."""
    img = _dusty_source()
    strokes = detect_luma_regions(img, 0.66, 4, gate=0.0)
    assert strokes
    regions = build_heal_regions(strokes, [], (160, 160), 0, 0.0, False, False, 0.0, (160, 160))
    out = apply_manual_heals(img, *regions)
    assert out[80:83, 80:83].mean() > 0.1, "speck not cloned over"


def test_detect_luma_regions_precomputed_stats_equivalent():
    from negpy.features.retouch.logic import compute_dust_stats

    img = _dusty_source()
    stats = compute_dust_stats(img, 4)
    assert detect_luma_regions(img, 0.66, 4, stats=stats) == detect_luma_regions(img, 0.66, 4)
