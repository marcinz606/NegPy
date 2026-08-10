import json

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.retouch.logic import (
    _IR_SCORE_FLOOR,
    apply_hair_inpaint,
    apply_score_repair,
    compute_dust_stats,
    detect_luma_score,
    film_scale,
    hair_bake_token,
    ir_defect_score,
    lines_to_score,
    manual_bake_token,
    normalize_ir,
    repair_components,
    route_wide_defects,
    scratch_detect_bar,
    strokes_to_score,
    trace_scratch,
)
from negpy.features.retouch.models import HEAL_SIZE_REF, RetouchConfig


def _size_at_ref(diameter_px, shape):
    """Convert a brush diameter in image px to the stored HEAL_SIZE_REF-scale size."""
    return diameter_px * HEAL_SIZE_REF / max(shape)


def _grainy(h, w, level=0.5, sigma=0.01, seed=21):
    rng = np.random.default_rng(seed)
    return (np.full((h, w, 3), level) + rng.normal(0, sigma, (h, w, 3))).astype(np.float32)


def _heal(img, points, diameter_px):
    """Paint one stroke over ``img`` and repair it, the way the source bake does."""
    size = _size_at_ref(diameter_px, img.shape[:2])
    score = strokes_to_score(img, [(points, size, 0.0, 0.0)], [])
    if score is None:
        return img, None
    return np.asarray(repair_components(img, score, floor=False)), score


def test_brush_repairs_the_defect_and_leaves_clean_grain_alone():
    """The brush marks a search area, not a stamp: a speck under a generous brush is
    rebuilt, and the clean film around it inside the same brush is byte-identical."""
    img = _grainy(100, 100)
    img[49:52, 49:52] = 0.95

    out, _ = _heal(img, [[0.5, 0.5]], 15.0)

    assert out[49:52, 49:52].max() < 0.7, "dust inside the brush was not repaired"
    yy, xx = np.mgrid[0:100, 0:100]
    dist = np.hypot(xx - 50, yy - 50)
    clean_in_brush = (dist < 7) & (dist > 5.5)  # outside the defect's own skirt pad
    assert np.array_equal(out[clean_in_brush], img[clean_in_brush]), "clean grain inside the brush changed"


def test_brush_repairs_a_dark_scratch():
    """#791: a scratch has lost emulsion, so it reads *brighter* in transmittance than the
    film around it. The old bright-only gate could never repair one — the two-sided gate
    plus floor=False must."""
    img = _grainy(100, 160)
    for x in range(30, 130):
        img[58:61, x] = 0.85  # a scratch running across the frame

    out, _ = _heal(img, [[30.0 / 160, 59.0 / 100], [80.0 / 160, 59.0 / 100], [129.0 / 160, 59.0 / 100]], 10.0)

    err_before = float(np.abs(img[58:61, 30:130] - 0.5).mean())
    err_after = float(np.abs(out[58:61, 30:130] - 0.5).mean())
    assert err_after < err_before * 0.2, f"scratch not repaired ({err_before:.3f} -> {err_after:.3f})"


def test_brush_on_clean_film_is_a_noop():
    """Painting over film with nothing wrong with it must not smooth its grain away."""
    img = _grainy(100, 100)
    assert strokes_to_score(img, [([[0.5, 0.5]], _size_at_ref(15.0, (100, 100)), 0.0, 0.0)], []) is None


def test_faint_defect_still_repairs():
    """A mark too weak for the absolute bar is rescaled against the stroke's own peak,
    so the tool never silently does nothing where the user saw something."""
    img = _grainy(100, 100, sigma=0.004)
    img[49:52, 49:52] = 0.53  # ~1.5 sigma of density: under the absolute bar

    out, score = _heal(img, [[0.5, 0.5]], 15.0)
    assert score is not None, "a faint but real defect must still be found"
    assert abs(float(out[49:52, 49:52].mean()) - 0.5) < abs(float(img[49:52, 49:52].mean()) - 0.5)


def test_repair_footprint_stays_within_the_brush():
    """Nothing outside the painted capsule may change — the repaired footprint must not
    exceed the on-screen cursor."""
    img = _grainy(100, 100, level=0.4, seed=31)
    img[48:52, :] = 0.95  # dust strip across the whole frame

    out, _ = _heal(img, [[0.5, 0.5]], 16.0)  # radius 8 px

    changed = np.abs(out.astype(np.float64) - img).max(axis=2) > 5e-3
    ys, xs = np.where(changed)
    assert len(ys) > 0, "strip inside the brush was not repaired"
    dist = np.hypot(xs + 0.5 - 50.0, ys + 0.5 - 50.0)
    assert dist.max() <= 9.0, f"repair leaked {dist.max():.2f}px from centre, brush radius is 8"
    assert out[48:52, 80:].min() > 0.9, "strip outside the brush must stay untouched"


def test_stroke_radius_matches_cursor_fraction():
    """The repaired footprint must equal the overlay cursor circle: the cursor
    (overlay._brush_screen_radius) draws size/(2·HEAL_SIZE_REF) of the view, and the
    same fraction must hold at any render resolution."""
    size = 24.0
    for h, w in ((300, 450), (600, 900)):
        img = _grainy(h, w, seed=17)
        img[h // 2 - 1 : h // 2 + 2, :] = 0.95  # a strip crossing the whole frame
        score = strokes_to_score(img, [([[0.5, 0.5]], size, 0.0, 0.0)], [])
        assert score is not None

        _ys, xs = np.where(score < 1.0)
        reach = float(np.abs(xs - w / 2.0).max())
        expected = size / (2.0 * HEAL_SIZE_REF) * max(h, w)
        assert abs(reach - expected) <= 2.0, f"{h}x{w}: footprint {reach:.2f}px, cursor {expected:.2f}px"


def test_legacy_spots_repair():
    """Pre-stroke (nx, ny, size) spots still repair — they are just a one-point stroke."""
    img = _grainy(100, 100)
    img[48:53, 48:53] = 0.95
    score = strokes_to_score(img, [], [(0.5, 0.5, _size_at_ref(12.0, (100, 100)))])
    assert score is not None
    out = np.asarray(repair_components(img, score, floor=False))
    assert out[48:53, 48:53].max() < 0.7


def test_repair_components_matches_the_whole_frame_repair():
    """Cropping per defect is an optimization, not a different result."""
    img = _grainy(120, 120, seed=5)
    img[58:62, 58:62] = 0.95
    score = strokes_to_score(img, [([[0.5, 0.5]], _size_at_ref(14.0, (120, 120)), 0.0, 0.0)], [])
    assert score is not None
    cropped = np.asarray(repair_components(img, score, floor=False))
    whole = np.asarray(apply_score_repair(img, score, floor=False))
    np.testing.assert_allclose(cropped, whole, atol=2e-3)


def test_repair_leaves_unscored_pixels_byte_identical():
    img = _grainy(120, 120, seed=8)
    img[58:62, 58:62] = 0.95
    score = strokes_to_score(img, [([[0.5, 0.5]], _size_at_ref(14.0, (120, 120)), 0.0, 0.0)], [])
    out = np.asarray(repair_components(img, score, floor=False))
    untouched = np.repeat((score >= 1.0)[..., None], 3, axis=2)
    assert np.array_equal(out[untouched], img[untouched])


def test_manual_bake_token_tracks_the_strokes():
    empty = RetouchConfig()
    one = RetouchConfig(manual_heal_strokes=[([[0.3, 0.4]], 6.0, 0.0, 0.0)])
    two = RetouchConfig(manual_heal_strokes=[([[0.3, 0.4]], 6.0, 0.0, 0.0), ([[0.6, 0.6]], 6.0, 0.0, 0.0)])
    assert manual_bake_token(empty) == ""
    assert manual_bake_token(one) != manual_bake_token(two)
    assert manual_bake_token(one) == manual_bake_token(RetouchConfig(manual_heal_strokes=list(one.manual_heal_strokes)))


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


def test_preset_save_excludes_frame_specific_heals():
    # Enforcement moved from the presets service to the catalog: dust spots and
    # heal strokes have no SettingRow, so even saving every row can't leak them.
    from negpy.desktop.settings_catalog import all_rows, selected_flat_dict

    cfg = WorkspaceConfig(
        retouch=RetouchConfig(
            dust_remove=True,
            manual_dust_spots=[(0.1, 0.2, 6.0)],
            manual_heal_strokes=[([[0.3, 0.4]], 5.0, 0.02, -0.01)],
            scratch_lines=[(0.0, 0.5, 1.0, 0.51, 3.0)],
        )
    )
    data = selected_flat_dict(cfg, all_rows())
    assert "manual_heal_strokes" not in data
    assert "manual_dust_spots" not in data
    assert "scratch_lines" not in data
    assert data["dust_remove"] is True


def _dusty_source(h=160, w=160, seed=42):
    rng = np.random.default_rng(seed)
    img = (np.full((h, w, 3), 0.18) * (1.0 + rng.normal(0, 0.02, (h, w, 3)))).astype(np.float32)
    img[80:83, 80:83] = 0.005  # dust blocks scanner light: dark on the scan
    return img


def test_detect_luma_score_finds_dark_speck():
    img = _dusty_source()
    score, hairs = detect_luma_score(img, dust_threshold=0.66, dust_size=4)
    assert score is not None and hairs is None
    assert score[80:83, 80:83].max() < 1.0, "the speck must be scored as a defect"
    ys, xs = np.where(score < 1.0)
    assert abs(float(xs.mean()) - 81.5) < 5.0 and abs(float(ys.mean()) - 81.5) < 5.0


def test_detect_luma_score_exposure_invariant():
    """The density proxy is self-normalized: a 2-stop exposure shift must yield the
    identical score (no detection flicker while grading)."""
    img = _dusty_source()
    np.testing.assert_array_equal(detect_luma_score(img, 0.66, 4)[0], detect_luma_score(img * 4.0, 0.66, 4)[0])


def test_detect_luma_score_clean_frame_is_empty():
    rng = np.random.default_rng(9)
    img = (np.full((160, 160, 3), 0.18) * (1.0 + rng.normal(0, 0.02, (160, 160, 3)))).astype(np.float32)
    assert detect_luma_score(img, 0.66, 4)[0] is None


def test_ir_long_scratch_is_healed_by_the_fill():
    """A long thin scratch stays with the score-weighted fill (every pixel sits within
    reach of clean film) and is actually rebuilt — the #563 'cloned blobs' came from
    handing this class to a single-offset clone."""
    ir = np.full((200, 200), 0.9, dtype=np.float32)
    img = np.clip(np.random.default_rng(4).normal(0.5, 0.01, (200, 200, 3)), 0, 1).astype(np.float32)
    for t in range(80):
        x, y = 40 + t, 60 + t // 2
        ir[y : y + 2, x : x + 2] = 0.1
        img[y : y + 2, x : x + 2] = 0.06
    score = ir_defect_score(normalize_ir(ir), 0.5)
    assert route_wide_defects(score) is None, "thin: the fill's job, not the inpaint's"
    out = np.asarray(apply_score_repair(img, score))
    scratch = img[:, :, 0] < 0.1
    assert float(out[scratch].min()) > 0.35, "the scratch is rebuilt from its flanks"


def test_ir_mild_speck_stays_with_the_fill():
    """Small defects stay with the score-weighted fill — routing is reserved for
    components the fill's support can't see across."""
    ir = np.full((120, 120), 0.9, dtype=np.float32)
    ir[60:62, 55:66] = 0.1  # ~11px long, ~1px wide: well inside the fill's reach
    assert route_wide_defects(ir_defect_score(normalize_ir(ir), 0.5)) is None


def test_detected_specks_repair_end_to_end():
    """detect → shared fill removes the speck, in the source frame the meters read."""
    img = _dusty_source()
    score, _ = detect_luma_score(img, 0.66, 4)
    assert score is not None
    out = np.asarray(repair_components(img, score))
    assert out[80:83, 80:83].mean() > 0.1, "speck not rebuilt"


def test_detect_luma_score_precomputed_stats_equivalent():
    img = _dusty_source()
    stats = compute_dust_stats(img, 4)
    np.testing.assert_array_equal(detect_luma_score(img, 0.66, 4, stats=stats)[0], detect_luma_score(img, 0.66, 4)[0])


def test_apply_hair_inpaint_removes_hair_and_preserves_rest():
    """A hair over a gradient is filled from its surroundings; every non-masked
    pixel stays byte-identical (only fabricated pixels touch the 8-bit encode)."""
    h, w = 60, 60
    grad = np.linspace(0.2, 0.8, w, dtype=np.float32)[None, :].repeat(h, axis=0)
    img = np.stack([grad] * 3, axis=-1)
    hair = img.copy()
    hair[10:50, 30] = 0.95  # bright vertical hair
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:50, 30] = 1

    out = apply_hair_inpaint(hair, [mask], dilate_px=0)
    mb = mask.astype(bool)
    assert np.array_equal(out[~mb], hair[~mb]), "non-masked pixels must be untouched"
    # Healed hairline matches the local gradient (near its clean left/right neighbours).
    assert abs(float(out[30, 30, 0]) - float(img[30, 30, 0])) < 0.05


def test_apply_hair_inpaint_noop_on_empty_mask():
    img = np.full((20, 20, 3), 0.4, dtype=np.float32)
    assert apply_hair_inpaint(img, [np.zeros((20, 20), np.uint8)]) is img


def test_apply_hair_inpaint_upsamples_detection_mask():
    """A detection-scale mask is resized to the buffer before filling."""
    img = np.full((80, 80, 3), 0.3, dtype=np.float32)
    img[20:60, 40] = 0.95
    small = np.zeros((40, 40), np.uint8)
    small[10:30, 20] = 1  # half-res mask over the hair
    out = apply_hair_inpaint(img, [small])
    assert out[40, 40, 0] < 0.7, "hair not removed via upsampled mask"


def test_hair_bake_token_tracks_detection_params():
    a = RetouchConfig(dust_remove=True, dust_threshold=0.5, dust_size=4)
    assert hair_bake_token(a) != hair_bake_token(RetouchConfig(dust_remove=True, dust_threshold=0.6, dust_size=4))
    assert hair_bake_token(a) == hair_bake_token(RetouchConfig(dust_remove=True, dust_threshold=0.5, dust_size=4))


def _transport_scratch(h=400, w=1200, depth=0.12, slope=0.006, seed=3):
    """A faint, slightly sloped full-length scratch — the #788 defect. Per pixel it sits
    well under the brush's seed bar; only its length makes it findable."""
    rng = np.random.default_rng(seed)
    img = (np.full((h, w, 3), 0.45) + rng.normal(0, 0.012, (h, w, 3))).astype(np.float32)
    row0 = h // 2
    for x in range(w):
        y = int(round(row0 + slope * (x - w / 2)))
        img[y : y + 2, x] *= 1.0 + depth
    return img, row0, slope


def test_trace_scratch_finds_a_faint_sloped_line():
    img, row0, slope = _transport_scratch()
    h, w = img.shape[:2]
    line = trace_scratch(img, 0.5, (row0 + 0.5) / h)
    assert line is not None, "a real transport scratch must be traced from one click"
    nx0, ny0, nx1, ny1, _width = line
    assert nx1 - nx0 > 0.8, "the trace must follow the scratch across the frame"
    fitted = (ny1 - ny0) * h / max((nx1 - nx0) * w, 1e-6)
    assert abs(fitted - slope) < 0.002, f"slope {fitted:.4f} does not match the scratch's {slope:.4f}"


def test_trace_scratch_is_none_on_clean_film():
    rng = np.random.default_rng(5)
    img = (np.full((300, 900, 3), 0.45) + rng.normal(0, 0.012, (300, 900, 3))).astype(np.float32)
    assert trace_scratch(img, 0.5, 0.5) is None, "clean film must not yield a line"


def test_traced_line_repairs_the_scratch_and_spares_the_film():
    img, row0, _slope = _transport_scratch()
    h, w = img.shape[:2]
    line = trace_scratch(img, 0.5, (row0 + 0.5) / h)
    score = lines_to_score(img, [line])
    assert score is not None
    out = np.asarray(repair_components(img, score, floor=False, factor=film_scale((h, w))))

    # Follow the scratch's own path — it drifts with x, so a fixed row band is mostly film.
    xs = np.arange(50, w - 50)
    ys = np.round(row0 + _slope * (xs - w / 2)).astype(int)
    before = float(np.abs(img[ys, xs] - 0.45).mean())
    after = float(np.abs(out[ys, xs] - 0.45).mean())
    assert after < before * 0.4, f"scratch not repaired ({before:.4f} -> {after:.4f})"
    # Film a few rows away is untouched: the line is a search area like the brush.
    far = np.s_[row0 + 20 : row0 + 60, :]
    assert np.array_equal(out[far], img[far]), "film away from the line must be untouched"


def test_line_score_skips_stretches_with_no_scratch():
    """A transport scratch fades in and out, so the repair follows the evidence rather
    than painting the full width."""
    img, row0, _ = _transport_scratch()
    h, w = img.shape[:2]
    clean = img.copy()
    rng = np.random.default_rng(9)
    clean[:, : w // 2] = (np.full((h, w // 2, 3), 0.45) + rng.normal(0, 0.012, (h, w // 2, 3))).astype(np.float32)
    line = trace_scratch(clean, 0.75, (row0 + 0.5) / h)
    assert line is not None
    score = lines_to_score(clean, [line])
    assert score is not None
    left = (score[:, : w // 4] < 1.0).sum()
    right = (score[:, 3 * w // 4 :] < 1.0).sum()
    assert right > 0 and left == 0, f"scored the clean half too (left={left}, right={right})"


def test_manual_bake_token_tracks_scratch_lines():
    base = RetouchConfig()
    one = RetouchConfig(scratch_lines=[(0.0, 0.5, 1.0, 0.51, 3.0)])
    two = RetouchConfig(scratch_lines=[(0.0, 0.5, 1.0, 0.51, 3.0), (0.0, 0.2, 1.0, 0.21, 3.0)])
    assert manual_bake_token(base) == ""
    assert manual_bake_token(one) != manual_bake_token(two)


def test_scratch_lines_serialization_roundtrip():
    cfg = WorkspaceConfig(retouch=RetouchConfig(scratch_lines=[(0.1, 0.5, 0.9, 0.52, 3.0)]))
    restored = WorkspaceConfig.from_flat_dict(json.loads(json.dumps(cfg.to_dict())))
    assert [tuple(line) for line in restored.retouch.scratch_lines] == [(0.1, 0.5, 0.9, 0.52, 3.0)]


def test_hand_placed_repairs_are_not_refused_by_the_frame_budget():
    """The budget guards an automatic detector: a garbage IR plane can call half the frame a
    defect. A hand-placed repair is deliberate, and a full-width scratch covers more than the
    cap on its own, so it must not be judged by it."""
    h, w = 400, 1600
    score = np.ones((h, w), dtype=np.float32)
    score[190:214, :] = _IR_SCORE_FLOOR  # a wide full-width band: 6% of the frame

    assert route_wide_defects(score) is None, "over budget, as it would be for IR"
    assert route_wide_defects(score, budget=None) is not None, "a hand-placed repair must still route"


def test_wide_scratch_is_repaired():
    """The band is grown from the scratch, so a thick one is covered rather than cut to a
    fixed width and left with its flanks showing."""
    h, w, thick = 400, 1600, 8
    rng = np.random.default_rng(3)
    img = (np.full((h, w, 3), 0.45) + rng.normal(0, 0.012, (h, w, 3))).astype(np.float32)
    row0 = h // 2
    for x in range(w):
        img[row0 : row0 + thick, x] *= 1.12

    line = trace_scratch(img, 0.5, (row0 + thick / 2) / h)
    assert line is not None
    score = lines_to_score(img, [line])
    assert score is not None
    out = np.asarray(repair_components(img, score, floor=False, factor=film_scale((h, w))))
    routed = route_wide_defects(score, budget=None)
    if routed is not None:
        out = np.asarray(apply_hair_inpaint(out, [routed]))

    band = np.s_[row0 : row0 + thick, 60 : w - 60]
    before = float(np.abs(img[band] - 0.45).mean())
    after = float(np.abs(out[band] - 0.45).mean())
    assert after < before * 0.35, f"wide scratch not repaired ({before:.4f} -> {after:.4f})"


def test_line_sensitivity_trades_reach_for_restraint():
    """The slider must actually move the bar: a looser setting follows a scratch further and
    repairs a wider band, a tighter one holds back."""
    img, row0, _ = _transport_scratch(depth=0.08)
    h, w = img.shape[:2]
    line = trace_scratch(img, 0.5, (row0 + 0.5) / h, threshold=0.5)
    assert line is not None

    loose = lines_to_score(img, [line], threshold=0.05)
    tight = lines_to_score(img, [line], threshold=0.95)
    assert loose is not None
    n_loose = int((loose < 1.0).sum())
    n_tight = 0 if tight is None else int((tight < 1.0).sum())
    assert n_loose > n_tight, f"sensitivity did nothing (loose {n_loose}, tight {n_tight})"


def test_scratch_detect_bar_is_monotonic():
    bars = [scratch_detect_bar(v) for v in (0.05, 0.5, 0.95)]
    assert bars == sorted(bars), "higher must be more conservative"
