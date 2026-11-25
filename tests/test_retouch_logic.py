import dataclasses
import math
import json

import cv2
import negpy_core
import numpy as np
import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.retouch.logic import (
    _IR_SCORE_FLOOR,
    _IR_WRITE_HI,
    _density,
    _dust_stats_px,
    apply_hair_inpaint,
    apply_score_repair,
    compute_dust_stats,
    detect_bar,
    detect_luma_score,
    drop_exclusions,
    exclusion_cover,
    exclusion_token,
    film_scale,
    hair_bake_token,
    ir_defect_score,
    lines_to_score,
    luma_bake_token,
    manual_bake_token,
    normalize_ir,
    repair_components,
    route_wide_defects,
    scratch_detect_bar,
    trace_scratch,
)
from negpy.features.retouch.heal import apply_heal_stroke, apply_heal_strokes
from negpy.features.retouch.models import HEAL_SIZE_REF, RetouchConfig


def _size_at_ref(diameter_px, shape):
    """Convert a brush diameter in image px to the stored HEAL_SIZE_REF-scale size."""
    return diameter_px * HEAL_SIZE_REF / max(shape)


def _grainy(h, w, level=0.5, sigma=0.01, seed=21):
    rng = np.random.default_rng(seed)
    return (np.full((h, w, 3), level) + rng.normal(0, sigma, (h, w, 3))).astype(np.float32)


def _heal(img, points, diameter_px):
    """Paint one stroke over ``img`` and repair it, the way the source bake does."""
    return apply_heal_strokes(img, [(points, _size_at_ref(diameter_px, img.shape[:2]), 0.0, 0.0)])


def _disc(shape, cx, cy, r):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return np.hypot(xx - cx, yy - cy) <= r


def test_heal_click_removes_a_speck_and_keeps_the_grain():
    img = _grainy(120, 120)
    clean = img.copy()
    img[58:63, 58:63] = 0.95

    out = _heal(img, [[60.5 / 120, 60.5 / 120]], 14.0)

    core = np.s_[58:63, 58:63]
    assert abs(float(out[core].mean()) - 0.5) < 0.02, "the speck was not rebuilt to the film level"
    inside = _disc(img.shape, 60.5, 60.5, 6.0)
    ratio = float(out[inside].std()) / float(clean[inside].std())
    assert ratio > 0.6, f"the repair smoothed the grain away (std ratio {ratio:.2f})"


def test_heal_drag_repairs_a_faint_soft_blob():
    """Heal rebuilds the brush footprint unconditionally, including faint ~3-sigma soft blobs."""
    img = _grainy(120, 160, sigma=0.01)
    yy, xx = np.mgrid[0:120, 0:160]
    blob = np.exp(-(((xx - 80.0) ** 2 + (yy - 60.0) ** 2) / (2.0 * 4.0**2)))
    img += (0.03 * blob)[..., None].astype(np.float32)  # about 3 sigma of grain at its peak

    out = _heal(img, [[0.45, 0.5], [0.5, 0.5], [0.55, 0.5]], 20.0)

    core = _disc(img.shape, 80, 60, 3.0)
    before = float(np.abs(img[core] - 0.5).mean())
    after = float(np.abs(out[core] - 0.5).mean())
    assert after < 0.5 * before, f"faint blob not repaired ({before:.4f} -> {after:.4f})"


def test_heal_repairs_a_bright_scratch_along_a_drag():
    img = _grainy(100, 160)
    img[58:61, 30:130] = 0.85

    out = _heal(img, [[30.0 / 160, 59.0 / 100], [80.0 / 160, 59.0 / 100], [129.0 / 160, 59.0 / 100]], 10.0)

    err_before = float(np.abs(img[58:61, 32:128] - 0.5).mean())
    err_after = float(np.abs(out[58:61, 32:128] - 0.5).mean())
    assert err_after < err_before * 0.2, f"scratch not repaired ({err_before:.3f} -> {err_after:.3f})"


def test_heal_continues_a_tonal_edge_through_the_brush():
    """A speck on a horizon is rebuilt from both sides: the edge stays sharp, with no grey smear."""
    img = _grainy(120, 160, level=0.3, sigma=0.008)
    img[:60] += 0.4
    clean = img.copy()
    img[55:66, 75:86] = 0.95

    out = _heal(img, [[80.5 / 160, 60.5 / 120]], 16.0)

    assert abs(float(out[52:58, 78:84].mean()) - float(clean[52:58, 78:84].mean())) < 0.05, "bright side"
    assert abs(float(out[62:68, 78:84].mean()) - float(clean[62:68, 78:84].mean())) < 0.05, "dark side"


def test_heal_footprint_stays_within_the_brush():
    img = _grainy(100, 100, level=0.4, seed=31)
    img[48:52, :] = 0.95  # dust strip across the whole frame

    out = _heal(img, [[0.5, 0.5]], 16.0)  # radius 8 px

    changed = np.abs(out.astype(np.float64) - img).max(axis=2) > 0.0
    ys, xs = np.where(changed)
    assert len(ys) > 0, "strip inside the brush was not repaired"
    assert float(np.hypot(xs - 50.0, ys - 50.0).max()) <= 9.0, "the repair leaked past the brush"
    assert out[48:52, 80:].min() > 0.9, "the strip outside the brush stays as it is"


def test_heal_edge_cases_stay_finite():
    for img, stroke in (
        (_grainy(100, 100), ([[0.0, 0.0]], _size_at_ref(12.0, (100, 100)), 0.0, 0.0)),
        (np.ones((100, 100, 3), dtype=np.float32), ([[0.5, 0.5]], _size_at_ref(12.0, (100, 100)), 0.0, 0.0)),
        (_grainy(600, 900, seed=3), ([[0.5, 0.5]], 64.0, 0.0, 0.0)),
        (_grainy(40, 40), ([[0.5, 0.5]], 64.0, 0.0, 0.0)),
    ):
        out = apply_heal_strokes(img, [stroke])
        assert out.shape == img.shape and np.isfinite(out).all()


def test_legacy_spots_repair():
    """Pre-stroke (nx, ny, size) spots still repair: they run as one-point strokes."""
    img = _grainy(100, 100)
    img[48:53, 48:53] = 0.95
    out = apply_heal_strokes(img, [], [(0.505, 0.505, _size_at_ref(12.0, (100, 100)))])
    assert out[48:53, 48:53].max() < 0.7


def test_heal_is_deterministic_and_the_batch_matches_one_stroke_at_a_time():
    """Disjoint strokes heal in parallel waves; the result must equal applying them in order."""
    img = _grainy(240, 240, seed=7)
    centres = ((30, 30), (120, 120), (60, 190), (125, 135))  # the last two overlap the second
    for cx, cy in centres:
        img[cy - 2 : cy + 3, cx - 2 : cx + 3] = 0.92
    strokes = [([[(cx + 0.5) / 240, (cy + 0.5) / 240]], _size_at_ref(14.0, img.shape[:2]), 0.0, 0.0) for cx, cy in centres]

    batch = apply_heal_strokes(img, strokes)
    seq = img.copy()
    for points, size, _, _ in strokes:
        apply_heal_stroke(seq, points, size)
    assert np.array_equal(batch, seq)
    assert np.array_equal(batch, apply_heal_strokes(img, strokes))


def _square_score(shape, y0, y1, x0, x1):
    from negpy.features.retouch.logic import _mask_to_score

    mask = np.zeros(shape, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return _mask_to_score(mask, 2.5)


def test_repair_components_matches_the_whole_frame_repair():
    """Cropping per defect is an optimization, not a different result."""
    img = _grainy(120, 120, seed=5)
    img[58:62, 58:62] = 0.95
    score = _square_score((120, 120), 58, 62, 58, 62)
    cropped = np.asarray(repair_components(img, score, floor=False))
    whole = np.asarray(apply_score_repair(img, score, floor=False))
    np.testing.assert_allclose(cropped, whole, atol=2e-3)


def test_repair_leaves_unscored_pixels_byte_identical():
    img = _grainy(120, 120, seed=8)
    img[58:62, 58:62] = 0.95
    score = _square_score((120, 120), 58, 62, 58, 62)
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


def _two_speck_source(h=200, w=200, seed=7):
    rng = np.random.default_rng(seed)
    img = (np.full((h, w, 3), 0.18) * (1.0 + rng.normal(0, 0.02, (h, w, 3)))).astype(np.float32)
    spots = [(40, 40), (160, 160)]
    for y, x in spots:
        img[y : y + 3, x : x + 3] = 0.005
    return img, spots


def test_dust_stats_px_on_a_crop_matches_the_frame_windows():
    img = _grainy(400, 2400, seed=5)
    img[199:202, 1199:1202] *= 0.12
    dens = _density(img)
    base = 4 * film_scale(img.shape[:2])
    crop = np.ascontiguousarray(dens[100:300, 1100:1300])
    z = _dust_stats_px(crop, base)[2]
    assert z.shape == (200, 200)
    assert z[100, 100] >= 9.0


def test_detect_luma_score_finds_dark_speck():
    img = _dusty_source()
    score, hairs = detect_luma_score(img, dust_threshold=0.66, dust_size=4)
    assert score is not None and hairs is None
    assert score[80:83, 80:83].max() < 1.0, "the speck must be scored as a defect"
    ys, xs = np.where(score < 1.0)
    assert abs(float(xs.mean()) - 81.5) < 5.0 and abs(float(ys.mean()) - 81.5) < 5.0


def test_detect_luma_score_finds_a_speck_at_the_smallest_size():
    img = _dusty_source()
    score, _ = detect_luma_score(img, 0.66, 2)
    assert score is not None and (score[80:83, 80:83] < _IR_WRITE_HI).all()


def test_detect_luma_score_exposure_invariant():
    """The density proxy is self-normalized: a 2-stop exposure shift must yield the
    identical score (no detection flicker while grading)."""
    img = _dusty_source()
    np.testing.assert_array_equal(detect_luma_score(img, 0.66, 4)[0], detect_luma_score(img * 4.0, 0.66, 4)[0])


def test_detect_luma_score_clean_frame_is_empty():
    rng = np.random.default_rng(9)
    img = (np.full((160, 160, 3), 0.18) * (1.0 + rng.normal(0, 0.02, (160, 160, 3)))).astype(np.float32)
    assert detect_luma_score(img, 0.66, 4)[0] is None


def test_exclusion_releases_the_detection_under_it():
    img = _dusty_source()
    score, _ = detect_luma_score(img, 0.66, 4)
    assert score is not None
    # Wide enough to cover the speck and the skirt the score ramps over.
    size = _size_at_ref(30, img.shape)
    out, hairs = drop_exclusions(score, None, [([[81.5 / 160.0, 81.5 / 160.0]], size)])
    assert out is None and hairs is None, "nothing left to repair, so nothing is baked"


def test_exclusion_elsewhere_leaves_the_detection_alone():
    img = _dusty_source()
    score, _ = detect_luma_score(img, 0.66, 4)
    out, _ = drop_exclusions(score, None, [([[0.1, 0.1]], _size_at_ref(30, img.shape))])
    assert out is not None
    np.testing.assert_array_equal(out, score)


def test_exclusion_releases_only_the_film_it_covers():
    """The band is the cut, so a mark the brush clips keeps its repair on the side the
    brush missed. Releasing the whole mark from a clipped edge is what this replaced."""
    img = _dusty_source()
    score, _ = detect_luma_score(img, 0.66, 4)
    assert score is not None
    marked = score < 1.0
    ys, xs = np.where(marked)
    near, far = (int(ys[0]), int(xs[0])), (int(ys[-1]), int(xs[-1]))
    # A brush covering the near end of the mark and nowhere near the far one.
    clip = ([[near[1] / 160.0, near[0] / 160.0]], _size_at_ref(6, img.shape))
    out, _ = drop_exclusions(score, None, [clip])
    assert out is not None, "the far side of the mark is still a defect"
    assert out[near] > score[near], "the covered end is released"
    assert out[far] == score[far], "the end it never reached is untouched"


def test_exclusion_rim_does_not_step():
    """The score is a ramp, so a hard cut at the brush edge would print the edge. The rim
    is feathered, as a manual heal feathers its own."""
    img = _dusty_source()
    score, _ = detect_luma_score(img, 0.66, 4)
    assert score is not None
    out, _ = drop_exclusions(score, None, [([[0.5, 0.5]], _size_at_ref(8, img.shape))])
    assert out is not None
    row = out[80, :]
    assert float(np.abs(np.diff(row)).max()) < 0.5, "no cliff across the released band"


def test_exclusion_releases_only_the_covered_span_of_a_hair():
    """A hair is long and thin, so whole-component release and covered-span release differ
    most here. The mask is binary, so the footprint applies unfeathered."""
    hair = np.zeros((200, 200), dtype=np.uint8)
    hair[100, 20:180] = 1
    out_score, out_hair = drop_exclusions(None, hair, [([[0.15, 0.5]], _size_at_ref(6, (200, 200)))])
    assert out_score is None
    assert out_hair is not None, "the rest of the hair is still repaired"
    assert not out_hair[100, 30], "the covered span is released"
    assert out_hair[100, 170], "the far end of the hair is not"


def test_exclusion_clearing_the_last_hair_skips_the_bake():
    hair = np.zeros((200, 200), dtype=np.uint8)
    hair[100, 98:102] = 1
    _score, out_hair = drop_exclusions(None, hair, [([[0.5, 0.5]], _size_at_ref(40, (200, 200)))])
    assert out_hair is None, "nothing left to inpaint, so nothing is baked"


def test_exclusion_leaves_untouched_specks_repaired():
    """A band must not spill onto a defect it never reached."""
    img, _ = _two_speck_source()
    score, _ = detect_luma_score(img, 0.66, 4)
    assert score is not None
    near = ([[41.5 / 200.0, 41.5 / 200.0]], _size_at_ref(24, img.shape))
    out, _ = drop_exclusions(score, None, [near])
    assert out is not None
    assert not (out < 1.0)[40:43, 40:43].any(), "the covered speck is released"
    np.testing.assert_array_equal(out[150:175, 150:175], score[150:175, 150:175])


def test_exclusion_stroke_covers_the_film_between_its_points():
    """A drag is sampled sparsely, so the band has to be swept along the path — loose dabs
    at the sample points would leave the film between them repaired."""
    stroke = ([[0.2, 0.5], [0.8, 0.5]], _size_at_ref(6, (200, 200)))
    cover = exclusion_cover([stroke], (200, 200))
    assert cover[100, 40] and cover[100, 160], "the ends are covered"
    assert cover[100, 60:140].all(), "and so is every pixel between them"


def test_dust_busy_label_does_not_contradict_an_exclusion():
    """The bake's busy toast reads back to the user, so it must not say it is repairing
    dust on the pass a right-click asked it to stop repairing."""
    from negpy.services.rendering.image_processor import _dust_step_label

    base = RetouchConfig(dust_remove=True)
    assert _dust_step_label(base) == "repairing dust"
    assert "repairing" not in _dust_step_label(dataclasses.replace(base, dust_exclusion_strokes=[([[0.5, 0.5]], 6.0)]))


def test_exclusion_token_tracks_the_strokes():
    base = RetouchConfig(dust_remove=True)
    assert exclusion_token(base) == ""
    excluded = dataclasses.replace(base, dust_exclusion_strokes=[([[0.5, 0.5]], 6.0)])
    assert exclusion_token(excluded) != ""
    assert luma_bake_token(excluded) != luma_bake_token(base)
    assert hair_bake_token(excluded) != hair_bake_token(base)


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


def _marked(img, thr=0.66, size=4):
    score, hair = detect_luma_score(img, thr, size)
    mark = np.zeros(img.shape[:2], bool)
    if score is not None:
        mark |= score < _IR_WRITE_HI
    if hair is not None:
        mark |= hair > 0
    return mark


def test_detect_luma_score_covers_the_speck_footprint():
    """The mark covers the whole speck, not its brightest pixel."""
    img = _dusty_source()
    img[80:87, 80:87] = 0.005
    assert _marked(img)[80:87, 80:87].mean() >= 0.9


def test_detect_luma_score_joins_a_hair_into_one_component():
    img = _dusty_source(h=200, w=200)
    img[80:83, 80:83] = 0.18
    img[100:102, 40:120] = 0.02
    score, hair = detect_luma_score(img, 0.66, 4)
    assert hair is not None, "an 80 px hair routes to inpaint"
    n, _ = cv2.connectedComponents(hair.astype(np.uint8), connectivity=8)
    assert n == 2, "one hair, one component"
    assert hair[100:102, 45:115].all()
    assert score is None or not (score[100:102, 40:120] < 1.0).any(), "the hair is not also a speck"


def test_detect_bar_is_monotonic():
    bars = [detect_bar(s) for s in np.linspace(0.0, 1.0, 21)]
    assert all(a < b for a, b in zip(bars, bars[1:]))
    assert detect_bar(0.0) == 3.0 and detect_bar(0.66) == 9.0 and 40.0 < detect_bar(0.99) < 48.0
    assert detect_bar(1.0) == math.inf
    assert abs(detect_bar(0.66 - 1e-6) - detect_bar(0.66 + 1e-6)) < 1e-3


def test_lower_threshold_marks_more_and_the_deepest_speck_always():
    img = _dusty_source(h=200, w=200)
    img[80:83, 80:83] = 0.18
    for y, level in ((40, 0.005), (100, 0.06), (160, 0.11)):
        img[y : y + 3, 100:103] = level
    sweep = (0.3, 0.66, 0.9, 0.99)
    counts = [int(_marked(img, thr).sum()) for thr in sweep]
    assert counts[0] >= counts[1] >= counts[2] >= counts[3] > 0
    for thr in sweep:
        assert _marked(img, thr)[40:43, 100:103].any(), f"deepest speck lost at {thr}"


def test_detect_bar_top_rejects_a_strong_mark():
    rng = np.random.default_rng(42)
    img = (np.full((200, 200, 3), 0.18) * (1.0 + rng.normal(0, 0.06, (200, 200, 3)))).astype(np.float32)
    img[100:103, 100:103] = 0.10
    assert _marked(img, 0.66)[100:103, 100:103].any()
    assert not _marked(img, 0.99).any()


def test_threshold_at_max_turns_its_detection_off():
    img = _dusty_source(h=200, w=200)
    img[100:102, 40:120] = 0.02
    score, hair = detect_luma_score(img, 1.0, 4)
    assert score is None and hair is None
    score, hair = detect_luma_score(img, 0.66, 4, hair_threshold=1.0)
    assert hair is None and score is not None, "specks keep their own bar"
    score, hair = detect_luma_score(img, 1.0, 4, hair_threshold=0.66)
    assert score is None and hair is not None, "hairs keep theirs"


def test_detect_luma_score_grainy_clean_frame_is_empty():
    rng = np.random.default_rng(3)
    grain = cv2.GaussianBlur(rng.normal(0, 0.06, (240, 240, 3)).astype(np.float32), (0, 0), 0.8)
    img = np.clip(0.18 * (1.0 + grain), 0.001, None).astype(np.float32)
    assert detect_luma_score(img, 0.66, 4) == (None, None)


def test_texture_protects_a_compact_mark_and_a_loose_hair_bar_reaches_a_hair():
    rng = np.random.default_rng(5)
    img = _dusty_source(h=200, w=200)
    img[80:83, 80:83] = 0.18
    blocks = rng.uniform(0.85, 1.18, (25, 13)).repeat(8, axis=0).repeat(8, axis=1)
    img[:, 100:, :] *= blocks[:, :100, None]
    img[60:65, 40:45] = 0.05  # speck on flat film
    img[60:65, 150:155] = 0.05  # same speck inside the structure
    img[140:142, 110:190] = 0.005  # hair across the structure
    mark = _marked(img)
    assert mark[60:65, 40:45].any()
    assert not mark[55:70, 145:160].any()
    assert not mark[140:142, 120:180].any()
    _, hair = detect_luma_score(img, 0.66, 4, hair_threshold=0.01)
    assert hair is not None and hair[140:142, 120:180].mean() > 0.8


def test_bright_rim_along_a_tonal_edge_is_not_a_hair():
    rng = np.random.default_rng(42)
    img = (np.full((200, 200, 3), 0.18) * (1.0 + rng.normal(0, 0.02, (200, 200, 3)))).astype(np.float32)
    img[100:, :] *= 0.3
    img[99:101, 20:180] = 0.035  # rim on the edge
    img[40:42, 20:180] = 0.11  # hair on flat film
    mark = _marked(img)
    assert not mark[99:101, 30:170].any()
    assert mark[40:42, 30:170].mean() > 0.8


def _speck_and_hair_source():
    rng = np.random.default_rng(42)
    img = (np.full((200, 200, 3), 0.18) * (1.0 + rng.normal(0, 0.06, (200, 200, 3)))).astype(np.float32)
    img[40:43, 40:43] = 0.10  # z ≈ 37
    img[120:122, 30:170] = 0.10  # z ≈ 25
    return img


def test_hair_threshold_moves_hairs_only():
    img = _speck_and_hair_source()
    score, hair = detect_luma_score(img, 0.66, 4, hair_threshold=1.0)
    assert hair is None, "the hair sits under the tightest hair bar"
    assert score is not None and (score[40:43, 40:43] < _IR_WRITE_HI).any(), "the speck keeps its own bar"


def test_loose_hair_threshold_finds_a_hair_under_the_spot_bar():
    img = _speck_and_hair_source()
    score, hair = detect_luma_score(img, 1.0, 4, hair_threshold=0.66)
    assert hair is not None and hair[120:122, 40:160].all()
    assert score is None or not (score[40:43, 40:43] < _IR_WRITE_HI).any()


def test_hair_threshold_defaults_to_the_spot_bar():
    img = _speck_and_hair_source()
    a, b = detect_luma_score(img, 0.66, 4), detect_luma_score(img, 0.66, 4, hair_threshold=0.66)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


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
    assert hair_bake_token(a) != hair_bake_token(dataclasses.replace(a, dust_hair_threshold=0.8))


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


def test_luma_bake_token_tracks_the_toggle_and_params():
    from negpy.features.retouch.logic import luma_bake_token

    off = luma_bake_token(RetouchConfig(dust_remove=False))
    on = luma_bake_token(RetouchConfig(dust_remove=True))
    assert off == ""
    assert on != off
    assert luma_bake_token(RetouchConfig(dust_remove=True, dust_threshold=0.5)) != on
    assert luma_bake_token(RetouchConfig(dust_remove=True, dust_hair_threshold=0.8)) != on


def test_dust_toggle_changes_engine_source_hash():
    """The speck bake mutates the source, so the toggle must change the hash both
    engine caches key on — or the GPU keeps rendering the stale uploaded texture."""
    import dataclasses
    from unittest.mock import patch

    from negpy.domain.models import WorkspaceConfig
    from negpy.services.rendering.image_processor import ImageProcessor

    proc = ImageProcessor()
    proc.engine_gpu = None  # CPU branch; both engines receive the same hash
    img = np.random.default_rng(0).random((64, 96, 3)).astype(np.float32)
    cfg = WorkspaceConfig()
    seen = []

    def spy(buf, settings, source_hash, context=None):
        seen.append(source_hash)
        return buf

    with patch.object(proc.engine_cpu, "process", side_effect=spy):
        proc.run_pipeline(img, cfg, "src", render_size_ref=1600.0, prefer_gpu=False, readback_metrics=False)
        cfg_on = dataclasses.replace(cfg, retouch=dataclasses.replace(cfg.retouch, dust_remove=True))
        proc.run_pipeline(img, cfg_on, "src", render_size_ref=1600.0, prefer_gpu=False, readback_metrics=False)

    assert seen[0] != seen[1], "dust_remove toggle left source_hash unchanged"


def test_trace_scratch_follows_a_steep_line_across_a_wide_hq_buffer():
    """At the steepest slope a wide frame moves the line far past the search rows."""
    img, row0, slope = _transport_scratch(h=600, w=5000, depth=0.2, slope=0.018)
    h, w = img.shape[:2]
    line = trace_scratch(img, 0.5, (row0 + 0.5) / h)
    assert line is not None
    nx0, ny0, nx1, ny1, _width = line
    assert nx1 - nx0 > 0.8
    fitted = (ny1 - ny0) * h / max((nx1 - nx0) * w, 1e-6)
    assert abs(fitted - slope) < 0.002


def test_heal_stroke_respects_avoid_mask():
    """Pixels flagged in ``avoid`` are excluded from the PatchMatch donor pool."""
    import negpy_core

    img = _grainy(96, 96, level=0.35, sigma=0.01, seed=17)
    # Contaminate the upper half with bright green; leave the lower half clean at 0.35.
    img[:44, :, :] = 0.35
    img[:44, :, 1] = 0.90
    img[46:51, 46:51] = 0.98

    avoid = np.zeros((96, 96), dtype=np.uint8)
    avoid[:45, :] = 1
    path = np.array([[48, 48]], dtype=np.int32)
    out = img.copy()
    negpy_core.heal_stroke(out, (0, 0, 96, 96), path, 5, avoid=avoid)

    core = out[46:51, 46:51]
    assert abs(float(core[:, :, 1].mean()) - 0.35) < 0.04, "donor came from avoided region"


def test_heal_vignette_gradient_has_no_internal_step_seam():
    """Across a strong tonal gradient, donor patches carry their DC shift so interior seams stay smooth."""
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:140, 0:180].astype(np.float32)
    base = 0.12 + 0.55 * (xx / 179.0) + 0.18 * (yy / 139.0)
    grain = rng.normal(0.0, 0.012, (140, 180, 3)).astype(np.float32)
    clean = np.clip(base[..., None] + grain, 0.02, 0.98).astype(np.float32)
    img = clean.copy()
    img[58:82, 78:102] = 0.99

    out = _heal(img, [[90.0 / 180, 70.0 / 140]], 34.0)

    patch_err = np.abs(out[56:84, 76:104].mean(axis=2) - clean[56:84, 76:104].mean(axis=2))
    assert float(patch_err.mean()) < 0.025
    # Adjacent interior differences should remain bounded by local grain, never a tonal cliff.
    interior = out[60:80, 80:100].mean(axis=2)
    max_step = max(float(np.abs(np.diff(interior, axis=0)).max()), float(np.abs(np.diff(interior, axis=1)).max()))
    assert max_step < 0.08, f"internal tonal step cliff ({max_step:.3f})"


def test_heal_sanitizes_non_finite_and_extreme_inputs():
    """NaN/Inf/zero/negative pixels, 1x1 frames, full-frame brushes, off-frame strokes, duplicate points and F-order arrays."""
    import negpy_core
    import pytest

    img = _grainy(80, 80, level=0.4, sigma=0.01, seed=9)
    img[10, 10] = np.nan
    img[12, 14] = np.inf
    img[15, 16] = -np.inf
    img[18, 20] = -0.5
    img[22, 24] = 0.0
    img[38:43, 38:43] = np.inf

    # Fortran-ordered input + duplicate points (size 180 -> brush radius 4.5 px on 80x80, covering 38:43).
    f_img = np.asfortranarray(img)
    out = apply_heal_strokes(f_img, [([[40.5 / 80, 40.5 / 80], [40.5 / 80, 40.5 / 80], [40.5 / 80, 40.5 / 80]], 180.0, 0.0, 0.0)])
    assert out.flags["C_CONTIGUOUS"]
    assert np.isfinite(out[35:46, 35:46]).all()
    assert abs(float(out[38:43, 38:43].mean()) - 0.4) < 0.05

    # 1x1 image and out-of-frame strokes safely no-op.
    tiny = np.full((1, 1, 3), 0.5, dtype=np.float32)
    assert np.array_equal(apply_heal_strokes(tiny, [([[0.5, 0.5]], 10.0, 0.0, 0.0)]), tiny)
    off = apply_heal_strokes(_grainy(40, 40), [([[-2.0, -2.0]], 8.0, 0.0, 0.0), ([[0.5, 0.5]], float("nan"), 0.0, 0.0)])
    assert off.shape == (40, 40, 3)

    # Brush covering 100% of frame falls back cleanly without panicking.
    small = _grainy(24, 24, level=0.45)
    full = apply_heal_strokes(small, [([[0.5, 0.5]], 400.0, 0.0, 0.0)])
    assert np.isfinite(full).all()

    # Invalid brush radii and malformed path shapes raise ValueError cleanly.
    buf = _grainy(32, 32)
    with pytest.raises(ValueError):
        negpy_core.heal_stroke(buf, (0, 0, 32, 32), np.array([[16, 16]], dtype=np.int32), 0)
    with pytest.raises(ValueError):
        negpy_core.heal_stroke(buf, (0, 0, 32, 32), np.array([[16, 16]], dtype=np.int32), 100_000)
    with pytest.raises(ValueError):
        negpy_core.heal_stroke(buf, (0, 0, 32, 32), np.array([16, 16, 17], dtype=np.int32), 4)


def test_heal_long_diagonal_scratch_partitions_bounded_at_export_scale():
    """A full-frame diagonal scratch at 2400x3600 subdivides into tight crops instead of allocating a full-frame bbox."""
    import time

    h, w = 2400, 3600
    img = np.full((h, w, 3), 0.42, dtype=np.float32)
    xs = np.linspace(180, w - 180, 1200).astype(np.int32)
    ys = np.linspace(120, h - 120, 1200).astype(np.int32)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            img[np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)] = 0.95

    t0 = time.perf_counter()
    out = apply_heal_strokes(img, [([[180.0 / w, 120.0 / h], [(w - 180.0) / w, (h - 120.0) / h]], 5.0, 0.0, 0.0)])
    elapsed = time.perf_counter() - t0

    mid_vals = out[ys[100:-100], xs[100:-100]]
    assert abs(float(mid_vals.mean()) - 0.42) < 0.02, "diagonal scratch not healed"
    assert elapsed < 1.5, f"diagonal scratch took {elapsed:.2f}s (subdivision regression)"


def test_heal_continues_curved_line_across_perpendicular_scratch():
    """A narrow high-contrast curved arc crossed by a long perpendicular scratch heals without cutting a gap in the arc."""
    h, w = 160, 160
    clean = _grainy(h, w, level=0.48, sigma=0.008, seed=23)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Dense dark feature in transmittance (bright white curved arc in print), 4 px wide.
    arc_x = 52.0 + 0.006 * (yy - 30.0) ** 2
    arc = np.exp(-0.5 * ((xx - arc_x) / 1.8) ** 2)
    clean = np.clip(clean * (1.0 - 0.82 * arc[..., None]), 0.04, 0.98).astype(np.float32)

    img = clean.copy()
    # Horizontal scratch cutting across both dark background and the curved arc at y = 79..81.
    img[79:82, 30:130] = 0.92
    out = _heal(img, [[30.0 / w, 80.5 / h], [80.0 / w, 80.5 / h], [130.0 / w, 80.5 / h]], 11.0)

    # Check every row inside the healed horizontal band right along the curved arc centerline:
    for y in range(75, 86):
        cx = int(round(52.0 + 0.006 * (y - 30.0) ** 2))
        got = float(out[y, cx - 1 : cx + 2].mean())
        want = float(clean[y, cx - 1 : cx + 2].mean())
        assert abs(got - want) < 0.06, f"arc severed at row {y}: got {got:.3f}, want {want:.3f}"
    # And background along the scratch away from the arc is healed back to ~0.48.
    assert abs(float(out[79:82, 95:120].mean()) - 0.48) < 0.03


def test_heal_reconstructs_tight_circular_arc_via_rotation():
    """A scratch across the top apex of a tight semicircle heals by rotating angled shoulder patches."""
    h, w = 80, 120
    clean = _grainy(h, w, level=0.22, sigma=0.008, seed=41)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.abs(np.hypot(xx - 60.0, yy - 58.0) - 26.0)
    w_ring = np.clip((2.2 - dist) / 1.4, 0.0, 1.0)[..., None]
    clean = (clean * (1.0 - w_ring) + 0.82 * w_ring).astype(np.float32)

    img = clean.copy()
    img[22:45, 57:64] = 0.04
    pts = [[60.5 / w, (y + 0.5) / h] for y in range(22, 45, 4)]
    # Restrict upper semicircle so no unrotated horizontal arc exists anywhere in the crop.
    out = _heal(img[:56], [[p[0], p[1] * h / 56.0] for p in pts], 10.0)

    assert float(out[32, 60].mean()) > 0.58, f"circle apex severed: {float(out[32, 60].mean()):.3f}"
    assert float(out[24, 60].mean()) < 0.34 and float(out[40, 60].mean()) < 0.34


def test_heal_reconstructs_tapered_stripe_via_scale():
    """A scratch across a perspective-foreshortened stripe heals at the local tapered width."""
    h, w = 120, 120
    clean = _grainy(h, w, level=0.20, sigma=0.008, seed=53)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    hw = 1.6 + 0.045 * yy
    t = np.clip((hw + 1.0 - np.abs(xx - 60.0)) / 1.2, 0.0, 1.0)[..., None]
    clean = (clean * (1.0 - t) + 0.80 * t).astype(np.float32)

    img = clean.copy()
    img[54:59, 40:81] = 0.04
    out = _heal(img, [[40.0 / w, 56.5 / h], [60.0 / w, 56.5 / h], [80.0 / w, 56.5 / h]], 8.0)

    assert float(out[56, 60].mean()) > 0.68
    assert float(out[56, 52].mean()) < 0.30


def test_heal_preserves_clumpy_multi_pixel_dye_clouds():
    """Wide strokes on coarse-grained scans preserve multi-pixel dye-cloud clumping rather than 1x1 confetti."""
    rng = np.random.default_rng(77)
    h, w = 160, 160
    # Correlated ~6 px dye clouds via 2x upsampled + smoothed random field, plus fine sensor noise.
    coarse = rng.normal(0.0, 0.09, (h // 3, w // 3, 3)).astype(np.float32)
    clouds = np.repeat(np.repeat(coarse, 3, axis=0), 3, axis=1)
    pad_h, pad_w = h - clouds.shape[0], w - clouds.shape[1]
    clouds = np.pad(clouds, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    clouds = 0.25 * (np.roll(clouds, -1, axis=0) + np.roll(clouds, 1, axis=0) + np.roll(clouds, -1, axis=1) + np.roll(clouds, 1, axis=1))
    clean = np.clip(0.30 + clouds + rng.normal(0.0, 0.008, (h, w, 3)).astype(np.float32), 0.05, 0.90)

    img = clean.copy()
    img[40:120, 75:86] = 0.94
    pts = [[80.5 / w, (y + 0.5) / h] for y in range(40, 121, 8)]
    out = _heal(img, pts, 16.0)

    c_strip = clean[45:115, 74:87]
    o_strip = out[45:115, 74:87]
    c_lag4 = float(np.sqrt(np.mean((c_strip[:, 4:] - c_strip[:, :-4]) ** 2)))
    o_lag4 = float(np.sqrt(np.mean((o_strip[:, 4:] - o_strip[:, :-4]) ** 2)))
    c_lag1 = float(np.sqrt(np.mean((c_strip[:, 1:] - c_strip[:, :-1]) ** 2)))
    o_lag1 = float(np.sqrt(np.mean((o_strip[:, 1:] - o_strip[:, :-1]) ** 2)))
    assert o_lag4 > 0.78 * c_lag4, f"3-8px dye-cloud clumps collapsed: {o_lag4:.4f} vs {c_lag4:.4f}"
    assert (o_lag1 / o_lag4) < 1.25 * (c_lag1 / c_lag4), "clumpy dye clouds turned into 1x1 sandpaper"


def test_heal_rejects_fortran_ordered_arrays():
    """PyO3 and Python heal entry points reject Fortran-ordered image, path and avoid arrays."""
    c_img = np.full((32, 40, 3), 0.4, dtype=np.float32, order="C")
    f_img = np.asfortranarray(c_img)
    c_path = np.array([[10, 10], [20, 15]], dtype=np.int32, order="C")
    f_path = np.asfortranarray(c_path)
    f_avoid = np.asfortranarray(np.zeros((32, 40), dtype=np.uint8))

    with pytest.raises(ValueError, match="C-contiguous"):
        apply_heal_stroke(f_img, [[0.5, 0.5]], 8.0)
    with pytest.raises(ValueError, match="C-contiguous"):
        negpy_core.heal_stroke(f_img, (0, 0, 40, 32), c_path, 4)
    with pytest.raises(ValueError, match="C-contiguous"):
        negpy_core.heal_stroke(c_img, (0, 0, 40, 32), f_path, 4)
    with pytest.raises(ValueError, match="C-contiguous"):
        negpy_core.heal_stroke(c_img, (0, 0, 40, 32), c_path, 4, f_avoid)
    with pytest.raises(ValueError, match="C-contiguous"):
        negpy_core.heal_strokes(f_img, [((0, 0, 40, 32), c_path, 4)])
    with pytest.raises(ValueError, match="C-contiguous"):
        negpy_core.heal_strokes(c_img, [((0, 0, 40, 32), f_path, 4)])


def test_heal_handles_extreme_i32_coordinates_and_dense_scribbles():
    """Extreme i32 path coordinates and dense back-and-forth scribbles clip and finish cleanly."""
    img = _grainy(560, 560, level=0.40, sigma=0.012, seed=91)
    img[276:284, 276:284] = 0.95

    extreme = np.array([[-2_147_483_648, -2_147_483_648], [280, 280], [2_147_483_647, 2_147_483_647]], dtype=np.int32)
    out_ext = img.copy()
    negpy_core.heal_stroke(out_ext, (0, 0, 560, 560), extreme, 6)
    assert np.isfinite(out_ext).all()
    assert float(out_ext[278:282, 278:282].mean()) < 0.52

    # 8,000-point back-and-forth scribble inside a compact box must heal as one mask without OOM/hang.
    t = np.arange(8000, dtype=np.int32)
    sx = 260 + (t % 2) * 40
    sy = 260 + ((t // 2) % 40)
    scribble = np.ascontiguousarray(np.stack([sx, sy], axis=1), dtype=np.int32)
    out_scr = img.copy()
    negpy_core.heal_stroke(out_scr, (0, 0, 560, 560), scribble, 5)
    assert np.isfinite(out_scr).all()
    assert abs(float(out_scr[278:282, 278:282].mean()) - 0.40) < 0.04


def test_heal_handles_fully_covered_crop_without_overflow():
    """Brushes covering 100% or 98% of `rect` complete with finite values and zero overflow."""
    img = _grainy(32, 32, level=0.42, sigma=0.01, seed=93)
    orig = img.copy()
    negpy_core.heal_stroke(img, (0, 0, 32, 32), np.array([[16, 16]], dtype=np.int32), 48)
    assert np.isfinite(img).all()
    np.testing.assert_array_equal(img, orig)

    nearly_full = orig.copy()
    negpy_core.heal_stroke(nearly_full, (0, 0, 32, 32), np.array([[16, 16]], dtype=np.int32), 21)
    assert np.isfinite(nearly_full).all()
    assert abs(float(nearly_full.mean()) - 0.42) < 0.03


def test_heal_respects_avoid_mask_with_narrow_border_or_full_exclusion():
    """`avoid` is never relaxed to copy excluded pixels, even when only a 2 px border or 0 px remain."""
    h, w = 48, 48
    img = _grainy(h, w, level=0.32, sigma=0.008, seed=95)
    # Leave only a 2 px clean border at ~0.32; fill interior non-hole region with bright 0.96 avoid tone.
    avoid = np.ones((h, w), dtype=np.uint8)
    avoid[:2, :] = 0
    avoid[-2:, :] = 0
    avoid[:, :2] = 0
    avoid[:, -2:] = 0
    img[avoid != 0] = 0.96
    img[22:27, 22:27] = 0.85
    avoid[20:29, 20:29] = 0
    # Mark collar around the brush as avoid too so only the 2 px outer frame is clean.
    avoid_tight = np.ones((h, w), dtype=np.uint8)
    avoid_tight[:2, :] = 0
    avoid_tight[-2:, :] = 0
    avoid_tight[:, :2] = 0
    avoid_tight[:, -2:] = 0

    out_border = img.copy()
    negpy_core.heal_stroke(out_border, (0, 0, w, h), np.array([[24, 24]], dtype=np.int32), 5, avoid_tight)
    assert float(out_border[23:26, 23:26].mean()) < 0.42, "avoid pixels leaked when only 2 px border was clean"

    # 100% exterior masked by avoid: hole must not absorb 0.98 avoid tones.
    full_avoid = np.ones((h, w), dtype=np.uint8)
    out_none = np.full((h, w, 3), 0.98, dtype=np.float32)
    out_none[20:29, 20:29] = 0.25
    negpy_core.heal_stroke(out_none, (0, 0, w, h), np.array([[24, 24]], dtype=np.int32), 3, full_avoid)
    assert float(out_none[24, 24].mean()) < 0.35, "100% avoid exterior leaked into hole"


def test_heal_subdivides_hooked_stroke_and_matches_batch_bits():
    """A long hooked scratch masks its returning tail in early sub-strokes and matches heal_strokes."""
    h, w = 620, 620
    img = _grainy(h, w, level=0.36, sigma=0.012, seed=97)
    # Hooked U-turn scratch whose return leg passes within 22 px of the outgoing leg.
    pts_px = [(80, y) for y in range(80, 521, 20)] + [(92, 530), (104, 520)] + [(104, y) for y in range(500, 79, -20)]
    path = np.array(pts_px, dtype=np.int32)
    img[78:533, 78:83] = 0.95
    img[518:533, 80:107] = 0.95
    img[78:523, 102:107] = 0.95

    out_single = img.copy()
    out_batch = img.copy()
    negpy_core.heal_stroke(out_single, (0, 0, w, h), path, 5)
    negpy_core.heal_strokes(out_batch, [((0, 0, w, h), path, 5)])
    np.testing.assert_array_equal(out_single, out_batch)
    assert float(out_single[120:480, 79:82].max()) < 0.50, "returning unhealed leg leaked into outgoing sub-stroke"
    assert float(out_single[120:480, 103:106].max()) < 0.50, "return leg left unhealed"


def test_heal_dodges_wide_multiscale_neighbour_dust_blob():
    """High-DPI unclicked dust blobs (~13x13 px) are caught on coarse pyramid levels and excluded."""
    h, w = 180, 180
    img = _grainy(h, w, level=0.34, sigma=0.01, seed=99)
    # Clicked scratch spot at (90, 90) surrounded by four wide 13x13 unclicked dust blobs.
    img[85:96, 85:96] = 0.94
    for cy, cx in [(90, 114), (90, 66), (66, 90), (114, 90)]:
        img[cy - 6 : cy + 7, cx - 6 : cx + 7] = 0.96

    out = img.copy()
    negpy_core.heal_stroke(out, (10, 10, 170, 170), np.array([[80, 80]], dtype=np.int32), 8)
    assert float(out[86:95, 86:95].max()) < 0.50, "wide neighbour dust blob copied into healed spot"


def test_heal_tightens_oversized_full_frame_rect():
    """Passing a full-frame `rect` for a small spot heals cleanly without scanning the whole frame."""
    img = _grainy(900, 1200, level=0.38, sigma=0.01, seed=101)
    img[448:453, 598:603] = 0.94
    out = img.copy()
    negpy_core.heal_stroke(out, (0, 0, 1200, 900), np.array([[600, 450]], dtype=np.int32), 6)
    assert abs(float(out[448:453, 598:603].mean()) - 0.38) < 0.04
    assert np.array_equal(out[:200, :200], img[:200, :200])
