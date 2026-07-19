import os
import tempfile

import cv2
import numpy as np
import tifffile

from negpy.domain.models import WorkspaceConfig
from negpy.features.retouch.logic import (
    _HAIR_INPAINT_GAMMA,
    _HAIR_INPAINT_RADIUS,
    _IR_GAMMA_FALLBACK,
    _IR_GAMMA_HI,
    _fit_refraction_gammas,
    _mask_to_strokes,
    apply_hair_inpaint,
    apply_ir_attenuation,
    apply_manual_heals,
    build_heal_regions,
    detect_ir_regions,
    downsample_ir,
    ir_bake_token,
    ir_detect_cutoff,
    ir_ratio_and_gain,
    normalize_ir,
)
from negpy.features.retouch.models import RetouchConfig
from negpy.infrastructure.loaders.factory import LoaderFactory


def test_retouch_config_defaults_include_ir_fields():
    cfg = RetouchConfig()
    assert cfg.ir_dust_remove is False
    assert 0.05 < cfg.ir_threshold < 0.95
    assert cfg.ir_inpaint_radius >= 1


def test_workspace_config_backcompat_for_ir_fields():
    """Old config dicts without IR fields must deserialize with sane defaults."""
    cfg = WorkspaceConfig.from_flat_dict({})
    assert cfg.retouch.ir_dust_remove is False


def test_workspace_config_roundtrip_ir_fields():
    cfg = WorkspaceConfig(
        retouch=RetouchConfig(ir_dust_remove=True, ir_threshold=0.4, ir_inpaint_radius=5),
    )
    flat = cfg.to_dict()
    assert flat["ir_dust_remove"] is True
    assert flat["ir_threshold"] == 0.4

    restored = WorkspaceConfig.from_flat_dict(flat)
    assert restored.retouch.ir_dust_remove is True
    assert restored.retouch.ir_threshold == 0.4


def test_detect_ir_regions_heals_defect_end_to_end():
    """IR speck → synthesized ungated stroke → membrane clone removes it."""
    h, w = 80, 80
    rng = np.random.default_rng(17)
    img = (np.full((h, w, 3), 0.5) + rng.normal(0, 0.01, (h, w, 3))).astype(np.float32)
    img[39:42, 39:42] = 0.95
    ir = np.full((h, w), 0.9, dtype=np.float32)
    ir[39:42, 39:42] = 0.05

    strokes, _ = detect_ir_regions(normalize_ir(ir), 0.5, pad_px=3.0)
    assert len(strokes) == 1
    assert strokes[0][4] == 0.0  # IR regions are ungated

    regions = build_heal_regions(strokes, [], (h, w), 0, 0.0, False, False, 0.0, (w, h))
    out = apply_manual_heals(img, *regions)
    assert out[40, 40, 0] < 0.7


def test_detect_ir_regions_no_defect_is_empty():
    ir = np.full((40, 40), 0.9, dtype=np.float32)
    strokes, hair = detect_ir_regions(normalize_ir(ir), 0.5)
    assert strokes == [] and hair is None


def test_ir_detect_cutoff_mapping_and_direction():
    """The slider→ratio-cutoff map: lower slider catches more (higher cutoff) in
    both modes; the attenuation band sits lower than detection-only."""
    assert ir_detect_cutoff(0.1, True) > ir_detect_cutoff(0.9, True)
    assert ir_detect_cutoff(0.1, False) > ir_detect_cutoff(0.9, False)
    assert ir_detect_cutoff(0.35, True) < ir_detect_cutoff(0.35, False)
    assert abs(ir_detect_cutoff(0.35, True) - 0.71) < 1e-6


def test_normalize_ir_flat_plane_is_unity():
    """Clean film → ratio ~1.0 everywhere; a dust dip on a mild illumination
    gradient is still detected at the default cutoff (raw-IR thresholding missed
    dips that sat above the global cutoff)."""
    ir = np.full((120, 120), 0.8, dtype=np.float32)
    assert abs(float(normalize_ir(ir).mean()) - 1.0) < 0.01

    grad = np.linspace(0.7, 0.85, 120, dtype=np.float32)[:, None].repeat(120, axis=1)
    grad[60:63, 60:63] = grad[60:63, 60:63] * 0.4  # dust dip on the gradient
    strokes, _ = detect_ir_regions(normalize_ir(grad), ir_detect_cutoff(0.35, True))
    assert len(strokes) == 1


def test_detect_ir_regions_coverage_abort():
    """A cutoff that marks the whole frame returns nothing (never smears the preview)."""
    ratio = np.full((80, 80), 0.5, dtype=np.float32)  # 100% below any sane cutoff
    assert detect_ir_regions(ratio, 0.8)[0] == []


def test_tiff_loader_reads_ir_from_extrasamples():
    h, w = 16, 24
    rgb = np.full((h, w, 3), 30000, dtype=np.uint16)
    ir = np.full((h, w), 50000, dtype=np.uint16)
    rgba_with_ir = np.dstack([rgb, ir])
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "scan.tif")
        tifffile.imwrite(path, rgba_with_ir, photometric="rgb", extrasamples=("unspecified",))
        ctx_mgr, metadata = LoaderFactory().get_loader(path)
        with ctx_mgr:
            pass
        assert metadata["ir"] is not None
        assert metadata["ir"].shape == (h, w)
        assert metadata["ir"].dtype == np.float32
        assert abs(float(metadata["ir"].mean()) - (50000.0 / 65535.0)) < 1e-3


def test_tiff_loader_sidecar_ir_file():
    h, w = 12, 18
    rgb = np.full((h, w, 3), 20000, dtype=np.uint16)
    ir = np.full((h, w), 60000, dtype=np.uint16)
    with tempfile.TemporaryDirectory() as td:
        rgb_path = os.path.join(td, "scan.tif")
        ir_path = os.path.join(td, "scan_IR.tif")
        tifffile.imwrite(rgb_path, rgb, photometric="rgb")
        tifffile.imwrite(ir_path, ir, photometric="minisblack")
        ctx_mgr, metadata = LoaderFactory().get_loader(rgb_path)
        with ctx_mgr:
            pass
        assert metadata["ir"] is not None
        assert metadata["ir"].shape == (h, w)
        assert abs(float(metadata["ir"].mean()) - (60000.0 / 65535.0)) < 1e-3


def test_tiff_loader_no_ir_when_rgb_only():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "rgb_only.tif")
        tifffile.imwrite(path, np.full((10, 10, 3), 30000, dtype=np.uint16), photometric="rgb")
        _, metadata = LoaderFactory().get_loader(path)
        assert metadata["ir"] is None


def test_rawpy_loader_dng_thumbnail_subifd_ir():
    """VueScan/Adobe-style DNG: thumbnail IFD0 + SubIFD carrying the 4-sample LinearRaw
    RGB+IR data (img02.dng's structure) — not NegPy's own single-IFD DNG output."""
    h, w = 8, 10
    thumb = np.zeros((4, 5, 3), dtype=np.uint8)
    full = np.random.randint(0, 65535, (h, w, 4)).astype(np.uint16)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "scan.dng")
        with tifffile.TiffWriter(path) as tw:
            tw.write(thumb, photometric="rgb", subfiletype=1, subifds=1)
            tw.write(full, photometric=34892, subfiletype=0, planarconfig="CONTIG")
        ctx_mgr, metadata = LoaderFactory().get_loader(path)
        with ctx_mgr:
            pass
        assert metadata["ir"] is not None
        assert metadata["ir"].shape == (h, w)


def test_tiff_loader_silverfast_multipage_ir():
    """SilverFast iSRD: IR stored as page 2 with NewSubfileType=4 (transparency mask)."""
    h, w = 16, 24
    rgb = np.full((h, w, 3), 30000, dtype=np.uint16)
    ir = np.full((h, w), 50000, dtype=np.uint16)
    thumb = np.full((4, 6, 3), 30000, dtype=np.uint16)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "silverfast.tif")
        with tifffile.TiffWriter(path) as tw:
            tw.write(rgb, photometric="rgb", subfiletype=0)
            tw.write(thumb, photometric="rgb", subfiletype=1)
            tw.write(ir, photometric="minisblack", subfiletype=0)
        ctx_mgr, metadata = LoaderFactory().get_loader(path)
        with ctx_mgr:
            pass
        assert metadata["ir"] is not None
        assert metadata["ir"].shape == (h, w)
        assert metadata["ir"].dtype == np.float32
        assert abs(float(metadata["ir"].mean()) - (50000.0 / 65535.0)) < 1e-3


def test_ir_dust_remove_field_invalidates_retouch_hash():
    from negpy.kernel.caching.logic import calculate_config_hash

    a = RetouchConfig(ir_dust_remove=False)
    b = RetouchConfig(ir_dust_remove=True)
    assert calculate_config_hash(a) != calculate_config_hash(b)


def test_ir_attenuation_field_invalidates_retouch_hash():
    from negpy.kernel.caching.logic import calculate_config_hash

    assert calculate_config_hash(RetouchConfig(ir_attenuation=True)) != calculate_config_hash(RetouchConfig(ir_attenuation=False))


def test_ir_bake_token_active_and_empty():
    on = RetouchConfig(ir_dust_remove=True, ir_attenuation=True)
    assert ir_bake_token(on, has_ir=True) == "|irdiv1"
    assert ir_bake_token(on, has_ir=False) == ""  # no IR plane → nothing to bake
    assert ir_bake_token(RetouchConfig(ir_dust_remove=True, ir_attenuation=False), True) == ""
    assert ir_bake_token(RetouchConfig(ir_dust_remove=False, ir_attenuation=True), True) == ""


def test_ir_ratio_and_gain_properties():
    """Gain never darkens a clean pixel, clamps at 2.0, and is identity on clean
    film (ratio≈1); γ stays inside the clamp. Dust the visible really carries still gets its
    correction — the clean-base cap may only bite where the visible is clean."""
    h = w = 200
    ir = np.full((h, w), 0.9, dtype=np.float32)
    ir[95:105, 95:105] = 0.2  # opaque-ish core
    ir[60:64, 60:64] = 0.78 * 0.9  # semi-transparent speck (ratio ≈ 0.78)
    img = np.full((h, w, 3), 0.5, dtype=np.float32)
    img[95:105, 95:105] = 0.15
    img[60:64, 60:64] = 0.42

    ratio, gain, degenerate, gammas = ir_ratio_and_gain(ir, img)
    assert not degenerate
    assert gain.shape == (h, w, 3)
    assert gain.min() >= 1.0 - 1e-4
    assert gain.max() <= 2.0 + 1e-4
    clean = ratio > 0.99
    assert clean.any()
    assert abs(float(gain[np.broadcast_to(clean[..., None], gain.shape)].reshape(-1, 3).mean()) - 1.0) < 1e-3
    assert all(1.0 <= g <= 2.2 for g in gammas)
    assert gain[62, 62].min() > 1.05  # the semi-transparent speck is still corrected
    assert gain[100, 100].min() > 1.5  # ...and so is the opaque core


def test_ir_ratio_and_gain_degenerate_on_image_content():
    """An IR plane carrying image content (a broad gradient, like B&W silver) is
    flagged degenerate so the caller skips both the bake and IR strokes. Guards that
    the check stays on the raw ratio — decontamination would unmix the gradient away."""
    grad = np.linspace(0.2, 0.9, 200, dtype=np.float32)[None, :].repeat(200, axis=0)
    img = np.stack([grad] * 3, axis=-1)
    _, _, degenerate, _ = ir_ratio_and_gain(grad.copy(), img)
    assert degenerate


def test_ir_dead_margins_do_not_mask_the_frame():
    """Scan overrun past the film gate reads at the sensor floor. No-signal, not dust:
    left as a dip it swamps the coverage abort and no speck is found."""
    h, w = 400, 300
    rng = np.random.default_rng(3)
    ir = np.clip(0.85 + rng.normal(0, 0.025, (h, w)), 0.0, 1.0).astype(np.float32)
    img = np.full((h, w, 3), 0.5, dtype=np.float32)
    ir[:, :20] = 0.004  # holder mask, left
    ir[:, -20:] = 0.004  # ...and right
    img[:, :20] = 0.002
    img[:, -20:] = 0.002
    speck = (slice(200, 208), slice(150, 158))
    ir[speck] *= 0.35
    img[speck] *= 0.55

    ratio, gain, degenerate, _ = ir_ratio_and_gain(ir, img)
    assert not degenerate
    assert float(ratio[:, :20].min()) > 0.99  # dead margin reads as clean film
    assert float(gain[:, :20].max()) < 1.01  # ...and is never "corrected"
    strokes, _ = detect_ir_regions(ratio, ir_detect_cutoff(0.66, True), pad_px=3.0)
    assert len(strokes) >= 1


def test_ir_noisy_clean_film_is_not_degenerate():
    """A real IR plane carries a few percent of noise, deepened by the min-preserving
    downsample. Clean C41 must not read as B&W silver just for being noisy."""
    rng = np.random.default_rng(11)
    ir = np.clip(0.8 + rng.normal(0, 0.03, (400, 300)), 0.0, 1.0).astype(np.float32)
    img = np.clip(0.5 + rng.normal(0, 0.03, (400, 300, 3)), 1e-3, 1.0).astype(np.float32)

    _, _, degenerate, _ = ir_ratio_and_gain(ir, img)
    assert not degenerate


def _ghosted_frame(ghost: float, size: int = 240):
    """Synthetic scan: sharp-edged image content, an IR plane that partially absorbs
    it (the ghost normalize_ir's spatial high-pass can't remove), and one dust speck
    attenuating both planes."""
    rng = np.random.default_rng(7)
    img = np.full((size, size, 3), 0.55, dtype=np.float32)
    img[:, size // 2 :] = 0.18  # hard vertical edge — survives dilate+blur
    img[40:80, 40:200] = 0.30  # bar
    img += rng.normal(0, 0.004, img.shape).astype(np.float32)
    img = np.clip(img, 1e-3, 1.0)

    ir = np.full((size, size), 0.9, dtype=np.float32)
    if ghost > 0.0:
        ir *= (img.mean(axis=-1) / 0.55) ** ghost  # dye blocks IR where density is high
    speck = (slice(150, 158), slice(60, 68))
    ir[speck] *= 0.35
    img[speck] *= 0.55
    return img.astype(np.float32), ir.astype(np.float32), speck


def test_ir_decontaminate_removes_ghost():
    """A visible-image ghost in the IR plane is unmixed out: the clean-film ratio
    flattens, while the dust speck still dips and is still detected."""
    img, ir, speck = _ghosted_frame(ghost=0.15)
    raw = normalize_ir(ir)
    clean, _, degenerate, _ = ir_ratio_and_gain(ir, img)
    assert not degenerate

    off = np.ones(raw.shape, dtype=bool)
    off[speck] = False  # clean film only — the speck is signal, not spread
    assert clean[off].std() < raw[off].std() / 2.0
    assert abs(float(np.median(clean[off])) - 1.0) < 0.02

    assert float(clean[speck].min()) < 0.75  # the defect survives unmixing
    strokes, _ = detect_ir_regions(clean, ir_detect_cutoff(0.66, True), pad_px=3.0)
    assert len(strokes) >= 1


def test_ir_decontaminate_noop_on_clean_ir():
    """No ghost → the fit lands at noise level and the ratio passes through untouched,
    so a clean scanner renders exactly as it did before."""
    img, ir, _ = _ghosted_frame(ghost=0.0)
    raw = normalize_ir(ir)
    clean, _, _, _ = ir_ratio_and_gain(ir, img)
    assert np.allclose(clean, raw, atol=1e-6)


def test_ir_gain_median_near_unity_with_ghost():
    """The reported bug: without unmixing the ghost pushes the ratio below the identity
    point frame-wide, so the bake lifts the whole buffer before the meters read it."""
    img, ir, _ = _ghosted_frame(ghost=0.15)
    _, gain, _, _ = ir_ratio_and_gain(ir, img)
    assert abs(float(np.median(gain)) - 1.0) < 1e-3


def _fit_inputs(gamma_true: float, residue_slope: float, size: int = 200):
    """Inputs for the γ fit: dust on flat film with a known slope, plus a *larger*
    population of `_ir_decontaminate` residue over textured content carrying a steeper
    spurious one. Returns (ratio, vis_log, img_det).

    Residue outnumbering dust is the real case (a treeline frame put ~1300 residue pixels
    in the band against ~860 of dust), and is why the median needs the flat restriction
    under it. Built directly rather than rendered from a synthetic frame: residue only
    appears where real content beats normalize_ir's base, and a frame tuned until that
    happens would test the tuning, not the estimator."""
    rng = np.random.default_rng(3)
    img = np.full((size, size, 3), 0.5, dtype=np.float32)
    img[100:, :] = np.where(rng.random((size - 100, size, 1)) > 0.5, 0.5, 0.12)  # foliage
    img += rng.normal(0, 0.002, img.shape).astype(np.float32)  # grain: no exact-zero Laplacian
    ratio = np.ones((size, size), dtype=np.float32)
    vis_log = np.zeros((size, size, 3), dtype=np.float32)

    def _paint(rows: slice, cols: slice, slope: float) -> None:
        a = rng.uniform(0.72, 0.90, (rows.stop - rows.start, cols.stop - cols.start)).astype(np.float32)
        ratio[rows, cols] = a
        vis_log[rows, cols, :] = (slope * np.log(a))[:, :, None]

    _paint(slice(10, 40), slice(10, 40), gamma_true)  # 900 px of dust, on flat film
    _paint(slice(110, 150), slice(10, 50), residue_slope)  # 1600 px of residue, on foliage
    return ratio, vis_log.astype(np.float32), img.astype(np.float32)


def test_gamma_fit_ignores_decontamination_residue_at_image_edges():
    """The reported bug: dust on sky came back darker than the sky, tinted cyan — the fit
    read the edge residue as dust and the bake overshot the film base. It must return the
    dust's slope, not a blend."""
    ratio, vis_log, img = _fit_inputs(gamma_true=1.1, residue_slope=6.0)
    gammas = _fit_refraction_gammas(ratio, vis_log, img)
    assert all(abs(g - 1.1) < 0.15 for g in gammas), gammas

    # The least-squares fit this replaced, on the very same input: dragged to the cap.
    band = (ratio > 0.70) & (ratio < 0.92)
    xb = np.log(ratio[band])
    ls = [float(np.sum(xb * vis_log[:, :, c][band]) / np.sum(xb * xb)) for c in range(3)]
    assert all(g > 2.0 for g in ls), ls


def test_gamma_fit_still_reaches_the_cap_for_strongly_scattering_dust():
    """Dust that genuinely attenuates visible far harder than IR still pins γ at the cap —
    robustness must not cost the scans (iSRD-style) that legitimately want a high γ."""
    ratio, vis_log, img = _fit_inputs(gamma_true=3.0, residue_slope=3.0)
    assert all(abs(g - _IR_GAMMA_HI) < 1e-5 for g in _fit_refraction_gammas(ratio, vis_log, img))


def test_ir_gain_never_lifts_a_pixel_past_its_local_clean_base():
    """The reported dark outline: downsample_ir is min-preserving while the visible arrives
    area-averaged, so the IR dip is wider than the defect the visible carries (9 px against
    3 here) and the uncapped gain skirt lifts clean film."""
    h = w = 300  # roomy enough that the specks stay under the degenerate-coverage guard
    ir = np.full((h, w), 0.9, dtype=np.float32)
    img = np.full((h, w, 3), 0.30, dtype=np.float32)
    ir_dip = np.zeros((h, w), dtype=bool)
    vis_dip = np.zeros((h, w), dtype=bool)
    rng = np.random.default_rng(5)
    for _ in range(40):
        y, x = int(rng.integers(12, h - 14)), int(rng.integers(12, w - 14))
        ir[y - 3 : y + 6, x - 3 : x + 6] *= 0.82  # IR: 9 px wide (the min-pooled footprint)
        img[y : y + 3, x : x + 3] *= 0.82  # visible: only the middle 3 px are really dust
        ir_dip[y - 3 : y + 6, x - 3 : x + 6] = True
        vis_dip[y : y + 3, x : x + 3] = True
    _, gain, degenerate, _ = ir_ratio_and_gain(ir, img)
    assert not degenerate

    out = np.asarray(apply_ir_attenuation(img, gain))
    skirt = ir_dip & ~vis_dip  # clean film the IR dips over: the bake must leave it alone
    assert skirt.sum() > 2000
    assert np.abs(out[skirt] / 0.30 - 1.0).max() < 0.02, "the gain skirt lifted clean film"
    # These cores can't also check that real dust survives the cap: the IR and visible dips
    # are exactly proportional here, so _ir_decontaminate unmixes them away. That side is
    # asserted in test_ir_ratio_and_gain_properties.


def test_gamma_fit_falls_back_when_the_band_is_too_small():
    """A frame with almost no semi-transparent dust has nothing to fit — the gain is ~1
    regardless of γ, so the fallback stands rather than fitting noise."""
    ratio = np.ones((200, 200), dtype=np.float32)
    ratio[:10, :10] = 0.8  # 100 px, under the 500-px gate
    vis_log = np.zeros((200, 200, 3), dtype=np.float32)
    img = np.full((200, 200, 3), 0.5, dtype=np.float32)
    assert _fit_refraction_gammas(ratio, vis_log, img) == (_IR_GAMMA_FALLBACK,) * 3


def test_hysteresis_grows_from_cores_but_admits_no_new_defects():
    """Both halves of Canny's rule: a shallow dip touching a strong core is pulled in
    whole, while an equally shallow dip with no core stays out."""
    ratio = np.ones((80, 200), dtype=np.float32)
    ratio[40, 20:60] = 0.40  # core, below the cutoff
    ratio[40, 60:100] = 0.65  # its shallow continuation — above the cutoff
    ratio[40, 140:180] = 0.65  # same depth, but no core anywhere near it

    strokes, hair = detect_ir_regions(ratio, 0.586, pad_px=1.0, min_area=1)
    covered = np.zeros((80, 200), dtype=bool)
    for pts, size, _sdx, _sdy, _g in strokes:
        for px, py in np.atleast_2d(pts):
            covered[int(round(py * 80)), int(round(px * 200))] = True
    if hair is not None:
        covered |= hair.astype(bool)

    assert covered[40, 20:60].all(), "the core itself"
    assert covered[40, 60:100].any(), "its above-cutoff continuation is pulled in by the core"
    assert not covered[40, 140:180].any(), "an identical dip with no core must stay rejected"


def _curled_hair_mask(size: int = 120) -> np.ndarray:
    """A hair that curls back on itself: thin everywhere, but its PCA extent/width
    reads compact, exactly like the real one on samples/ir/18.tiff."""
    m = np.zeros((size, size), dtype=np.uint8)
    t = np.linspace(0, 2.2 * np.pi, 600)
    xs = (size / 2 + (size / 3) * np.cos(t)).astype(int)
    ys = (size / 2 + (size / 3.5) * np.sin(2 * t)).astype(int)
    for x, y in zip(xs, ys):
        m[max(y - 1, 0) : y + 2, max(x - 1, 0) : x + 2] = 1  # ~3 px thick
    return m


def test_twisted_hair_routes_to_inpaint():
    """The reported bug: a hair that curls scores PCA aspect < 3 and used to fall through
    to a compact membrane disc. Thinness is twist-invariant, so it routes to the inpaint."""
    m = _curled_hair_mask()
    comps, hair = _mask_to_strokes(m, 3.0, 512)
    assert hair is not None, "curled hair must reach the inpaint mask"
    assert int(hair.sum()) == int(m.sum()), "the whole hair, not part of it"
    assert not comps, "and it must not also become membrane strokes"


def test_round_speck_stays_a_membrane_stroke():
    """The other side of the same rule: a compact speck of comparable area must not be
    dragged into the inpaint by the thinness test."""
    m = np.zeros((120, 120), dtype=np.uint8)
    cv2.circle(m, (60, 60), 18, 1, -1)  # area ~1000, same order as the hair above
    comps, hair = _mask_to_strokes(m, 3.0, 512)
    assert hair is None
    assert len(comps) == 1


def test_downsample_ir_preserves_a_subpixel_hair():
    """The reported bug: INTER_AREA averages a sub-pixel hair's dip away and shatters its
    component. Min-preserving keeps the dip deep and the hair in one piece."""
    ir = np.full((900, 1350), 0.9, dtype=np.float32)
    ir[400:404, 200:1100] = 0.25  # a 4 px hair, sub-pixel once downsampled 4.5x
    target = 300  # 1350 -> 300 == the 4.5x of a 7184px scan at preview size

    area = cv2.resize(ir, (target, 200), interpolation=cv2.INTER_AREA)
    mine = downsample_ir(ir, target)
    assert mine.shape == (200, target)

    cut = 0.586
    r_area, r_mine = normalize_ir(area), normalize_ir(mine)
    assert r_mine.min() < r_area.min(), "the dip must survive better than INTER_AREA"
    n_mine = cv2.connectedComponentsWithStats((r_mine < cut).astype(np.uint8), 8)[0] - 1
    assert n_mine == 1, "the hair stays one component"
    assert float((r_mine < cut).sum()) > float((r_area < cut).sum())


def test_downsample_ir_is_a_noop_when_not_downsampling():
    """A scan already at or below preview size must pass through untouched — the erode
    is a resample artefact fix, not a filter to apply unconditionally."""
    ir = np.full((200, 300), 0.9, dtype=np.float32)
    ir[100:104, 50:250] = 0.25
    assert np.array_equal(downsample_ir(ir, 300), ir)


def test_downsample_ir_matches_across_preview_and_export():
    """WYSIWYG: the preview decode passes explicit dims, export lets the helper compute
    them. Same full-res plane must give the same buffer, or the two paths detect
    different region sets."""
    rng = np.random.default_rng(3)
    ir = np.clip(rng.normal(0.9, 0.01, (900, 1350)), 0, 1).astype(np.float32)
    ir[400:404, 200:1100] = 0.25
    export = downsample_ir(ir, 300)
    preview = downsample_ir(ir, 300, dims=(export.shape[1], export.shape[0]))
    assert np.array_equal(export, preview)


def _noisy_frame(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(0.5, 0.15, (h, w, 3)), 0, 1).astype(np.float32)


def _inpaint_whole_frame(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The pre-optimisation implementation, kept as the oracle: gamma-encode the entire
    buffer, inpaint, decode, keep the masked pixels. apply_hair_inpaint now does this per
    connected component's bbox, which must be bit-exact, not merely close."""
    enc = np.clip(np.ascontiguousarray(img, dtype=np.float32), 0.0, 1.0) ** (1.0 / _HAIR_INPAINT_GAMMA)
    filled = cv2.inpaint((enc * 255.0 + 0.5).astype(np.uint8), mask, _HAIR_INPAINT_RADIUS, cv2.INPAINT_NS)
    dec = (filled.astype(np.float32) / 255.0) ** _HAIR_INPAINT_GAMMA
    out = np.ascontiguousarray(img, dtype=np.float32).copy()
    mb = mask.astype(bool)
    out[mb] = dec[mb]
    return out


def test_hair_inpaint_per_component_matches_whole_frame():
    """cv2.inpaint only propagates outward from the mask boundary, so filling each defect
    inside its own bbox is exactly the whole-frame result — without gamma-encoding 34MP to
    serve a hairline (1.7s -> 0.3s on a full scan)."""
    img = _noisy_frame(300, 400)
    m = np.zeros((300, 400), dtype=np.uint8)
    cv2.line(m, (60, 80), (200, 190), 1, 2)
    mb = m.astype(bool)

    out = apply_hair_inpaint(img, [m], dilate_px=0)
    assert np.array_equal(out, _inpaint_whole_frame(img, m))
    assert not np.array_equal(out[mb], img[mb]), "the hair must actually be filled"
    assert np.array_equal(out[~mb], img[~mb]), "clean pixels stay byte-identical"


def test_hair_inpaint_matches_for_scattered_hairs():
    """Hairs in opposite corners — their union bbox is the whole frame, which is why the
    crop is per-component and not one bbox over the union."""
    img = _noisy_frame(300, 400, seed=1)
    m = np.zeros((300, 400), dtype=np.uint8)
    cv2.line(m, (20, 20), (70, 60), 1, 2)
    cv2.line(m, (330, 240), (380, 280), 1, 2)
    assert cv2.connectedComponentsWithStats(m, connectivity=8)[0] - 1 == 2, "fixture must be two components"
    assert np.array_equal(apply_hair_inpaint(img, [m], dilate_px=0), _inpaint_whole_frame(img, m))


def test_hair_inpaint_neighbour_in_bbox_is_not_cloned_as_source():
    """Two hairs closer than the bbox pad: each one's crop contains the other. The neighbour
    has to stay masked inside that crop — mask only the component being filled and its dust
    becomes clone source, filling the defect straight back in."""
    img = _noisy_frame(200, 200, seed=2)
    m = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(m, (80, 40), (80, 160), 1, 2)
    cv2.line(m, (86, 40), (86, 160), 1, 2)  # 6 px away — well inside _HAIR_INPAINT_PAD
    assert cv2.connectedComponentsWithStats(m, connectivity=8)[0] - 1 == 2, "fixture must be two components"
    assert np.array_equal(apply_hair_inpaint(img, [m], dilate_px=0), _inpaint_whole_frame(img, m))


def test_ir_attenuation_is_the_upsampled_gain_product():
    """apply_ir_attenuation uses cv2.multiply to skip numpy's redundant float32 copy of the
    whole frame; it must stay exactly the gain product."""
    img = _noisy_frame(64, 96, seed=4)
    rng = np.random.default_rng(5)
    gain_det = rng.uniform(1.0, 2.0, (16, 24, 3)).astype(np.float32)
    expected = img * cv2.resize(gain_det, (96, 64), interpolation=cv2.INTER_LINEAR)
    out = apply_ir_attenuation(img, gain_det)
    assert out.dtype == np.float32
    assert np.array_equal(out, expected.astype(np.float32))


def test_ir_attenuation_passes_through_matched_gain():
    img = _noisy_frame(32, 32, seed=6)
    gain = np.full((32, 32, 3), 2.0, dtype=np.float32)
    assert np.array_equal(apply_ir_attenuation(img, gain), (img * 2.0).astype(np.float32))
