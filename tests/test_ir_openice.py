import contextlib
from dataclasses import replace

import cv2
import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.retouch import openice as oi
from negpy.features.retouch.logic import ir_bake_token
from negpy.features.retouch.models import IR_METHOD_NEGPY, IR_METHOD_OPENICE, RetouchConfig

_H = _W = 256
_CLEAN_IR = 0.74
_IR_SIGMA = 0.011


def _frame(defects=(), seed=7, ir_sigma=_IR_SIGMA, texture=1.0, clean_ir=_CLEAN_IR):
    """Synthetic linear negative + IR. ``defects`` are (cx, cy, r, transmittance) discs,
    neutral: a real defect blocks the visible channels and IR by the same fraction."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:_H, 0:_W]
    tex = 0.5 + texture * (0.16 * np.sin(x / 7.0) * np.cos(y / 11.0) + 0.03 * rng.standard_normal((_H, _W)))
    rgb = np.stack([tex * 0.95, tex * 0.55, tex * 0.30], -1)
    rgb = np.clip(rgb + 0.008 * rng.standard_normal(rgb.shape), 0.02, 1.0).astype(np.float32)
    ir = np.clip(clean_ir + ir_sigma * rng.standard_normal((_H, _W)), 0.0, 1.0).astype(np.float32)
    for cx, cy, r, t in defects:
        m = ((x - cx) ** 2 + (y - cy) ** 2) <= r * r
        rgb[m] *= t
        ir[m] *= t
    return rgb, ir.astype(np.float32)


@contextlib.contextmanager
def _no_dither():
    """§8 grain is a random per-pixel write, so it floors how close a repair can land to the
    clean frame. Tests that measure repair accuracy pin it off (ICE's own ice_no_dither)."""
    amp = oi._DITHER_AMP
    oi._DITHER_AMP = np.zeros(3, np.float32)
    try:
        yield
    finally:
        oi._DITHER_AMP = amp


def _disc(cx, cy, r):
    y, x = np.mgrid[0:_H, 0:_W]
    return ((x - cx) ** 2 + (y - cy) ** 2) <= r * r


def test_density_round_trips_within_one_16bit_step():
    v = np.linspace(0.0, 1.0, 20001, dtype=np.float32)
    assert np.abs(oi.density_inv(oi.density(v)) - v).max() * 65535.0 < 1.0


def test_density_endpoints_match_the_16bit_scale():
    assert float(oi.density(np.float32([0.0]))[0]) == 0.0
    assert abs(float(oi.density(np.float32([1.0]))[0]) - 65535.0) < 0.5


def test_pyramid_kernels_have_the_ice_cell_counts():
    """69/21/16 weight units — the octagon supports and the 1-2-1 tent. A miscounted
    kernel still runs and still looks plausible, so the shape is pinned here."""
    for k, cells in ((oi._K9, 69), (oi._K5, 21), (oi._K3, 16)):
        units = (k / k[k > 0].min()).sum()
        assert round(float(units)) == cells
        assert abs(float(k.sum()) - 1.0) < 1e-6


def test_calibration_recovers_an_injected_red_to_ir_crosstalk():
    rgb, ir = _frame()
    c_true = 0.05
    # The relation the gate inverts: d_ir_measured = (1-c)*d_ir_true + c*d_red, so a
    # frame built this way must calibrate back to c.
    contaminated = oi.density_inv((1.0 - c_true) * oi.density(ir) + c_true * oi.density(rgb[:, :, 0]))
    cal = oi.calibrate(rgb, contaminated, 0.66)
    assert abs(cal.crosstalk - c_true) < 0.015


def test_gate_is_flat_once_the_crosstalk_is_removed():
    """The point of the crosstalk term: the gate must respond to defects, not to the
    picture. Without it the red channel's texture prints straight through into the gate."""
    rgb, ir = _frame()
    c_true = 0.08
    contaminated = oi.density_inv((1.0 - c_true) * oi.density(ir) + c_true * oi.density(rgb[:, :, 0]))
    cal = oi.calibrate(rgb, contaminated, 0.66)
    d_rgb = oi.density(rgb)
    corrected, _ = oi._gate_and_weight(d_rgb, oi.density(contaminated), np.ones_like(ir, bool), cal)
    raw, _ = oi._gate_and_weight(d_rgb, oi.density(contaminated), np.ones_like(ir, bool), replace(cal, crosstalk=0.0))

    def corr(g):
        return abs(float(np.corrcoef(g.ravel(), d_rgb[:, :, 0].ravel())[0, 1]))

    assert corr(raw) > 0.5
    assert corr(corrected) < 0.1


def test_clean_film_is_left_bit_identical():
    rgb, ir = _frame()
    out, trigger, w = oi.reconstruct(rgb, ir, oi.calibrate(rgb, ir, 0.66))
    assert not trigger.any()
    # A fully-confident pixel is excluded by `keep`, so it is never written.
    clean = w >= 1.0
    assert clean.mean() > 0.9
    assert np.array_equal(out[clean], rgb[clean])


def test_reconstruction_lifts_a_neutral_speck_towards_the_clean_film():
    spot = (128, 128, 3, 0.35)
    rgb, ir = _frame([spot])
    clean_rgb, _ = _frame()
    out, _, _ = oi.reconstruct(rgb, ir, oi.calibrate(rgb, ir, 0.66))
    m = _disc(*spot[:3])
    before = float(np.abs(rgb[m] - clean_rgb[m]).mean())
    after = float(np.abs(out[m] - clean_rgb[m]).mean())
    assert after < 0.35 * before


def test_a_shallow_speck_is_repaired_at_full_strength():
    """ICE writes max(L3, acc) for every pixel under full confidence. Scaling that write
    by confidence spares clean film nothing the margin does not already gate, and costs a
    shallow speck — which never reaches the weight floor — most of its repair."""
    spot = (128, 128, 3, 0.96)  # detected, but nowhere near the weight floor
    rgb, ir = _frame([spot])
    clean_rgb, _ = _frame()
    with _no_dither():
        out, _, w = oi.reconstruct(rgb, ir, oi.calibrate(rgb, ir, 0.66))
    m = _disc(*spot[:3])
    # w ≈ 0.88 here; a smoothstep ramp from 1.0 down to 0.6 would pass ~20% of the lift.
    assert 0.6 < float(np.median(w[m])) < 1.0, "shallow by construction: off the weight floor"
    before = float(np.abs(rgb[m] - clean_rgb[m]).mean())
    after = float(np.abs(out[m] - clean_rgb[m]).mean())
    assert after < 0.7 * before


def test_a_repair_never_darkens():
    """ "Only fill, never darken": a defect steals light, so the bake may only add it."""
    rgb, ir = _frame([(128, 128, 4, 0.25), (60, 190, 2, 0.5)])
    out, _, _ = oi.reconstruct(rgb, ir, oi.calibrate(rgb, ir, 0.66))
    assert (out >= rgb - 1e-6).all()


def test_give_up_triggers_on_a_defect_wider_than_the_window_but_not_on_a_speck():
    """Both conditions are required: past the 9×9 window *and* under the dust floor.

    The floor is ICE's 6.5%-of-clear-film transmittance, which only clears the dead-margin
    cut when clear film sits near full scale — the Coolscan case ICE was built around. On a
    scanner whose IR runs darker the dead margin claims those pixels first and this trigger
    never fires, which is why the frame here is built with a bright, quiet IR plane.
    """
    speck, blob = (70, 70, 2, 0.058), (180, 180, 14, 0.058)
    rgb, ir = _frame([speck, blob], ir_sigma=0.002, clean_ir=1.0)
    cal = oi.calibrate(rgb, ir, 0.66)
    assert oi._DEAD_FLOOR < ir[_disc(*blob[:3])].max() < oi.density_inv(np.float32([cal.dust_floor]))[0]
    _, trigger, _ = oi.reconstruct(rgb, ir, cal)
    assert not trigger[_disc(*speck[:3])].any()
    assert trigger[_disc(*blob[:3])].mean() > 0.5


def test_a_wide_but_translucent_defect_is_reconstructed_not_given_up():
    """The floor is a transmittance, not a depth in noise σ. Anchored to σ it lands ~10x
    shallower and hands ordinary deep dust to the router instead of repairing it."""
    blob = (180, 180, 14, 0.30)  # wider than the window, but far from opaque
    rgb, ir = _frame([blob])
    out, trigger, _ = oi.reconstruct(rgb, ir, oi.calibrate(rgb, ir, 0.66))
    m = _disc(*blob[:3])
    assert trigger[m].mean() < 0.05
    assert (out[m] > rgb[m]).mean() > 0.9


def test_degenerate_guard_fires_when_the_ir_plane_mirrors_the_image():
    """B&W silver and Kodachrome block infrared the way dust does, so the whole frame
    would read as a defect. Detected by correlation against the picture, not by depth."""
    rgb, _ = _frame()
    cal = oi.calibrate(rgb, np.ascontiguousarray(rgb[:, :, 1]), 0.66)
    assert cal.degenerate
    out, corrected, degenerate, routed = oi.run(rgb, np.ascontiguousarray(rgb[:, :, 1]), 0.66)
    assert degenerate and corrected is None and routed is None
    assert out is rgb


def test_degenerate_guard_passes_ordinary_colour_film():
    rgb, ir = _frame([(128, 128, 5, 0.2)])
    assert not oi.calibrate(rgb, ir, 0.66).degenerate


def test_route_drops_a_rebate_sized_component_but_keeps_dust():
    """The film rebate sits above the dead floor on some scanners and arrives as one
    enormous defect (measured: two full-height strips, 1.2% of a flatbed frame). Routed,
    it blows the budget and takes the real dust down with it."""
    trigger = np.zeros((_H, _W), bool)
    trigger[:, :20] = True  # rebate strip
    trigger[100:104, 100:104] = True  # a dust core
    mask = oi.route(trigger)
    assert mask is not None
    assert mask[:, :20].sum() == 0
    assert mask[98:106, 98:106].any()


def test_route_returns_none_when_nothing_or_everything_is_hopeless():
    assert oi.route(np.zeros((_H, _W), bool)) is None
    over = np.zeros((_H, _W), bool)
    # Many small components, each under the per-component cap, together over budget.
    over[::4, ::2] = True
    assert oi.route(over) is None


def test_run_returns_the_ir_bake_contract():
    rgb, ir = _frame([(128, 128, 4, 0.2)])
    dims = (64, 64)
    out, corrected, degenerate, routed = oi.run(rgb, ir, 0.66, dims)
    assert out.shape == rgb.shape and out.dtype == np.float32
    assert degenerate is False
    assert corrected is not None and corrected.shape == (dims[1], dims[0])
    assert routed is None or routed.shape == (dims[1], dims[0])


def test_banding_matches_a_single_pass():
    """Bands overlap by the pyramid's own reach, so a seam must not show. Guards the
    halo against anyone widening a kernel without widening _HALO."""
    rgb, ir = _frame([(128, 128, 4, 0.25), (128, 40, 3, 0.3)])
    cal = oi.calibrate(rgb, ir, 0.66)
    banded, _, _ = oi.reconstruct(rgb, ir, cal)
    original = oi._BAND_ROWS
    try:
        oi._BAND_ROWS = 4096  # one band for the whole frame
        whole, _, _ = oi.reconstruct(rgb, ir, cal)
    finally:
        oi._BAND_ROWS = original
    assert np.abs(banded - whole).max() < 1e-6


def test_dither_grains_a_repair_without_shifting_its_level():
    """A pixel fully covered by dust has no grain of its own left for the finest band to
    restore, so the repair reads as a smooth patch against film unless §8 adds grain (#732)."""
    blob = (128, 128, 8, 0.15)
    rgb, ir = _frame([blob])
    cal = oi.calibrate(rgb, ir, 0.66)
    grained, _, w = oi.reconstruct(rgb, ir, cal)
    with _no_dither():
        smooth, _, _ = oi.reconstruct(rgb, ir, cal)

    core = _disc(128, 128, 6)  # inside the blob, clear of its edge
    delta = grained - smooth
    assert float(delta[core].std()) > 0.01 * float(smooth[core].mean())
    assert abs(float(delta[core].mean())) < 0.1 * float(delta[core].std())
    assert not delta[w >= 1.0].any(), "grain belongs to the repair, not to clean film"


def test_dither_is_zero_mean_in_band_and_silent_outside_it():
    """ICE's envelope: a parabola over [D(0.01M), D(0.99M)] peaking at 1, times ±amount/2 of
    the pixel's own density. Outside the band, or if the grain would leave it, no grain."""
    mid = np.full((256, 256, 3), 0.5 * (oi._DITHER_LO + oi._DITHER_HI), np.float32)
    d = oi._dither(mid, 0)
    assert abs(float(d.mean())) < 0.01 * float(np.abs(d).max())
    peak = np.abs(d).max(axis=(0, 1)) / (0.5 * oi._DITHER_AMP * mid[0, 0])
    assert np.allclose(peak, 1.0, atol=0.02)
    for x in (oi._DITHER_LO - 1.0, oi._DITHER_HI + 1.0):
        assert not oi._dither(np.full((16, 16, 3), x, np.float32), 0).any()


def test_threshold_biases_the_ramp_monotonically():
    rgb, ir = _frame([(128, 128, 4, 0.4)])
    ramps = [oi.calibrate(rgb, ir, t).ramp for t in (0.05, 0.5, 0.95)]
    assert ramps[0] < ramps[1] < ramps[2]
    # A wider ramp is more conservative: the same pixels read as defective (the margin
    # sets that), but less deeply, so less is rewritten.
    depth = [float((1.0 - oi.reconstruct(rgb, ir, oi.calibrate(rgb, ir, t))[2]).mean()) for t in (0.05, 0.95)]
    assert depth[0] > depth[1]


def test_dead_margin_is_neither_corrected_nor_routed():
    """No film under the head is not a defect — and a strip scan's margin is big enough
    to swallow the whole inpaint budget if it is treated as one."""
    rgb, ir = _frame()
    ir[:, :30] = 0.0
    out, _, w = oi.reconstruct(rgb, ir, oi.calibrate(rgb, ir, 0.66))
    assert (w[:, :30] == 1.0).all()
    assert np.array_equal(out[:, :30], rgb[:, :30])


def test_config_defaults_to_the_negpy_method_and_round_trips():
    assert RetouchConfig().ir_method == IR_METHOD_NEGPY
    flat = WorkspaceConfig(retouch=RetouchConfig(ir_dust_remove=True, ir_method=IR_METHOD_OPENICE)).to_dict()
    assert flat["ir_method"] == IR_METHOD_OPENICE
    assert WorkspaceConfig.from_flat_dict(flat).retouch.ir_method == IR_METHOD_OPENICE


def test_legacy_config_without_ir_method_loads_as_negpy():
    flat = WorkspaceConfig(retouch=RetouchConfig(ir_dust_remove=True)).to_dict()
    del flat["ir_method"]
    assert WorkspaceConfig.from_flat_dict(flat).retouch.ir_method == IR_METHOD_NEGPY


def test_bake_token_distinguishes_the_methods():
    negpy = RetouchConfig(ir_dust_remove=True)
    ice = RetouchConfig(ir_dust_remove=True, ir_method=IR_METHOD_OPENICE)
    assert ir_bake_token(negpy, True) != ir_bake_token(ice, True)
    # Unchanged for the default method, so existing render caches still hit.
    assert ir_bake_token(negpy, True) == "|ir1r0.66"
    assert ir_bake_token(ice, False) == ""


def test_openice_leaves_more_of_the_clean_frame_untouched_than_it_changes():
    """The whole point of the give-up path: a bake that rewrites the frame is wrong even
    if every defect goes away."""
    rgb, ir = _frame([(60, 60, 3, 0.3), (190, 100, 2, 0.4), (100, 190, 4, 0.25)])
    out, _, _ = oi.reconstruct(rgb, ir, oi.calibrate(rgb, ir, 0.66))
    changed = np.abs(out - rgb).max(-1) > 1e-6
    assert changed.mean() < 0.05
    # And what did change is where the IR said so.
    defects = _disc(60, 60, 3) | _disc(190, 100, 2) | _disc(100, 190, 4)
    near = cv2.dilate(defects.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
    assert changed[~near].mean() < 0.02
