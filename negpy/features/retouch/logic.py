import hashlib
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
from numba import njit  # type: ignore

from negpy.domain.types import ImageBuffer
from negpy.features.geometry.logic import smooth_polyline
from negpy.features.retouch.models import HEAL_SIZE_REF, IR_METHOD_NEGPY
from negpy.kernel.image.logic import get_luminance
from negpy.kernel.image.validation import ensure_image
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

# Spread floor: stops noise on low-contrast sources (fog, flat frames) from
# being amplified to full range; dust sits ≥ ~1 density unit above surroundings.
_PROXY_MIN_SPREAD = 0.8
# Pad heals past the detected bright core — an unhealed soft skirt reads as a halo.
_DETECT_PAD_PX = 2.5

# Manual heal gate. A painted stroke marks a *search area*, not a stamp: the pixels
# repaired inside it are the ones that stand out from the film around them, so clean
# grain under a generous brush is left byte-identical. Two-sided is the point: dust is
# a bright outlier in density, a scratch a dark one, and a bright-only gate could never
# repair a scratch at all (#791).
#
# The excess is measured on a 3×3 mean of the local high-pass, against the σ of that
# same quantity taken robustly (MAD) over the stroke's neighbourhood. Both halves matter:
# averaging drops uncorrelated grain by 3 while a defect ≥2 px keeps its full excess, and
# measuring σ rather than assuming one keeps the bars valid on any film and any scanner.
# On clean film |z| peaks around 3 whatever the noise level, which is what puts the bars
# where they are — an absolute density threshold could not separate the two.
_MANUAL_Z_HI = 8.0
# Grow bar for the hysteresis below: the floor a connected pixel must clear to count as part
# of a defect already found. Absolute, or a fraction of the core's own strength for a defect
# whose skirt is proportionally deep.
_MANUAL_Z_GROW = 2.0
_MANUAL_Z_GROW_FRAC = 0.25
# Backstop bar: a stroke whose strongest pixel never reaches _MANUAL_Z_HI is rescaled
# against its own maximum, so a faint-but-real defect still repairs instead of the tool
# silently doing nothing. Under this it stays a no-op — the brush found clean film.
_MANUAL_Z_MIN = 5.0
# Local-statistics window as a multiple of the brush radius: wide enough that a defect
# filling the brush cannot set its own baseline.
_MANUAL_WIN_FACTOR = 3.0
# Soft edge on the search area itself, so a defect crossing the brush rim doesn't repair
# to a hard line.
_MANUAL_RIM_PX = 1.5

# Transport scratches (#788): a mark running the length of the film, nearly straight and a
# few px wide. Its length is what makes it findable at all — per pixel it sits under the
# manual-heal seed bar, so no per-pixel gate can reach it and the evidence has to be
# integrated along the line instead.
#
# Cross-section band-pass: film grain is finer, image structure broader.
_SCRATCH_FINE_PX = 1.2
_SCRATCH_BROAD_PX = 9.0
# The ridge response is normalized by a *local* noise scale. Global lets one busy corner of
# the frame set the bar and buries a faint scratch running through smooth sky.
_SCRATCH_NOISE_WIN = 151
# Slope search, rise per unit run. The scratch is straight but the film is rarely square to
# the sensor, and a fraction of a degree drifts tens of px across a frame — enough for an
# axis-aligned collapse to smear the ridge away.
_SCRATCH_SLOPE_MAX = 0.02
_SCRATCH_SLOPE_STEP = 0.00025
# Rows searched either side of the click, and how hard the fit is pulled back toward it.
# Without the pull it snaps to whatever ridge is strongest in the band, not the one clicked.
_SCRATCH_SEARCH_ROWS = 30
_SCRATCH_CLICK_PULL = 12.0
# Slider range for the bar a ridge must clear (see scratch_detect_bar); the default
# slider position sits in the middle of it.
_SCRATCH_Z_LOOSE = 0.4
_SCRATCH_Z_TIGHT = 1.6
# Presence along the line: the bar must hold over this fraction of a window this wide before
# a stretch is repaired. Transport scratches fade in and out, so extent is measured.
_SCRATCH_RUN_WIN = 151
_SCRATCH_RUN_FRAC = 0.35
# A trace this weak is not a scratch — the click found clean film.
_SCRATCH_MIN_EVIDENCE = 0.25
# Ceiling on the repaired half-width, px at HEAL_SIZE_REF. The band is grown from the
# scratch, so this only stops a runaway where the ridge never breaks.
_SCRATCH_WIDTH_MAX = 14.0
_SCRATCH_WIDTH_MIN = 3.0

# Detection follows the buffer it repairs, at most this far under it: the score is upsampled
# onto that buffer and the fill supports scale with the same factor, so coarse detection writes
# a fat mask and averages over a wide support. A defect straddling a tonal edge is then rebuilt
# from the bright side of it and prints as a dark blotch on the light one.
_IR_MAX_UPSAMPLE = 1.5
_IR_DETECT_MAX = 3600  # memory: ir_ratio_and_gain holds ~10 planes of it
# Film-footprint windows below are px at this detection long edge and scale with the plane
# (_ir_win): on a finer plane a wide hair fills an unscaled base window, depresses its own base
# and stops reading as a defect at all.
_IR_DETECT_REF = 1600
# IR ratio-normalization base window (px at _IR_DETECT_REF, pinned like HEAL_SIZE_REF).
# Defects wider than ~half of it depress their own base (max-area/Scratch territory).
_IR_BASE_WIN = 25
_IR_GAIN_IDENTITY = 0.97  # gain is identity at/above this ratio
_IR_GAIN_CLAMP = 2.0  # caps misregistration halos
# Per-channel refraction γ, fitted per frame (patent 1.03–1.10 under-correct file IR).
_IR_GAMMA_LO = 1.0
_IR_GAMMA_HI = 2.2
_IR_GAMMA_FALLBACK = 1.5
# Below this the beam is blocked outright — holder, not film. Coolscan rolls: margin 98% under
# it. ponytail: absolute; a low-IR-gain scanner would want a percentile. Opaque hairs pass under
# the floor too, so only below-floor regions this large (fraction of the frame) are holder —
# writing the rest off leaves them unrepairable once the plane resolves their cores.
_IR_DEAD_FLOOR = 0.05
_IR_DEAD_MIN_AREA = 0.002
# Clean-film pivot: normalize_ir's base is a local max, so clean film sits ~k·σ_IR under 1,
# where depending on the scanner. Left absolute, a Coolscan 5000 (ratio median 0.945) put
# 84% of the frame below _IR_GAIN_IDENTITY, starving _ir_clean_base into its local-max
# fallback — no cap, mottled film at every slider position (#647). Measured at detection
# scale, on the same ratio it corrects; not on the full-res plane.
_IR_NOISE_SIGMA = 3.0
# Bounds the rescale so >50% coverage (median inside the dust) can't scale the ratio clean
# and silently disable IR removal. ponytail: absolute; the knob if a scanner needs more.
_IR_PIVOT_LO = 0.60
# Dip depth is scanner-dependent — clean-film MAD σ is 0.038–0.063 on a Coolscan 5000 against
# 0.005 on a Plustek/SilverFast HDRi DNG, putting one speck at ratio 0.62 and the other at 0.92,
# past every absolute landmark below. Stretch-only, so σ ≥ _IR_REF_SIGMA is an exact no-op.
_IR_REF_SIGMA = 0.02
_IR_SCALE_MAX = 6.0  # bounds a degenerate/quantized plane measuring σ ~0
# Crosstalk unmixing: dye/silver absorbs some IR, so the IR plane carries a ghost of
# the image that normalize_ir's spatial high-pass can't see (a sharp edge survives it).
_IR_XTALK_MAX = 0.8  # per-channel exponent cap; ≥0 only — density can only block IR
_IR_XTALK_MIN = 0.02  # |b| sum below this is a noise-level fit → exact no-op
_IR_DEGENERATE_GHOST = 0.5  # fitted exponent sum above this: IR mirrors the image (B&W/Kodachrome)
_IR_XTALK_TRIM = 5.0  # fit drops this bottom-ratio percentile (the dust minority)
# γ fit sample: keep this flattest fraction of the band by visible Laplacian, dropping the
# restriction below _IR_FIT_MIN_PX rather than fitting a handful of pixels. See _fit_refraction_gammas.
_IR_FIT_FLAT_PCT = 40
_IR_FIT_MIN_PX = 200
# Fit sample cap: _ir_decontaminate and _fit_refraction_gammas resolve 3-4 per-frame scalars,
# so they stride their pixel set down to this rather than growing with the detection plane.
_IR_FIT_MAX_PX = 200_000
# Clean-base cap window (px at _IR_DETECT_REF, odd). The bake may never lift a pixel above its
# own local clean base — past that it invents signal rather than recovering it. Needed because
# downsample_ir is min-preserving while the visible arrives area-averaged, so at detection scale
# the ratio's dip runs deeper and ~1 px wider than the defect the visible carries (0.816 against
# 0.892); uncapped, that skirt lifts clean film and every speck and hair renders with a dark
# outline. Reaches ±4 px, past _DETECT_PAD_PX's skirt. Base = defect-excluded local mean −
# _IR_CAP_SIGMA·σ, not blur(dilate): the dilate is a local max, ~2σ of grain high, and re-admits
# the ring on grainy film (#563). Under _IR_CAP_MIN_SUPPORT clean pixels in the window → the max
# estimate returns (deep inside wide defects, where _IR_GAIN_CLAMP binds first).
_IR_CAP_WIN = 9
_IR_CAP_SIGMA = 1.0
_IR_CAP_MIN_SUPPORT = 0.1  # fraction of the window (~8 px at 9×9)

# IR reconstruction: concepts ported from digital-fauxice (MIT, © 2026 Rohan
# Pandula, see NOTICE.md) — continuous score, score-weighted fill, original-floor rule.
# Score: 1 = clean (ratio ≥ _IR_GAIN_IDENTITY), floor at/below the slider's cutoff.
# Never thresholded — no mask edge to halo, no coverage fraction to abort on.
_IR_SCORE_FLOOR = 0.02
# Fill supports (detection-scale px, × the buffer's upsample factor). Candidate per
# support: Σ(rgb·score·win)/Σ(score·win) — low-score neighbours self-exclude. A finer
# support wins once its clean fraction reaches _IR_FILL_TAU (edges continue through).
_IR_FILL_SCALES = (9, 5, 3)
_IR_FILL_TAU = 0.15
# Write ramp: untouched above HI (grain survives), full fill at/below LO.
_IR_WRITE_HI = 0.85
_IR_WRITE_LO = 0.40
# Route to inpaint only components with a core the fill can't see across (chebyshev
# radius ≥ 5 ⇔ solid 9×9 interior). Thin hairs stay with the fill: every pixel is
# within reach of clean film, and NS inpaint would smear structure the fill keeps.
# The budget bounds only this heavy path; the fill always runs.
_IR_ROUTE_RADIUS = 5
_IR_ROUTE_DILATE = 2
_IR_ROUTE_BUDGET = 0.02  # fraction of the frame
# Crop-per-defect stops paying past this count (see repair_components).
_REPAIR_MAX_COMPONENTS = 256

# Strong hairs/scratches route to structure-following inpaint instead of the weighted
# fill: a long twist crosses varied background, and averaging across it smears the
# structure the inpaint follows. Detection-scale px. See _is_hair.
_HAIR_MIN_AREA = 20
_HAIR_MIN_ELONG = 8.0  # area/thickness² ≈ length/thickness; round specks measure 1–3
# cv2.inpaint fill: dilate covers the PSF skirt at 1:1 (apply_hair_inpaint widens it to
# track the mask's upsample); NS radius; gamma gives the 8-bit encode a perceptual
# spread (cv2.inpaint is 8-bit only). Navier-Stokes only propagates outward from the mask
# boundary, so each defect is filled in its own bbox + _HAIR_INPAINT_PAD (>= the radius):
# same pixels, without gamma-encoding the whole frame to serve a hairline.
_HAIR_DILATE_PX = 1
_HAIR_INPAINT_RADIUS = 3
_HAIR_INPAINT_GAMMA = 2.2
_HAIR_INPAINT_PAD = 16


@njit(cache=True, fastmath=True)
def _detect_dust_mask_jit(
    luma: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    w_std: np.ndarray,
    dust_threshold: float,
) -> np.ndarray:
    """Local-contrast dust detector on the normalized-density plane; the
    wide-window texture penalty protects rocks/foliage."""
    h, w = luma.shape
    hit_mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            l_curr = luma[y, x]
            l_mean = mean[y, x]
            local_s = max(0.005, std[y, x])

            w_s = max(0.0, w_std[y, x] - 0.02)
            wide_penalty = (w_s * w_s * w_s) * 800.0
            thresh = (dust_threshold * 0.4) + (local_s * 1.0) + wide_penalty

            if (l_curr - l_mean) > thresh and l_curr > 0.15 and (l_curr - l_mean) / local_s > 3.0:
                is_strong = (l_curr - l_mean) > (thresh * 2.5) or (l_curr - l_mean) > 0.25
                if 0 < y < h - 1 and 0 < x < w - 1:
                    is_max = True
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if dy == 0 and dx == 0:
                                continue
                            if luma[y + dy, x + dx] >= l_curr:
                                is_max = False
                                break
                        if not is_max:
                            break
                    if is_max or is_strong:
                        hit_mask[y, x] = 1
                else:
                    hit_mask[y, x] = 1
    return hit_mask


def _proxy_norm(img: ImageBuffer) -> Tuple[float, float]:
    """(lo, spread) percentile normalization of the detection proxy."""
    dens = -np.log10(np.clip(get_luminance(img), 1e-6, None))
    lo, hi = np.percentile(dens, (0.5, 99.5))
    return float(lo), max(float(hi - lo), _PROXY_MIN_SPREAD)


def _is_hair(labels_sub: np.ndarray, area: int) -> bool:
    """Hair/scratch (thin) rather than speck: ``2*max(distanceTransform)`` is the widest
    the defect ever gets, so ``area/thickness²`` reads as length/thickness for a ribbon.

    Thin, not straight — bending moves no interior pixel further from its edge, so a
    twist scores like a straight hair, where PCA extent/width (the obvious measure)
    calls it compact. The real hair on samples/ir/18.tiff: PCA aspect 2.45 = "speck",
    thinness 26.3 = hair.
    """
    if area < _HAIR_MIN_AREA:
        return False
    # Pad, or a component touching the sub-image border reads as thin along that edge.
    dist = cv2.distanceTransform(np.pad(labels_sub.astype(np.uint8), 1), cv2.DIST_L2, 5)
    thickness = 2.0 * float(dist.max())
    return area / max(thickness * thickness, 1e-6) >= _HAIR_MIN_ELONG


def _box_mean_std(plane: np.ndarray, win: int) -> Tuple[np.ndarray, np.ndarray]:
    """Local mean and standard deviation of ``plane`` over a ``win``×``win`` box."""
    mean = cv2.blur(plane, (win, win))
    var = cv2.blur(plane * plane, (win, win)) - mean * mean
    return mean, np.sqrt(np.clip(var, 0.0, None))


def _density(img: ImageBuffer) -> np.ndarray:
    """Log density of the luminance. Local contrast measured here is exposure-invariant:
    a gain on the source is an offset in density and cancels in every difference."""
    return -np.log10(np.clip(get_luminance(img), 1e-6, None)).astype(np.float32)


def _mask_to_score(mask: np.ndarray, pad_px: float) -> np.ndarray:
    """Binary defect mask → the continuous score every repair consumes: at the floor on
    the defect, ramping to clean over ``pad_px``. The ramp is the skirt allowance a hard
    mask edge would leave behind as a halo."""
    d = cv2.distanceTransform((np.asarray(mask) == 0).astype(np.uint8), cv2.DIST_L2, 3)
    t = np.clip(d / max(pad_px, 1e-3), 0.0, 1.0)
    return (_IR_SCORE_FLOOR + (1.0 - _IR_SCORE_FLOOR) * (t * t * (3.0 - 2.0 * t))).astype(np.float32)


def split_hairs(mask: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Defect mask → ``(compact, hairs)``. Compact defects go to the weighted fill; long
    twisted ones to structure-following inpaint, which keeps detail the fill would average
    across. ``hairs`` is None when nothing is thin enough."""
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(np.ascontiguousarray(mask, dtype=np.uint8), connectivity=8)
    compact = np.zeros(mask.shape[:2], dtype=np.uint8)
    hairs: Optional[np.ndarray] = None
    for i in range(1, n_lbl):
        x0, y0 = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        labels_sub = labels[y0 : y0 + bh, x0 : x0 + bw] == i
        if _is_hair(labels_sub, int(stats[i, cv2.CC_STAT_AREA])):
            if hairs is None:
                hairs = np.zeros(mask.shape[:2], dtype=np.uint8)
            hairs[y0 : y0 + bh, x0 : x0 + bw][labels_sub] = 1
        else:
            compact[y0 : y0 + bh, x0 : x0 + bw][labels_sub] = 1
    return compact, hairs


def compute_dust_stats(img: ImageBuffer, dust_size: int) -> Tuple[np.ndarray, ...]:
    """Threshold-independent detection stat maps (proxy + blur windows) — the expensive
    ~2/3 of a detection pass, cacheable across threshold changes."""
    lo, spread = _proxy_norm(img)
    proxy = np.clip((_density(img) - lo) / spread, 0.0, 1.0).astype(np.float32)
    base_size = max(1.0, float(dust_size))
    v_win = int(max(3, base_size * 3.0)) * 2 + 1
    w_win = int(max(7, base_size * 4.0)) * 2 + 1
    mean, std = _box_mean_std(proxy, v_win)
    _, w_std = _box_mean_std(proxy, w_win)
    return (
        np.ascontiguousarray(proxy.astype(np.float32)),
        np.ascontiguousarray(mean.astype(np.float32)),
        np.ascontiguousarray(std.astype(np.float32)),
        np.ascontiguousarray(w_std.astype(np.float32)),
    )


def detect_luma_score(
    img: ImageBuffer,
    dust_threshold: float,
    dust_size: int,
    stats: Optional[Tuple[np.ndarray, ...]] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Statistical dust detection on the linear source → ``(score, hair_mask)``.

    Compact specks become a score for the shared fill; strongly elongated hairs a mask
    for structure-following inpaint, which follows detail a weighted average would smear.
    """
    if stats is None:
        stats = compute_dust_stats(img, dust_size)
    proxy, mean, std, w_std = stats[:4]
    hit = _detect_dust_mask_jit(proxy, mean, std, w_std, float(dust_threshold))
    if not np.any(hit):
        return None, None
    compact, hair_mask = split_hairs(hit)
    score = _mask_to_score(compact, _DETECT_PAD_PX) if compact.any() else None
    return score, hair_mask


def strokes_to_score(
    img: ImageBuffer,
    strokes: List[Tuple],
    legacy_spots: List[Tuple[float, float, float]],
) -> Optional[np.ndarray]:
    """Painted heal/scratch strokes → a defect score for the shared repair.

    The capsule a stroke paints is a **search area**, not a stamp: inside it, a pixel is
    repaired by how far it stands out from the film around it (a two-sided local z-score
    on density), so clean grain under a generous brush comes back untouched and only the
    defect is rewritten. Two-sided because dust is a bright outlier in density while a
    scratch, having lost emulsion, is a dark one — the direction a bright-only gate could
    never repair.

    Strokes carry raw-frame normalized coordinates, so this runs before geometry and needs
    no mapping. Returns ``None`` when no stroke found anything worth repairing.
    """
    entries: List[Tuple[List, float]] = [(list(s[0]), float(s[1])) for s in strokes]
    entries += [([[nx, ny]], float(size)) for nx, ny, size in legacy_spots]
    if not entries:
        return None

    h, w = img.shape[:2]
    dens = _density(img)
    score = np.ones((h, w), dtype=np.float32)
    # Brush size is a DIAMETER at HEAL_SIZE_REF scale, so the painted footprint matches the
    # cursor at any render resolution (overlay._brush_screen_radius draws size/(2·REF)).
    scale = max(w, h) / HEAL_SIZE_REF
    touched = False

    for points, size in entries:
        radius = max(1.0, size * scale * 0.5)
        chain = [(float(p[0]) * w, float(p[1]) * h) for p in points]
        if len(chain) >= 3:
            chain = smooth_polyline(chain, closed=False)
        pts = np.array(chain, dtype=np.float32)

        win = int(max(3.0, radius * _MANUAL_WIN_FACTOR)) * 2 + 1
        pad = int(radius) + win
        x0 = max(0, int(pts[:, 0].min()) - pad)
        y0 = max(0, int(pts[:, 1].min()) - pad)
        x1 = min(w, int(pts[:, 0].max()) + pad + 1)
        y1 = min(h, int(pts[:, 1].max()) + pad + 1)
        if x1 <= x0 or y1 <= y0:
            continue

        cover = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        local = np.round(pts - (x0, y0)).astype(np.int32)
        if len(local) > 1:
            cv2.polylines(cover, [local], False, 1, thickness=max(1, int(round(2.0 * radius))))
        for cx, cy in local:  # round caps and joins — a thick polyline alone leaves flat ends
            cv2.circle(cover, (int(cx), int(cy)), max(1, int(round(radius))), 1, -1)
        if not cover.any():
            continue

        crop = dens[y0:y1, x0:x1]
        detail = cv2.blur(crop - cv2.blur(crop, (win, win)), (3, 3))
        # MAD about zero over the whole crop, so the defect stays a minority in its own
        # noise estimate. 0.6745 = the half-normal median, turning it into a σ.
        sigma = float(np.median(np.abs(detail))) / 0.6745
        z = np.abs(detail) / max(sigma, 1e-9)

        inside = cover > 0
        peak = float(z[inside].max())
        if peak < _MANUAL_Z_MIN:
            continue  # the brush found clean film; repairing it would only smooth grain
        # Self-normalize a faint defect against its own peak, so a real mark the absolute bar
        # would miss still repairs rather than the tool doing nothing.
        hi = _MANUAL_Z_HI if peak >= _MANUAL_Z_HI else peak * 0.9
        # Hysteresis, the reason a single bar cannot do this job: a defect's bright core clears
        # any bar, but the soft skirt around it (and a hair lying over scene structure) does not,
        # and a bar low enough to catch those catches grain everywhere. Grow from the core down
        # to the low bar through connected pixels only — grain is isolated, so it never joins.
        lo = max(_MANUAL_Z_GROW, hi * _MANUAL_Z_GROW_FRAC)
        strong = inside & (z >= hi)
        if not strong.any():
            continue
        _n, lab = cv2.connectedComponents((inside & (z >= lo)).astype(np.uint8), connectivity=8)
        seeded = np.unique(lab[strong])
        keep = np.isin(lab, seeded[seeded > 0]).astype(np.uint8)

        # Pad past the defect, the way the detector does: a defect's soft PSF skirt falls under
        # any bar that keeps grain out, and leaving it unrepaired prints as a dark outline
        # around an otherwise clean repair.
        region = _mask_to_score(keep, _DETECT_PAD_PX * film_scale((h, w)))
        # Soften the search area's own rim, or a defect crossing it repairs to a hard line.
        d = cv2.distanceTransform(cover, cv2.DIST_L2, 3)
        alpha = np.clip(d / _MANUAL_RIM_PX, 0.0, 1.0)

        region = 1.0 - alpha * (1.0 - region)
        np.minimum(score[y0:y1, x0:x1], region.astype(np.float32), out=score[y0:y1, x0:x1])
        touched = True

    return score if touched else None


def manual_bake_token(retouch) -> str:
    """Config identity of the painted heals, folded into source_hash so a new stroke
    invalidates the render (mirrors ``ir_bake_token``)."""
    lines = getattr(retouch, "scratch_lines", [])
    if not (retouch.manual_heal_strokes or retouch.manual_dust_spots or lines):
        return ""
    payload = repr(
        (retouch.manual_heal_strokes, retouch.manual_dust_spots, lines, round(float(getattr(retouch, "scratch_threshold", 0.5)), 4))
    ).encode()
    return "|heal" + hashlib.sha1(payload).hexdigest()[:12]


def scratch_detect_bar(slider: float) -> float:
    """UI sensitivity (higher = conservative) -> the ridge bar a scratch must clear."""
    s = float(np.clip(slider, 0.0, 1.0))
    return _SCRATCH_Z_LOOSE + (_SCRATCH_Z_TIGHT - _SCRATCH_Z_LOOSE) * s


def _scratch_ridge(img: ImageBuffer) -> np.ndarray:
    """Cross-section ridge response of ``img``, in units of the *local* noise.

    Band-passed across the scratch only (film grain is finer, image structure broader), then
    divided by a local scale of that same response. Local, because a global scale lets one
    busy corner of the frame set the bar and bury a faint scratch running through smooth sky.
    """
    dens = _density(img)
    fine = cv2.GaussianBlur(dens, (1, 0), sigmaX=0, sigmaY=_SCRATCH_FINE_PX)
    broad = cv2.GaussianBlur(dens, (1, 0), sigmaX=0, sigmaY=_SCRATCH_BROAD_PX)
    ridge = fine - broad
    win = (_SCRATCH_NOISE_WIN, _SCRATCH_NOISE_WIN)
    # mean|x| is 0.8σ for gaussian noise; the ratio is what the bars above are quoted in.
    scale = cv2.blur(np.abs(ridge), win) / 0.8
    return ridge / np.maximum(scale, 1e-6)


def _shear_rows(plane: np.ndarray, slope: float, about_x: float, width: int) -> np.ndarray:
    """``plane`` sheared so a line of gradient ``slope`` through ``about_x`` becomes one row.

    warpAffine applies M forward (source → destination) unless asked otherwise, so the matrix
    carries -slope: a source point at ``y = row + slope·(x - about_x)`` has to land back on
    ``row``.
    """
    m = np.float32([[1.0, 0.0, 0.0], [-slope, 1.0, slope * about_x]])
    return cv2.warpAffine(plane, m, (width, plane.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def trace_scratch(img: ImageBuffer, nx: float, ny: float, threshold: float = 0.5) -> Optional[Tuple[float, float, float, float, float]]:
    """One click near a transport scratch → ``(nx0, ny0, nx1, ny1, width)``, or None.

    The slope is fitted, not assumed: film is rarely square to the sensor, and a fraction of a
    degree is enough to smear the ridge across many rows and lose the integration this depends
    on. ``width`` is for the on-screen guide; the repair re-grows the band itself.
    """
    h, w = img.shape[:2]
    cx, cy = float(nx) * w, float(ny) * h
    bar = scratch_detect_bar(threshold)
    z = _scratch_ridge(img)
    y0 = max(0, int(cy) - _SCRATCH_SEARCH_ROWS)
    y1 = min(h, int(cy) + _SCRATCH_SEARCH_ROWS)
    if y1 - y0 < 3:
        return None
    band = np.ascontiguousarray(z[y0:y1])
    pull = np.exp(-0.5 * ((np.arange(y0, y1) - cy) / _SCRATCH_CLICK_PULL) ** 2)

    best: Optional[Tuple[float, float, int, np.ndarray]] = None
    for slope in np.arange(-_SCRATCH_SLOPE_MAX, _SCRATCH_SLOPE_MAX + 1e-9, _SCRATCH_SLOPE_STEP):
        sheared = _shear_rows(band, float(slope), cx, w)
        strength = np.abs(sheared.mean(axis=1)) * pull
        k = int(np.argmax(strength))
        if best is None or strength[k] > best[0]:
            best = (float(strength[k]), float(slope), k, sheared[k])
    if best is None or best[0] < _SCRATCH_MIN_EVIDENCE:
        return None
    _, slope, k, along = best

    # A transport scratch fades in and out, so measure its extent rather than assume it spans
    # the frame.
    on = (along * np.sign(along.mean()) > bar).astype(np.float32)
    run = cv2.blur(on.reshape(1, -1), (_SCRATCH_RUN_WIN, 1)).ravel() >= _SCRATCH_RUN_FRAC
    if not run.any():
        return None
    cols = np.flatnonzero(run)
    x0, x1 = float(cols[0]), float(cols[-1])
    row = y0 + k
    # For the guide only — the repair grows its own band per column.
    scale = film_scale((h, w))
    max_half = max(1, int(round(0.5 * _SCRATCH_WIDTH_MAX * scale)))
    grown = _grow_band(_shear_rows(z, slope, cx, w), row, cols, max_half, float(np.sign(along.mean()) or 1.0), bar)
    width = float(np.clip(2.0 * float(np.median(grown.sum(axis=0))) / max(scale, 1e-6), _SCRATCH_WIDTH_MIN, _SCRATCH_WIDTH_MAX))
    return (x0 / w, (row + slope * (x0 - cx)) / h, x1 / w, (row + slope * (x1 - cx)) / h, width)


def _grow_band(sheared: np.ndarray, row: int, xs: np.ndarray, max_half: int, sign: float, bar: float) -> np.ndarray:
    """Per-column extent of the scratch either side of the line, by hysteresis on the ridge.

    The rule the brush already uses: grow outward from the centre while the response holds,
    stop where it breaks. On the normalized response, so it follows a scratch of any width
    without a pixel measurement that would disagree with itself between preview and export.
    """
    h = sheared.shape[0]
    band = np.zeros((2 * max_half + 1, xs.size), dtype=bool)
    band[max_half] = True
    for direction in (-1, 1):
        alive = np.ones(xs.size, dtype=bool)
        for step in range(1, max_half + 1):
            r = row + direction * step
            if not 0 <= r < h:
                break
            alive &= (sheared[r, xs] * sign) > bar
            if not alive.any():
                break
            band[max_half + direction * step] = alive
    return band


def lines_to_score(img: ImageBuffer, lines: List[Tuple], threshold: float = 0.5) -> Optional[np.ndarray]:
    """Traced scratch lines → a defect score for the shared repair.

    The line says where to look; presence and width are re-measured here, so stretches that
    carry no scratch are left alone. Same contract as a painted stroke: the geometry is a
    search area, the evidence decides.
    """
    if not lines:
        return None
    h, w = img.shape[:2]
    bar = scratch_detect_bar(threshold)
    z = _scratch_ridge(img)
    scale = film_scale((h, w))
    mask = np.zeros((h, w), dtype=np.uint8)
    touched = False

    for nx0, ny0, nx1, ny1, width in lines:
        x0, x1 = float(nx0) * w, float(nx1) * w
        y0, y1 = float(ny0) * h, float(ny1) * h
        if abs(x1 - x0) < 1.0:
            continue
        slope = (y1 - y0) / (x1 - x0)
        # ``width`` is only what the guide drew — the band is re-grown from the scratch here,
        # so it follows one that widens along its length and ignores the traced resolution.
        max_half = max(1, int(round(0.5 * _SCRATCH_WIDTH_MAX * scale)))
        sheared = _shear_rows(z, slope, x0, w)
        row = int(round(y0))
        if not 0 <= row < h:
            continue
        along = sheared[row]
        sign = np.sign(along.mean()) or 1.0
        on = (along * sign > bar).astype(np.float32)
        run = cv2.blur(on.reshape(1, -1), (_SCRATCH_RUN_WIN, 1)).ravel() >= _SCRATCH_RUN_FRAC
        lo, hi = int(max(0, min(x0, x1))), int(min(w, max(x0, x1)) + 1)
        keep = np.zeros(w, dtype=bool)
        keep[lo:hi] = run[lo:hi]
        if not keep.any():
            continue

        xs = np.flatnonzero(keep)
        grown = _grow_band(sheared, row, xs, max_half, float(sign), bar)
        # Undo the shear: the band was grown at a fixed row of the sheared frame, but the
        # scratch drifts with x in the frame the mask belongs to.
        centres = np.round(y0 + slope * (xs - x0)).astype(np.int64)
        offsets = np.arange(-max_half, max_half + 1)[:, None]
        rows = centres[None, :] + offsets
        valid = grown & (rows >= 0) & (rows < h)
        cols = np.broadcast_to(xs, rows.shape)
        mask[rows[valid], cols[valid]] = 1
        touched = True

    # One ramp for every line, the same skirt allowance the detector's regions get.
    return _mask_to_score(mask, _DETECT_PAD_PX * scale) if touched and mask.any() else None


def ir_detect_target(buffer_long_edge: int, preview_long_edge: int) -> int:
    """Long edge to detect IR defects at for a buffer this size: never finer than the buffer,
    never coarser than preview scale, capped by ``_IR_MAX_UPSAMPLE`` and ``_IR_DETECT_MAX``."""
    want = int(math.ceil(buffer_long_edge / _IR_MAX_UPSAMPLE))
    return int(min(max(preview_long_edge, want), _IR_DETECT_MAX, buffer_long_edge))


def _ir_live(plane: np.ndarray) -> np.ndarray:
    """Film under the head: everything but the below-floor regions large enough to be holder."""
    dead = plane < _IR_DEAD_FLOOR
    if not dead.any():
        return np.ones(plane.shape[:2], dtype=bool)
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(dead.astype(np.uint8), connectivity=8)
    holder = np.zeros(n_lbl, dtype=bool)
    holder[1:] = stats[1:, cv2.CC_STAT_AREA] >= _IR_DEAD_MIN_AREA * dead.size
    return ~holder[labels]


def _ir_detect_scale(plane: np.ndarray) -> float:
    """Detection-plane resolution over ``_IR_DETECT_REF``, floored at 1."""
    return max(1.0, max(plane.shape[:2]) / _IR_DETECT_REF)


def _ir_win(px: int, scale: float) -> int:
    """Pinned footprint → odd window on this detection plane."""
    return int(round(px * scale)) | 1


def _fit_sample(mask: np.ndarray) -> np.ndarray:
    """Flat indices of ``mask``, strided down to ``_IR_FIT_MAX_PX`` (exact under the cap)."""
    idx = np.flatnonzero(mask.ravel())
    step = -(-idx.size // _IR_FIT_MAX_PX)
    return idx[::step] if step > 1 else idx


def downsample_ir(plane: np.ndarray, target_long_edge: int, dims: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Min-preserving IR downsample to ``target_long_edge`` (no-op if already smaller).
    ``dims`` (w, h) overrides the computed target for callers that must land on an
    existing buffer's exact shape.

    A defect is a *minimum* in IR transmittance and INTER_AREA averages sub-pixel minima
    away: a ~4 px hair downsampled 4.5x lost its dip from 0.22 to 0.31 and shattered into
    stray pixels. Eroding by the resample footprint first carries the dip through;
    ``normalize_ir``'s ``blur(dilate(ir))`` base tracks the eroded plane back up, so clean
    film still sits at ~1.0. Every IR consumer routes through here or preview and export
    detect different region sets.
    """
    plane = np.ascontiguousarray(plane, dtype=np.float32)
    h, w = plane.shape[:2]
    long_edge = max(h, w)
    if long_edge <= target_long_edge and dims is None:
        return plane
    if dims is None:
        s = target_long_edge / long_edge
        dims = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    if dims == (w, h):
        return plane
    # Erode by the resample footprint — a 1.25x downsample must not fatten by a 4.5x kernel.
    k = max(1, int(round(long_edge / target_long_edge)) | 1)
    if k > 1:
        plane = cv2.erode(plane, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.resize(plane, dims, interpolation=cv2.INTER_AREA).astype(np.float32)


def normalize_ir(plane: np.ndarray) -> np.ndarray:
    """Locally-normalized IR: ``ir / blur(dilate(ir))`` — ~1.0 on clean film, dips on
    defects, illumination-independent. Separates dust from content that raw-IR
    thresholding conflated (dilate→max estimates the clean base, blur smooths it)."""
    plane = np.ascontiguousarray(plane, dtype=np.float32)
    win = _ir_win(_IR_BASE_WIN, _ir_detect_scale(plane))
    base = cv2.blur(cv2.dilate(plane, cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))), (win, win))
    return plane / np.maximum(base, 1e-4)


def ir_detect_cutoff(slider: float, attenuation: bool) -> float:
    """UI IR sensitivity (higher = conservative) → ratio cutoff; lower slider catches
    more. Attenuation-on band sits lower (division handles the rest, only cores need cloning)."""
    s = float(np.clip(slider, 0.0, 1.0))
    return (0.85 - 0.40 * s) if attenuation else (0.95 - 0.20 * s)


def ir_defect_score(ratio: np.ndarray, cutoff: float) -> np.ndarray:
    """Continuous defect score in ``[_IR_SCORE_FLOOR, 1]``: 1 = clean film, floor
    at/below ``cutoff`` (from ir_detect_cutoff). The 3×3 erode bleeds a defect's score
    one pixel outward, covering sub-pixel hairs and the min-pool skirt. 3×3 at any detection
    scale: a sampling allowance, not a film footprint — widening it with the plane re-fattens
    the mask the finer detection just tightened."""
    span = max(_IR_GAIN_IDENTITY - cutoff, 1e-4)
    t = (np.ascontiguousarray(ratio, dtype=np.float32) - cutoff) / span
    score = np.clip(t * (1.0 - _IR_SCORE_FLOOR) + _IR_SCORE_FLOOR, _IR_SCORE_FLOOR, 1.0)
    return cv2.erode(score, np.ones((3, 3), np.uint8))


def score_weighted_fill(
    img: np.ndarray,
    score: np.ndarray,
    scales: Tuple[int, ...] = _IR_FILL_SCALES,
    reject_floor_mass: bool = False,
) -> np.ndarray:
    """Multiscale score-normalized average, blended coarse→fine by clean fraction.
    Where no support holds clean film the quotient tends to zero and the original-floor
    rule in apply_score_repair keeps the source pixel.

    ``reject_floor_mass`` measures that clean fraction *above* the score floor. A defect
    pixel scores 0.02 rather than 0, so a support seeing nothing but defect still carries
    den ≈ 0.02 — enough confidence to win, with the defect's own value as its candidate,
    and the fill quietly puts back a share of what it was asked to remove. Callers keeping
    the original-floor rule leave this off: it corrects them downstream, and rejecting those
    rungs outright shifts weight onto the coarsest support, which is the one that reaches
    across a tonal edge and over-lifts the dense side of it. A repair allowed to darken has
    no such corrector, so it has to pay for honest confidence there instead.
    """
    s3 = score[..., None]
    weighted = img * s3
    fill: Optional[np.ndarray] = None
    for i, k in enumerate(scales):
        if i == len(scales) - 1:
            num = cv2.GaussianBlur(weighted, (k, k), 0)
            den = cv2.GaussianBlur(score, (k, k), 0)
        else:
            num = cv2.boxFilter(weighted, -1, (k, k))
            den = cv2.boxFilter(score, -1, (k, k))
        cand = num / np.maximum(den, 1e-6)[..., None]
        if fill is None:
            fill = cand
        else:
            mass = (den - _IR_SCORE_FLOOR) / (1.0 - _IR_SCORE_FLOOR) if reject_floor_mass else den
            conf = np.clip(mass / _IR_FILL_TAU, 0.0, 1.0)[..., None]
            fill = fill * (1.0 - conf) + cand * conf
    assert fill is not None  # scales is never empty
    return fill


def _fill_supports(buffer_long_edge: int, factor: float) -> Tuple[int, ...]:
    """Fill ladder in buffer px: ``_IR_FILL_SCALES`` at the detection plane, plus a coarse rung
    at the reference footprint when detection runs finer than it. Both ends carry: a small fine
    end keeps the average off the far side of a tonal edge, and only the coarse rung reaches
    clean film across a wide defect."""
    fine = [int(round(k * factor)) | 1 for k in _IR_FILL_SCALES]
    film = max(factor, buffer_long_edge / _IR_DETECT_REF)
    return tuple(dict.fromkeys([int(round(_IR_FILL_SCALES[0] * film)) | 1] + fine))


def _borrow_clean_grain(src: np.ndarray, clean: np.ndarray, sigma: float) -> np.ndarray:
    """Detail of the nearest clean pixel, per pixel, high-passed at ``sigma``.

    Real film rather than synthesized noise: the donor is the closest pixel the IR score calls
    clean, so it carries the same emulsion, density and scanner noise as the hole it fills.
    Ceiling: deep inside a wide defect every pixel resolves to the same few boundary donors and
    the paste flattens toward a constant, leaving those interiors to the fill's own blend.
    """
    h, w = clean.shape
    # DIST_LABEL_PIXEL numbers the zero pixels 1..N in raster order, so flatnonzero inverts it.
    _, labels = cv2.distanceTransformWithLabels((~clean).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    nearest = np.flatnonzero(clean.ravel())[labels.ravel().astype(np.intp) - 1]
    # Mirror through the donor, don't sample it: a whole row across a hair shares one nearest
    # clean pixel, and pasting that verbatim streaks the grain into bands.
    ny, nx = np.divmod(nearest, w)
    py, px = np.divmod(np.arange(h * w), w)
    mirrored = np.clip(2 * ny - py, 0, h - 1) * w + np.clip(2 * nx - px, 0, w - 1)
    take = np.where(clean.ravel()[mirrored], mirrored, nearest)
    grain = src - cv2.GaussianBlur(src, (0, 0), sigma)
    return grain.reshape(-1, 3)[take].reshape(src.shape)


def apply_score_repair(
    img: ImageBuffer,
    score_det: np.ndarray,
    *,
    floor: bool = True,
    long_edge: Optional[int] = None,
    factor: Optional[float] = None,
) -> ImageBuffer:
    """Bake the score-weighted fill into the linear source (new array). The detection-scale
    score is upsampled; the fill convolutions rerun at the buffer's own resolution with
    rescaled supports — filled pixels are never upsampled.

    One repair for every defect source: an IR score, a luma-detected speck or a painted
    stroke all arrive here as a score map. ``floor`` keeps the original-floor rule, which
    only holds where the defect is known to be dark in transmittance (dust). A painted
    stroke turns it off: a scratch has lost emulsion and reads *brighter* than the film
    around it, so its repair has to be free to darken. ``long_edge`` overrides the film
    footprint the support ladder is derived from, so repairing a crop picks the same
    supports as the whole frame would (see ``repair_components``).
    """
    h, w = img.shape[:2]
    src = np.ascontiguousarray(img, dtype=np.float32)
    if score_det.shape[:2] == (h, w):
        score = np.ascontiguousarray(score_det, dtype=np.float32)
        factor = factor or 1.0
    else:
        factor = max(h / score_det.shape[0], w / score_det.shape[1])
        score = cv2.resize(score_det, (w, h), interpolation=cv2.INTER_LINEAR)
    fill = score_weighted_fill(src, score, _fill_supports(long_edge or max(h, w), factor), reject_floor_mass=not floor)
    a = np.clip((_IR_WRITE_HI - score) / (_IR_WRITE_HI - _IR_WRITE_LO), 0.0, 1.0)
    a = (a * a * (3.0 - 2.0 * a))[..., None]
    out = src * (1.0 - a) + fill * a
    # A weighted average lands grainless, so a repair reads as a smooth patch against film.
    sigma = max(1.0, factor)
    clean = score >= _IR_WRITE_HI
    if clean.any():
        out += a * _borrow_clean_grain(src, clean, sigma)
    # Original-floor rule: dust is dark in negative transmittance — repairs only lighten. Compared
    # on the low-frequency deficit, since per pixel it is a half-wave rectifier: it keeps the
    # fill's grain peaks and clips its troughs, sitting the repair ~3% bright with half the
    # texture. Under the same ramp, so a pixel the fill never touched stays byte-identical.
    if floor:
        lo_deficit = cv2.GaussianBlur(src, (0, 0), sigma) - cv2.GaussianBlur(out, (0, 0), sigma)
        out += a * np.maximum(lo_deficit, 0.0)
    return ensure_image(np.maximum(out, 0.0))


def film_scale(shape: Tuple[int, int]) -> float:
    """Pixels per unit of film footprint for a buffer this size — the factor a score measured
    at the buffer's own resolution needs so the fill's supports stay film-scale rather than
    grain-scale (see ``_fill_supports``)."""
    return max(1.0, max(shape) / _IR_DETECT_REF)


def repair_components(img: ImageBuffer, score_det: np.ndarray, *, floor: bool = True, factor: Optional[float] = None) -> ImageBuffer:
    """``apply_score_repair`` per defect, each in its own padded crop.

    The fill is four convolutions over whatever buffer it is handed. That is right for an
    IR score, where defects are spread over the frame, and wasteful for the handful of
    painted strokes or detected specks this serves — at export resolution it would filter
    a hundred megapixels to repair a dozen. The support ladder and the mask's upsample
    factor still come from the whole frame, so a crop repairs exactly as it would there.
    """
    h, w = img.shape[:2]
    if score_det.shape[:2] == (h, w):
        score = np.ascontiguousarray(score_det, dtype=np.float32)
        factor = factor or 1.0
    else:
        factor = max(h / score_det.shape[0], w / score_det.shape[1])
        score = cv2.resize(score_det, (w, h), interpolation=cv2.INTER_LINEAR)
    m = (score < 1.0).astype(np.uint8)
    if not m.any():
        return img
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    # Past this many, cropping costs more than the whole-frame convolutions it avoids.
    if n_lbl - 1 > _REPAIR_MAX_COMPONENTS:
        return apply_score_repair(img, score, floor=floor, factor=factor)
    src = np.ascontiguousarray(img, dtype=np.float32)
    out = src.copy()
    # Reach of the coarsest support, so every crop holds the clean film the fill averages.
    pad = max(_fill_supports(max(h, w), factor))
    for i in range(1, n_lbl):
        bx, by = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        x0, y0 = max(0, bx - pad), max(0, by - pad)
        x1 = min(w, bx + int(stats[i, cv2.CC_STAT_WIDTH]) + pad)
        y1 = min(h, by + int(stats[i, cv2.CC_STAT_HEIGHT]) + pad)
        sub = apply_score_repair(src[y0:y1, x0:x1], score[y0:y1, x0:x1], floor=floor, long_edge=max(h, w), factor=factor)
        # This component only: a neighbour clipped by the crop repairs badly here and gets
        # its own correctly-padded crop anyway.
        mb = labels[y0:y1, x0:x1] == i
        out[y0:y1, x0:x1][mb] = np.asarray(sub)[mb]
    return ensure_image(out)


def route_wide_defects(score: np.ndarray, *, budget: Optional[float] = _IR_ROUTE_BUDGET) -> Optional[np.ndarray]:
    """Detection-scale mask of at-floor components past the fill's reach, for
    apply_hair_inpaint. Over ``budget`` (misregistered/garbage IR) → None + warning.

    ``budget=None`` lifts the cap for hand-placed repairs. The cap guards an *automatic*
    detector, where a misregistered IR plane can call half the frame a defect; a full-width
    line clears it on its own, so it would refuse exactly the case that needs the inpaint.
    """
    at_floor = (score <= _IR_SCORE_FLOOR + 1e-6).astype(np.uint8)
    if not at_floor.any():
        return None
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(at_floor, connectivity=8)
    routed = np.zeros_like(at_floor)
    scale = _ir_detect_scale(score)
    radius = int(round(_IR_ROUTE_RADIUS * scale))
    side = 2 * radius - 1
    hit = False
    for i in range(1, n_lbl):
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if min(bw, bh) < side:  # can't contain a side² solid → radius under the bar
            continue
        x0, y0 = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        sub = np.pad((labels[y0 : y0 + bh, x0 : x0 + bw] == i).astype(np.uint8), 1)
        if float(cv2.distanceTransform(sub, cv2.DIST_C, 3).max()) >= radius:
            routed[labels == i] = 1
            hit = True
    if not hit:
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(round(_IR_ROUTE_DILATE * scale)) + 1,) * 2)
    routed = cv2.dilate(routed, k)
    frac = float(routed.mean())
    if budget is not None and frac > budget:
        logger.warning("Retouch: routed defects cover %.1f%% of the frame — inpaint skipped, fill only", frac * 100.0)
        return None
    return routed


def _ir_decontaminate(ratio: np.ndarray, vis_log: np.ndarray) -> Tuple[np.ndarray, float]:
    """Divide the visible-image ghost out of the normalized IR: robust LS fit of
    log(ir) on log(vis) over clean film, then ``ratio / Π vis_c^b_c``. Exponents clamp
    to ≥0 (density can only block IR) and fit to ~0 on a clean scanner (→ no-op). Also
    returns the exponent sum — ghost strength, which is how ``ir_ratio_and_gain`` bails."""
    if ratio.size < 500:
        return ratio, 0.0
    # Fit on clean film only. Dust dips *both* planes, so a fit that sees it explains
    # the defect away as ghost and the division stops lifting it. Trim by ratio
    # percentile, not a fixed cutoff (a strong ghost drags clean film below any fixed
    # one) and not by residual (the dust fits itself perfectly — residual can't see it).
    keep = _fit_sample(ratio >= np.percentile(ratio, _IR_XTALK_TRIM))
    y = np.log(np.clip(ratio.ravel()[keep], 1e-4, 1.0))
    x = vis_log.reshape(-1, vis_log.shape[-1])[keep]
    if y.size < 500:
        return ratio, 0.0
    # Intercept column, dropped from the result: both logs sit below their own dilate+blur
    # envelope, so origin-forced least squares reads that shared negative offset as slope
    # and fits b≈0.6 on two *independent* noisy planes.
    x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    b = np.clip(np.linalg.lstsq(x, y, rcond=None)[0][:3], 0.0, _IR_XTALK_MAX)
    ghost = float(np.abs(b).sum())
    if ghost < _IR_XTALK_MIN:
        return ratio, ghost
    return np.clip(ratio / np.exp((vis_log * b).sum(-1)), 0.0, 1.5).astype(np.float32), ghost


def _fit_refraction_gammas(ratio: np.ndarray, vis_log: np.ndarray, img_det: np.ndarray) -> Tuple[float, ...]:
    """Per-channel refraction γ: the slope of log(vis_norm) on log(ratio) over the
    shallow-dust band, as the median of the per-pixel slopes over locally flat film.

    Median and flat restriction are both load-bearing. The band selects on the IR ratio
    alone, so besides dust it collects ``_ir_decontaminate``'s residue at hard image edges,
    and least squares through the origin is x²-weighted — that deep non-dust minority
    dominated it, reading γ 1.9/2.2/2.2 for dust measuring ~1.0/1.1/1.2 and over-correcting
    every speck into a dark cyan blob. Median alone reads 1.3/1.8/1.8, flat-only least
    squares 1.4/2.1/2.0, and γ 1.5 already tints."""
    band = (ratio > 0.70) & (ratio < 0.92)
    if int(band.sum()) < 500:
        return (_IR_GAMMA_FALLBACK,) * 3
    # ksize=5 carries its own smoothing, so no separate blur.
    edge = np.abs(cv2.Laplacian(img_det[:, :, 1], cv2.CV_32F, ksize=5))
    flat = band & (edge < np.percentile(edge[band], _IR_FIT_FLAT_PCT))
    fit = _fit_sample(flat if int(flat.sum()) >= _IR_FIT_MIN_PX else band)
    # The band bounds ratio away from 1, so the per-pixel slope needs no guard.
    xb = np.log(ratio.ravel()[fit])
    vl = vis_log.reshape(-1, 3)[fit]
    return tuple(float(np.clip(np.median(vl[:, c] / xb), _IR_GAMMA_LO, _IR_GAMMA_HI)) for c in range(3))


def _ir_clean_base(img_det: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    """Local clean-film level per channel over ``_IR_CAP_WIN``: mean of the pixels the
    IR ratio calls clean, minus ``_IR_CAP_SIGMA`` of their σ (see the constants block)."""
    win_px = _ir_win(_IR_CAP_WIN, _ir_detect_scale(ratio))
    win = (win_px, win_px)
    w_clean = (ratio >= _IR_GAIN_IDENTITY).astype(np.float32)
    den = np.maximum(cv2.blur(w_clean, win), 1e-6)[..., None]
    mean = cv2.blur(img_det * w_clean[..., None], win) / den
    var = cv2.blur(img_det * img_det * w_clean[..., None], win) / den - mean * mean
    base = mean - _IR_CAP_SIGMA * np.sqrt(np.clip(var, 0.0, None))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, win)
    dil = cv2.blur(cv2.dilate(img_det, kernel), win)
    return np.where(den > _IR_CAP_MIN_SUPPORT, base, dil)


def _ir_normalize_ratio(ratio: np.ndarray, live: np.ndarray) -> np.ndarray:
    """Both ratio landmarks onto what the absolute constants expect: clean-film floor on
    ``_IR_GAIN_IDENTITY``, dip scale on ``_IR_REF_SIGMA`` (see the constants block). ``live``
    only: the dead-margin 1.0s inflate σ ~20% on a strip scan. MAD, not std, so a dusty
    minority can't move either landmark."""
    sample = ratio[live]
    if sample.size < 500:  # nothing measurable: leave the landmarks alone
        return ratio
    med = float(np.median(sample))
    sigma = 1.4826 * float(np.median(np.abs(sample - med)))
    pivot = float(np.clip(med - _IR_NOISE_SIGMA * sigma, _IR_PIVOT_LO, _IR_GAIN_IDENTITY))
    if pivot <= _IR_PIVOT_LO:
        logger.warning("IR dust: clean-film pivot floored at %.2f (IR σ %.3f) — very noisy IR plane", _IR_PIVOT_LO, sigma)
    # A clipped pivot is an exact no-op: x/x is 1.0 in IEEE, float32 × 1.0 exact.
    scale = _IR_GAIN_IDENTITY / pivot
    ratio = (ratio * scale).astype(np.float32)
    # The rescale carries σ with it, so no second median pass. Dips only: a noiseless plane
    # takes the full _IR_SCALE_MAX, which would land its clean level at 1.15.
    k = float(np.clip(_IR_REF_SIGMA / max(sigma * scale, 1e-6), 1.0, _IR_SCALE_MAX))
    if k > 1.0:
        stretched = _IR_GAIN_IDENTITY - (_IR_GAIN_IDENTITY - ratio) * k
        ratio = np.where(ratio < _IR_GAIN_IDENTITY, stretched, ratio).astype(np.float32)
    return ratio


def ir_ratio_and_gain(ir_det: np.ndarray, img_det: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Tuple[float, ...]]:
    """Detection-scale ``(ratio, gain HxWx3, degenerate, gammas)`` for IR-division
    attenuation: semi-transparent dust recovered by ``RGB / ratio^γ``, γ per channel from
    ``_fit_refraction_gammas``. ``degenerate`` = IR carrying image content
    (B&W/Kodachrome) → caller skips the whole IR bake."""
    plane = ir_det[:, :, 0] if ir_det.ndim == 3 else ir_det
    ratio = normalize_ir(plane)
    # No film under the head is not a defect; left as a dip the holder margin would score
    # as one giant routed component and swamp the routing budget.
    live = _ir_live(plane)
    ratio[~live] = 1.0
    img_det = np.ascontiguousarray(img_det, dtype=np.float32)
    if img_det.shape[:2] != ratio.shape[:2]:
        img_det = cv2.resize(img_det, (ratio.shape[1], ratio.shape[0]), interpolation=cv2.INTER_AREA)

    vis_log = np.stack([np.log(np.clip(normalize_ir(img_det[:, :, c]), 1e-4, 1.0)) for c in range(3)], axis=-1)
    ratio, ghost = _ir_decontaminate(ratio, vis_log)
    # On the fitted exponent, not on how far the ratio dips: a few percent of IR noise
    # (deepened by the min-preserving downsample) read as silver on clean C41 rolls.
    degenerate = ghost > _IR_DEGENERATE_GHOST
    # After the unmixing, never before: it clips log(ratio) at 1.0, and a rescaled clean
    # population piles into that clip and flattens the fitted exponent.
    ratio = _ir_normalize_ratio(ratio, live)

    gammas = _fit_refraction_gammas(ratio, vis_log, img_det)
    base = np.clip(ratio / _IR_GAIN_IDENTITY, 1e-4, 1.0)
    gain = np.empty(ratio.shape + (3,), dtype=np.float32)
    for c in range(3):
        gain[:, :, c] = np.minimum(_IR_GAIN_CLAMP, base ** (-gammas[c]))
    # Never lift a pixel past its own local clean base (see _IR_CAP_WIN); floored at 1 so the
    # cap only ever holds the bake back, never darkens a pixel itself.
    clean = _ir_clean_base(img_det, ratio)
    np.minimum(gain, np.maximum(clean / np.maximum(img_det, 1e-5), 1.0), out=gain)
    return ratio, gain, degenerate, gammas


def apply_ir_attenuation(img: ImageBuffer, gain_det: np.ndarray) -> ImageBuffer:
    """Visible buffer × upsampled per-channel IR gain map (new array — buffers are read-only)."""
    h, w = img.shape[:2]
    gain = gain_det if gain_det.shape[:2] == (h, w) else cv2.resize(gain_det, (w, h), interpolation=cv2.INTER_LINEAR)
    # cv2.multiply, not `a * b`: the product of two float32 buffers is already float32,
    # so the astype numpy needs here would copy the whole frame a second time.
    return ensure_image(cv2.multiply(np.ascontiguousarray(img, dtype=np.float32), gain))


def ir_bake_token(retouch, has_ir: bool) -> str:
    """Config-identity token for the IR bake (mirrors ``flatfield_token``); folded into
    source_hash so a toggle or threshold drag invalidates the engine cache."""
    if not (retouch.ir_dust_remove and has_ir):
        return ""
    method = getattr(retouch, "ir_method", IR_METHOD_NEGPY)
    tail = "" if method == IR_METHOD_NEGPY else f"|{method}"
    return f"|ir{int(retouch.ir_attenuation)}r{round(float(retouch.ir_threshold), 3)}{tail}"


def apply_hair_inpaint(
    img: ImageBuffer,
    hair_masks: List[np.ndarray],
    radius: int = _HAIR_INPAINT_RADIUS,
    dilate_px: Optional[int] = None,
) -> ImageBuffer:
    """Structure-following fill of long/twisted defects (``cv2.inpaint``, Navier–Stokes)
    baked into the linear source. Each detection-scale mask is upsampled to the buffer,
    unioned and dilated to cover the PSF skirt; only masked pixels are overwritten (the
    rest stay byte-identical — the 8-bit encode cv2.inpaint requires touches only the
    fabricated hairline). Returns a new array (buffers are read-only)."""
    h, w = img.shape[:2]
    masks = [hm for hm in hair_masks if hm is not None]
    if not masks:
        return img
    factor = max(1.0, h / masks[0].shape[0], w / masks[0].shape[1])
    if dilate_px is None:
        # A detection-scale mask knows its boundary only to within the upsample factor,
        # so the dilate tracks it; at 1:1 that's _HAIR_DILATE_PX, the PSF skirt alone.
        dilate_px = max(_HAIR_DILATE_PX, round(factor))
    m = np.zeros((h, w), dtype=np.uint8)
    for hm in masks:
        r = hm if hm.shape[:2] == (h, w) else cv2.resize(hm.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        m |= (np.asarray(r) > 0.5).astype(np.uint8)
    if not m.any():
        return img
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        m = cv2.dilate(m, k)
    src = np.ascontiguousarray(img, dtype=np.float32)
    out = src.copy()
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for i in range(1, n_lbl):
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        x0, y0 = max(0, bx - _HAIR_INPAINT_PAD), max(0, by - _HAIR_INPAINT_PAD)
        x1 = min(w, bx + int(stats[i, cv2.CC_STAT_WIDTH]) + _HAIR_INPAINT_PAD)
        y1 = min(h, by + int(stats[i, cv2.CC_STAT_HEIGHT]) + _HAIR_INPAINT_PAD)
        # Mask the whole crop, not just this component: a neighbour reaching into the
        # bbox must stay unknown or it becomes clone source and its dust is filled back in.
        sub_m = np.ascontiguousarray(m[y0:y1, x0:x1])
        crop = src[y0:y1, x0:x1]
        # Encode against the crop's clean range — clip(0,1) posterizes fills in dark regions.
        ctx = crop[sub_m == 0]
        lo = float(np.percentile(ctx, 0.5)) if ctx.size else 0.0
        hi = float(np.percentile(ctx, 99.5)) if ctx.size else 1.0
        span = max(hi - lo, 1e-4)
        enc = np.clip((crop - lo) / span, 0.0, 1.0) ** (1.0 / _HAIR_INPAINT_GAMMA)
        filled = cv2.inpaint((enc * 255.0 + 0.5).astype(np.uint8), sub_m, radius, cv2.INPAINT_NS)
        dec = ((filled.astype(np.float32) / 255.0) ** _HAIR_INPAINT_GAMMA) * span + lo
        # ...but only keep this component (a neighbour clipped by the bbox fills badly here,
        # and gets its own correctly-padded crop anyway), alpha-feathered across the dilate
        # band: full fill on the detected defect, ramp over the skirt. dilate_px=0 → no feather.
        mb = labels[y0:y1, x0:x1] == i
        d = cv2.distanceTransform(mb.astype(np.uint8), cv2.DIST_C, 3)
        a = np.minimum(d / float(dilate_px + 1), 1.0)[..., None]
        blended = crop * (1.0 - a) + dec * a
        out[y0:y1, x0:x1][mb] = blended[mb]
    # Navier–Stokes propagates a smooth field, so the fill lands grainless (see
    # apply_score_repair, which fills the same kind of hole by a different route).
    filled_px = m.astype(bool)
    clean = ~filled_px
    if clean.any():
        out[filled_px] += _borrow_clean_grain(src, clean, max(1.0, factor))[filled_px]
    return out


def hair_bake_token(retouch) -> str:
    """Detection-param identity for the hair inpaint (folded into source_hash when a
    hair is actually detected). Distinct params → distinct inpainted source."""
    r = retouch
    return f"|hair{int(r.dust_remove)}_{round(float(r.dust_threshold), 3)}_{int(r.dust_size)}_{int(r.ir_dust_remove)}_{round(float(r.ir_threshold), 3)}"
