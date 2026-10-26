import hashlib
import math
from typing import List, Optional, Tuple

import negpy_core
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.geometry.logic import smooth_polyline
from negpy.features.retouch.models import HEAL_SIZE_REF
from negpy.kernel.image.logic import get_luminance
from negpy.kernel.image.validation import ensure_image
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

# Spread floor: stops noise on low-contrast sources (fog, flat frames) from being
# amplified to full range. Dust sits about a density unit above its surroundings.
_PROXY_MIN_SPREAD = 0.8
# Pad heals past the detected bright core: an unhealed soft skirt reads as a halo.
_DETECT_PAD_PX = 2.5
# Optical detection: density excess over a robust background, averaged over 3x3, in units of
# its local MAD σ. Under the average a speck keeps its full excess while grain drops by 3.
# The robust filters run on 8-bit planes (the uint8 median blur is O(1) there); the excess is
# scaled by _DETECT_MAD_GAIN first so grain σ spans enough levels.
_DETECT_AVG_PX = 3
_DETECT_MAD_GAIN = 4.0
_DETECT_SIGMA_MIN = 0.003
# Slider → seed bar in σ: linear to the default, geometric above it.
_DETECT_Z_LOOSE = 3.0
_DETECT_Z_DEFAULT = 9.0
_DETECT_Z_TIGHT = 48.0
_DETECT_DEFAULT_POS = 0.66
# Hysteresis: a component seeded above the bar grows through connected pixels down to this
# floor (absolute, or a fraction of the seed bar), within a reach of the seed. Grain is
# isolated, so it never joins; a defect's soft skirt does.
_DETECT_Z_GROW = 2.5
_DETECT_Z_GROW_FRAC = 0.3
_DETECT_GROW_REACH = 2  # times dust_size, px
# Texture raises the bar: at detection scale a thin dense image line is the same shape as a
# hair, and only its surroundings tell them apart. Measured candidates included, so a fat
# defect raises its own bar. Clean film of any grain sits under the knee.
_DETECT_TEXTURE_KNEE = 0.02
_DETECT_TEXTURE_GAIN = 40.0  # bar multiplier per unit of σ over the knee
# A hair-shaped component (negpy_core's is_hair) tests its median σ against a higher, steeper knee: a hair across
# busy film stays under it, a seed grown along a tonal edge does not.
_DETECT_HAIR_TEXTURE_KNEE = 0.075
_DETECT_HAIR_TEXTURE_GAIN = 150.0
_DETECT_HAIR_FILL_GAIN = 2.0  # cap on that σ, × the σ with hairs filled, so a deep hair cannot raise its own bar
# Below this normalized density the film is clear base or holder; noise there is not dust.
_DETECT_PROXY_MIN = 0.15

# Manual heal gate. A painted stroke marks a *search area*, not a stamp: only pixels that
# stand out from the film around them are repaired, so clean grain under a generous brush
# stays byte-identical. The gate is two-sided because dust is a bright outlier in density
# and a scratch a dark one (#791).
#
# The excess is measured on a 3x3 mean of the local high-pass, against the MAD σ of that
# same quantity over the stroke's neighbourhood. Averaging drops uncorrelated grain by 3
# while a defect of 2 px or more keeps its full excess, and measuring σ keeps the bars
# valid on any film and scanner. An absolute density threshold cannot separate the two.
_MANUAL_Z_HI = 8.0
# Grow bar for the hysteresis below: the floor a connected pixel must clear to join a
# defect already found. Absolute, or a fraction of the core's strength for a defect whose
# skirt is proportionally deep.
_MANUAL_Z_GROW = 2.0
_MANUAL_Z_GROW_FRAC = 0.25
# Backstop bar: a stroke whose strongest pixel never reaches _MANUAL_Z_HI is rescaled
# against its own maximum, so a faint but real defect still repairs. Under this bar the
# stroke stays a no-op: the brush found clean film.
_MANUAL_Z_MIN = 5.0
# Local-statistics window, as a multiple of the brush radius. Wide enough that a defect
# filling the brush cannot set its own baseline.
_MANUAL_WIN_FACTOR = 3.0
# Soft edge on the search area, so a defect crossing the brush rim does not repair to a
# hard line.
_MANUAL_RIM_PX = 1.5

# Transport scratches (#788): a nearly straight mark a few px wide running the length of
# the film. Per pixel it sits under the manual-heal seed bar, so no per-pixel gate can
# reach it. Only integrating along the line finds it.
#
# Cross-section band-pass: film grain is finer, image structure broader.
_SCRATCH_FINE_PX = 1.2
_SCRATCH_BROAD_PX = 9.0
# The ridge response is normalized by a *local* noise scale. A global one lets a busy
# corner set the bar and buries a faint scratch running through smooth sky.
_SCRATCH_NOISE_WIN = 151
# Slope search, rise per unit run. The scratch is straight but the film is rarely square to
# the sensor, and a fraction of a degree drifts tens of px across a frame, enough for an
# axis-aligned collapse to smear the ridge away.
_SCRATCH_SLOPE_MAX = 0.02
_SCRATCH_SLOPE_STEP = 0.00025
# Rows searched either side of the click, and how hard the fit is pulled back toward it.
# Without the pull it snaps to the strongest ridge in the band, not the one clicked.
_SCRATCH_SEARCH_ROWS = 30
_SCRATCH_CLICK_PULL = 12.0
# Slider range for the bar a ridge must clear (see scratch_detect_bar); the default
# slider position sits in the middle of it.
_SCRATCH_Z_LOOSE = 0.4
_SCRATCH_Z_TIGHT = 1.6
# Presence along the line: the bar must hold over this fraction of a window this wide
# before a stretch is repaired. Transport scratches fade in and out, so measure the extent.
_SCRATCH_RUN_WIN = 151
_SCRATCH_RUN_FRAC = 0.35
# A trace this weak is not a scratch: the click found clean film.
_SCRATCH_MIN_EVIDENCE = 0.25
# Ceiling on the repaired half-width, px at HEAL_SIZE_REF. The band grows from the
# scratch, so this only stops a runaway where the ridge never breaks.
_SCRATCH_WIDTH_MAX = 14.0
_SCRATCH_WIDTH_MIN = 3.0

# Detection follows the buffer it repairs, at most this far under it. The score is upsampled
# onto that buffer and the fill supports scale with it, so coarse detection writes a fat mask
# and averages over a wide support. A defect on a tonal edge is then rebuilt from the bright
# side and prints as a dark blotch on the light one.
_IR_MAX_UPSAMPLE = 1.5
_IR_DETECT_MAX = 3600  # memory: ir_ratio_and_gain holds ~10 planes of it
_IR_DOWNSAMPLE_WORK_BYTES = 64 * 1024 * 1024
# The film-footprint windows below are px at this detection long edge and scale with the
# plane (_ir_win). On a finer plane a wide hair fills an unscaled base window, depresses
# its own base and stops reading as a defect.
_IR_DETECT_REF = 1600
# IR ratio-normalization base window (px at _IR_DETECT_REF, pinned like HEAL_SIZE_REF).
# Defects wider than about half of it depress their own base.
_IR_BASE_WIN = 25
_IR_GAIN_IDENTITY = 0.97  # gain is identity at/above this ratio
_IR_GAIN_CLAMP = 2.0  # caps misregistration halos
# Per-channel refraction γ, fitted per frame. The patent values under-correct file IR.
_IR_GAMMA_LO = 1.0
_IR_GAMMA_HI = 2.2
_IR_GAMMA_FALLBACK = 1.5
# Below this the beam is blocked outright: holder, not film. ponytail: absolute; a
# low-IR-gain scanner would want a percentile. Opaque hairs also pass under the floor, so
# only below-floor regions this large (fraction of the frame) count as holder. Writing off
# the rest leaves them unrepairable once the plane resolves their cores.
_IR_DEAD_FLOOR = 0.05
_IR_DEAD_MIN_AREA = 0.002
# Clean-film pivot: normalize_ir's base is a local max, so clean film sits some multiple of
# σ_IR under 1, depending on the scanner. Left absolute, a Coolscan 5000 puts most of the
# frame below _IR_GAIN_IDENTITY, starves _ir_clean_base into its local-max fallback and
# leaves mottled film at every slider position (#647). Measured at detection scale, on the
# same ratio it corrects.
_IR_NOISE_SIGMA = 3.0
# Bounds the rescale, so coverage above 50% (a median inside the dust) cannot scale the
# ratio clean and disable IR removal. ponytail: absolute; the knob if a scanner needs more.
_IR_PIVOT_LO = 0.60
# Dip depth is scanner-dependent: clean-film σ differs by an order of magnitude between a
# Coolscan and a Plustek DNG, putting the same speck either side of every absolute landmark
# below. Stretch-only, so σ >= _IR_REF_SIGMA is an exact no-op.
_IR_REF_SIGMA = 0.02
_IR_SCALE_MAX = 6.0  # bounds a degenerate/quantized plane measuring σ ~0
# IR glare, in clean-film σ over the median. Dust only dims IR, so a brighter pixel is
# scanner glare, such as a lit frame edge; in normalize_ir's local max it would carry a base
# window into clean film and score that film as a defect strip.
_IR_GLARE_SIGMA = 4.0
# Crosstalk unmixing: dye and silver absorb some IR, so the IR plane carries a ghost of the
# image that normalize_ir's spatial high-pass cannot see. A sharp edge survives it.
_IR_XTALK_MAX = 0.8  # per-channel exponent cap; ≥0 only — density can only block IR
_IR_XTALK_MIN = 0.02  # |b| sum below this is a noise-level fit → exact no-op
_IR_DEGENERATE_GHOST = 0.5  # fitted exponent sum above this: IR mirrors the image (B&W/Kodachrome)
_IR_XTALK_TRIM = 5.0  # fit drops this bottom-ratio percentile (the dust minority)
# γ fit sample: keep this flattest fraction of the band by visible Laplacian. Below
# _IR_FIT_MIN_PX drop the restriction instead of fitting a handful of pixels.
_IR_FIT_FLAT_PCT = 40
_IR_FIT_MIN_PX = 200
# Fit sample cap: _ir_decontaminate and _fit_refraction_gammas resolve a few per-frame
# scalars, so they stride their pixel set down to this instead of growing with the plane.
_IR_FIT_MAX_PX = 200_000
# Clean-base cap window (px at _IR_DETECT_REF, odd). The bake may never lift a pixel above
# its own local clean base: past that it invents signal instead of recovering it. Needed
# because downsample_ir is min-preserving while the visible arrives area-averaged, so at
# detection scale the ratio's dip runs deeper and about 1 px wider than the defect the
# visible carries. Uncapped, that skirt lifts clean film and every speck renders with a dark
# outline. Reaches ±4 px, past _DETECT_PAD_PX's skirt. Base = defect-excluded local mean -
# _IR_CAP_SIGMA*σ, not blur(dilate): the dilate is a local max about 2σ of grain high and
# re-admits the ring on grainy film (#563). Under _IR_CAP_MIN_SUPPORT clean pixels in the
# window the max estimate returns, deep inside wide defects where _IR_GAIN_CLAMP binds first.
_IR_CAP_WIN = 9
_IR_CAP_SIGMA = 1.0
_IR_CAP_MIN_SUPPORT = 0.1  # fraction of the window (~8 px at 9×9)

# IR reconstruction: concepts ported from digital-fauxice (MIT, © 2026 Rohan Pandula, see
# NOTICE.md): continuous score, score-weighted fill, original-floor rule.
# Score: 1 = clean (ratio >= _IR_GAIN_IDENTITY), floor at or below the slider's cutoff.
# Never thresholded, so there is no mask edge to halo and no coverage fraction to abort on.
_IR_SCORE_FLOOR = 0.02
# Fill supports (detection-scale px, times the buffer's upsample factor). Candidate per
# support: Σ(rgb*score*win)/Σ(score*win), so low-score neighbours self-exclude. A finer
# support wins once its clean fraction reaches _IR_FILL_TAU, and edges continue through.
_IR_FILL_SCALES = (9, 5, 3)
_IR_FILL_TAU = 0.15
# Write ramp: untouched above HI so grain survives, full fill at or below LO.
_IR_WRITE_HI = 0.85
_IR_WRITE_LO = 0.40
# Route to inpaint only components with a core the fill cannot see across (chebyshev
# radius >= 5, a solid 9x9 interior). Thin hairs stay with the fill: every pixel is within
# reach of clean film, and NS inpaint would smear structure the fill keeps. The budget
# bounds only this heavy path; the fill always runs.
_IR_ROUTE_RADIUS = 5
_IR_ROUTE_DILATE = 2
_IR_ROUTE_BUDGET = 0.02  # fraction of the frame
# Crop-per-defect stops paying past this count (see repair_components).
_REPAIR_MAX_COMPONENTS = 256

# Strong hairs and scratches route to structure-following inpaint instead of the weighted
# fill: a long twist crosses varied background, and averaging across it smears the structure
# the inpaint follows. Detection-scale px. See is_hair in negpy_core/src/retouch.rs.
_HAIR_MIN_AREA = 20
_HAIR_MIN_ELONG = 8.0  # area/thickness² ≈ length/thickness; round specks measure 1–3
# Inpaint fill: the dilate covers the PSF skirt at 1:1 (apply_hair_inpaint widens it to
# track the mask's upsample), then the NS radius, then gamma to give the 8-bit encode a
# perceptual spread (the inpaint is 8-bit). Navier-Stokes propagates outward from the
# mask boundary only, so each defect is filled in its own bbox + _HAIR_INPAINT_PAD. Same
# pixels, without gamma-encoding the whole frame to serve a hairline.
_HAIR_DILATE_PX = 1
_HAIR_INPAINT_RADIUS = 3
_HAIR_INPAINT_GAMMA = 2.2
_HAIR_RANGE_SAMPLE_PX = 1 << 20  # clean pixels sampled for a crop's encode range
_HAIR_TILE_PX = 1024  # inpaint tile; the solver walks whatever image it is handed
_HAIR_TILE_HALO = 32  # px of neighbourhood a tile's core fill can see
_HAIR_INPAINT_PAD = 16


def _density(img: ImageBuffer) -> np.ndarray:
    """Log density of the luminance. Local contrast measured here is exposure-invariant:
    a gain on the source is an offset in density and cancels in every difference."""
    return -negpy_core.log_density_open(get_luminance(img))


def _mask_to_score(mask: np.ndarray, pad_px: float) -> np.ndarray:
    """Binary defect mask → the continuous score every repair consumes: at the floor on
    the defect, ramping to clean over ``pad_px``. The ramp is the skirt allowance a hard
    mask edge would leave behind as a halo."""
    return negpy_core.mask_to_score((np.asarray(mask) != 0).astype(np.uint8), float(pad_px), _IR_SCORE_FLOOR)


def split_hairs(mask: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Defect mask → ``(compact, hairs)``. Compact defects go to the weighted fill; long
    twisted ones to structure-following inpaint, which keeps detail the fill would average
    across. ``hairs`` is None when nothing is thin enough."""
    return negpy_core.split_hairs(np.ascontiguousarray(mask, dtype=np.uint8), (_HAIR_MIN_AREA, _HAIR_MIN_ELONG))


def _u8(plane: np.ndarray) -> np.ndarray:
    return (np.clip(plane, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _texture_win(shape: Tuple[int, ...], dust_size: int) -> int:
    return int(max(7, max(1.0, float(dust_size)) * film_scale(shape) * 4.0)) * 2 + 1


def compute_dust_stats(img: ImageBuffer, dust_size: int) -> Tuple[np.ndarray, ...]:
    """Threshold-independent detection maps ``(proxy, background, z, texture)``, the
    expensive part of a detection pass, cacheable across threshold changes. ``z`` is the
    3x3-averaged density excess over the median background in units of its own local MAD σ;
    ``texture`` the wide-window σ of the density."""
    shape = img.shape[:2]
    # Size is film footprint at _IR_DETECT_REF; the windows follow the plane (see _ir_win).
    base_size = max(1.0, float(dust_size)) * film_scale(shape)
    v_win = int(max(3, base_size * 3.0)) * 2 + 1
    # Median: the level of the film around a defect narrower than half the window. Opening:
    # identity on the soft ramp of a tonal edge and on anything wider than a speck, both of
    # which the median alone calls dense. Whichever is higher is the background.
    disk = negpy_core.ellipse_kernel((2 * int(round(base_size)) + 1,) * 2)
    constants = (_PROXY_MIN_SPREAD, _DETECT_AVG_PX, _DETECT_MAD_GAIN, _DETECT_SIGMA_MIN)
    return negpy_core.dust_stats(_density(img), v_win, _texture_win(shape, dust_size), disk, constants)


def detect_bar(slider: float) -> float:
    """UI Threshold (higher = conservative) → the seed bar in local σ; 1.0 is off (inf)."""
    s = float(np.clip(slider, 0.0, 1.0))
    if s >= 1.0:
        return math.inf
    if s <= _DETECT_DEFAULT_POS:
        return _DETECT_Z_LOOSE + (_DETECT_Z_DEFAULT - _DETECT_Z_LOOSE) * s / _DETECT_DEFAULT_POS
    t = (s - _DETECT_DEFAULT_POS) / (1.0 - _DETECT_DEFAULT_POS)
    return _DETECT_Z_DEFAULT * (_DETECT_Z_TIGHT / _DETECT_Z_DEFAULT) ** t


def detect_luma_score(
    img: ImageBuffer,
    dust_threshold: float,
    dust_size: int,
    stats: Optional[Tuple[np.ndarray, ...]] = None,
    hair_threshold: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Statistical dust detection on the linear source → ``(score, hair_mask)``: a fill score for
    compact specks, an inpaint mask for hairs. A speck must clear ``dust_threshold``'s bar, a hair
    ``hair_threshold``'s (default: the same), each raised by its own texture ramp."""
    if stats is None:
        stats = compute_dust_stats(img, dust_size)
    proxy, background, z, texture = stats[:4]
    hi = detect_bar(dust_threshold)
    hi_hair = hi if hair_threshold is None else detect_bar(hair_threshold)
    scale = film_scale(proxy.shape)
    lo = max(_DETECT_Z_GROW, min(hi, hi_hair) * _DETECT_Z_GROW_FRAC)
    reach = int(round(_DETECT_GROW_REACH * max(1, dust_size) * scale))
    disk = negpy_core.ellipse_kernel((2 * reach + 1,) * 2)
    planes = tuple(np.ascontiguousarray(a, dtype=np.float32) for a in (proxy, background, z, texture))
    return negpy_core.detect_luma_score(
        planes,
        (hi, hi_hair, lo),
        disk,
        (_DETECT_PROXY_MIN, _DETECT_TEXTURE_KNEE, _DETECT_TEXTURE_GAIN),
        (_HAIR_MIN_AREA, _HAIR_MIN_ELONG),
        (_DETECT_HAIR_TEXTURE_KNEE, _DETECT_HAIR_TEXTURE_GAIN, _DETECT_HAIR_FILL_GAIN),
        _texture_win(proxy.shape, dust_size),
        (_DETECT_PAD_PX * scale, _IR_SCORE_FLOOR),
    )


def exclusion_cover(strokes: List[Tuple], shape: Tuple[int, int]) -> np.ndarray:
    """Excluded strokes → their binary footprint on a plane of ``shape``. A stroke is the
    band its whole path sweeps, not its sampled points: the drag is sampled sparsely and
    loose dabs would leave the film between them repaired. Smoothed past two points and
    round-capped, so the band matches the one the overlay draws."""
    h, w = shape[:2]
    scale = max(w, h) / HEAL_SIZE_REF
    bands = []
    for points, size in strokes:
        radius = max(1, int(round(float(size) * scale * 0.5)))
        chain = [(float(p[0]) * w, float(p[1]) * h) for p in points]
        if len(chain) >= 3:
            chain = smooth_polyline(chain, closed=False)
        pts = np.ascontiguousarray(np.round(np.array(chain, dtype=np.float32)).astype(np.int32))
        bands.append((pts, radius, max(1, 2 * radius)))
    return negpy_core.stroke_cover((h, w), bands)


def drop_exclusions(
    score: Optional[np.ndarray],
    hair_mask: Optional[np.ndarray],
    strokes: List[Tuple],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Release the optical detections the excluded strokes cover, so the film there arrives
    at the render untouched. The band is the cut: a defect crossing its rim keeps the repair
    on the side the brush missed, so half a mark can be released without the rest. A score
    with nothing left below clean, or an emptied hair mask, comes back as None: the repair
    is then skipped rather than run over an identity."""
    if not strokes or (score is None and hair_mask is None):
        return score, hair_mask
    ref = score if score is not None else hair_mask
    cover = exclusion_cover(strokes, ref.shape[:2])  # type: ignore[union-attr]
    if score is not None:
        # Feather the rim, as the manual heal does its own: the score is a ramp, and cutting
        # one at the brush edge prints the edge as a step.
        score = np.array(score, dtype=np.float32)
        if not negpy_core.release_score(score, cover, _MANUAL_RIM_PX):
            score = None
    if hair_mask is not None:
        # Binary, so it takes the footprint unfeathered.
        hair_mask = np.where(cover > 0, 0, hair_mask).astype(hair_mask.dtype)
        if not hair_mask.any():
            hair_mask = None
    return score, hair_mask


def exclusion_token(retouch) -> str:
    """Config identity of the excluded strokes. Folded into the luma and hair tokens: a
    released detection changes the baked source as surely as a new one does."""
    if not retouch.dust_exclusion_strokes:
        return ""
    return "|ex" + hashlib.sha1(repr(retouch.dust_exclusion_strokes).encode()).hexdigest()[:12]


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
    score = np.ones((h, w), dtype=np.float32)
    # Brush size is a DIAMETER at HEAL_SIZE_REF scale, so the painted footprint matches the
    # cursor at any render resolution (overlay._brush_screen_radius draws size/(2*REF)).
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

        local = np.ascontiguousarray(np.round(pts - (x0, y0)).astype(np.int32))
        brush = (max(1, int(round(radius))), max(1, int(round(2.0 * radius))))
        # Inside the brush a pixel repairs by how far it stands out from the film around it:
        # MAD about zero over the whole crop, so the defect stays a minority in its own noise
        # estimate, then hysteresis from the core, because grain is isolated and never joins.
        constants = (
            win,
            _MANUAL_Z_MIN,
            _MANUAL_Z_HI,
            _MANUAL_Z_GROW,
            _MANUAL_Z_GROW_FRAC,
            _DETECT_PAD_PX * film_scale((h, w)),
            _IR_SCORE_FLOOR,
            _MANUAL_RIM_PX,
        )
        region = negpy_core.heal_region(_density(img[y0:y1, x0:x1]), local, brush, constants)
        if region is None:
            continue
        np.minimum(score[y0:y1, x0:x1], region, out=score[y0:y1, x0:x1])
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


# Rows the ridge response at one row can see: the broad Gaussian's kernel plus the noise
# window's half width. A band computed with this margin matches the whole-frame response.
_SCRATCH_RIDGE_REACH = 128


def _scratch_ridge_rows(img: ImageBuffer, r0: int, r1: int) -> Tuple[np.ndarray, int]:
    """``_scratch_ridge`` for rows ``[r0, r1)`` only, computed on a band with enough margin
    that those rows equal the whole-frame response. Returns ``(ridge, band_start)``: the
    array covers ``[band_start, band_end)``, a superset of the request."""
    h = img.shape[0]
    a, b = max(0, r0 - _SCRATCH_RIDGE_REACH), min(h, r1 + _SCRATCH_RIDGE_REACH)
    return _scratch_ridge(img[a:b]), a


def _scratch_ridge(img: ImageBuffer) -> np.ndarray:
    """Cross-section ridge response of ``img``, in units of the *local* noise.

    Band-passed across the scratch only (film grain is finer, image structure broader), then
    divided by a local scale of that same response. Local, because a global scale lets one
    busy corner of the frame set the bar and bury a faint scratch running through smooth sky.
    """
    # The noise scale is mean|x| over 0.8, its ratio to σ for gaussian noise: the bars above
    # are quoted in σ.
    return negpy_core.scratch_ridge(_density(img), (_SCRATCH_FINE_PX, _SCRATCH_BROAD_PX, _SCRATCH_NOISE_WIN))


def _shear_rows(plane: np.ndarray, slope: float, about_x: float, width: int) -> np.ndarray:
    """``plane`` sheared so a line of gradient ``slope`` through ``about_x`` becomes one row.

    warpAffine applies M forward (source → destination) unless asked otherwise, so the matrix
    carries -slope: a source point at ``y = row + slope·(x - about_x)`` has to land back on
    ``row``.
    """
    m = np.float32([[1.0, 0.0, 0.0], [-slope, 1.0, slope * about_x]])
    return negpy_core.warp_affine(np.ascontiguousarray(plane, dtype=np.float32), m.astype(np.float64), (width, plane.shape[0]))


def trace_scratch(img: ImageBuffer, nx: float, ny: float, threshold: float = 0.5) -> Optional[Tuple[float, float, float, float, float]]:
    """One click near a transport scratch → ``(nx0, ny0, nx1, ny1, width)``, or None.

    The slope is fitted, not assumed: film is rarely square to the sensor, and a fraction of a
    degree is enough to smear the ridge across many rows and lose the integration this depends
    on. ``width`` is for the on-screen guide; the repair re-grows the band itself.
    """
    h, w = img.shape[:2]
    cx, cy = float(nx) * w, float(ny) * h
    bar = scratch_detect_bar(threshold)
    y0 = max(0, int(cy) - _SCRATCH_SEARCH_ROWS)
    y1 = min(h, int(cy) + _SCRATCH_SEARCH_ROWS)
    if y1 - y0 < 3:
        return None
    scale = film_scale((h, w))
    max_half = max(1, int(round(0.5 * _SCRATCH_WIDTH_MAX * scale)))
    # The band grown below reads the sheared frame ±max_half around the line, and a shear
    # of the steepest slope moves rows by slope·w across the width: only that many rows of
    # ridge are needed, and the frame is a full-resolution buffer on hover.
    reach = max_half + int(math.ceil(_SCRATCH_SLOPE_MAX * w)) + 2
    z, z0 = _scratch_ridge_rows(img, y0 - reach, y1 + reach)
    pull = np.exp(-0.5 * ((np.arange(y0, y1) - cy) / _SCRATCH_CLICK_PULL) ** 2)

    slopes = np.arange(-_SCRATCH_SLOPE_MAX, _SCRATCH_SLOPE_MAX + 1e-9, _SCRATCH_SLOPE_STEP)
    # Each slope shears the ridge with its reach, then cuts the search rows: a line leaves them
    # across a wide frame.
    found = negpy_core.scratch_slope_search(np.ascontiguousarray(z, dtype=np.float32), cx, (y0 - z0, y1 - z0), pull, slopes)
    if found is None or found[0] < _SCRATCH_MIN_EVIDENCE:
        return None
    _, index, k, along = found
    slope = float(slopes[index])

    # A transport scratch fades in and out, so measure its extent instead of assuming it
    # spans the frame.
    on = (along * np.sign(along.mean()) > bar).astype(np.float32)
    run = negpy_core.box_filter(on.reshape(1, -1), (_SCRATCH_RUN_WIN, 1)).ravel() >= _SCRATCH_RUN_FRAC
    if not run.any():
        return None
    cols = np.flatnonzero(run)
    x0, x1 = float(cols[0]), float(cols[-1])
    row = y0 + k
    # For the guide only: the repair grows its own band per column.
    grown = _grow_band(_shear_rows(z, slope, cx, w), row - z0, cols, max_half, float(np.sign(along.mean()) or 1.0), bar)
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
    scale = film_scale((h, w))
    mask = np.zeros((h, w), dtype=np.uint8)
    touched = False

    for nx0, ny0, nx1, ny1, width in lines:
        x0, x1 = float(nx0) * w, float(nx1) * w
        y0, y1 = float(ny0) * h, float(ny1) * h
        if abs(x1 - x0) < 1.0:
            continue
        slope = (y1 - y0) / (x1 - x0)
        # ``width`` is only what the guide drew. The band is re-grown from the scratch here,
        # so it follows a scratch that widens and ignores the traced resolution.
        max_half = max(1, int(round(0.5 * _SCRATCH_WIDTH_MAX * scale)))
        row = int(round(y0))
        if not 0 <= row < h:
            continue
        # Ridge on the rows the sheared band reads, not the frame (see trace_scratch).
        reach = max_half + int(math.ceil(abs(slope) * max(x0, w - x0))) + 2
        z, z0 = _scratch_ridge_rows(img, row - reach, row + reach + 1)
        sheared = _shear_rows(z, slope, x0, w)
        row -= z0
        along = sheared[row]
        sign = np.sign(along.mean()) or 1.0
        on = (along * sign > bar).astype(np.float32)
        run = negpy_core.box_filter(on.reshape(1, -1), (_SCRATCH_RUN_WIN, 1)).ravel() >= _SCRATCH_RUN_FRAC
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
    return negpy_core.ir_live(np.ascontiguousarray(plane, dtype=np.float32), _IR_DEAD_FLOOR, _IR_DEAD_MIN_AREA)


def _ir_glare(plane: np.ndarray, live: np.ndarray) -> Optional[np.ndarray]:
    """Pixels more than ``_IR_GLARE_SIGMA`` over the median of the live film, or None."""
    return negpy_core.ir_glare(np.ascontiguousarray(plane, dtype=np.float32), np.ascontiguousarray(live), _IR_FIT_MAX_PX, _IR_GLARE_SIGMA)


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


def _erode_resize_bounded(plane: np.ndarray, dims: Tuple[int, int], kernel: np.ndarray) -> np.ndarray:
    """Apply the min-preserving downsample in source-row blocks."""
    if plane.ndim == 3 and plane.shape[2] == 1:
        plane = plane[:, :, 0]
    h, w = plane.shape[:2]
    dw, dh = dims
    scale_y = h / dh
    channels = plane.shape[2] if plane.ndim == 3 else 1
    bytes_per_row = max(1, w * channels * plane.dtype.itemsize)
    source_rows = max(1, (_IR_DOWNSAMPLE_WORK_BYTES // 4) // bytes_per_row)
    output_rows = max(1, int(source_rows * dh / h))
    radius = kernel.shape[0] // 2
    output = np.empty((dh, dw, channels), dtype=np.float32) if plane.ndim == 3 else np.empty((dh, dw), dtype=np.float32)

    for top in range(0, dh, output_rows):
        bottom = min(dh, top + output_rows)
        source_top = math.floor(top * scale_y)
        source_bottom = min(h, math.ceil((bottom - 1) * scale_y + scale_y))
        read_top = max(0, source_top - radius)
        read_bottom = min(h, source_bottom + radius)
        source = np.ascontiguousarray(plane[read_top:read_bottom], dtype=np.float32)
        eroded = negpy_core.morphology_ex(source, negpy_core.MORPH_ERODE, kernel)
        core = np.ascontiguousarray(eroded[source_top - read_top : source_bottom - read_top])
        horizontal = negpy_core.resize_area(core, (dw, len(core)))
        # The vertical pass keeps the full-image sampling grid across fractional block boundaries.
        negpy_core.area_rows_into(horizontal, output[top:bottom], top, source_top, scale_y, h)
    return output


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
    plane = np.asarray(plane, dtype=np.float32)
    h, w = plane.shape[:2]
    long_edge = max(h, w)
    if long_edge <= target_long_edge and dims is None:
        return np.ascontiguousarray(plane)
    if dims is None:
        s = target_long_edge / long_edge
        dims = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    if dims == (w, h):
        return np.ascontiguousarray(plane)
    # An HxWx1 plane comes back as HxW.
    if plane.ndim == 3 and plane.shape[2] == 1:
        plane = plane[:, :, 0]
    # Erode by the resample footprint: a 1.25x downsample must not fatten by a 4.5x kernel.
    k = max(1, int(round(long_edge / max(dims))) | 1)
    if k > 1:
        kernel = negpy_core.ellipse_kernel((k, k))
        if plane.nbytes > _IR_DOWNSAMPLE_WORK_BYTES:
            return _erode_resize_bounded(plane, dims, kernel)
        plane = negpy_core.morphology_ex(np.ascontiguousarray(plane), negpy_core.MORPH_ERODE, kernel)
    return negpy_core.resize_area(np.ascontiguousarray(plane), dims)


def normalize_ir(plane: np.ndarray, glare: Optional[np.ndarray] = None) -> np.ndarray:
    """Locally-normalized IR: ``ir / blur(dilate(ir))`` — ~1.0 on clean film, dips on
    defects, illumination-independent. Separates dust from content that raw-IR
    thresholding conflated (dilate→max estimates the clean base, blur smooths it).
    ``glare`` pixels stay out of the base."""
    plane = np.ascontiguousarray(plane, dtype=np.float32)
    win = _ir_win(_IR_BASE_WIN, _ir_detect_scale(plane))
    return negpy_core.normalize_ir(plane, None if glare is None else np.ascontiguousarray(glare), win)


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
    constants = (cutoff, span, 1.0 - _IR_SCORE_FLOOR, _IR_SCORE_FLOOR)
    return negpy_core.ir_defect_score(np.ascontiguousarray(ratio, dtype=np.float32), constants)


def _fill_supports(buffer_long_edge: int, factor: float) -> Tuple[int, ...]:
    """Fill ladder in buffer px: ``_IR_FILL_SCALES`` at the detection plane, plus a coarse rung
    at the reference footprint when detection runs finer than it. Both ends carry: a small fine
    end keeps the average off the far side of a tonal edge, and only the coarse rung reaches
    clean film across a wide defect."""
    fine = [int(round(k * factor)) | 1 for k in _IR_FILL_SCALES]
    film = max(factor, buffer_long_edge / _IR_DETECT_REF)
    return tuple(dict.fromkeys([int(round(_IR_FILL_SCALES[0] * film)) | 1] + fine))


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
        score = negpy_core.resize_linear(np.ascontiguousarray(score_det, dtype=np.float32), (w, h))
    supports = list(_fill_supports(long_edge or max(h, w), factor))
    constants = (_IR_SCORE_FLOOR, _IR_FILL_TAU, _IR_WRITE_HI, _IR_WRITE_HI - _IR_WRITE_LO)
    out = negpy_core.apply_score_repair(src, score, supports, bool(floor), max(1.0, factor), constants)
    return ensure_image(out)


def film_scale(shape: Tuple[int, int]) -> float:
    """Pixels per unit of film footprint for a buffer this size — the factor a score measured
    at the buffer's own resolution needs so the fill's supports stay film-scale rather than
    grain-scale (see ``_fill_supports``)."""
    return max(1.0, max(shape) / _IR_DETECT_REF)


def repair_components(
    img: ImageBuffer,
    score_det: np.ndarray,
    *,
    floor: bool = True,
    factor: Optional[float] = None,
    base_out: Optional[np.ndarray] = None,
    dirty: Optional[np.ndarray] = None,
) -> ImageBuffer:
    """``apply_score_repair`` per defect, each in its own padded crop.

    The fill is four convolutions over whatever buffer it is handed. That is right for an
    IR score, where defects are spread over the frame, and wasteful for the handful of
    painted strokes or detected specks this serves — at export resolution it would filter
    a hundred megapixels to repair a dozen. The support ladder and the mask's upsample
    factor still come from the whole frame, so a crop repairs exactly as it would there.

    ``base_out``/``dirty`` support an incremental re-bake, for a session that adds one
    stroke at a time: with both given, a component that contains no ``dirty`` pixel is
    copied from ``base_out`` unchanged rather than re-filled, since neither its score nor
    its source pixels moved since that buffer was made. ``dirty=None`` (every caller but
    the painted-strokes bake) fills every component, matching the non-incremental result
    exactly — the manual-bake caller reads ``base_out`` itself when nothing is dirty at
    all, so an all-False mask never has to be passed in here.
    """
    h, w = img.shape[:2]
    if score_det.shape[:2] == (h, w):
        score = np.ascontiguousarray(score_det, dtype=np.float32)
        factor = factor or 1.0
    else:
        factor = max(h / score_det.shape[0], w / score_det.shape[1])
        score = negpy_core.resize_linear(np.ascontiguousarray(score_det, dtype=np.float32), (w, h))
        if dirty is not None and dirty.shape[:2] != (h, w):
            dirty = negpy_core.resize_nearest(dirty.astype(np.uint8), (w, h)) > 0
    src = np.ascontiguousarray(img, dtype=np.float32)
    out = np.ascontiguousarray(base_out, dtype=np.float32).copy() if base_out is not None else src.copy()
    supports = _fill_supports(max(h, w), factor)
    constants = (_IR_SCORE_FLOOR, _IR_FILL_TAU, _IR_WRITE_HI, _IR_WRITE_HI - _IR_WRITE_LO)
    # Past the cap, cropping costs more than the whole-frame convolutions it avoids. The pad is
    # the coarsest support's reach, so every crop holds the clean film the fill averages.
    dirty_mask = None if dirty is None else np.ascontiguousarray(dirty, dtype=bool)
    args = (list(supports), bool(floor), max(1.0, factor), constants, max(supports), _REPAIR_MAX_COMPONENTS)
    if not negpy_core.repair_components(src, score, out, *args, dirty_mask):
        return img
    return ensure_image(out)


def route_wide_defects(score: np.ndarray, *, budget: Optional[float] = _IR_ROUTE_BUDGET) -> Optional[np.ndarray]:
    """Detection-scale mask of at-floor components past the fill's reach, for
    apply_hair_inpaint. Over ``budget`` (misregistered/garbage IR) → None + warning.

    ``budget=None`` lifts the cap for hand-placed repairs. The cap guards an *automatic*
    detector, where a misregistered IR plane can call half the frame a defect; a full-width
    line clears it on its own, so it would refuse exactly the case that needs the inpaint.
    """
    scale = _ir_detect_scale(score)
    k = negpy_core.ellipse_kernel((2 * int(round(_IR_ROUTE_DILATE * scale)) + 1,) * 2)
    bar = np.float32(_IR_SCORE_FLOOR + 1e-6)
    routed = negpy_core.route_wide_defects(np.ascontiguousarray(score, dtype=np.float32), bar, int(round(_IR_ROUTE_RADIUS * scale)), k)
    if routed is None:
        return None
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
    # Fit on clean film only. Dust dips *both* planes, so a fit that sees it explains the
    # defect away as ghost and the division stops lifting it. Trim by ratio percentile: a
    # fixed cutoff fails because a strong ghost drags clean film below it, and residual
    # fails because the dust fits itself perfectly.
    ratio = np.ascontiguousarray(ratio, dtype=np.float32)
    keep = _fit_sample(ratio >= np.float32(negpy_core.percentiles(ratio.ravel(), [_IR_XTALK_TRIM])[0]))
    y = np.log(np.clip(ratio.ravel()[keep], 1e-4, 1.0))
    x = vis_log.reshape(-1, vis_log.shape[-1])[keep]
    if y.size < 500:
        return ratio, 0.0
    # Intercept column, dropped from the result. Both logs sit below their own dilate+blur
    # envelope, so origin-forced least squares reads that shared negative offset as slope
    # and fits a large b on two *independent* noisy planes.
    x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    b = np.clip(np.linalg.lstsq(x, y, rcond=None)[0][:3], 0.0, _IR_XTALK_MAX)
    ghost = float(np.abs(b).sum())
    if ghost < _IR_XTALK_MIN:
        return ratio, ghost
    return negpy_core.ir_ghost_divide(ratio, np.ascontiguousarray(vis_log, dtype=np.float32), b.tolist()), ghost


def _fit_refraction_gammas(ratio: np.ndarray, vis_log: np.ndarray, img_det: np.ndarray) -> Tuple[float, ...]:
    """Per-channel refraction γ: the slope of log(vis_norm) on log(ratio) over the
    shallow-dust band, as the median of the per-pixel slopes over locally flat film.

    Median and flat restriction are both load-bearing. The band selects on the IR ratio
    alone, so besides dust it collects ``_ir_decontaminate``'s residue at hard image edges,
    and least squares through the origin is x²-weighted — that deep non-dust minority
    dominated it, reading γ 1.9/2.2/2.2 for dust measuring ~1.0/1.1/1.2 and over-correcting
    every speck into a dark cyan blob. Median alone reads 1.3/1.8/1.8, flat-only least
    squares 1.4/2.1/2.0, and γ 1.5 already tints."""
    f32 = (np.ascontiguousarray(a, dtype=np.float32) for a in (ratio, vis_log, img_det))
    gamma = (_IR_GAMMA_LO, _IR_GAMMA_HI)
    return negpy_core.fit_refraction_gammas(*f32, (0.70, 0.92), _IR_FIT_FLAT_PCT, _IR_FIT_MIN_PX, _IR_FIT_MAX_PX, _IR_GAMMA_FALLBACK, gamma)


def _ir_clean_base(img_det: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    """Local clean-film level per channel over ``_IR_CAP_WIN``: mean of the pixels the
    IR ratio calls clean, minus ``_IR_CAP_SIGMA`` of their σ (see the constants block)."""
    win = _ir_win(_IR_CAP_WIN, _ir_detect_scale(ratio))
    constants = (win, _IR_GAIN_IDENTITY, _IR_CAP_SIGMA, _IR_CAP_MIN_SUPPORT)
    return negpy_core.ir_clean_base(
        np.ascontiguousarray(img_det, dtype=np.float32), np.ascontiguousarray(ratio, dtype=np.float32), constants
    )


def _ir_normalize_ratio(ratio: np.ndarray, live: np.ndarray) -> np.ndarray:
    """Both ratio landmarks onto what the absolute constants expect: clean-film floor on
    ``_IR_GAIN_IDENTITY``, dip scale on ``_IR_REF_SIGMA`` (see the constants block). ``live``
    only: the dead-margin 1.0s inflate σ ~20% on a strip scan. MAD, not std, so a dusty
    minority can't move either landmark."""
    ratio = np.array(ratio, dtype=np.float32)
    constants = (_IR_GAIN_IDENTITY, _IR_NOISE_SIGMA, _IR_PIVOT_LO, _IR_REF_SIGMA, _IR_SCALE_MAX)
    measured = negpy_core.ir_normalize_ratio(ratio, np.ascontiguousarray(live), constants)
    if measured is not None and measured[0] <= _IR_PIVOT_LO:
        logger.warning("IR dust: clean-film pivot floored at %.2f (IR σ %.3f) — very noisy IR plane", _IR_PIVOT_LO, measured[1])
    return ratio


def ir_ratio_and_gain(ir_det: np.ndarray, img_det: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Tuple[float, ...]]:
    """Detection-scale ``(ratio, gain HxWx3, degenerate, gammas)`` for IR-division
    attenuation: semi-transparent dust recovered by ``RGB / ratio^γ``, γ per channel from
    ``_fit_refraction_gammas``. ``degenerate`` = IR carrying image content
    (B&W/Kodachrome) → caller skips the whole IR bake."""
    plane = ir_det[:, :, 0] if ir_det.ndim == 3 else ir_det
    # No film under the head is not a defect. Left as a dip, the holder margin scores as one
    # giant routed component and swamps the routing budget.
    live = _ir_live(plane)
    ratio = normalize_ir(plane, _ir_glare(plane, live))
    ratio[~live] = 1.0
    img_det = np.ascontiguousarray(img_det, dtype=np.float32)
    if img_det.shape[:2] != ratio.shape[:2]:
        img_det = negpy_core.resize_area(np.ascontiguousarray(img_det, dtype=np.float32), (ratio.shape[1], ratio.shape[0]))

    vis_log = negpy_core.ir_vis_log(img_det, _ir_win(_IR_BASE_WIN, _ir_detect_scale(img_det)))
    ratio, ghost = _ir_decontaminate(ratio, vis_log)
    # On the fitted exponent, not on how far the ratio dips. A few percent of IR noise,
    # deepened by the min-preserving downsample, reads as silver on clean C41 rolls.
    degenerate = ghost > _IR_DEGENERATE_GHOST
    # After the unmixing, never before: it clips log(ratio) at 1.0, and a rescaled clean
    # population piles into that clip and flattens the fitted exponent.
    ratio = _ir_normalize_ratio(ratio, live)

    gammas = _fit_refraction_gammas(ratio, vis_log, img_det)
    # Never lift a pixel past its own local clean base (see _IR_CAP_WIN). Floored at 1, so
    # the cap only holds the bake back and never darkens a pixel.
    clean = _ir_clean_base(img_det, ratio)
    gain = negpy_core.ir_gain(
        ratio, img_det, np.ascontiguousarray(clean, dtype=np.float32), list(gammas), (_IR_GAIN_IDENTITY, _IR_GAIN_CLAMP)
    )
    return ratio, gain, degenerate, gammas


def apply_ir_attenuation(img: ImageBuffer, gain_det: np.ndarray) -> ImageBuffer:
    """Visible buffer × upsampled per-channel IR gain map (new array — buffers are read-only)."""
    h, w = img.shape[:2]
    gain = gain_det if gain_det.shape[:2] == (h, w) else negpy_core.resize_linear(np.ascontiguousarray(gain_det, dtype=np.float32), (w, h))
    return ensure_image(negpy_core.multiply(np.ascontiguousarray(img, dtype=np.float32), np.ascontiguousarray(gain, dtype=np.float32)))


def luma_bake_token(retouch) -> str:
    """Config identity of the auto speck repair, folded into source_hash so the
    Dust Removal toggle invalidates the uploaded source (mirrors ir_bake_token).
    hair_bake_token cannot cover this: it joins the hash only when a hair was
    actually detected, and the speck fill runs regardless."""
    if not retouch.dust_remove:
        return ""
    return (
        f"|dust{round(float(retouch.dust_threshold), 3)}_{round(float(retouch.dust_hair_threshold), 3)}_{int(retouch.dust_size)}"
        + exclusion_token(retouch)
    )


def ir_bake_token(retouch, has_ir: bool) -> str:
    """Config-identity token for the IR bake (mirrors ``flatfield_token``); folded into
    source_hash so a toggle or threshold drag invalidates the engine cache."""
    if not (retouch.ir_dust_remove and has_ir):
        return ""
    return f"|ir{int(retouch.ir_attenuation)}r{round(float(retouch.ir_threshold), 3)}"


def apply_hair_inpaint(
    img: ImageBuffer,
    hair_masks: List[np.ndarray],
    radius: int = _HAIR_INPAINT_RADIUS,
    dilate_px: Optional[int] = None,
) -> ImageBuffer:
    """Structure-following fill of long/twisted defects (Navier–Stokes inpaint, ``negpy_core::inpaint``)
    baked into the linear source. Each detection-scale mask is upsampled to the buffer,
    unioned and dilated to cover the PSF skirt; only masked pixels are overwritten (the
    rest stay byte-identical — the 8-bit encode of the inpaint touches only the
    fabricated hairline). Returns a new array (buffers are read-only)."""
    h, w = img.shape[:2]
    masks = [hm for hm in hair_masks if hm is not None]
    if not masks:
        return img
    factor = max(1.0, h / masks[0].shape[0], w / masks[0].shape[1])
    if dilate_px is None:
        # A detection-scale mask knows its boundary only to the upsample factor, so the
        # dilate tracks it. At 1:1 that is _HAIR_DILATE_PX, the PSF skirt alone.
        dilate_px = max(_HAIR_DILATE_PX, round(factor))
    m = np.zeros((h, w), dtype=np.uint8)
    for hm in masks:
        r = hm if hm.shape[:2] == (h, w) else negpy_core.resize_linear(np.ascontiguousarray(hm, dtype=np.float32), (w, h))
        m |= (np.asarray(r) > 0.5).astype(np.uint8)
    if not m.any():
        return img
    if dilate_px > 0:
        k = negpy_core.ellipse_kernel((2 * dilate_px + 1, 2 * dilate_px + 1))
        m = negpy_core.morphology_ex(m, negpy_core.MORPH_DILATE, k)
    src = np.ascontiguousarray(img, dtype=np.float32)
    # Each component fills in its padded box, tiled with a halo; a neighbour reaching into the
    # box stays unknown, or it becomes clone source and its dust is filled back in.
    out = negpy_core.hair_fill(
        src,
        m,
        (float(radius), dilate_px, _HAIR_INPAINT_PAD, _HAIR_TILE_PX, _HAIR_TILE_HALO, _HAIR_RANGE_SAMPLE_PX, _HAIR_INPAINT_GAMMA),
    )
    # Navier-Stokes propagates a smooth field, so the fill lands grainless. See
    # apply_score_repair, which fills the same kind of hole by a different route.
    negpy_core.add_clean_grain(out, np.ascontiguousarray(src, dtype=np.float32), m == 0, max(1.0, factor))
    return out


def hair_bake_token(retouch) -> str:
    """Detection-param identity for the hair inpaint (folded into source_hash when a
    hair is actually detected). Distinct params → distinct inpainted source."""
    r = retouch
    return (
        f"|hair{int(r.dust_remove)}_{round(float(r.dust_threshold), 3)}_{round(float(r.dust_hair_threshold), 3)}_{int(r.dust_size)}_{int(r.ir_dust_remove)}_{round(float(r.ir_threshold), 3)}"
        + exclusion_token(r)
    )


def repair_coverage(
    ir_mask: Optional[np.ndarray],
    dust_mask: Optional[np.ndarray],
    hair_masks: Optional[List[np.ndarray]],
) -> Tuple[float, float, float]:
    """(ir, dust, painted) share of the scan each repair route rewrote.

    Each mask is measured against its own grid: detection, IR and the manual routes run
    at different resolutions, and a fraction is scale-free. Same-grid hair masks overlap
    by design (a routed IR defect is also a hair), so they union rather than sum; across
    grids the largest stands in, which cannot understate the coverage."""

    def _frac(mask: Optional[np.ndarray]) -> float:
        arr = np.asarray(mask) if mask is not None else None
        return float(np.count_nonzero(arr) / arr.size) if arr is not None and arr.size else 0.0

    by_shape: dict = {}
    for m in hair_masks or []:
        arr = np.asarray(m).astype(bool)
        prev = by_shape.get(arr.shape)
        by_shape[arr.shape] = arr if prev is None else (prev | arr)
    painted = max((_frac(u) for u in by_shape.values()), default=0.0)
    return _frac(ir_mask), _frac(dust_mask), painted
