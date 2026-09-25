"""
Pure heuristics for auto-detecting the film process mode (C41 / B&W / E-6)
from a raw linear scan, before any inversion or normalization.
"""

from typing import Optional, Sequence

import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.exposure.normalization import get_analysis_crop
from negpy.features.process.models import ProcessConfig, ProcessMode


def effective_linear_raw(process: ProcessConfig) -> bool:
    """Whether the decode skips the camera's as-shot white balance.

    True when the user asked for Linear RAW, and **always** on an as-captured Slide.
    That render applies the camera's own matrix, which folds the
    as-shot multipliers back in itself (`camera_to_working_matrix`), and the
    Calibration panel already documents Linear RAW as inert there — but the decode
    was reading the stored flag regardless, so a hidden, stale toggle silently decided
    whether white balance was applied. `positive_source` exempts this forced case on
    any mode: a finished positive has no camera matrix to fold multipliers back in, so
    it decodes on its own embedded profile, same as when the source is a plain Color
    or B&W scan already.

    Every site that decides `use_camera_wb` must ask this one question. The decode and the
    matrix have to agree: apply white balance at both, or at neither. Splitting them tints
    the render by the raw green-to-red ratio, which is roughly 2:1.

    It matters most to a bracket. `use_camera_wb` applies each *file's* own multipliers, and
    a camera left on auto white balance records different ones per frame — on a real
    8-frame slide bracket the darkest frame's B/G came out 1.41 against ~1.8-2.1 for the
    rest. Frames then sit on different scales, and the exposure ratios solved between them
    absorb the difference: that bracket's shortest link solved to 0.75 EV instead of 1.00,
    which prints as contour rings around a blown highlight.
    """
    from negpy.features.process.path import RenderPath, render_path

    return process.linear_raw or render_path(process) is RenderPath.TRANSFER


def linear_raw_token(process: ProcessConfig) -> str:
    """Decode-mode identity, folded into the render source hash so the auto-meter
    re-runs when Linear RAW toggles (the decode changes the source pixels).

    Keyed on the *effective* value: the transfer path decodes without white balance
    whatever the stored flag says, so keying on the flag alone would serve a buffer decoded
    the other way.
    """
    return f"|lr:{int(effective_linear_raw(process))}"


def demosaic_token(mode: str) -> str:
    """CFA-interpolation identity for a cache key. Preview and export choose separately, so
    the caller passes the mode for its own path."""
    return f"|dm:{mode}"


def should_fold_camera_wb(process: ProcessConfig) -> bool:
    """Whether `camera_to_working_matrix` should fold the as-shot multipliers back in.

    True when the decode skipped white balance (`effective_linear_raw`) *and* the capture
    was not made under narrowband light. A camera's as-shot WB is a continuous-spectrum
    estimate, and narrowband light has no color temperature that estimate can describe — the
    same reason white balance cannot fix the hue rotation narrowband light imposes (see
    docs/USER_GUIDE.md's Hue Trim). Folding it back in for a narrowband capture is not a
    milder version of the correct fix, it is the wrong correction: there is no scene white
    balance for the fold to reconstruct, whatever the camera happened to read.

    The narrowband condition is `narrowband_profile_active`, not the stored flag: the flag
    survives a mode switch and is inert on a slide, where nothing narrowband applies.

    Also false whenever `highlight_reconstruction_bakes_wb` is true: the decode already
    carries the real white balance in that case, and folding it again would double-apply
    it — the same "decode and matrix must agree" rule this function exists for in the
    first place.

    Every site that folds `camera_wb` into the capture matrix must ask this one question,
    the same way every decode asks `effective_linear_raw`.
    """
    if highlight_reconstruction_bakes_wb(process):
        return False
    return effective_linear_raw(process) and not narrowband_profile_active(process)


VALID_HIGHLIGHT_LEVELS = frozenset({0, 2, 3, 4, 5, 6, 7, 8, 9})


def effective_highlight_reconstruction(process: ProcessConfig) -> int:
    """The libraw `highlight_mode` the decode should actually request.

    Reconstruction only makes sense against a slide's own blown highlights, so it is
    forced to 0 (Clip, today's decode) off `ProcessMode.E6` — a stored value surviving a
    mode switch, or a hand-edited sidecar, must not silently start reconstructing a
    negative's genuinely-clipped base. `1` (Ignore) is excluded even on E-6: it only
    changes libraw's WB-multiplier scaling, a separate decode-correctness fix, and is
    never a level this control should expose. An out-of-range stored value resolves to 0
    rather than raising or passing a bogus number to rawpy.

    Also 0 under `narrowband_scan`: a triplet's single raw channel per exposure carries no
    "highlight color" for libraw to reconstruct — the same reasoning `should_fold_camera_wb`
    applies to the WB fold.

    On a merged bracket the stored value is already 0 (`WorkspaceConfig.__post_init__`), so
    nothing here has to ask: a reconstructed pixel no longer reads near the sensor ceiling,
    which is what the merge's own clip detection trusts to find a genuine highlight.

    Every site that sets `highlight_mode` on a decode must ask this one question, the same
    way every decode asks `effective_linear_raw`.
    """
    if process.process_mode != ProcessMode.E6 or process.narrowband_scan:
        return 0
    value = int(process.highlight_reconstruction)
    return value if value in VALID_HIGHLIGHT_LEVELS else 0


def highlight_reconstruction_token(process: ProcessConfig) -> str:
    """Reconstruction-level identity for a cache key, keyed on the *effective* value —
    the stored one can be nonzero off the E-6 path, where it never reaches the decode.
    """
    return f"|hr:{effective_highlight_reconstruction(process)}"


def highlight_reconstruction_bakes_wb(process: ProcessConfig) -> bool:
    """Whether an active reconstruction should decode with the real white balance baked
    in, instead of the transfer path's usual neutral decode plus downstream matrix fold.

    Libraw's own reconstruction reads its decode's per-channel multipliers (`pre_mul`) to
    decide what counts as clipped. On a neutral decode those are all `1.0`, so its clip
    threshold sits at the raw ADC ceiling and essentially never fires — the clipping
    reconstruction targets only exists after NegPy's own camera-matrix white-balance fold
    runs, downstream of where libraw already decided there was nothing to do. Baking the
    real white balance in at decode time is the only way reconstruction sees what actually
    clips.

    Only overrides the *default* reason for a neutral decode: being on the transfer path
    itself (`render_path`). An explicit Linear RAW request stays neutral
    regardless — that toggle is the user asking for it directly, and reconstruction must
    not reach around it. `positive_source` has no camera matrix to fold in the first
    place, so there is nothing to bake either. False whenever
    `effective_highlight_reconstruction` resolves to 0.

    Every site that decides whether to bake real white balance into a decode must ask
    this one question, the same discipline `effective_linear_raw` and
    `should_fold_camera_wb` are held to.
    """
    if process.linear_raw or process.positive_source:
        return False
    if not effective_highlight_reconstruction(process):
        return False
    from negpy.features.process.path import RenderPath, render_path

    return render_path(process) is not RenderPath.PRINT


def highlight_reconstruction_bakes_wb_token(process: ProcessConfig) -> str:
    """Cache-key identity for `highlight_reconstruction_bakes_wb`, distinct from
    `linear_raw_token`: an explicit Linear RAW request and the transfer path's own default
    neutral decode both read as `effective_linear_raw() == True`, so `linear_raw_token`
    alone cannot tell a config where reconstruction bakes white balance in apart from one
    where it stays neutral because the user asked for it — yet the two decode differently
    once reconstruction is active. See `highlight_reconstruction_bakes_wb`.
    """
    return f"|hrwb:{int(highlight_reconstruction_bakes_wb(process))}"


def highlight_reconstruction_bright_gain(wb: Optional[Sequence[float]], highlight_mode: int) -> float:
    """The rawpy `bright` value that offsets libraw's own highlight-mode scaling.

    Libraw normalizes its white-balance gains against the *smallest* per-channel
    multiplier when `highlight_mode` is Clip, and against the *largest* one otherwise —
    reserving headroom for the reconstruction algorithm at the cost of scaling the whole
    decode down by `max(wb)/min(wb)`. `bright` multiplies libraw's own scale factors
    before its final bit-depth quantization, so passing this ratio back restores the
    Clip-equivalent exposure without re-touching pixels already reconstructed above the
    naive white level.

    1.0 (no-op) when `highlight_mode` is 0 — libraw already normalizes by the minimum in
    that case — or `wb` is neutral, missing, or degenerate.
    """
    if not highlight_mode or wb is None:
        return 1.0
    values = [float(v) for v in wb[:3]]
    if len(values) != 3 or not all(v > 0.0 and np.isfinite(v) for v in values):
        return 1.0
    return max(values) / min(values)


def narrowband_allowed(process: ProcessConfig) -> bool:
    """Whether narrowband capture applies to this film at all. Never to a transparency.

    The bundled profile characterises narrowband capture of *negative* dyes; E-6 is a
    different dye set, so on a slide it is a fixed 3x3 derived from the wrong film — an
    approximate correction for dyes that are not there, which is worse than none.
    Narrowband's real payoffs (defeating the orange mask, clean separation ahead of a
    high-gain inversion) belong to negatives, and a slide has neither.

    Single source of truth for the rule: the sensor panel greys Narrowband and the scan
    setup on it, `narrowband_profile_active` and `unmix_block_reason` refuse on it.
    """
    return process.process_mode != ProcessMode.E6


def narrowband_profile_active(process: ProcessConfig) -> bool:
    """Whether the bundled RGBScan input profile applies (`effective_input_icc` reads it).
    An explicit Input ICC is a deliberate choice about the user's own source and still
    wins — that decision is not made here."""
    return process.narrowband_scan and narrowband_allowed(process)


# Tuned against real sample scans; see tests/test_process_detect.py.
_ANALYSIS_BUFFER = 0.12  # centre-crop ratio: drops film rebate / borders
_MAX_ANALYSIS_DIM = 256  # downsample longest edge to this for speed
_BW_CORR_THRESHOLD = 0.99  # min channel correlation above this -> monochrome
_C41_ORANGE_THRESHOLD = 1.5  # red-over-blue cast above this -> orange mask (C41)
_PURPLE_G_DEFICIT = 0.05  # min absolute linear deficit: (R+B)/2 - G (purple mask)
_PURPLE_RB_BALANCE = 1.05  # min(R,B)/G must exceed this (both R and B above G)


def _downsample(img: ImageBuffer, max_dim: int) -> ImageBuffer:
    """Strided downsample so analysis stays cheap on full-res previews."""
    longest = max(img.shape[0], img.shape[1])
    if longest <= max_dim:
        return img
    step = int(np.ceil(longest / max_dim))
    return img[::step, ::step]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two flattened channels."""
    a = a.ravel() - float(a.mean())
    b = b.ravel() - float(b.mean())
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b))) + 1e-12
    return float(np.sum(a * b) / denom)


def _has_purple_mask(r_v: float, g_v: float, b_v: float) -> bool:
    """True iff a single (r,g,b) triplet shows the purple-mask pattern (R≈B>>G)."""
    deficit = (r_v + b_v) / 2 - g_v
    balance = min(r_v, b_v) / (g_v + 1e-6)
    return deficit > _PURPLE_G_DEFICIT and balance > _PURPLE_RB_BALANCE


def detect_process_mode(raw: Optional[ImageBuffer]) -> ProcessMode:
    """
    Classify a raw linear scan as C41, B&W or E-6.
    Falls back to C41 (the default) on ambiguous or invalid input.
    """
    if raw is None or raw.ndim != 3 or raw.shape[2] < 3:
        return ProcessMode.C41

    img = get_analysis_crop(raw[:, :, :3].astype(np.float32), _ANALYSIS_BUFFER)
    img = _downsample(img, _MAX_ANALYSIS_DIM)
    img = np.clip(np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    if img.size == 0:
        return ProcessMode.C41

    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # B&W: the channels stay near-perfectly correlated even with a color tint, while real
    # color (C41, E-6) has varied hues and lower correlation.
    min_corr = min(_corr(r, g), _corr(g, b), _corr(r, b))
    if min_corr > _BW_CORR_THRESHOLD:
        return ProcessMode.BW

    r_mean, b_mean = float(np.mean(r)), float(np.mean(b))
    r_p25, b_p25 = float(np.percentile(r, 25)), float(np.percentile(b, 25))
    r_p98, g_p98, b_p98 = float(np.percentile(r, 98)), float(np.percentile(g, 98)), float(np.percentile(b, 98))

    # Orange mask (standard C41): R much greater than B. Scanners sometimes correct the mask
    # only in bright areas, so check across density levels.
    orange_score = max(
        (r_mean + 1e-6) / (b_mean + 1e-6),
        (r_p25 + 1e-6) / (b_p25 + 1e-6),
        (r_p98 + 1e-6) / (b_p98 + 1e-6),
    )
    if orange_score > _C41_ORANGE_THRESHOLD:
        return ProcessMode.C41

    # Purple mask, as on Harman Phoenix: R about equal to B with G suppressed. Check at p98,
    # the clearest film areas, where the base color is most visible.
    if _has_purple_mask(r_p98, g_p98, b_p98):
        return ProcessMode.C41

    return ProcessMode.E6
