"""
Pure heuristics for auto-detecting the film process mode (C41 / B&W / E-6)
from a raw linear scan, before any inversion or normalization.
"""

from typing import Optional, Sequence

import negpy_core
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

    True when the decode skipped white balance (`effective_linear_raw`). False whenever
    `highlight_reconstruction_bakes_wb` is true: the decode already carries the real white
    balance in that case, and folding it again would double-apply it — the same "decode and
    matrix must agree" rule this function exists for in the first place.

    Every site that folds `camera_wb` into the capture matrix must ask this one question,
    the same way every decode asks `effective_linear_raw`.
    """
    if highlight_reconstruction_bakes_wb(process):
        return False
    return effective_linear_raw(process)


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

    On a merged bracket the stored value is already 0 (`WorkspaceConfig.__post_init__`), so
    nothing here has to ask: a reconstructed pixel no longer reads near the sensor ceiling,
    which is what the merge's own clip detection trusts to find a genuine highlight.

    Every site that sets `highlight_mode` on a decode must ask this one question, the same
    way every decode asks `effective_linear_raw`.
    """
    if process.process_mode != ProcessMode.E6:
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


# Tuned against real sample scans; see tests/test_process_detect.py.
_ANALYSIS_BUFFER = 0.12  # centre-crop ratio: drops film rebate / borders
_MAX_ANALYSIS_DIM = 256  # downsample longest edge to this for speed
_BW_CORR_THRESHOLD = 0.99  # min channel correlation above this -> monochrome, when too few blocks give a chroma spread
_BW_CHROMA_THRESHOLD = 0.022  # chroma spread (log10) below this -> monochrome
_CHROMA_BLOCK = 4  # block edge in analysis pixels: averages out sensor noise, keeps scene hues
_CHROMA_LO, _CHROMA_HI = 0.003, 0.97  # a block with a pixel outside this in any channel is noise or clipped
_CHROMA_MIN_SHARE = 0.1  # usable blocks needed, as a share of all blocks
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
    return detect_process_and_source(raw)[0]


def detect_source_color(raw: Optional[ImageBuffer]) -> bool:
    """True when the scan is of color film: its channels decorrelate, or it carries an orange or purple mask."""
    return detect_process_and_source(raw)[1]


def detect_process_and_source(raw: Optional[ImageBuffer]) -> tuple[ProcessMode, bool]:
    """(process mode, source is color film) from one pass of the scan stats; invalid input reads as C41."""
    if raw is None or raw.ndim != 3 or raw.shape[2] < 3:
        return ProcessMode.C41, True

    img = _downsample(get_analysis_crop(raw[:, :, :3], _ANALYSIS_BUFFER), _MAX_ANALYSIS_DIM).astype(np.float32)
    img = np.clip(np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    if img.size == 0:
        return ProcessMode.C41, True

    min_corr, (r_mean, b_mean, r_p25, b_p25, r_p98, g_p98, b_p98) = negpy_core.process_mode_stats(np.ascontiguousarray(img))

    # Orange mask (standard C41): R much greater than B. Scanners sometimes correct the mask
    # only in bright areas, so check across density levels.
    orange_score = max(
        (r_mean + 1e-6) / (b_mean + 1e-6),
        (r_p25 + 1e-6) / (b_p25 + 1e-6),
        (r_p98 + 1e-6) / (b_p98 + 1e-6),
    )
    # Purple mask, as on Harman Phoenix: R about equal to B with G suppressed. Check at p98,
    # the clearest film areas, where the base color is most visible.
    masked = orange_score > _C41_ORANGE_THRESHOLD or _has_purple_mask(r_p98, g_p98, b_p98)
    spread = negpy_core.chroma_spread(np.ascontiguousarray(img), _CHROMA_BLOCK, _CHROMA_LO, _CHROMA_HI, _CHROMA_MIN_SHARE)
    monochrome = spread < _BW_CHROMA_THRESHOLD if spread is not None else min_corr > _BW_CORR_THRESHOLD

    # B&W: silver leaves only a tint and noise around the neutral axis; dyes carry the scene's hues.
    # Channel correlation falls on a flat or clipped B&W frame, so it decides only without a spread.
    # A gray scene on masked color film still reads B&W for the mode, but its source stays color.
    if monochrome:
        return ProcessMode.BW, masked
    return (ProcessMode.C41 if masked else ProcessMode.E6), True
