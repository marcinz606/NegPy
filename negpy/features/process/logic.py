"""
Pure heuristics for auto-detecting the film process mode (C41 / B&W / E-6)
from a raw linear scan, before any inversion or normalization.
"""

from typing import Optional

import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.exposure.normalization import get_analysis_crop
from negpy.features.process.models import ProcessConfig, ProcessMode


def effective_linear_raw(process: ProcessConfig, render_intent: Optional[str] = None) -> bool:
    """Whether the decode skips the camera's as-shot white balance.

    True when the user asked for Linear RAW, and **always** on the transparency transfer
    path. That render applies the camera's own matrix, which folds the as-shot multipliers
    back in itself (`camera_to_working_matrix`), and the Calibration panel already documents
    Linear RAW as inert there — but the decode was reading the stored flag regardless, so a
    hidden, stale toggle silently decided whether white balance was applied.

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
    from negpy.features.exposure.transfer import is_transparency_transfer

    if process.linear_raw:
        return True
    return is_transparency_transfer(process.process_mode, process.e6_normalize, render_intent)


def linear_raw_token(process: ProcessConfig, render_intent: Optional[str] = None) -> str:
    """Decode-mode identity, folded into the render source hash so the auto-meter
    re-runs when Linear RAW toggles (the decode changes the source pixels).

    Keyed on the *effective* value: the transfer path decodes without white balance
    whatever the stored flag says, so keying on the flag alone would serve a buffer decoded
    the other way.
    """
    return f"|lr:{int(effective_linear_raw(process, render_intent))}"


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

    # B&W: channels stay near-perfectly correlated even with a color tint;
    # real color (C41/E-6) has varied hues and lower correlation.
    min_corr = min(_corr(r, g), _corr(g, b), _corr(r, b))
    if min_corr > _BW_CORR_THRESHOLD:
        return ProcessMode.BW

    r_mean, b_mean = float(np.mean(r)), float(np.mean(b))
    r_p25, b_p25 = float(np.percentile(r, 25)), float(np.percentile(b, 25))
    r_p98, g_p98, b_p98 = float(np.percentile(r, 98)), float(np.percentile(g, 98)), float(np.percentile(b, 98))

    # Orange mask (standard C41): R >> B.
    # Scanners sometimes correct the mask only in bright areas; check across density levels.
    orange_score = max(
        (r_mean + 1e-6) / (b_mean + 1e-6),
        (r_p25 + 1e-6) / (b_p25 + 1e-6),
        (r_p98 + 1e-6) / (b_p98 + 1e-6),
    )
    if orange_score > _C41_ORANGE_THRESHOLD:
        return ProcessMode.C41

    # Purple mask (e.g. Harman Phoenix): R≈B with G suppressed.
    # Check at p98 (clearest film areas) where the base color is most visible.
    if _has_purple_mask(r_p98, g_p98, b_p98):
        return ProcessMode.C41

    return ProcessMode.E6
