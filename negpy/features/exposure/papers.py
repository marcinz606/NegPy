"""
Darkroom paper profiles — per-paper overrides of the H&D print character.

A profile overrides a few EXPOSURE_CONSTANTS keys (the paper's characteristic
curve) plus optional color terms. It only sets the curve *shape*; Grade still
owns contrast and Density/toe/shoulder still trim on top. The default profile
reproduces EXPOSURE_CONSTANTS exactly. B&W profiles are tonal only under normal
development (the B&W path collapses to luminance, so the RA4 color terms are
inert — paper tone is a Toning job). The exception is Lith: `lith_path` is the paper's
infectious-development color and the only color term a B&W profile acts on.

Values were loosely mapped by Claude from published datasheets (Ilford, Kodak
Endura, Foma, Fuji), not a precise calibration. Mainly d_max is grounded; the
knee/midtone tweaks are light touches for character. Note these stack on the
Grade slope, so over-soft knees read flat — keep them gentle.

To replace a profile with a real calibration, digitize the datasheet's D-logH
curve and run scripts/fit_paper_profile.py — it fits this exact parametric
family and prints the PaperProfile kwargs (accept when RMS ≤ ~0.05 D).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.process.models import ProcessMode

DEFAULT_PROFILE_KEY = "neutral"

DyeMatrix = Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]

_IDENTITY_DYE: DyeMatrix = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

# Lith hue path: four (a*, b*) anchors at the density fractions in
# LITH_CONSTANTS["path_u"], reading peach, ochre, olive, neutral. The olive knot is not
# decorative: the green transition between warm highlights and cold blacks is the
# signature of a lith print on warmtone paper.
LithPath = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]

# The default belongs to the *Neutral* profile, so it stays restrained: a hint of warmth,
# not a look. Pick a warmtone paper to get the color.
_DEFAULT_LITH_PATH: LithPath = ((4.0, 8.0), (2.0, 8.0), (-1.0, 4.0), (0.0, 0.0))

# Paper-character keys a profile overrides in the effective constants dict.
_TONAL_KEYS = (
    "d_max",
    "d_min",
    "toe_sharpness_base",
    "shoulder_sharpness_base",
    "toe_height",
    "shoulder_height",
    "paper_midtone_gamma",
    "paper_gamma_width",
)


@dataclass(frozen=True)
class PaperProfile:
    """
    One paper's print character. Tonal fields default to the current
    EXPOSURE_CONSTANTS values; color fields are identity (neutral).

    channel_gamma — per-channel (R, G, B) slope multipliers (dye-layer contrast
    crossover). base_tint_cmy — per-channel (C, M, Y) additions to the minimum
    density floor (base tint, shows in highlights). dye_matrix — dye coupling
    D_rgb = M · D_dye above base (unwanted absorptions), row-normalized at use.
    lith_path — the paper's lith color path, read by the Lith stage.
    kind drives dropdown grouping.
    """

    label: str
    kind: str = "ra4"  # "default" | "bw" | "ra4"
    d_max: float = EXPOSURE_CONSTANTS["d_max"]
    d_min: float = EXPOSURE_CONSTANTS["d_min"]
    toe_sharpness_base: float = EXPOSURE_CONSTANTS["toe_sharpness_base"]
    shoulder_sharpness_base: float = EXPOSURE_CONSTANTS["shoulder_sharpness_base"]
    toe_height: float = EXPOSURE_CONSTANTS["toe_height"]
    shoulder_height: float = EXPOSURE_CONSTANTS["shoulder_height"]
    paper_midtone_gamma: float = EXPOSURE_CONSTANTS["paper_midtone_gamma"]
    paper_gamma_width: float = EXPOSURE_CONSTANTS["paper_gamma_width"]
    channel_gamma: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    base_tint_cmy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dye_matrix: DyeMatrix = _IDENTITY_DYE
    lith_path: LithPath = _DEFAULT_LITH_PATH


PAPER_PROFILES: Dict[str, PaperProfile] = {
    DEFAULT_PROFILE_KEY: PaperProfile(label="Neutral (default)", kind="default"),
    # ── B&W (tonal only) ──────────────────────────────────────────────────────
    "ilford_mg_rc": PaperProfile(
        label="Ilford Multigrade RC",
        kind="bw",
        # Neutral VC workhorse; Dmax ~2.1, normal contrast.
        d_max=2.10,
        d_min=0.04,
        paper_midtone_gamma=0.15,
        # Multigrade resists lith. Rudman's lithability test reads an incorporated accelerator,
        # which short-circuits the semiquinone cascade, so the restrained default path is right
        # for it.
    ),
    "ilford_fb_classic": PaperProfile(
        label="Ilford Multigrade FB Classic",
        kind="bw",
        # Baryta, deeper blacks + crisper shadow knee than RC.
        d_max=2.15,
        d_min=0.04,
        toe_sharpness_base=5.0,
        paper_midtone_gamma=0.15,
    ),
    "foma_fomatone": PaperProfile(
        label="Foma Fomatone MG Classic",
        kind="bw",
        # Warm chlorobromide; gentler rendering, Dmax ~2.0.
        d_max=2.0,
        d_min=0.05,
        toe_sharpness_base=3.5,
        paper_midtone_gamma=0.10,
        # The canonical lith paper: reddish-yellow highlights through an olive transition to
        # green-black shadows (Moersch's per-paper tables).
        lith_path=((14.0, 22.0), (7.0, 26.0), (-8.0, 14.0), (-2.0, 2.0)),
    ),
    "foma_fomabrom": PaperProfile(
        label="Foma Fomabrom Variant",
        kind="bw",
        # Neutral baryta, Dmax 2.0.
        d_max=2.0,
        d_min=0.04,
        paper_midtone_gamma=0.15,
        # Yellowish highlights to greenish black, less colorful than Fomatone.
        lith_path=((6.0, 18.0), (2.0, 18.0), (-9.0, 10.0), (-2.0, 1.0)),
    ),
    # ── RA4 color ───────────────────────────────────────────────────────────
    "kodak_endura": PaperProfile(
        label="Kodak Endura Premier",
        kind="ra4",
        # Neutral, with deep blacks and a punchy midtone S. The datasheet R/G/B diverge only at
        # Dmax, red densest, which cools the deep shadows. Approximated with a small
        # channel_gamma.
        d_max=2.55,
        d_min=0.06,
        toe_sharpness_base=3.5,
        paper_midtone_gamma=0.22,
        channel_gamma=(1.04, 1.0, 0.98),
        # Estimated, not measured: cyan absorbs some green, magenta some blue.
        dye_matrix=(
            (0.95, 0.04, 0.01),
            (0.08, 0.88, 0.04),
            (0.04, 0.14, 0.82),
        ),
    ),
    "fuji_crystal": PaperProfile(
        label="Fujicolor Crystal Archive",
        kind="ra4",
        # No published curve, so this is a rough estimate: brilliant whites, vivid blue and
        # green, a slightly cool base. The tint is a per-channel density offset, where positive
        # darkens that channel, so a negative M/Y lifts green and blue for the vivid look.
        d_max=2.35,
        d_min=0.03,
        paper_midtone_gamma=0.15,
        channel_gamma=(1.0, 1.03, 1.05),
        base_tint_cmy=(0.0, -0.01, -0.015),
        # Estimated, not measured: slightly cleaner dyes than Endura.
        dye_matrix=(
            (0.96, 0.03, 0.01),
            (0.06, 0.91, 0.03),
            (0.03, 0.11, 0.86),
        ),
    ),
}


def resolve_paper(key: str) -> PaperProfile:
    """Profile for `key`, falling back to the neutral default on unknown keys."""
    return PAPER_PROFILES.get(key, PAPER_PROFILES[DEFAULT_PROFILE_KEY])


def resolve_dye_matrix(paper: PaperProfile | None) -> Optional[np.ndarray]:
    """
    Row-normalized dye coupling matrix (rows sum to 1, so neutrals are preserved),
    or None for identity so the default path stays byte-exact.
    """
    if paper is None or paper.dye_matrix == _IDENTITY_DYE:
        return None
    m = np.array(paper.dye_matrix, dtype=np.float64)
    return m / np.maximum(m.sum(axis=1, keepdims=True), 1e-6)


# Achromatic projection in density space (all rows 1/3) -- see resolve_saturation_matrix.
_ACHROMATIC_J = np.full((3, 3), 1.0 / 3.0, dtype=np.float64)


def resolve_saturation_matrix(k_rgb: Tuple[float, float, float]) -> Optional[np.ndarray]:
    """
    Density-domain saturation matrix, applied to density above paper base
    (same coordinate as dye_matrix). Row ch is k_rgb[ch]*I[ch] + (1-k_rgb[ch])*J[ch]
    — each row still sums to 1 regardless of its own k, so per-channel k's
    diverge (R/G/B pushed independently) without breaking neutral preservation;
    no per-row normalization needed. k=1 on every channel is identity (returns
    None so the default path stays byte-exact); k=0 full desaturation (collapse
    to the achromatic mean) for that channel's row; k>1 boosts density
    separation.
    """
    if k_rgb == (1.0, 1.0, 1.0):
        return None
    k = np.array(k_rgb, dtype=np.float64)
    return np.diag(k) + (1.0 - k)[:, np.newaxis] * _ACHROMATIC_J


def compose_density_matrices(dye: Optional[np.ndarray], sat: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Composite density-domain matrix for the dye_mix kernel slot, or None if
    both factors are identity (mirrors resolve_dye_matrix's allocation-free
    common case).

    sat is applied outermost, dye innermost (sat @ dye): dye_matrix keeps
    acting on the print curve's real density exactly as it always has, and
    saturation is layered on top as a final creative step rather than being
    fed back through the paper's physical crosstalk (that would make the
    slider's effective strength paper-dependent, since some of the push
    would get reabsorbed by the crosstalk before it reaches display). Do not
    flip this order.
    """
    if dye is None and sat is None:
        return None
    d = np.eye(3, dtype=np.float64) if dye is None else dye
    s = np.eye(3, dtype=np.float64) if sat is None else sat
    return s @ d


# Which paper kind each process mode exposes. E-6 (slide) has no entry and gets only the
# neutral default. Keyed by ProcessMode, a StrEnum, so plain-string process_mode values
# look up fine.
_MODE_KIND: Dict[str, str] = {ProcessMode.C41: "ra4", ProcessMode.BW: "bw"}


def profiles_for_mode(process_mode: str) -> List[Tuple[str, PaperProfile]]:
    """Selectable (key, profile) pairs for `process_mode`: neutral default first,
    then the papers whose kind matches the mode (default only for E-6)."""
    allowed = _MODE_KIND.get(process_mode)
    out = [(DEFAULT_PROFILE_KEY, PAPER_PROFILES[DEFAULT_PROFILE_KEY])]
    if allowed is not None:
        out += [(k, p) for k, p in PAPER_PROFILES.items() if p.kind == allowed]
    return out


def effective_paper_profile(key: str, process_mode: str | None) -> PaperProfile:
    """Mode-aware resolve: the stored profile only when its kind matches the mode,
    otherwise the neutral default. E-6 and any cross-mode/stale value collapse to
    default, so an incompatible `paper_profile` can never leak into a render."""
    paper = resolve_paper(key)
    if paper.kind == "default":
        return paper
    if process_mode is not None and _MODE_KIND.get(process_mode) == paper.kind:
        return paper
    return PAPER_PROFILES[DEFAULT_PROFILE_KEY]


def effective_constants(paper: PaperProfile | None) -> Dict[str, Any]:
    """
    EXPOSURE_CONSTANTS with the profile's tonal overrides applied. Returns the
    shared dict unchanged when paper is None or the neutral default, so the
    common path stays allocation-free and byte-for-byte identical.
    """
    if paper is None or paper.kind == "default":
        return EXPOSURE_CONSTANTS
    c = dict(EXPOSURE_CONSTANTS)
    for k in _TONAL_KEYS:
        c[k] = getattr(paper, k)
    return c
