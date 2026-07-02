"""
Darkroom paper profiles — per-paper overrides of the H&D print character.

A profile overrides a few EXPOSURE_CONSTANTS keys (the paper's characteristic
curve) plus optional colour terms. It only sets the curve *shape*; Grade still
owns contrast and Density/toe/shoulder still trim on top. The default profile
reproduces EXPOSURE_CONSTANTS exactly. B&W profiles are tonal only (the B&W path
collapses to luminance, so colour terms are inert — paper tone is a Toning job).

Values were loosely mapped by Claude from published datasheets (Ilford, Kodak
Endura, Foma, Fuji), not a precise calibration. Mainly d_max is grounded; the
knee/midtone tweaks are light touches for character. Note these stack on the
Grade slope, so over-soft knees read flat — keep them gentle.

To replace a profile with a real calibration, digitize the datasheet's D-logH
curve and run scripts/fit_paper_profile.py — it fits this exact parametric
family and prints the PaperProfile kwargs (accept when RMS ≤ ~0.05 D).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.process.models import ProcessMode

DEFAULT_PROFILE_KEY = "neutral"

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
    EXPOSURE_CONSTANTS values; colour fields are identity (neutral).

    channel_gamma — per-channel (R, G, B) slope multipliers (dye-layer contrast
    crossover). base_tint_cmy — per-channel (C, M, Y) pre-curve density offsets
    (paper-base warmth). kind drives dropdown grouping.
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
    ),
    "foma_fomabrom": PaperProfile(
        label="Foma Fomabrom Variant",
        kind="bw",
        # Neutral baryta, Dmax 2.0.
        d_max=2.0,
        d_min=0.04,
        paper_midtone_gamma=0.15,
    ),
    # ── RA4 colour ───────────────────────────────────────────────────────────
    "kodak_endura": PaperProfile(
        label="Kodak Endura Premier",
        kind="ra4",
        # Neutral, deep blacks (Dmax ~2.55), punchy midtone S. Datasheet R/G/B
        # diverge only at Dmax (R densest) → cool deep shadows; approximated with a
        # small channel_gamma.
        d_max=2.55,
        d_min=0.06,
        toe_sharpness_base=3.5,
        paper_midtone_gamma=0.22,
        channel_gamma=(1.04, 1.0, 0.98),
    ),
    "fuji_crystal": PaperProfile(
        label="Fujicolor Crystal Archive",
        kind="ra4",
        # No published curve; rough estimate — brilliant whites, vivid blue/green,
        # slight cool base. Tint is a per-channel density offset (+darkens that
        # channel): negative M/Y lifts green/blue for the cool, vivid look.
        d_max=2.35,
        d_min=0.03,
        paper_midtone_gamma=0.15,
        channel_gamma=(1.0, 1.03, 1.05),
        base_tint_cmy=(0.0, -0.01, -0.015),
    ),
}


def resolve_paper(key: str) -> PaperProfile:
    """Profile for `key`, falling back to the neutral default on unknown keys."""
    return PAPER_PROFILES.get(key, PAPER_PROFILES[DEFAULT_PROFILE_KEY])


# Which paper kind each process mode exposes. E-6 (slide) has no entry — it gets
# only the neutral default. Keyed by ProcessMode (a StrEnum), so plain-string
# process_mode values look up fine.
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
