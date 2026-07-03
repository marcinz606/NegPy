"""
Darkroom paper profiles — per-paper overrides of the H&D print character.

A profile overrides a few EXPOSURE_CONSTANTS keys (the paper's characteristic
curve) plus optional colour terms. It only sets the curve *shape*; Grade still
owns contrast and Density/toe/shoulder still trim on top. The default profile
reproduces EXPOSURE_CONSTANTS exactly. B&W profiles are tonal only (the B&W path
collapses to luminance, so colour terms are inert — paper tone is a Toning job).

Tonal values are least-squares fits of digitized datasheet D-logH curves
(scripts/fit_paper_profile.py; curves extracted from the vendor PDFs' vector
art, 2026-07 — digitized CSVs + fit logs in papers/fits/). Family fits share
one paper shape across the published grade/channel curves; per-curve slope is
nuisance (Grade owns contrast). All accepted fits have RMS ≤ 0.03 D except
where noted. Fuji publishes no D-logH curve, so that profile stays a hand
estimate. Note d_min is the softplus model parameter, not the literal paper
floor — with very soft knees (Endura) the rendered floor emerges lower from
the toe/shoulder interplay.

To recalibrate, digitize a datasheet's D-logH curve and run
scripts/fit_paper_profile.py — it fits this exact parametric family and
prints the PaperProfile kwargs (accept when RMS ≤ ~0.05 D).
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
        # MULTIGRADE-IV-RC-Papers-060619.pdf p3. Curve chart is a low-res raster,
        # so the shape is the FB Classic family fit (Ilford: MG curves "broadly
        # similar"); Dmax 2.20 measured off the RC chart plateau — deeper than the
        # FB Classic chart, contrary to RC/FB folklore. Toe unreadable → FB d_min.
        d_max=2.20,
        d_min=0.012,
        toe_sharpness_base=5.64,
        shoulder_sharpness_base=3.46,
        paper_midtone_gamma=0.0,
    ),
    "ilford_fb_classic": PaperProfile(
        label="Ilford Multigrade FB Classic",
        kind="bw",
        # MULTIGRADE FB CLASSIC datasheet p2, grades 00-5 family fit, RMS 0.028 D.
        # Crisp toe, soft-ish shoulder, no extra midtone S (mg fit → 0).
        d_max=2.08,
        d_min=0.012,
        toe_sharpness_base=5.64,
        shoulder_sharpness_base=3.46,
        paper_midtone_gamma=0.0,
    ),
    "foma_fomatone": PaperProfile(
        label="Foma Fomatone MG Classic",
        kind="bw",
        # fomatone datasheet p2, grades 0-4 family fit, RMS 0.017 D. Strong
        # chlorobromide midtone S with snappy knees; chart plateau is semi-glossy
        # (1.83) — d_max set to the glossy 2.0 from the same sheet's table.
        d_max=2.0,
        d_min=0.085,
        toe_sharpness_base=8.13,
        shoulder_sharpness_base=7.15,
        paper_midtone_gamma=0.80,
        paper_gamma_width=0.62,
    ),
    "foma_fomabrom": PaperProfile(
        label="Foma Fomabrom Variant",
        kind="bw",
        # fomabrom datasheet p2, grades 0-5 family fit, RMS 0.014 D. Near-linear
        # mid (mg ≈ 0), firm knees; d_max 2.0 per the chart's own "Dmax=2,0" mark
        # (fit read 1.97, within axis-calibration slop).
        d_max=2.0,
        d_min=0.105,
        toe_sharpness_base=6.23,
        shoulder_sharpness_base=6.55,
        paper_midtone_gamma=0.03,
        paper_gamma_width=0.37,
    ),
    # ── RA4 colour ───────────────────────────────────────────────────────────
    "kodak_endura": PaperProfile(
        label="Kodak Endura Premier",
        kind="ra4",
        # paper-endura-techpub-e4070.pdf p4 (Status A). Tonal shape = green-channel
        # fit (RMS 0.005 D): very soft, wide knees over the 3-decade axis; the
        # rendered floor emerges ≈0.10 from the knee interplay, not d_min itself.
        # channel_gamma from per-channel slope refits with the G shape fixed
        # (R 9.46 / G 8.27 / B 8.52) — R steepest ≈ its deeper Dmax (2.75 vs 2.52),
        # cool deep shadows. The shared-shape family can't also hold R's Dmax
        # (R-channel residual 0.10 D, concentrated in the last ~0.3 D of shadow).
        # Channel floors are near-equal (R/G 0.105, B 0.079) → no base tint.
        d_max=2.52,
        d_min=0.20,
        toe_sharpness_base=0.99,
        shoulder_sharpness_base=0.90,
        paper_midtone_gamma=0.0,
        channel_gamma=(1.14, 1.0, 1.03),
    ),
    "fuji_crystal": PaperProfile(
        label="Fujicolor Crystal Archive",
        kind="ra4",
        # Fuji publishes no D-logH curve for CA papers (Type II / DPII / AF3-198E
        # sheets checked 2026-07: spectral data only), so this stays a hand
        # estimate — brilliant whites, vivid blue/green, slight cool base. Tint is
        # a per-channel density offset (+darkens that channel): negative M/Y lifts
        # green/blue for the cool, vivid look.
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
