"""
Darkroom paper profiles — per-paper overrides of the H&D print character.

A profile overrides a few EXPOSURE_CONSTANTS keys (the paper's characteristic
curve) plus optional color terms. It only sets the curve *shape*; Grade still
owns contrast and Density/toe/shoulder still trim on top. The default profile
reproduces EXPOSURE_CONSTANTS exactly. B&W profiles are tonal only under normal
development (the B&W path collapses to luminance, so the RA4 color terms are
inert — paper tone is a Toning job). The exception is Lith: `lith_path` is the paper's
infectious-development color and the only color term a B&W profile acts on.

A profile whose comment names a datasheet is fitted to it with scripts/fit_paper_profile.py,
which renders through print_curve at the slope the Grade gives the curve's ISO R (accept when
RMS ≤ ~0.05 D). The others were loosely mapped from datasheets: mainly d_max is grounded, and
their knee/midtone values are light touches for character. paper_midtone_gamma stays within
±0.25, because Snap and its per-layer trims add to it below the kernel's |gamma| < 1 bound.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.exposure.print_lut import DyeLut, DyeMatrix, DyeMatrixRows, PrintColor, PrintLut, load_print_lut
from negpy.features.process.models import ProcessMode

DEFAULT_PROFILE_KEY = "neutral"

_IDENTITY_DYE: DyeMatrixRows = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

# Lith hue path: four (a*, b*) anchors at the density fractions in
# LITH_CONSTANTS["path_u"], reading peach, ochre, olive, neutral. The olive knot is not
# decorative: the green transition between warm highlights and cold blacks is the
# signature of a lith print on warmtone paper.
LithPath = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]

# The default belongs to the *Digital* profile, so it stays restrained: a hint of warmth,
# not a look. Pick a warmtone paper to get the color.
_DEFAULT_LITH_PATH: LithPath = ((4.0, 8.0), (2.0, 8.0), (-1.0, 4.0), (0.0, 0.0))

# A color stock's dyes are a LUT baked by scripts/build_print_luts.py (negpy/features/exposure/print_luts/):
# a datasheet's spectral dye density curves, or a minilab's measured profile. Stocks on one datasheet's
# dyes share its LUT.

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
    density floor (base tint, shows in highlights). print_color — the stock's dyes: a
    DyeMatrix coupling D_rgb = M · D_dye above base, or a DyeLut from dye density to color.
    lith_path — the paper's lith color path, read by the Lith stage.
    kind picks the process mode the paper shows in; medium is its group in the dropdown.
    """

    label: str
    kind: str = "ra4"  # "default" | "bw" | "ra4"
    medium: str = "paper"  # "digital" | "paper" | "film"
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
    print_color: PrintColor = DyeMatrix(_IDENTITY_DYE)
    lith_path: LithPath = _DEFAULT_LITH_PATH


PAPER_PROFILES: Dict[str, PaperProfile] = {
    DEFAULT_PROFILE_KEY: PaperProfile(label="Digital (Adobe RGB)", kind="default", medium="digital"),
    # Digital's curve with a Pakon minilab's colors: the LUT is RGBScan.icc applied to Digital's print.
    "pakon": PaperProfile(label="Digital (Pakon Minilab)", kind="ra4", medium="digital", print_color=DyeLut("pakon")),
    # ── B&W (tonal only) ──────────────────────────────────────────────────────
    "ilford_mg_rc": PaperProfile(
        label="Ilford Multigrade RC",
        toe_sharpness_base=4.0,
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
        toe_sharpness_base=4.0,
        kind="bw",
        # Neutral baryta, Dmax 2.0.
        d_max=2.0,
        d_min=0.04,
        paper_midtone_gamma=0.15,
        # Yellowish highlights to greenish black, less colorful than Fomatone.
        lith_path=((6.0, 18.0), (2.0, 18.0), (-9.0, 10.0), (-2.0, 1.0)),
    ),
    # Fitted to Kodak G-26 (seven filter grades, F surface, Dektol 1:2).
    "kodak_polymax_ii": PaperProfile(
        label="Kodak Polymax II RC",
        kind="bw",
        d_max=2.13,
        d_min=0.062,
        toe_sharpness_base=3.58,
        shoulder_sharpness_base=3.07,
        paper_midtone_gamma=-0.111,
        paper_gamma_width=0.47,
    ),
    # Fitted to Kodak G-4037 (four filter grades, E surface, Dektol 1:2).
    "kodak_polycontrast_iv": PaperProfile(
        label="Kodak Polycontrast IV RC",
        kind="bw",
        d_max=2.15,
        d_min=0.062,
        toe_sharpness_base=5.99,
        shoulder_sharpness_base=2.96,
        paper_midtone_gamma=-0.148,
        paper_gamma_width=0.85,
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
        # Dyes from E-4070's spectral dye density curves, under D50.
        print_color=DyeLut("kodak_e4070"),
    ),
    # Fuji publishes no characteristic curve. The knees are fitted to the grey wedge koraks.nl
    # measured on ADOX Color Mission, a Crystal Archive paper; the midtone takes Endura Premier's
    # shape, a consumer RA4 paper of like contrast. Paper white, base tint, Dmax and dyes come from
    # Fujifilm's Frontier-PD profile for the paper; the HD and Pearl stocks keep this curve with their
    # own profiles. Pearl keeps DPII's white: its measured white is the
    # measuring angle on a pearl surface.
    "fuji_crystal": PaperProfile(
        label="Fujicolor Crystal Archive",
        kind="ra4",
        d_max=2.42,
        d_min=0.083,
        toe_sharpness_base=5.96,
        shoulder_sharpness_base=2.89,
        paper_midtone_gamma=0.22,
        base_tint_cmy=(0.008, 0.007, -0.016),
        print_color=DyeLut("fuji_ca_dpii"),
    ),
    "fuji_crystal_hd": PaperProfile(
        label="Fujicolor Crystal Archive HD",
        kind="ra4",
        d_max=2.14,
        d_min=0.071,
        toe_sharpness_base=5.96,
        shoulder_sharpness_base=2.89,
        paper_midtone_gamma=0.22,
        base_tint_cmy=(0.015, 0.011, -0.026),
        print_color=DyeLut("fuji_ca_hd"),
    ),
    "fuji_crystal_pearl": PaperProfile(
        label="Fujicolor Crystal Archive Pearl",
        kind="ra4",
        d_max=2.23,
        d_min=0.083,
        toe_sharpness_base=5.96,
        shoulder_sharpness_base=2.89,
        paper_midtone_gamma=0.22,
        base_tint_cmy=(0.008, 0.007, -0.016),
        print_color=DyeLut("fuji_ca_pearl"),
    ),
    # Fitted to Kodak E-4021 (Status A R/G/B, shared with Supra Endura), with its dyes: spektrafilm's
    # trace of the 2009 sheet (CC BY-SA 4.0, NOTICE.md), under D50.
    "kodak_portra_endura": PaperProfile(
        label="Kodak Portra Endura",
        kind="ra4",
        d_max=2.65,
        d_min=0.093,
        toe_sharpness_base=1.78,
        shoulder_sharpness_base=2.81,
        paper_midtone_gamma=0.25,
        paper_gamma_width=0.89,
        channel_gamma=(0.968, 1.0, 1.013),
        base_tint_cmy=(-0.002, 0.012, -0.009),
        print_color=DyeLut("kodak_e4021"),
    ),
    "kodak_supra_endura": PaperProfile(
        label="Kodak Supra Endura",
        kind="ra4",
        d_max=2.60,
        d_min=0.095,
        toe_sharpness_base=7.99,
        shoulder_sharpness_base=3.23,
        paper_midtone_gamma=0.165,
        paper_gamma_width=0.62,
        channel_gamma=(1.008, 1.0, 1.0),
        base_tint_cmy=(0.011, -0.009, -0.003),
        print_color=DyeLut("kodak_e4021"),
    ),
    # Fitted to Kodak E-4020 (Status A R/G/B). Dyes are E-4070's: the E-4020 sheet draws each dye
    # over a density floor in every band, which desaturates past any RA4 print, and spektrafilm's
    # Ultra Endura is the later High Definition paper (E-4044).
    "kodak_ultra_endura": PaperProfile(
        label="Kodak Ultra Endura",
        kind="ra4",
        d_max=2.61,
        d_min=0.096,
        toe_sharpness_base=6.04,
        shoulder_sharpness_base=5.74,
        paper_midtone_gamma=0.197,
        paper_gamma_width=0.60,
        channel_gamma=(1.109, 1.0, 1.051),
        base_tint_cmy=(-0.003, -0.003, 0.005),
        print_color=DyeLut("kodak_e4070"),
    ),
    # ── Color print film (kind "ra4": the kind gates the C-41 mode) ─────────
    # Fitted to Kodak H-1-2383 (Status A R/G/B) with the `print` fit, which scales the
    # projection Dmax (near 4) into the paper range. The datasheet draws one shared base, so
    # no tint. Dyes from its spectral dye density curves, viewed under a xenon projector.
    "kodak_2383": PaperProfile(
        label="Kodak Vision 2383",
        kind="ra4",
        medium="film",
        d_max=2.29,
        d_min=0.065,
        toe_sharpness_base=2.65,
        shoulder_sharpness_base=2.86,
        paper_midtone_gamma=-0.019,
        paper_gamma_width=0.31,
        channel_gamma=(1.034, 1.0, 0.989),
        print_color=DyeLut("kodak_2383"),
    ),
    # Fitted like 2383 to Kodak H-1-2393t (1998); its curves and dyes are spektrafilm's trace of
    # that sheet (CC BY-SA 4.0, NOTICE.md). The base tint is within the trace's noise.
    "kodak_2393": PaperProfile(
        label="Kodak Vision Premier 2393",
        kind="ra4",
        medium="film",
        d_max=2.25,
        d_min=0.025,
        toe_sharpness_base=3.22,
        shoulder_sharpness_base=2.28,
        paper_midtone_gamma=0.074,
        paper_gamma_width=0.58,
        channel_gamma=(1.075, 1.0, 1.049),
        print_color=DyeLut("kodak_2393"),
    ),
}


def resolve_paper(key: str) -> PaperProfile:
    """Profile for `key`, falling back to the Digital default on unknown keys."""
    return PAPER_PROFILES.get(key, PAPER_PROFILES[DEFAULT_PROFILE_KEY])


def resolve_print_color(paper: PaperProfile | None) -> Tuple[Optional[np.ndarray], Optional[PrintLut]]:
    """
    (dye matrix, LUT) for the print kernel. The matrix is row-normalized (rows sum to 1, so
    neutrals are preserved) and None for identity, so the default path stays byte-exact.
    """
    color = None if paper is None else paper.print_color
    if isinstance(color, DyeLut):
        return None, load_print_lut(color.key)
    if color is None or color.rows == _IDENTITY_DYE:
        return None, None
    m = np.array(color.rows, dtype=np.float64)
    return m / np.maximum(m.sum(axis=1, keepdims=True), 1e-6), None


# Achromatic projection in density space (all rows 1/3) -- see resolve_saturation_matrix.
_ACHROMATIC_J = np.full((3, 3), 1.0 / 3.0, dtype=np.float64)


def resolve_saturation_matrix(k_rgb: Tuple[float, float, float]) -> Optional[np.ndarray]:
    """
    Density-domain saturation matrix, applied to density above paper base
    (same coordinate as a dye matrix). Row ch is k_rgb[ch]*I[ch] + (1-k_rgb[ch])*J[ch]
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
    both factors are identity (mirrors resolve_print_color's allocation-free
    common case).

    sat is applied outermost, dye innermost (sat @ dye), so a matrix stock's coupling
    does not reabsorb the push. A LUT stock passes dye=None: separation then acts on the
    dye amounts before the LUT, as a more saturated negative would on that paper.
    """
    if dye is None and sat is None:
        return None
    d = np.eye(3, dtype=np.float64) if dye is None else dye
    s = np.eye(3, dtype=np.float64) if sat is None else sat
    return s @ d


# Which paper kind each process mode exposes. E-6 (slide) has no entry and gets only the
# Digital default. Keyed by ProcessMode, a StrEnum, so plain-string process_mode values
# look up fine.
_MODE_KIND: Dict[str, str] = {ProcessMode.C41: "ra4", ProcessMode.BW: "bw"}


def profiles_for_mode(process_mode: str) -> List[Tuple[str, PaperProfile]]:
    """Selectable (key, profile) pairs for `process_mode`: the Digital default first,
    then the papers whose kind matches the mode (default only for E-6)."""
    allowed = _MODE_KIND.get(process_mode)
    out = [(DEFAULT_PROFILE_KEY, PAPER_PROFILES[DEFAULT_PROFILE_KEY])]
    if allowed is not None:
        out += [(k, p) for k, p in PAPER_PROFILES.items() if p.kind == allowed]
    return out


def effective_paper_profile(key: str, process_mode: str | None) -> PaperProfile:
    """Mode-aware resolve: the stored profile only when its kind matches the mode,
    otherwise the Digital default. E-6 and any cross-mode/stale value collapse to
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
    shared dict unchanged when paper is None or the Digital default, so the
    common path stays allocation-free and byte-for-byte identical.
    """
    if paper is None or paper.kind == "default":
        return EXPOSURE_CONSTANTS
    c = dict(EXPOSURE_CONSTANTS)
    for k in _TONAL_KEYS:
        c[k] = getattr(paper, k)
    return c
