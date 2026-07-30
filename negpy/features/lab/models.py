from dataclasses import dataclass
from enum import StrEnum


class SharpenMethod(StrEnum):
    USM = "usm"
    RL = "rl"


@dataclass(frozen=True)
class LabConfig:
    """
    Scanner emulation (Sharpening, CLAHE).

    Spectral crosstalk moved to ProcessConfig (capture-side, negative-density
    domain) — `color_separation`/`crosstalk_*` here are migrated by
    WorkspaceConfig.from_flat_dict.
    """

    saturation: float = 1.0
    # PROTOTYPE: A/B comparison toggle against the pre-gamut-aware naive flat
    # scale + hard clamp -- True (default) is the shipped gamut-aware behavior.
    # Not meant to ship; here so the same saturation value can be compared
    # against the old behavior directly instead of by re-deriving it by hand.
    saturation_gamut_aware: bool = True
    clahe_strength: float = 0.0
    sharpen: float = 0.25
    sharpen_method: SharpenMethod = SharpenMethod.USM
    sharpen_radius: float = 1.0
    sharpen_masking: float = 0.0
    chroma_denoise: float = 0.0
    glow_amount: float = 0.0
    halation_strength: float = 0.0
