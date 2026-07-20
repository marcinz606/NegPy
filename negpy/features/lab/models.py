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
    vibrance: float = 1.0
    chroma_damping: float = 0.5
    clahe_strength: float = 0.0
    sharpen: float = 0.25
    sharpen_method: SharpenMethod = SharpenMethod.USM
    sharpen_radius: float = 1.0
    sharpen_masking: float = 0.0
    chroma_denoise: float = 0.0
    glow_amount: float = 0.0
    halation_strength: float = 0.0
