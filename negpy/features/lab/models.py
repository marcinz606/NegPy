from dataclasses import dataclass


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
    clahe_strength: float = 0.0
    sharpen: float = 0.25
    chroma_denoise: float = 0.0
    glow_amount: float = 0.0
    halation_strength: float = 0.0
