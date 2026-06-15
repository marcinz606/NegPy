from dataclasses import dataclass


@dataclass(frozen=True)
class FlatFieldConfig:
    """Flat-field (illumination falloff) correction."""

    # Per-image toggle.
    enabled: bool = False
    # Resolved path of the globally active reference profile (seeded on file load).
    reference_path: str = ""
