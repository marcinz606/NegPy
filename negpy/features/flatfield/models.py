from dataclasses import dataclass


@dataclass(frozen=True)
class FlatFieldConfig:
    """Flat-field (illumination falloff) correction."""

    # Per-image toggle. Named 'apply', not 'enabled', to stay unique in the flat
    # config dict (WorkspaceConfig.to_dict) where RgbScanConfig.enabled also lives.
    apply: bool = False
    # Resolved path of the globally active reference profile (seeded on file load).
    reference_path: str = ""
