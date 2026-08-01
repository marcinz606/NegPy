from dataclasses import dataclass


@dataclass(frozen=True)
class FlatFieldConfig:
    """Flat-field (illumination falloff) correction."""

    # Per-image toggle. Named 'apply', not 'enabled', to stay unique in the flat
    # config dict (WorkspaceConfig.to_dict) where RgbScanConfig.enabled also lives.
    apply: bool = False
    # Opaque id of the active reference profile — a baked gain map stored in
    # APP_CONFIG.flatfield_dir (services/assets/flatfield.py). Stable across renames
    # and machines; the render path resolves the gain by this id, so the original
    # reference image can be moved or deleted without breaking the correction.
    profile_id: str = ""
    # Radial lens-distortion coefficient. A rig property, so it's mirrored from the
    # active profile (re-seeded on load) and consumed by the geometry stage — not
    # owned by the per-image edit.
    k1: float = 0.0
