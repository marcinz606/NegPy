"""
Backward-compat migrations for persisted configs (edits.db rows, .negpy sidecars,
export presets). Every legacy key rename, value rewrite and retired-field drop
belongs here — add new ones to this module, not to ``from_flat_dict``.

Not here: coercions that must run on *every* construction rather than only on load
— ``ExposureConfig.__post_init__`` (legacy 0-5 paper grade → ISO R, cast_removal
bool → strength) and the tuple-rehydrating ``__post_init__``s. String literals only
(no ``domain.models`` import — that module imports this one).
"""

from typing import Any, Dict

# Old field name → new field name.
KEY_RENAMES: Dict[str, str] = {
    "export_border_size": "border_size",
    "export_border_color": "border_color",
    # Shadow-neutral + density-balance consolidated into Cast Removal, then Cast
    # Removal itself became a 0..1 strength (bool True/False coerced to 1.0/0.0 in
    # ExposureConfig.__post_init__). Preserve a user's saved on/off.
    "auto_shadow_neutral": "cast_removal_strength",
    "cast_removal": "cast_removal_strength",
    # D-Range Clip split into independent luma + colour range clips; the old single
    # slider maps to the luma axis (colour defaults to its aggressive baseline).
    "drange_clip": "luma_range_clip",
}

# Fields removed from the config over time. Every edit saved before the removal
# still carries them, so they'd warn on every file load; drop them silently and
# keep the warning for truly unknown keys.
DROPPED_KEYS: frozenset[str] = frozenset(
    {
        "flare",  # veiling-glare floor
        "ir_inpaint_radius",  # IR stroke synthesis, replaced by score-weighted fill
        "auto_cast_removal",  # cast removal became always-on
        "DEFAULT_MATRIX",  # serialized crosstalk default, never a real field
        "surround",  # dim-surround print gamma, removed in #432 (replaced by Snap)
        # density_chroma_damping / density_divergence_damping: abandoned Dye Mute
        # reference-signal prototypes, dropped as mathematically redundant with
        # density_saturation itself.
        "density_chroma_damping",
        "density_divergence_damping",
    }
)

# Export formats no longer offered → the closest surviving one. DNG export was
# removed; TIFF is the other 16-bit master, so a saved DNG preset keeps its bit
# depth instead of falling through the encoder to 8-bit JPEG.
RETIRED_EXPORT_FORMATS: Dict[str, str] = {"DNG": "TIFF"}


def migrate_export_fmt(fmt: str) -> str:
    return RETIRED_EXPORT_FORMATS.get(fmt, fmt)


def migrate_flat_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite a flat config dict from any older NegPy version in place, and return it.

    Value rewrites run before DROPPED_KEYS is popped — several of them read a key
    on their way to deleting it.
    """
    for old_key, new_key in KEY_RENAMES.items():
        if old_key in data:
            data[new_key] = data.pop(old_key)

    # True Black (BPC on) renamed to Paper Black with inverted polarity.
    if "true_black" in data:
        data["paper_black"] = not bool(data.pop("true_black"))

    # Single roll-average toggle split into independent luma + colour axes.
    if "use_roll_average" in data:
        legacy = bool(data.pop("use_roll_average"))
        data.setdefault("use_luma_average", legacy)
        data.setdefault("use_colour_average", legacy)

    # Lab "Separation" moved to the capture side (ProcessConfig crosstalk):
    # the 1.0–2.0 slider maps to strength 0–1. crosstalk_matrix/crosstalk_profile
    # keep their key names and re-route to ProcessConfig by field membership;
    # the old serialized DEFAULT_MATRIX field is dropped.
    if "color_separation" in data and "crosstalk_strength" not in data:
        data["crosstalk_strength"] = min(max(float(data.pop("color_separation")) - 1.0, 0.0), 1.0)
    data.pop("color_separation", None)

    # Vignette became an exposure-domain burn: old ±1 strength (neg = darken)
    # maps to stops (pos = burn) with an approximate look-preserving factor.
    if "vignette_strength" in data and "vignette_stops" not in data:
        data["vignette_stops"] = -2.0 * float(data.pop("vignette_strength"))
    data.pop("vignette_strength", None)

    # Filed carrier's separate on/off toggle was folded into carrier_width itself
    # (0 = off) in #542. A pre-#542 save always serialized both keys together, so
    # if the toggle was off, force width to 0 too — otherwise the leftover numeric
    # width (2.0 by the old default) reads as "on" under the new width>0 gating,
    # silently re-enabling a carrier the user had switched off.
    if "carrier_enabled" in data and not data.pop("carrier_enabled"):
        data["carrier_width"] = 0.0

    if "use_original_res" in data and "export_resolution_mode" not in data:
        data["export_resolution_mode"] = "original" if data.pop("use_original_res") else "print"
    else:
        data.pop("use_original_res", None)

    if "same_as_source" in data and "output_mode" not in data:
        data["output_mode"] = "same_as_source" if data.pop("same_as_source") else "absolute"
    else:
        data.pop("same_as_source", None)

    # Metadata: migrate legacy combined override fields to structured gear fields.
    if data.get("camera_override") and not data.get("camera_model"):
        co = str(data.pop("camera_override", "")).strip()
        if co and not data.get("camera_make"):
            parts = co.split(None, 1)
            if len(parts) == 2:
                data["camera_make"], data["camera_model"] = parts[0], parts[1]
            else:
                data["camera_model"] = co
        elif co:
            data["camera_model"] = co
    else:
        data.pop("camera_override", None)
    if data.get("lens_override") and not data.get("lens_model"):
        data["lens_model"] = str(data.pop("lens_override", "")).strip()
    else:
        data.pop("lens_override", None)

    if "export_fmt" in data:
        data["export_fmt"] = migrate_export_fmt(str(data["export_fmt"]))

    for key in DROPPED_KEYS:
        data.pop(key, None)

    return data
