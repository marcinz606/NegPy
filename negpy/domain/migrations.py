"""
Backward-compat migrations for persisted configs (edits.db rows, .negpy sidecars,
export presets). Every legacy key rename, value rewrite and retired-field drop
belongs here — add new ones to this module, not to ``from_flat_dict``.

Not here: coercions that must run on *every* construction rather than only on load
— ``ExposureConfig.__post_init__`` (legacy 0-5 paper grade → ISO R, cast_removal
bool → strength), ``ProcessConfig.__post_init__`` (pre-rename process-mode names,
which also arrive from sticky settings and asset dicts) and the tuple-rehydrating
``__post_init__``s. String literals only (no ``domain.models`` import — that module
imports this one).

Also not here: migrations that rewrite *rows* rather than a config payload, since those
need a repository and this module stays dependency-free. They live beside the feature
they belong to — ``services/assets/hash_migration.py`` (edits saved under a superseded
content hash) and ``services/assets/flatfield_migration.py`` (legacy profile table).
"""

from typing import Any, Dict

# Old field name → new field name.
KEY_RENAMES: Dict[str, str] = {
    "export_border_size": "border_size",
    "export_border_color": "border_color",
    # Shadow-neutral + density-balance became Cast Removal, then Cast Removal
    # became a 0..1 strength (bool coerced in ExposureConfig.__post_init__).
    "auto_shadow_neutral": "cast_removal_strength",
    "cast_removal": "cast_removal_strength",
    # D-Range Clip split into luma + color clips; the old slider is the luma axis.
    "drange_clip": "luma_range_clip",
    # The frame-wide density control (ex "Print Saturation") absorbed the per-pixel
    # one and took its name (see the pop in migrate_flat_config).
    "density_saturation": "dye_separation",
    "density_saturation_trim_red": "dye_separation_trim_red",
    "density_saturation_trim_green": "dye_separation_trim_green",
    "density_saturation_trim_blue": "dye_separation_trim_blue",
    "use_colour_average": "use_color_average",
    # The contrast mask's radius is named for the darkroom spacer, not for the blur.
    "mask_blur": "mask_spacer",
    # An old save with True and no rect lands armed, so it resolves on the next render.
    "manual_crop_rect": "crop_rect",
    "auto_crop_enabled": "crop_from_auto",
}

# Fields removed over time. Old saves still carry them, so drop them silently and
# keep the warning for truly unknown keys.
DROPPED_KEYS: frozenset[str] = frozenset(
    {
        "flare",  # veiling-glare floor
        "ir_inpaint_radius",  # IR stroke synthesis, replaced by score-weighted fill
        "auto_cast_removal",  # cast removal became always-on
        "DEFAULT_MATRIX",  # serialized crosstalk default, never a real field
        "surround",  # dim-surround print gamma, removed in #432 (replaced by Snap)
        "density_chroma_damping",
        "density_divergence_damping",
        "chroma_damping",  # Dye Mute, first form
        "density_saturation_damping",
        "density_damping_spatial",
        "vibrance",  # Lab Vibrance, retired in favour of the per-pixel Dye Separation
        # Flat-field moved from a reference *path* to an opaque profile_id. The DB
        # migration rewrites the edits it can reach; this drops the legacy key from
        # sidecars and presets that slip through. Those need the profile re-picked.
        "reference_path",
        # TIFF-with-JXL-compression never shipped: too few readers support the tag.
        # TIFF is zlib-only and lossless JXL stays a standalone format.
        "tiff_compression",
        # Lith's bool became the alt_process enum it shares with Cyanotype
        # (migrate_flat_config reads it before this pop).
        "lith_enabled",
    }
)

# Retired export formats -> the closest surviving one. DNG maps to TIFF, the other
# 16-bit master, so a saved DNG preset keeps its bit depth instead of 8-bit JPEG.
RETIRED_EXPORT_FORMATS: Dict[str, str] = {"DNG": "TIFF"}


def migrate_export_fmt(fmt: str) -> str:
    return RETIRED_EXPORT_FORMATS.get(fmt, fmt)


def migrate_flat_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite a flat config dict from any older NegPy version in place, and return it.

    Value rewrites run before DROPPED_KEYS is popped — several of them read a key
    on their way to deleting it.
    """
    # The per-pixel Dye Separation merged into the frame-wide control and took its
    # name, so the same key means different things either side of the merge and an old
    # ±0.5 value would read as heavy desaturation. A dict carrying density_saturation
    # predates the merge, and anything under the new slider floor (0.25) can only be
    # the old signed control. Drop those before the rename writes that key.
    # Not DROPPED_KEYS: those pops run last and would take it back out.
    if "density_saturation" in data or float(data.get("dye_separation", 1.0)) < 0.25:
        data.pop("dye_separation", None)
    data.pop("density_vibrance", None)

    for old_key, new_key in KEY_RENAMES.items():
        if old_key in data:
            data[new_key] = data.pop(old_key)

    # True Black (BPC on) renamed to Paper Black with inverted polarity.
    if "true_black" in data:
        data["paper_black"] = not bool(data.pop("true_black"))

    # Single roll-average toggle split into independent luma + color axes.
    if "use_roll_average" in data:
        legacy = bool(data.pop("use_roll_average"))
        data.setdefault("use_luma_average", legacy)
        data.setdefault("use_color_average", legacy)

    # Lab "Separation" moved to ProcessConfig crosstalk: the 1.0-2.0 slider maps to
    # strength 0-1. crosstalk_matrix/crosstalk_profile keep their names and re-route
    # by field membership; the old serialized DEFAULT_MATRIX field is dropped.
    if "color_separation" in data and "crosstalk_strength" not in data:
        data["crosstalk_strength"] = min(max(float(data.pop("color_separation")) - 1.0, 0.0), 1.0)
    data.pop("color_separation", None)

    # The built-in crosstalk profile "Default" became "Generic C41". Saved edits store
    # the display name, so without this the dropdown selects the wrong row. The render
    # is unaffected: a None matrix still falls back to the built-in.
    if data.get("crosstalk_profile") == "Default":
        data["crosstalk_profile"] = "Generic C41"

    # Vignette became an exposure-domain burn: old ±1 strength (neg = darken) maps
    # to stops (pos = burn) with an approximate look-preserving factor.
    if "vignette_strength" in data and "vignette_stops" not in data:
        data["vignette_stops"] = -2.0 * float(data.pop("vignette_strength"))
    data.pop("vignette_strength", None)

    # The filed carrier's toggle folded into carrier_width (0 = off). An old save
    # serialized both keys, so a stored width of 2.0 with the toggle off would read
    # as "on" under the new width>0 gating. Force the width to 0.
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

    # Lith and Cyanotype are mutually exclusive, so the panel keeps one enum instead
    # of a bool each. The three lith_* sliders kept their names.
    if "lith_enabled" in data and "alt_process" not in data:
        data["alt_process"] = "lith" if data["lith_enabled"] else "none"

    if "export_fmt" in data:
        data["export_fmt"] = migrate_export_fmt(str(data["export_fmt"]))

    for key in DROPPED_KEYS:
        data.pop(key, None)

    return data
