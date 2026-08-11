"""
Backward-compat migrations for persisted configs (edits.db rows, .negpy sidecars,
export presets). Every legacy key rename, value rewrite and retired-field drop
belongs here — add new ones to this module, not to ``from_flat_dict``.

Not here: coercions that must run on *every* construction rather than only on load
— ``ExposureConfig.__post_init__`` (legacy 0-5 paper grade → ISO R, cast_removal
bool → strength) and the tuple-rehydrating ``__post_init__``s. String literals only
(no ``domain.models`` import — that module imports this one).

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
    # Shadow-neutral + density-balance consolidated into Cast Removal, then Cast
    # Removal itself became a 0..1 strength (bool True/False coerced to 1.0/0.0 in
    # ExposureConfig.__post_init__). Preserve a user's saved on/off.
    "auto_shadow_neutral": "cast_removal_strength",
    "cast_removal": "cast_removal_strength",
    # D-Range Clip split into independent luma + colour range clips; the old single
    # slider maps to the luma axis (colour defaults to its aggressive baseline).
    "drange_clip": "luma_range_clip",
    # The frame-wide density-domain control (ex "Print Saturation") absorbed the
    # per-pixel one beside it and took over its name (see the pop in migrate_flat_config).
    "density_saturation": "dye_separation",
    "density_saturation_trim_red": "dye_separation_trim_red",
    "density_saturation_trim_green": "dye_separation_trim_green",
    "density_saturation_trim_blue": "dye_separation_trim_blue",
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
        "density_chroma_damping",
        "density_divergence_damping",
        "chroma_damping",  # Dye Mute, first form
        "density_saturation_damping",
        "density_damping_spatial",
        "vibrance",  # Lab Vibrance, retired in favour of the per-pixel Dye Separation
        # Flat-field moved from a stored reference *path* to an opaque profile_id
        # (baked gain in the file store). The DB migration rewrites edits it can
        # reach; this drops the legacy key from any that slip through (sidecars,
        # presets) so it doesn't warn — that edit just needs its profile re-picked.
        "reference_path",
        # TIFF-with-JXL-compression option: too few readers actually support the
        # jpegxl TIFF compression tag, so it never shipped past this branch — TIFF
        # is zlib-only again and lossless JXL stays a standalone format.
        "tiff_compression",
        # Lith's on/off bool became one alt_process enum shared with Cyanotype
        # (see migrate_flat_config, which reads it before this pop).
        "lith_enabled",
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
    # The per-pixel Dye Separation was consolidated into the frame-wide density-domain
    # control (ex "Print Saturation"), which took over its name — so the same key means
    # different things either side of the merge, and the old ±0.5 value would read as a
    # heavy desaturation. A dict still carrying density_saturation predates the merge,
    # and anything under the new slider floor (0.25 at merge time) can only be the old
    # signed-around-zero control; drop those before the rename lands the new value on
    # that key. Not DROPPED_KEYS: those pops run last and would take it back out.
    if "density_saturation" in data or float(data.get("dye_separation", 1.0)) < 0.25:
        data.pop("dye_separation", None)
    data.pop("density_vibrance", None)

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

    # The built-in crosstalk profile "Default" was renamed "Generic C41". Saved edits store the
    # display name; the render is unaffected (a None matrix still falls back to the built-in),
    # but the dropdown would match no row and show the wrong profile as selected.
    if data.get("crosstalk_profile") == "Default":
        data["crosstalk_profile"] = "Generic C41"

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

    # Lith and Cyanotype are mutually exclusive, so the panel keeps one enum
    # instead of a bool each. The three lith_* sliders kept their names.
    if "lith_enabled" in data and "alt_process" not in data:
        data["alt_process"] = "lith" if data["lith_enabled"] else "none"

    if "export_fmt" in data:
        data["export_fmt"] = migrate_export_fmt(str(data["export_fmt"]))

    for key in DROPPED_KEYS:
        data.pop(key, None)

    return data
