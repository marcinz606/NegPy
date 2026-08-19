"""Storage for the settings that carry onto a freshly-opened file.

The set is a user choice, held as a list of `SettingRow` ids, and the values come from a
flat snapshot of the last persisted edit. Everything the picker can offer is a catalog
row; the few carries that are not plain config values — the rig-global flat-field
profile, the Kelvin roll-locks — stay in `AppState._apply_sticky_settings`.
"""

from __future__ import annotations

from typing import Any, Optional

from negpy.domain.interfaces import IRepository
from negpy.domain.models import ExportConfig, WorkspaceConfig
from negpy.desktop.settings_catalog import (
    CATALOG,
    DEFAULT_STICKY_IDS,
    SettingRow,
    all_rows,
    rows_by_id,
    selected_flat_dict,
)

STICKY_CONFIG_KEY = "sticky_config"
STICKY_ROWS_KEY = "sticky_rows"

_CATALOG_EXPORT_FIELDS = frozenset(f for title, rows in CATALOG if title == "Export" for r in rows for f in r.fields)

# Export fields the catalog deliberately does not list: the output folder, ICC paths and
# the contact-sheet layout. They are workspace state rather than a look, so they carry
# unconditionally and never appear in the picker. Derived, so a new Export row cannot
# leave one silently uncarried.
EXPORT_REMAINDER: frozenset[str] = frozenset(ExportConfig.__dataclass_fields__) - _CATALOG_EXPORT_FIELDS

# Scan-setup preferences, carried unconditionally for the same reason. Kept out of the
# catalog because that would put them in the preset "Replace look" set, where resetting
# the decode mode would force a re-decode.
ALWAYS_STICKY_PROCESS: tuple[tuple[str, str], ...] = (
    ("last_linear_raw", "linear_raw"),
    ("last_narrowband_scan", "narrowband_scan"),
)


def load_sticky_rows(repo: IRepository) -> list[SettingRow]:
    """The rows the user has chosen to carry, defaults when they never chose."""
    stored = repo.get_global_setting(STICKY_ROWS_KEY)
    ids = set(stored) if isinstance(stored, list) else set(DEFAULT_STICKY_IDS)
    by_id = rows_by_id()
    return [row for row_id, row in by_id.items() if row_id in ids]


def save_sticky_rows(repo: IRepository, ids: list[str]) -> None:
    repo.save_global_setting(STICKY_ROWS_KEY, sorted(ids))


# Kept out of the snapshot and written only by the Description… dialog, so the last
# confirm wins for the roll instead of whichever frame was saved last.
DESCRIPTION_FIELDS_KEY = "last_description_fields"


def sticky_snapshot(config: WorkspaceConfig) -> dict[str, Any]:
    """Every catalog-reachable field, so any row can be made sticky later."""
    flat = selected_flat_dict(config, all_rows())
    flat.pop("description_fields", None)
    return flat


def load_sticky_config(repo: IRepository) -> Optional[WorkspaceConfig]:
    stored = repo.get_global_setting(STICKY_CONFIG_KEY)
    if not isinstance(stored, dict) or not stored:
        return None
    return WorkspaceConfig.from_flat_dict(stored)


# Legacy per-key sticky store, superseded by STICKY_CONFIG_KEY. Only the keys the old
# _apply_sticky_settings actually read are worth carrying forward.
_LEGACY_KEYS: dict[str, str] = {
    "last_process_mode": "process_mode",
    "last_analysis_buffer": "analysis_buffer",
    "last_luma_range_clip": "luma_range_clip",
    "last_color_range_clip": "color_range_clip",
    "last_crosstalk_strength": "crosstalk_strength",
    "last_crosstalk_matrix": "crosstalk_matrix",
    "last_crosstalk_profile": "crosstalk_profile",
    "last_sensor_matrix": "sensor_matrix",
    "last_sensor_profile": "sensor_profile",
    "last_hue_trim": "hue_trim",
    "last_aspect_ratio": "autocrop_ratio",
    "last_autocrop_mode": "autocrop_mode",
    "last_autocrop_offset": "autocrop_offset",
    "last_autocrop_rebate_trim": "autocrop_rebate_trim",
    "last_flip_horizontal": "flip_horizontal",
    "last_flip_vertical": "flip_vertical",
    "last_auto_exposure": "auto_exposure",
    "last_auto_normalize_contrast": "auto_normalize_contrast",
    "last_paper_dmin": "paper_dmin",
    "last_paper_black": "paper_black",
    "last_cast_removal_strength": "cast_removal_strength",
    "last_paper_profile": "paper_profile",
    "last_dust_remove": "dust_remove",
    "last_protect_original_metadata": "protect_original_metadata",
}

_LEGACY_DICT_KEYS = ("last_lab_config", "last_export_config")


def migrate_legacy(repo: IRepository) -> None:
    """Seed the snapshot from the superseded per-key store, once."""
    if repo.get_global_setting(STICKY_CONFIG_KEY) is not None:
        return
    flat: dict[str, Any] = {}
    for legacy_key, field in _LEGACY_KEYS.items():
        value = repo.get_global_setting(legacy_key)
        if value is not None:
            flat[field] = value
    for legacy_key in _LEGACY_DICT_KEYS:
        stored = repo.get_global_setting(legacy_key)
        if isinstance(stored, dict):
            flat.update(stored)
    # True Black was renamed to Paper Black, inverted.
    if "paper_black" not in flat:
        legacy_bpc = repo.get_global_setting("last_true_black")
        if legacy_bpc is not None:
            flat["paper_black"] = not bool(legacy_bpc)
    if not flat:
        return
    known = {f for r in all_rows() for f in r.fields}
    repo.save_global_setting(STICKY_CONFIG_KEY, {k: v for k, v in flat.items() if k in known})
