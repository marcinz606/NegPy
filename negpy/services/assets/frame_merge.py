"""Carries an assembled frame's edit and roll records onto the TIFF merged from it."""

import os
from dataclasses import replace
from typing import Any, List

from negpy.domain.models import WorkspaceConfig
from negpy.features.flatfield.models import FlatFieldConfig
from negpy.features.rgbscan.models import RgbScanConfig
from negpy.features.stitch.models import StitchConfig
from negpy.services.assets import rolls
from negpy.services.assets.sidecar import sidecar_path_for, write_sidecar

#: Cards the merged file bakes, per kind; locked on the new frame so an Apply cannot re-apply them.
_BAKED_CARDS = {"rgb": ("sensor",), "stitch": ("sensor", "flatfield")}


def merged_edit(config: WorkspaceConfig, kind: str) -> WorkspaceConfig:
    """The frame's edit for its merged TIFF, without the other files and the baked corrections."""
    out = replace(
        config,
        rgbscan=RgbScanConfig(),
        process=replace(config.process, sensor_matrix=None, sensor_profile="None"),
    )
    if kind == "stitch":
        out = replace(out, stitch=StitchConfig(), flatfield=FlatFieldConfig())
    return out


def carry_edit(
    repo: Any,
    old_hash: str,
    old_path: str,
    new_hash: str,
    new_path: str,
    dropped_paths: List[str],
    config: WorkspaceConfig,
    kind: str,
    keep_source: bool = False,
) -> None:
    """Copy the frame's edit, forks, card locks, scene and roll membership to the merged file.
    *config* is the hydrated edit; with *keep_source* the source frame stays in its rolls."""

    def transform(c: WorkspaceConfig) -> WorkspaceConfig:
        return merged_edit(c, kind)

    owns_new = repo.copy_file_edits(old_hash, new_hash, new_path, transform)
    roll_ids = rolls.adopt_replacement(repo, old_hash, new_hash, old_path, new_path, dropped_paths, keep_source)
    for roll_id in roll_ids:
        repo.copy_file_edits(rolls.roll_edit_hash(old_hash, roll_id), rolls.roll_edit_hash(new_hash, roll_id), new_path, transform)
    member_rolls = rolls.rolls_containing_path(repo, new_path)
    # Resolved against the roll: a lock freezes its whole card, so each field must hold the roll's value.
    # A never-opened composite needs this row too, or it hydrates with the rig's flat field back on.
    if owns_new:
        roll_id = next(iter(member_rolls), None)
        resolved = config if roll_id is None else rolls.resolve_roll_config(repo, roll_id, old_hash, config)
        repo.save_file_settings(new_hash, merged_edit(resolved, kind), file_path=new_path)
    for roll_id in member_rolls:
        defaults = rolls.roll_defaults(repo, roll_id)
        for card in _BAKED_CARDS.get(kind, ()):
            if _roll_would_reapply(defaults, card):
                rolls.set_frame_override(repo, roll_id, new_hash, card, True)


def _roll_would_reapply(defaults: dict, card: str) -> bool:
    if card == "sensor":
        return defaults.get("sensor_matrix") is not None
    if card == "flatfield":
        return bool(defaults.get("apply"))
    return False


def carry_sidecar(old_path: str, new_path: str, config: WorkspaceConfig, kind: str) -> bool:
    if not os.path.exists(sidecar_path_for(old_path)):
        return False
    write_sidecar(new_path, merged_edit(config, kind))
    return True
