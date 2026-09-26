"""One-time lock for fields that became roll defaults after frames already carried their
own value: White/Black Point and their per-layer trims (Normalization), and Highlight
Recovery (Raw Decode). No roll has a default for them yet, so a frame's own value still
renders; the first Roll push would overwrite it on every frame that is not locked. This
locks the card on each frame whose saved value differs from the default, in every roll
that holds it. Best-effort and idempotent: guarded by a done flag, and it never raises
into app startup.
"""

import json
import sqlite3
from contextlib import closing

from negpy.features.exposure.models import ExposureConfig
from negpy.features.process.models import ProcessConfig, cast_removal_for_mode
from negpy.kernel.system.logging import get_logger
from negpy.services.assets import rolls

logger = get_logger(__name__)

_DONE_FLAG = "roll_field_locks_migrated_v1"
_BASELINE_SPLIT_FLAG = "baseline_card_split_v1"
_CAST_REMOVAL_FLAG = "cast_removal_roll_card_v1"

NEW_ROLL_FIELDS = {
    "process": (
        "white_point_offset",
        "black_point_offset",
        "white_point_trim_red",
        "white_point_trim_green",
        "white_point_trim_blue",
        "black_point_trim_red",
        "black_point_trim_green",
        "black_point_trim_blue",
    ),
    "demosaic": ("highlight_reconstruction",),
}


def _rolls_for_row(repo, file_hash: str, file_path: str) -> list:
    """A fork belongs to its own roll alone; a shared edit to every roll holding its path."""
    base = rolls.unforked_hash(file_hash)
    if base != file_hash:
        return [file_hash[len(base) + len(rolls._FORK_SEP) :]]
    return rolls.rolls_containing_path(repo, file_path) if file_path else []


def migrate_new_roll_field_locks(repo) -> None:
    if repo.get_global_setting(_DONE_FLAG):
        return
    try:
        defaults = ProcessConfig()
        locks = []
        with closing(sqlite3.connect(repo.edits_db_path)) as conn:
            for file_hash, settings_json, file_path in conn.execute("SELECT file_hash, settings_json, file_path FROM file_settings"):
                try:
                    data = json.loads(settings_json) if settings_json else {}
                except (ValueError, TypeError):
                    continue
                cards = [
                    card
                    for card, fields in NEW_ROLL_FIELDS.items()
                    if any(data.get(f, getattr(defaults, f)) != getattr(defaults, f) for f in fields)
                ]
                if not cards:
                    continue
                for roll_id in _rolls_for_row(repo, file_hash, file_path):
                    locks.extend((roll_id, rolls.unforked_hash(file_hash), card) for card in cards)
        for roll_id, file_hash, card in locks:
            rolls.set_frame_override(repo, roll_id, file_hash, card, True)
    except Exception:
        logger.exception("New roll-field lock migration failed; continuing without it")
    repo.save_global_setting(_DONE_FLAG, True)


def migrate_baseline_card_split(repo) -> None:
    """Use Luma/Color Average left the Normalization (``process``) card for their own
    ``baseline`` card. A frame locked on the old card keeps both halves of that lock, so
    nothing it renders changes. Runs before migrate_new_roll_field_locks, whose new
    ``process`` locks are about White/Black Point alone."""
    if repo.get_global_setting(_BASELINE_SPLIT_FLAG):
        return
    try:
        for roll_id, entry in rolls.saved_rolls(repo).items():
            for file_hash, cards in entry.get("frame_overrides", {}).items():
                if "process" in cards:
                    rolls.set_frame_override(repo, roll_id, file_hash, "baseline", True)
    except Exception:
        logger.exception("Roll Analysis card split migration failed; continuing without it")
    repo.save_global_setting(_BASELINE_SPLIT_FLAG, True)


def migrate_cast_removal_roll_locks(repo) -> None:
    """Cast Removal became a roll default (the ``cast_removal`` card). Locks that card on
    every frame whose saved strength differs from its own film mode's default, so the
    first Roll push does not overwrite a strength set by hand. Same guards as
    migrate_new_roll_field_locks."""
    if repo.get_global_setting(_CAST_REMOVAL_FLAG):
        return
    try:
        default_mode = ProcessConfig().process_mode
        default_strength = float(ExposureConfig.cast_removal_strength)
        locks = []
        with closing(sqlite3.connect(repo.edits_db_path)) as conn:
            for file_hash, settings_json, file_path in conn.execute("SELECT file_hash, settings_json, file_path FROM file_settings"):
                try:
                    data = json.loads(settings_json) if settings_json else {}
                except (ValueError, TypeError):
                    continue
                if "cast_removal_strength" not in data:
                    continue
                mode_default = cast_removal_for_mode(data.get("process_mode", default_mode), default_strength)
                if float(data["cast_removal_strength"]) == mode_default:
                    continue
                for roll_id in _rolls_for_row(repo, file_hash, file_path):
                    locks.append((roll_id, rolls.unforked_hash(file_hash)))
        for roll_id, file_hash in locks:
            rolls.set_frame_override(repo, roll_id, file_hash, "cast_removal", True)
    except Exception:
        logger.exception("Cast Removal roll-card lock migration failed; continuing without it")
    repo.save_global_setting(_CAST_REMOVAL_FLAG, True)
