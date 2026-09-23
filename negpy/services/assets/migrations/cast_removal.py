"""One-time cleanup of Cast Removal on legacy slide edits.

Cast Removal reached Transparency/E-6 after every such edit had already been saved
carrying the shipped default (0.5) — a value nobody chose, since the control had
nothing to do on a slide yet. This sweeps file_settings, edit_history and work_prints
once, zeroing cast_removal_strength wherever a slide row still carries that default,
so old edits render the way they always did. A value saved after this ran is a
deliberate choice and is left alone.

Best-effort and idempotent — guarded by a done flag, and it never raises into app
startup.
"""

import json
import sqlite3
from contextlib import closing

from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

_DONE_FLAG = "cast_removal_slide_migrated_v1"
#: Legacy raw process_mode strings for a slide, "E-6" predating the rename to
#: "Transparency".
_SLIDE_MODES = ("Transparency", "E-6")
#: The default every legacy slide row was saved with. Fixed, not ExposureConfig's current
#: default: the rows this sweep repairs were all written under it.
_SHIPPED_CAST_STRENGTH = 0.5


def _zero_legacy_rows(conn: sqlite3.Connection, table: str, key_cols: tuple[str, ...]) -> None:
    cols = ", ".join(key_cols)
    rows = conn.execute(f"SELECT {cols}, settings_json FROM {table}").fetchall()
    where = " AND ".join(f"{c} = ?" for c in key_cols)
    for *keys, settings_json in rows:
        if not settings_json:
            continue
        try:
            data = json.loads(settings_json)
        except (ValueError, TypeError):
            continue
        if str(data.get("process_mode", "")) not in _SLIDE_MODES:
            continue
        if float(data.get("cast_removal_strength", 0.0)) != _SHIPPED_CAST_STRENGTH:
            continue
        data["cast_removal_strength"] = 0.0
        conn.execute(f"UPDATE {table} SET settings_json = ? WHERE {where}", (json.dumps(data, default=str), *keys))


def migrate_legacy_slide_cast_removal(repo) -> None:
    """Zero the shipped-default Cast Removal on legacy slide edits, once."""
    if repo.get_global_setting(_DONE_FLAG):
        return
    try:
        with closing(sqlite3.connect(repo.edits_db_path)) as conn, conn:
            _zero_legacy_rows(conn, "file_settings", ("file_hash",))
            _zero_legacy_rows(conn, "edit_history", ("file_hash", "step_index"))
            _zero_legacy_rows(conn, "work_prints", ("file_hash", "name"))
    except Exception:
        logger.exception("Cast Removal migration failed; continuing without it")
    repo.save_global_setting(_DONE_FLAG, True)
