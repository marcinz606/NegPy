"""One-time migration of the legacy name-keyed normalization_rolls table onto the
Library roll each row matches by name. Best-effort and idempotent -- it never raises
into app startup.

A row whose name matches no roll is left where it is and the migration stays pending:
rolls are recognized on import, so at first launch after the upgrade there are none to
match yet, and dropping the row then would discard a baseline the user could still
address once the folder is imported.
"""

import json
import sqlite3
from contextlib import closing

from negpy.kernel.system.logging import get_logger
from negpy.services.assets import rolls

logger = get_logger(__name__)

_DONE_FLAG = "normalization_rolls_migrated_v1"


def migrate_legacy_normalization_rolls(repo) -> None:
    """Copies each normalization_rolls row onto the roll with a matching name, dropping
    the table once every row has found one."""
    if repo.get_global_setting(_DONE_FLAG):
        return
    try:
        by_name = {entry.get("name"): roll_id for roll_id, entry in rolls.saved_rolls(repo).items()}
        with closing(sqlite3.connect(repo.edits_db_path)) as conn, conn:
            row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='normalization_rolls'").fetchone()
            if row is None:
                repo.save_global_setting(_DONE_FLAG, True)
                return
            unmatched = 0
            for name, floors_json, ceils_json, cast_json in conn.execute(
                "SELECT name, floors_json, ceils_json, cast_json FROM normalization_rolls"
            ).fetchall():
                roll_id = by_name.get(name)
                if roll_id is None:
                    unmatched += 1
                    continue
                floors = tuple(json.loads(floors_json))
                ceils = tuple(json.loads(ceils_json))
                cast = tuple(json.loads(cast_json)) if cast_json else (0.0, 0.0, 0.0)
                rolls.set_roll_normalization(repo, roll_id, floors, ceils, cast)
                conn.execute("DELETE FROM normalization_rolls WHERE name = ?", (name,))
            if unmatched:
                return
            conn.execute("DROP TABLE normalization_rolls")
    except Exception:
        logger.exception("Normalization-roll migration failed; continuing without it")
        return
    repo.save_global_setting(_DONE_FLAG, True)
