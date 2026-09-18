"""One-time migration of the legacy name-keyed normalization_rolls table onto the
Library roll each row matches by name. A saved baseline whose name matches no roll is
dropped -- nothing can address it once rolls own their own baselines directly. Best-
effort and idempotent -- guarded by a done flag, and it never raises into app startup.
"""

import json
import sqlite3
from contextlib import closing

from negpy.kernel.system.logging import get_logger
from negpy.services.assets import rolls

logger = get_logger(__name__)

_DONE_FLAG = "normalization_rolls_migrated_v1"


def migrate_legacy_normalization_rolls(repo) -> None:
    """Copies each normalization_rolls row onto the roll with a matching name, then
    drops the table."""
    if repo.get_global_setting(_DONE_FLAG):
        return
    try:
        by_name = {entry.get("name"): roll_id for roll_id, entry in rolls.saved_rolls(repo).items()}
        with closing(sqlite3.connect(repo.edits_db_path)) as conn, conn:
            row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='normalization_rolls'").fetchone()
            if row is not None:
                for name, floors_json, ceils_json, cast_json in conn.execute(
                    "SELECT name, floors_json, ceils_json, cast_json FROM normalization_rolls"
                ):
                    roll_id = by_name.get(name)
                    if roll_id is None:
                        continue
                    floors = tuple(json.loads(floors_json))
                    ceils = tuple(json.loads(ceils_json))
                    cast = tuple(json.loads(cast_json)) if cast_json else (0.0, 0.0, 0.0)
                    rolls.set_roll_normalization(repo, roll_id, floors, ceils, cast)
                conn.execute("DROP TABLE normalization_rolls")
    except Exception:
        logger.exception("Normalization-roll migration failed; continuing without it")
    repo.save_global_setting(_DONE_FLAG, True)
