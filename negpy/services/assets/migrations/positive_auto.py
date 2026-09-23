"""One-time correction for a frame already marked Positive before Auto Density/Auto
Grade were restated for the transparency transfer path. Both used to be inert there,
so a Positive frame's saved edit carries whatever they happened to be -- almost always
still the negative default (on), since nothing gave anyone a reason to touch them.
Left alone, every one of those frames would suddenly render metered the moment this
shipped. This bakes the same rule AppController.set_positive_source now applies
(auto_meter_for_positive_source) into every already-saved row once, so a user's own
later choice, before or after this runs, is never touched again. Best-effort and
idempotent -- guarded by a done flag, and it never raises into app startup.
"""

import json
import sqlite3
from contextlib import closing

from negpy.features.process.models import ProcessMode
from negpy.kernel.system.logging import get_logger
from negpy.services.assets import rolls

logger = get_logger(__name__)

_DONE_FLAG = "auto_meter_positive_migrated_v1"


def migrate_auto_meter_for_positive_frames(repo) -> None:
    if repo.get_global_setting(_DONE_FLAG):
        return
    try:
        # positive_source predates the roll's "film" card -- no saved row has ever
        # locked it away, so a frame's own value or its roll's default (if either is
        # set) is the complete answer for every row this migration will ever see.
        positive_rolls = {rid for rid, entry in rolls.saved_rolls(repo).items() if entry.get("defaults", {}).get("positive_source")}
        with closing(sqlite3.connect(repo.edits_db_path)) as conn, conn:
            for file_hash, settings_json, file_path in conn.execute("SELECT file_hash, settings_json, file_path FROM file_settings"):
                if not settings_json:
                    continue
                try:
                    data = json.loads(settings_json)
                except (ValueError, TypeError):
                    continue
                # A negative-mode row loses the flag on load, so it meters again.
                effective_positive = str(data.get("process_mode", "")) == ProcessMode.E6 and (
                    bool(data.get("positive_source"))
                    or (bool(file_path) and bool(positive_rolls & set(rolls.rolls_containing_path(repo, file_path))))
                )
                if not effective_positive:
                    continue
                changed = False
                for field in ("auto_exposure", "auto_normalize_contrast"):
                    if data.get(field, True):
                        data[field] = False
                        changed = True
                if changed:
                    conn.execute(
                        "UPDATE file_settings SET settings_json = ? WHERE file_hash = ?", (json.dumps(data, default=str), file_hash)
                    )
    except Exception:
        logger.exception("Auto Density/Grade Positive-frame migration failed; continuing without it")
    repo.save_global_setting(_DONE_FLAG, True)
