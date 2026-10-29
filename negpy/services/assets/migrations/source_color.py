"""One-time re-measure of source_color on B&W frames.

The chroma-spread test replaced a channel-correlation test that read many camera captures of
B&W film as color film. source_color is measured once, when it is None, so this clears it on
B&W rows with no contrast filter and the next load measures it again. Without a filter the
print takes the default weights on either source, so no render changes. Best-effort and
idempotent: guarded by a done flag, and it never raises into app startup.
"""

import json
import sqlite3
from contextlib import closing

from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

_DONE_FLAG = "source_color_chroma_spread_v1"
#: "B&W" is the legacy spelling of "B&W Negative".
_BW_MODES = ("B&W Negative", "B&W")


def _clear_rows(conn: sqlite3.Connection, table: str, key_cols: tuple[str, ...]) -> None:
    cols = ", ".join(key_cols)
    where = " AND ".join(f"{c} = ?" for c in key_cols)
    for *keys, settings_json in conn.execute(f"SELECT {cols}, settings_json FROM {table}").fetchall():
        if not settings_json:
            continue
        try:
            data = json.loads(settings_json)
        except (ValueError, TypeError):
            continue
        if str(data.get("process_mode", "")) not in _BW_MODES or data.get("source_color") is None:
            continue
        if str(data.get("bw_filter", "none")) != "none":
            continue
        data["source_color"] = None
        conn.execute(f"UPDATE {table} SET settings_json = ? WHERE {where}", (json.dumps(data, default=str), *keys))


def migrate_bw_source_color(repo) -> None:
    """Clear source_color on B&W rows with no filter, once, so the next load measures it again."""
    if repo.get_global_setting(_DONE_FLAG):
        return
    try:
        with closing(sqlite3.connect(repo.edits_db_path)) as conn, conn:
            _clear_rows(conn, "file_settings", ("file_hash",))
            _clear_rows(conn, "edit_history", ("file_hash", "step_index"))
            _clear_rows(conn, "work_prints", ("file_hash", "name"))
    except Exception:
        logger.exception("source_color migration failed; continuing without it")
    repo.save_global_setting(_DONE_FLAG, True)
