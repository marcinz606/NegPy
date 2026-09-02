import os
import tomllib
from typing import Any, Optional

# (path) -> (mtime_ns, size, parsed). Profile folders are re-listed on every sidebar sync, so the
# parse is cached per file and only the stat runs per sync.
_cache: dict[str, tuple[int, int, Optional[dict]]] = {}


def load_toml_cached(path: str) -> Optional[dict[str, Any]]:
    """Parsed TOML for ``path``, re-read when the file changes; None when unreadable or invalid."""
    try:
        st = os.stat(path)
    except OSError:
        _cache.pop(path, None)
        return None
    hit = _cache.get(path)
    if hit is not None and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
        return hit[2]
    try:
        with open(path, "rb") as f:
            data: Optional[dict] = tomllib.load(f)
    except Exception:
        data = None
    _cache[path] = (st.st_mtime_ns, st.st_size, data)
    return data
