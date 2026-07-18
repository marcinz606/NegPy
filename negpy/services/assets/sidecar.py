import json
import os
import tempfile
from typing import Optional

from negpy.domain.models import WorkspaceConfig
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

SIDECAR_EXT = ".negpy"


def sidecar_path_for(source_path: str, half: int = 0) -> str:
    """Sidecar path next to the source file: ``<basename>.negpy`` (``<basename>.<half>.negpy`` for half-frame assets)."""
    base = os.path.splitext(os.path.basename(source_path))[0]
    suffix = f".{half}" if half else ""
    return os.path.join(os.path.dirname(source_path), base + suffix + SIDECAR_EXT)


def write_sidecar(source_path: str, config: WorkspaceConfig, half: int = 0) -> str:
    """Write the full edit (``config.to_dict()``) as JSON next to the source. Returns the path written."""
    path = sidecar_path_for(source_path, half)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = json.dumps(config.to_dict(), default=str, indent=2)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(path), delete=False, suffix=".part", encoding="utf-8") as tmp:
            tmp_path = tmp.name
            tmp.write(payload)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return path


def load_sidecar(source_path: str, half: int = 0) -> Optional[WorkspaceConfig]:
    """Load edits from a sidecar next to the source file. None if absent or malformed."""
    path = sidecar_path_for(source_path, half)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f_in:
            data = json.load(f_in)
        if not isinstance(data, dict):
            return None
        return WorkspaceConfig.from_flat_dict(data)
    except Exception as exc:
        logger.warning("Failed to load sidecar %s: %s", path, exc)
        return None


def load_or_promote(repo, file_hash: str, source_path: str, half: int = 0) -> Optional[WorkspaceConfig]:
    """DB first; on miss, try path-based fallback (handles EXIF-modified files),
    then fall back to sidecar. Re-homes on successful path match."""
    cfg = repo.load_file_settings(file_hash)
    if cfg is not None:
        return cfg

    # Path-based fallback: hash changed due to EXIF edits. Half-frame assets share
    # a path, so a path match would steal the sibling half's edit — skip it.
    if not half:
        path_result = repo.load_file_settings_by_path(source_path)
        if path_result is not None:
            old_hash, cfg = path_result
            repo.rehome_file_settings(old_hash, file_hash, source_path)
            return cfg

    # Sidecar fallback
    cfg = load_sidecar(source_path, half)
    if cfg is not None:
        repo.save_file_settings(file_hash, cfg, file_path=source_path)
    return cfg
