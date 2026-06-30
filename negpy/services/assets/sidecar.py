import json
import os
import tempfile
from typing import Optional

from negpy.domain.models import WorkspaceConfig
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

SIDECAR_EXT = ".negpy"


def sidecar_path_for(source_path: str) -> str:
    """Sidecar path next to the source file: ``<basename>.negpy``."""
    base = os.path.splitext(os.path.basename(source_path))[0]
    return os.path.join(os.path.dirname(source_path), base + SIDECAR_EXT)


def write_sidecar(source_path: str, config: WorkspaceConfig) -> str:
    """Write the full edit (``config.to_dict()``) as JSON next to the source. Returns the path written."""
    path = sidecar_path_for(source_path)
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


def load_sidecar(source_path: str) -> Optional[WorkspaceConfig]:
    """Load edits from a sidecar next to the source file. None if absent or malformed."""
    path = sidecar_path_for(source_path)
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


def load_or_promote(repo, file_hash: str, source_path: str) -> Optional[WorkspaceConfig]:
    """DB first; on miss, load a beside-source sidecar and write it into the DB (promote)."""
    cfg = repo.load_file_settings(file_hash)
    if cfg is None:
        cfg = load_sidecar(source_path)
        if cfg is not None:
            repo.save_file_settings(file_hash, cfg)
    return cfg
