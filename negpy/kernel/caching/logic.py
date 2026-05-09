import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict
from negpy.domain.types import ImageBuffer, ROI


@dataclass
class CacheEntry:
    """
    Intermediate pipeline stage result.
    """

    config_hash: str
    data: ImageBuffer
    metrics: Dict[str, Any]
    active_roi: Optional[ROI] = None


def calculate_config_hash(config: Any) -> str:
    """
    Stable hash of config state.

    Fast path: frozen dataclasses and tuples support __hash__; use str(hash(config))
    directly and skip JSON serialization.

    Fallback: configs with to_dict (e.g. WorkspaceConfig) or unhashable frozen
    dataclasses (lists/arrays in fields) go through JSON+MD5.
    """
    if not hasattr(config, "to_dict"):
        try:
            return str(hash(config))
        except TypeError:
            pass

    if hasattr(config, "to_dict"):
        data = config.to_dict()
    else:
        data = asdict(config)

    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()
