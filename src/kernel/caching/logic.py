import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict
from src.domain.types import ImageBuffer, ROI


@dataclass
class CacheEntry:
    """
    Represents a cached intermediate processing result.
    """

    config_hash: str
    data: ImageBuffer
    metrics: Dict[str, Any]
    active_roi: Optional[ROI] = None


def calculate_config_hash(config: Any) -> str:
    """
    Calculates a stable MD5 hash for a dataclass configuration.
    Values are sorted to ensure consistency.
    """
    if hasattr(config, "to_dict"):
        data = config.to_dict()
    elif hasattr(config, "__dataclass_fields__"):
        data = asdict(config)
    else:
        data = str(config)

    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()
