import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict
from src.domain.types import ImageBuffer, ROI


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
    Stable MD5 of config state.
    """
    if hasattr(config, "to_dict"):
        data = config.to_dict()
    elif hasattr(config, "__dataclass_fields__"):
        data = asdict(config)
    else:
        data = str(config)

    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()
