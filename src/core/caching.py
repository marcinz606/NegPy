import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict
from src.core.types import ImageBuffer, ROI


@dataclass
class CacheEntry:
    """
    Represents a cached intermediate processing result.
    """

    config_hash: str
    data: ImageBuffer
    metrics: Dict[str, Any]
    active_roi: Optional[ROI] = None


class PipelineCache:
    """
    Holds intermediate results of the processing pipeline for the ACTIVE image.
    This cache is reset when switching source files.
    """

    source_hash: str = ""

    # Checkpoints
    base: Optional[CacheEntry] = None
    exposure: Optional[CacheEntry] = None
    retouch: Optional[CacheEntry] = None
    lab: Optional[CacheEntry] = None

    def clear(self) -> None:
        self.base = None
        self.exposure = None
        self.retouch = None
        self.lab = None
        self.source_hash = ""


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
