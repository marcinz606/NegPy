import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict
from src.core.types import ImageBuffer


@dataclass
class CacheEntry:
    """
    Represents a cached intermediate processing result.
    """

    config_hash: str
    data: ImageBuffer
    metrics: Dict[str, Any]


class PipelineCache:
    """
    Holds intermediate results of the processing pipeline for the ACTIVE image.
    This cache is reset when switching source files.
    """

    source_hash: str = ""  # Hash of the currently loaded RAW file

    # Checkpoints
    base: Optional[CacheEntry] = None  # After Geometry + Norm (Stage 1)
    exposure: Optional[CacheEntry] = None  # After Photometric (Stage 2)
    retouch: Optional[CacheEntry] = None  # After Retouch (Stage 3)
    lab: Optional[CacheEntry] = None  # After Lab (Stage 4)

    def clear(self) -> None:
        """Invalidates all cache entries."""
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
