from typing import Optional
from src.kernel.caching.logic import CacheEntry


class PipelineCache:
    """
    Stores intermediate stage results for the active image.
    """

    source_hash: str = ""
    process_mode: str = ""

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
