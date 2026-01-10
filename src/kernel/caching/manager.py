from typing import Optional
from src.kernel.caching.logic import CacheEntry


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
