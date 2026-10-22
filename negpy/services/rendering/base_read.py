"""The clear-base read, measured once per source and shared by both engines and the preview prefetch."""

import threading
from collections import OrderedDict
from typing import Any, Hashable, Optional

from negpy.domain.types import ImageBuffer
from negpy.features.exposure.normalization import measure_base_log

CAPACITY = 64
# A cached None is a source known to have no base, so a miss needs its own marker.
NOT_READ: Any = object()

_reads: "OrderedDict[Hashable, Optional[tuple]]" = OrderedDict()
_lock = threading.Lock()


def cached_base_log(key: Optional[Hashable]) -> Any:
    """The stored read for `key`, or NOT_READ. Never measures."""
    if key is None:
        return NOT_READ
    with _lock:
        if key not in _reads:
            return NOT_READ
        _reads.move_to_end(key)
        return _reads[key]


def base_log_for(key: Optional[Hashable], img: ImageBuffer) -> Optional[tuple]:
    """The stored read for `key`, else a new one off `img`. A None key measures and stores nothing."""
    found = cached_base_log(key)
    if found is not NOT_READ:
        return found
    # Outside the lock: the prefetch and the render may read one source twice, never wait on each other.
    value = measure_base_log(img)
    if key is not None:
        with _lock:
            _reads[key] = value
            _reads.move_to_end(key)
            while len(_reads) > CAPACITY:
                _reads.popitem(last=False)
    return value


def clear() -> None:
    with _lock:
        _reads.clear()
