import rawpy
from typing import Any, ContextManager, Tuple
from src.domain.interfaces import IImageLoader


class RawpyLoader(IImageLoader):
    """
    Standard RAW loader (libraw).
    """

    def load(self, file_path: str) -> Tuple[ContextManager[Any], dict]:
        raw = rawpy.imread(file_path)

        metadata = {
            "orientation": 0,
            "raw_flip": 0,
            "color_space": "Adobe RGB",
        }

        return raw, metadata
