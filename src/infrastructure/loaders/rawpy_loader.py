import rawpy
from typing import Any, ContextManager
from src.domain.interfaces import IImageLoader


class RawpyLoader(IImageLoader):
    """
    Standard RAW loader (libraw).
    """

    def load(self, file_path: str) -> ContextManager[Any]:
        from typing import cast

        return cast(ContextManager[Any], rawpy.imread(file_path))
