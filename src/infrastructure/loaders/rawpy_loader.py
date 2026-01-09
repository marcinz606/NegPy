import rawpy
from typing import Any, ContextManager
from src.core.interfaces import IImageLoader


class RawpyLoader(IImageLoader):
    """
    Standard loader for digital RAW files (DNG, CR2, NEF, etc.)
    """

    def load(self, file_path: str) -> ContextManager[Any]:
        from typing import cast

        return cast(ContextManager[Any], rawpy.imread(file_path))
