import os
from typing import Any, ContextManager, Tuple
from negpy.infrastructure.loaders.pakon_loader import PakonLoader
from negpy.infrastructure.loaders.tiff_loader import TiffLoader
from negpy.infrastructure.loaders.jpeg_loader import JpegLoader
from negpy.infrastructure.loaders.rawpy_loader import RawpyLoader
from negpy.infrastructure.loaders.constants import (
    SUPPORTED_TIFF_EXTENSIONS,
    SUPPORTED_JPEG_EXTENSIONS,
)


class LoaderFactory:
    """
    Selects loader based on file ext/header.
    """

    def __init__(self) -> None:
        self._pakon = PakonLoader()
        self._tiff = TiffLoader()
        self._jpeg = JpegLoader()
        self._rawpy = RawpyLoader()

    def get_loader(self, file_path: str) -> Tuple[ContextManager[Any], dict]:
        ext = os.path.splitext(file_path)[1].lower()

        if ext in SUPPORTED_TIFF_EXTENSIONS:
            return self._tiff.load(file_path)

        if ext in SUPPORTED_JPEG_EXTENSIONS:
            return self._jpeg.load(file_path)

        if PakonLoader.can_handle(file_path):
            return self._pakon.load(file_path)

        return self._rawpy.load(file_path)


# Global instance for shared use
loader_factory = LoaderFactory()
