import os
from typing import Any, ContextManager
from src.infrastructure.loaders.pakon_loader import PakonLoader
from src.infrastructure.loaders.tiff_loader import TiffLoader
from src.infrastructure.loaders.rawpy_loader import RawpyLoader


class LoaderFactory:
    """
    Selects loader based on file ext/header.
    """

    def __init__(self) -> None:
        self._pakon = PakonLoader()
        self._tiff = TiffLoader()
        self._rawpy = RawpyLoader()

    def get_loader(self, file_path: str) -> ContextManager[Any]:
        ext = os.path.splitext(file_path)[1].lower()

        if PakonLoader.can_handle(file_path):
            return self._pakon.load(file_path)

        if ext in [".tif", ".tiff"]:
            return self._tiff.load(file_path)

        return self._rawpy.load(file_path)


# Global instance for shared use
loader_factory = LoaderFactory()
