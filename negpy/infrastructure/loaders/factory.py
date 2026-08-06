import os
from typing import Any, ContextManager, Tuple
from negpy.infrastructure.loaders.pakon_loader import PakonLoader
from negpy.infrastructure.loaders.tiff_loader import TiffLoader
from negpy.infrastructure.loaders.jpeg_loader import JpegLoader
from negpy.infrastructure.loaders.fff_loader import FffLoader, is_flextight_fff
from negpy.infrastructure.loaders.nef_loader import NefLoader, is_coolscan_nef
from negpy.infrastructure.loaders.noritsu_loader import NoritsuLoader, is_noritsu_raw
from negpy.infrastructure.loaders.jxl_loader import JxlLoader
from negpy.infrastructure.loaders.rawpy_loader import RawpyLoader
from negpy.infrastructure.loaders.constants import (
    SUPPORTED_TIFF_EXTENSIONS,
    SUPPORTED_JPEG_EXTENSIONS,
    SUPPORTED_JXL_EXTENSIONS,
)


class LoaderFactory:
    """
    Selects loader based on file ext/header.
    """

    def __init__(self) -> None:
        self._pakon = PakonLoader()
        self._tiff = TiffLoader()
        self._jpeg = JpegLoader()
        self._nef = NefLoader()
        self._fff = FffLoader()
        self._noritsu = NoritsuLoader()
        self._jxl = JxlLoader()
        self._rawpy = RawpyLoader()

    def get_loader(self, file_path: str, linear_raw: bool = False) -> Tuple[ContextManager[Any], dict]:
        ext = os.path.splitext(file_path)[1].lower()

        if ext in SUPPORTED_TIFF_EXTENSIONS:
            return self._tiff.load(file_path, linear_raw=linear_raw)

        if ext in SUPPORTED_JPEG_EXTENSIONS:
            return self._jpeg.load(file_path)

        if ext in SUPPORTED_JXL_EXTENSIONS:
            return self._jxl.load(file_path)

        if PakonLoader.can_handle(file_path):
            return self._pakon.load(file_path)

        if is_noritsu_raw(file_path):
            return self._noritsu.load(file_path)

        if is_coolscan_nef(file_path):
            return self._nef.load(file_path, linear_raw=linear_raw)

        if is_flextight_fff(file_path):
            return self._fff.load(file_path, linear_raw=linear_raw)

        return self._rawpy.load(file_path)


# Global instance for shared use
loader_factory = LoaderFactory()
