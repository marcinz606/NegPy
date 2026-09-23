import os
from collections.abc import Callable
from typing import Any, ContextManager, Optional, Tuple
from negpy.infrastructure.loaders.pakon_loader import PakonLoader
from negpy.infrastructure.loaders.tiff_loader import TiffLoader
from negpy.infrastructure.loaders.jpeg_loader import JpegLoader
from negpy.infrastructure.loaders.fff_loader import FffLoader, is_flextight_fff
from negpy.infrastructure.loaders.nef_loader import NefLoader, is_coolscan_nef
from negpy.infrastructure.loaders.noritsu_loader import NoritsuLoader, is_noritsu_raw
from negpy.infrastructure.loaders.jxl_loader import JxlLoader
from negpy.infrastructure.loaders.memory import PreviewMemoryEstimate, estimate_preview_memory
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

    def _select_loader(self, file_path: str) -> Any:
        ext = os.path.splitext(file_path)[1].lower()

        if ext in SUPPORTED_TIFF_EXTENSIONS:
            return self._tiff

        if ext in SUPPORTED_JPEG_EXTENSIONS:
            return self._jpeg

        if ext in SUPPORTED_JXL_EXTENSIONS:
            return self._jxl

        if PakonLoader.can_handle(file_path):
            return self._pakon

        if is_noritsu_raw(file_path):
            return self._noritsu

        if is_coolscan_nef(file_path):
            return self._nef

        if is_flextight_fff(file_path):
            return self._fff

        return self._rawpy

    def get_loader(
        self,
        file_path: str,
        linear_raw: bool = False,
        positive_source: bool = False,
        preview_max_edge: Optional[int] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> Tuple[ContextManager[Any], dict]:
        loader = self._select_loader(file_path)

        if loader is self._tiff:
            return loader.load(file_path, linear_raw=linear_raw, positive_source=positive_source)
        if loader is self._nef or loader is self._fff:
            return loader.load(file_path, linear_raw=linear_raw)
        if loader is self._rawpy:
            return loader.load(file_path, preview_max_edge=preview_max_edge, should_cancel=should_cancel)
        return loader.load(file_path)

    def load_bounded_preview(
        self,
        file_path: str,
        max_edge: int,
        *,
        fast_only: bool = False,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> Optional[Any]:
        """Ask the selected format loader for a memory-bounded preview."""
        return self._select_loader(file_path).load_bounded_preview(
            file_path,
            max_edge,
            fast_only=fast_only,
            should_cancel=should_cancel,
        )

    def estimate_preview_memory(self, file_path: str, max_edge: int) -> PreviewMemoryEstimate:
        """Estimate a loader's preview working set without decoding image pixels."""
        loader = self._select_loader(file_path)
        if loader is self._jpeg:
            profile = "display"
        elif loader in (self._rawpy, self._fff, self._jxl):
            profile = "raw"
        else:
            profile = "scan"
        return estimate_preview_memory(file_path, max_edge, profile)

    def estimate_linear_preview_prefetch_memory(self, file_path: str, max_edge: int) -> PreviewMemoryEstimate:
        """Estimate a neighbor decode before pixel decode: the segmented LinearRaw path where
        it applies, else the loader's whole-decode working set."""
        loader = self._select_loader(file_path)
        if loader is self._rawpy:
            streamed = loader.estimate_cancellable_linear_preview_memory(file_path, max_edge)
            if streamed is not None:
                return streamed
        return self.estimate_preview_memory(file_path, max_edge)


# Global instance for shared use
loader_factory = LoaderFactory()
