import os
import io
import numpy as np
import rawpy
import imageio.v3 as iio
from abc import ABC, abstractmethod
from typing import Any, List, Dict
from src.logging_config import get_logger

logger = get_logger(__name__)


class ImageContext(ABC):
    """
    Abstract interface mimicking the rawpy.RawPy behavior.
    """

    @abstractmethod
    def __enter__(self) -> "ImageContext": ...

    @abstractmethod
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...

    @abstractmethod
    def postprocess(self, **kwargs: Any) -> np.ndarray: ...


class NonStandardFileWrapper(ImageContext):
    """
    Wraps pre-loaded numpy data to provide a rawpy-compatible interface.
    """

    def __init__(self, data: np.ndarray):
        self.data = data

    def __enter__(self) -> "NonStandardFileWrapper":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def postprocess(self, **kwargs: Any) -> np.ndarray:
        bps = kwargs.get("output_bps", 8)
        half_size = kwargs.get("half_size", False)
        data = self.data
        if half_size:
            data = data[::2, ::2]

        if bps == 16:
            return (data * 65535.0).astype(np.uint16)
        return (data * 255.0).astype(np.uint8)


class BaseLoader(ABC):
    """
    Base class for all image loaders.
    """

    @abstractmethod
    def can_handle(self, file_path: str) -> bool: ...

    @abstractmethod
    def load(self, file_path: str) -> Any: ...


class RawpyLoader(BaseLoader):
    """
    Standard loader for digital RAW files (DNG, CR2, NEF, etc.)
    """

    def can_handle(self, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        # rawpy supported list is huge, we mostly exclude known non-raws
        return ext not in [".tif", ".tiff", ".jpg", ".jpeg", ".png"]

    def load(self, file_path: str) -> Any:
        # We return the actual rawpy object which already implements context manager
        with open(file_path, "rb") as f:
            return rawpy.imread(io.BytesIO(f.read()))


class TiffLoader(BaseLoader):
    """
    Loader for TIFF scans.
    """

    def can_handle(self, file_path: str) -> bool:
        return file_path.lower().endswith((".tif", ".tiff"))

    def load(self, file_path: str) -> NonStandardFileWrapper:
        img = iio.imread(file_path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        if img.dtype == np.uint8:
            f32 = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            f32 = img.astype(np.float32) / 65535.0
        else:
            f32 = np.clip(img.astype(np.float32), 0, 1)

        return NonStandardFileWrapper(f32)


class PakonLoader(BaseLoader):
    """
    Specialty loader for Pakon planar RAW files.
    """

    PAKON_SPECS: List[Dict[str, Any]] = [
        {"size": 36000000, "res": (2000, 3000), "desc": "F135 Plus High Res"},
        {"size": 9000000, "res": (1000, 1500), "desc": "F135 Plus Low Res"},
        {"size": 24000000, "res": (2000, 2000), "desc": "Pakon 2k Square"},
        {"size": 48000000, "res": (2000, 4000), "desc": "Pakon Panoram"},
        {"size": 72000000, "res": (4000, 3000), "desc": "F335 High Res"},
    ]

    def can_handle(self, file_path: str) -> bool:
        file_size = os.path.getsize(file_path)
        return any(abs(file_size - s["size"]) < 1024 for s in self.PAKON_SPECS)

    def load(self, file_path: str) -> NonStandardFileWrapper:
        file_size = os.path.getsize(file_path)
        spec = next(s for s in self.PAKON_SPECS if abs(file_size - s["size"]) < 1024)
        h, w = spec["res"]
        expected_pixels = h * w * 3

        with open(file_path, "rb") as f:
            data = np.fromfile(f, dtype="<u2", count=expected_pixels)

        data = data.reshape((3, h, w)).transpose((1, 2, 0))
        return NonStandardFileWrapper(data.astype(np.float32) / 65535.0)


class ImageLoaderFactory:
    """
    Dispatches the appropriate loader for a given file.
    """

    def __init__(self) -> None:
        # Ordered by specificity (Pakon first as it identifies by size)
        self.loaders: List[BaseLoader] = [PakonLoader(), TiffLoader(), RawpyLoader()]

    def get_loader(self, file_path: str) -> Any:
        for loader in self.loaders:
            if loader.can_handle(file_path):
                return loader.load(file_path)
        raise ValueError(f"No loader found for: {file_path}")


# Global instance
loader_factory = ImageLoaderFactory()
