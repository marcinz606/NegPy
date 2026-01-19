import numpy as np
import imageio.v3 as iio
from typing import Any, ContextManager, Tuple
from src.domain.interfaces import IImageLoader
from src.kernel.image.logic import uint8_to_float32, uint16_to_float32


class NonStandardFileWrapper:
    """
    numpy -> rawpy-like interface.
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


class TiffLoader(IImageLoader):
    """
    Loader for TIFF scans.
    """

    def load(self, file_path: str) -> Tuple[ContextManager[Any], dict]:
        img = iio.imread(file_path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        if img.dtype == np.uint8:
            f32 = uint8_to_float32(np.ascontiguousarray(img))
        elif img.dtype == np.uint16:
            f32 = uint16_to_float32(np.ascontiguousarray(img))
        else:
            f32 = np.clip(img.astype(np.float32), 0, 1)

        metadata = {"orientation": 0, "color_space": "Adobe RGB"}
        return NonStandardFileWrapper(f32), metadata
