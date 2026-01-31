import numpy as np
import imageio.v3 as iio
from typing import Any, ContextManager, Tuple
from src.domain.interfaces import IImageLoader
from src.kernel.image.logic import uint8_to_float32
from src.infrastructure.loaders.helpers import NonStandardFileWrapper


class JpegLoader(IImageLoader):
    """
    Loader for JPEG scans.
    """

    def load(self, file_path: str) -> Tuple[ContextManager[Any], dict]:
        img = iio.imread(file_path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        if img.dtype == np.uint8:
            f32 = uint8_to_float32(np.ascontiguousarray(img))
        else:
            f32 = np.clip(img.astype(np.float32) / 255.0, 0, 1)

        metadata = {"orientation": 0, "color_space": "sRGB"}
        return NonStandardFileWrapper(f32), metadata
