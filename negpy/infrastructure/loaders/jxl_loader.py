from collections.abc import Callable
from typing import Any, ContextManager, Optional, Tuple

import imagecodecs
import numpy as np
from PIL import Image

from negpy.domain.interfaces import IImageLoader
from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper
from negpy.kernel.image.logic import uint8_to_float32, uint16_to_float32


class JxlLoader(IImageLoader):
    def load(self, file_path: str) -> Tuple[ContextManager[Any], dict]:
        with open(file_path, "rb") as f:
            data = f.read()

        img = imagecodecs.jpegxl_decode(data)

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        if img.dtype == np.uint8:
            f32 = uint8_to_float32(np.ascontiguousarray(img))
        elif img.dtype == np.uint16:
            f32 = uint16_to_float32(np.ascontiguousarray(img))
        elif img.dtype == np.float32:
            f32 = np.clip(img, 0.0, 1.0)
        else:
            f32 = np.clip(img.astype(np.float32), 0.0, 1.0)

        metadata = {"orientation": 1, "color_space": None, "icc_profile": None, "ir": None}
        return NonStandardFileWrapper(f32), metadata

    def load_bounded_preview(
        self,
        file_path: str,
        max_edge: int,
        *,
        fast_only: bool = False,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> Optional[Image.Image]:
        if should_cancel is not None and should_cancel():
            raise InterruptedError("preview cancelled")
        return None
