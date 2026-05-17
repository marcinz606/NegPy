from typing import Any, ContextManager, Tuple

import imageio.v3 as iio
import numpy as np
from PIL import Image

from negpy.domain.interfaces import IImageLoader
from negpy.domain.models import ColorSpace
from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper, identify_color_space_from_icc
from negpy.kernel.image.logic import srgb_to_linear, uint8_to_float32


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

        icc_bytes: bytes | None = None
        try:
            with Image.open(file_path) as pil_img:
                icc_bytes = pil_img.info.get("icc_profile")
        except Exception:
            icc_bytes = None

        color_space = identify_color_space_from_icc(icc_bytes) or ColorSpace.SRGB.value
        if color_space == ColorSpace.SRGB.value:
            f32 = srgb_to_linear(f32)
        metadata = {"orientation": 0, "color_space": color_space, "icc_profile": icc_bytes, "ir": None}
        return NonStandardFileWrapper(f32), metadata
