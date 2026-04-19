import imageio.v3 as iio
import numpy as np
import tifffile
from typing import Any, ContextManager, Tuple
from negpy.domain.interfaces import IImageLoader
from negpy.domain.models import ColorSpace
from negpy.kernel.image.logic import uint8_to_float32, uint16_to_float32
from negpy.infrastructure.loaders.helpers import NonStandardFileWrapper, identify_color_space_from_icc


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

        icc_bytes: bytes | None = None
        try:
            with tifffile.TiffFile(file_path) as tif:
                tag = tif.pages[0].tags.get("InterColorProfile")
                if tag is not None and tag.value:
                    icc_bytes = bytes(tag.value)
        except Exception:
            icc_bytes = None

        color_space = identify_color_space_from_icc(icc_bytes) or ColorSpace.SRGB.value
        metadata = {"orientation": 0, "color_space": color_space, "icc_profile": icc_bytes}
        return NonStandardFileWrapper(f32), metadata
