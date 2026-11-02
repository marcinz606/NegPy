from collections.abc import Callable
from typing import Any, ContextManager, Optional, Tuple

import imageio.v3 as iio
import numpy as np
from PIL import Image

from negpy.domain.interfaces import IImageLoader
from negpy.domain.models import ColorSpace
from negpy.features.process.models import SourceEncoding
from negpy.infrastructure.loaders.helpers import (
    NonStandardFileWrapper,
    fit_bounded_preview,
    identify_color_space_from_icc,
    read_orientation,
)
from negpy.features.process.capture_color import source_cam_xyz
from negpy.infrastructure.loaders.source_encoding import PROFILE, decode_positive, resolve_decode
from negpy.kernel.image.logic import srgb_to_linear, uint8_to_float32


class JpegLoader(IImageLoader):
    """
    Loader for JPEG scans.
    """

    def load(self, file_path: str, positive_decode: str = "") -> Tuple[ContextManager[Any], dict]:
        """A non-empty *positive_decode* (path.positive_decode) decodes the file as a Positive. A JPEG
        has no Encoding control, so it always takes Auto."""
        img = iio.imread(file_path)
        # Adobe CMYK JPEGs store inverted ink; PIL's conversion knows that and imageio does not.
        with Image.open(file_path) as probe:
            if probe.mode == "CMYK":
                img = np.asarray(probe.convert("RGB"))
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        icc_bytes: bytes | None = None
        try:
            with Image.open(file_path) as pil_img:
                icc_bytes = pil_img.info.get("icc_profile")
        except Exception:
            icc_bytes = None

        if positive_decode:
            ran, decode = resolve_decode(SourceEncoding.AUTO, icc_bytes, img.dtype)
            metadata = {
                "orientation": read_orientation(file_path),
                "color_space": identify_color_space_from_icc(icc_bytes) if ran == PROFILE else None,
                "icc_profile": icc_bytes,
                "ir": None,
                "source_encoding": ran,
            }
            linear, to_working = decode_positive(np.ascontiguousarray(img), decode)
            wrapper = NonStandardFileWrapper(linear)
            wrapper.rgb_xyz_matrix = source_cam_xyz(to_working)
            return wrapper, metadata

        if img.dtype == np.uint8:
            f32 = uint8_to_float32(np.ascontiguousarray(img))
        else:
            f32 = np.clip(img.astype(np.float32) / 255.0, 0, 1)

        color_space = identify_color_space_from_icc(icc_bytes) or ColorSpace.SRGB.value
        if color_space == ColorSpace.SRGB.value:
            f32 = srgb_to_linear(f32)
        metadata = {"orientation": read_orientation(file_path), "color_space": color_space, "icc_profile": icc_bytes, "ir": None}
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
        with Image.open(file_path) as image:
            image.draft("RGB", (max_edge, max_edge))
            if image.width * image.height * 3 > 64 * 1024 * 1024:
                return None
            return fit_bounded_preview(image, max_edge, read_orientation(file_path))
