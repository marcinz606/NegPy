from typing import Optional

import numpy as np
from PyQt6.QtGui import QImage
from negpy.infrastructure.display.color_mgmt import apply_display_transform
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE
from negpy.kernel.image.logic import float_to_uint8


class ImageConverter:
    """
    Handles conversion between NumPy/PIL and PyQt6 image types.
    """

    @staticmethod
    def to_qimage(buffer: np.ndarray, color_space: str = WORKING_COLOR_SPACE, monitor_icc_bytes: Optional[bytes] = None) -> QImage:
        """
        Safely converts a NumPy float32 or uint8 buffer to a QImage.
        Performs a deep copy to prevent memory corruption (harsh noise).

        ``color_space`` is the working space of ``buffer``; it is color-managed to
        the monitor's display profile (``monitor_icc_bytes``, or sRGB when None) so
        the preview matches a color-managed view of the export.
        """
        # 1. Color-manage working space → display profile, then quantize to uint8
        if buffer.dtype == np.float32:
            buffer = apply_display_transform(buffer, color_space, monitor_icc_bytes)
            u8_buffer = float_to_uint8(buffer)
        else:
            u8_buffer = buffer

        # 2. Expand monochrome (H,W) or (H,W,1) to (H,W,3)
        if u8_buffer.ndim == 2 or (u8_buffer.ndim == 3 and u8_buffer.shape[2] == 1):
            u8_buffer = np.stack([u8_buffer.squeeze()] * 3, axis=-1)
        if not u8_buffer.flags["C_CONTIGUOUS"]:
            u8_buffer = np.ascontiguousarray(u8_buffer)

        h, w = u8_buffer.shape[:2]

        # 3. Create QImage
        # RGB888 is standard for our 3-channel processed output
        qimg = QImage(u8_buffer.data, w, h, w * 3, QImage.Format.Format_RGB888)

        # CRITICAL: QImage from data does NOT own the memory.
        # We MUST return a deep copy so that if the numpy buffer is cleared,
        # the QImage remains valid. This fixes the "harsh noise" bug.
        return qimg.copy()
