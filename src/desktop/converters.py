import sys
import numpy as np
from PyQt6.QtGui import QImage
from src.kernel.image.logic import float_to_uint8


class ImageConverter:
    """
    Handles conversion between NumPy/PIL and PyQt6 image types.
    """

    @staticmethod
    def to_qimage(buffer: np.ndarray, color_space: str = "sRGB") -> QImage:
        """
        Safely converts a NumPy float32 or uint8 buffer to a QImage.
        Performs a deep copy to prevent memory corruption (harsh noise).
        """
        # 1. Ensure uint8 for display
        if buffer.dtype == np.float32:
            u8_buffer = float_to_uint8(buffer)
        else:
            u8_buffer = buffer

        # 2. Handle dimensions
        h, w = u8_buffer.shape[:2]

        # Windows-specific alignment fix:
        # Windows GDI/DirectX expects 4-byte aligned scanlines. RGB888 (3-byte)
        # will look skewed/split if width is not a multiple of 4.
        # We convert to RGB32 (4-byte) to ensure perfect alignment.
        if sys.platform == "win32":
            # Create a 4-channel BGRA/RGBA buffer (Qt Format_RGB32)
            # RGB32 is actually 0xffRRGGBB in memory
            bgra = np.empty((h, w, 4), dtype=np.uint8)
            bgra[..., 0:3] = u8_buffer[..., 0:3]
            bgra[..., 3] = 255
            # We don't need to flip R/B for RGB32 if we use the right format
            qimg = QImage(bgra.data, w, h, w * 4, QImage.Format.Format_RGB32)
            return qimg.copy()

        # Ensure data is contiguous for QImage
        if not u8_buffer.flags["C_CONTIGUOUS"]:
            u8_buffer = np.ascontiguousarray(u8_buffer)

        # 3. Create QImage
        # RGB888 is standard for our 3-channel processed output
        qimg = QImage(u8_buffer.data, w, h, w * 3, QImage.Format.Format_RGB888)

        # CRITICAL: QImage from data does NOT own the memory.
        # We MUST return a deep copy so that if the numpy buffer is cleared,
        # the QImage remains valid. This fixes the "harsh noise" bug.
        return qimg.copy()
