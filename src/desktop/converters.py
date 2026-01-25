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
