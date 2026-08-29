import io

import numpy as np
from PIL import Image, JpegImagePlugin

from negpy.features.metadata.resolution import Resolution
from negpy.services.export.encoders import encode_jpeg


def test_jpeg_export_uses_444_subsampling() -> None:
    """Regression for #224: JPEG export must use 4:4:4, not libjpeg default 4:2:0."""
    arr = np.full((16, 16, 3), (128, 64, 200), dtype=np.uint8)

    bits = encode_jpeg(arr, resolution=Resolution.from_dpi(300))

    reopened = Image.open(io.BytesIO(bits))
    # 0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0
    assert JpegImagePlugin.get_sampling(reopened) == 0
