import io

from PIL import Image, JpegImagePlugin

from negpy.domain.models import ExportConfig, ExportFormat
from negpy.services.rendering.image_processor import ImageProcessor


def test_jpeg_export_uses_444_subsampling() -> None:
    """Regression for #224: JPEG export must use 4:4:4, not libjpeg default 4:2:0."""
    service = ImageProcessor()
    pil_img = Image.new("RGB", (16, 16), (128, 64, 200))
    settings = ExportConfig(export_fmt=ExportFormat.JPEG)

    buf = io.BytesIO()
    service._save_to_pil_buffer(pil_img, buf, settings, icc_bytes=None)

    buf.seek(0)
    reopened = Image.open(buf)
    # 0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0
    assert JpegImagePlugin.get_sampling(reopened) == 0
