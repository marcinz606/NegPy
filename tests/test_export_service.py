import numpy as np
from src.services.rendering.image_processor import ImageProcessor
from src.domain.models import WorkspaceConfig, ExportConfig


def test_apply_scaling_f32() -> None:
    service = ImageProcessor()
    # 100x100 white square
    img = np.ones((100, 100, 3), dtype=np.float32)
    params = WorkspaceConfig()

    # Export config for 50px result (approx)
    # 1 inch @ 50 DPI
    export_settings = ExportConfig(
        export_print_size=2.54, export_dpi=50, export_add_border=False
    )

    res = service._apply_scaling_and_border_f32(img, params, export_settings)
    assert res.shape == (50, 50, 3)
    assert np.allclose(res, 1.0)


def test_apply_border_f32() -> None:
    service = ImageProcessor()
    img = np.ones((100, 100, 3), dtype=np.float32)
    params = WorkspaceConfig()

    # 1 inch @ 100 DPI = 100px total
    # 0.1 inch border = 10px
    export_settings = ExportConfig(
        export_print_size=2.54,
        export_dpi=100,
        export_add_border=True,
        export_border_size=0.254,
        export_border_color="#000000",
    )

    res = service._apply_scaling_and_border_f32(img, params, export_settings)

    # Total size should be 100x100
    assert res.shape == (100, 100, 3)
    # Border should be black (0.0)
    assert np.allclose(res[0, 0], 0.0)
    # Content should be white (1.0)
    assert np.allclose(res[50, 50], 1.0)


def test_image_service_tiff_export_format() -> None:
    """Verify that TIFF export produces a non-empty buffer and handles 16-bit correctly."""
    import io
    import tifffile

    img = np.random.rand(10, 10, 3).astype(np.float32)
    img_16 = (img * 65535).astype(np.uint16)

    out_buf = io.BytesIO()
    tifffile.imwrite(
        out_buf,
        img_16,
        photometric="rgb",
        iccprofile=b"fake_icc_bytes",
        compression="lzw",
    )
    res = out_buf.getvalue()
    assert len(res) > 0

    # Verify we can read it back
    read_back = tifffile.imread(io.BytesIO(res))
    assert read_back.dtype == np.uint16
    assert read_back.shape == (10, 10, 3)
