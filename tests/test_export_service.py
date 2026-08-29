import inspect

import numpy as np
from negpy.domain.models import ExportConfig, ExportResolutionMode, WorkspaceConfig
from negpy.desktop.workers import export as export_worker_mod
from negpy.services.rendering.image_processor import ImageProcessor


def test_export_worker_uses_full_res_render_not_preview_load() -> None:
    """Export must keep using full-res `render_export(file_path, ...)` — not `PreviewManager` decode."""
    src = inspect.getsource(export_worker_mod.ExportWorker.run_batch)
    assert "render_export" in src
    assert "load_linear_preview" not in src


def test_apply_scaling_f32() -> None:
    service = ImageProcessor()
    # 100x100 white square
    img = np.ones((100, 100, 3), dtype=np.float32)
    params = WorkspaceConfig()

    # Export config for 50px result (approx)
    # 1 inch @ 50 DPI
    export_settings = ExportConfig(export_resolution_mode=ExportResolutionMode.PRINT.value, export_print_size=2.54, export_dpi=50)

    res = service._apply_scaling_and_border_f32(img, params, export_settings)
    assert res.shape == (50, 50, 3)
    assert np.allclose(res, 1.0)


def test_apply_border_f32() -> None:
    img = np.ones((100, 100, 3), dtype=np.float32)

    # 1 inch @ 100 DPI = 100px total
    # 0.1 inch border = 10px
    from negpy.services.export.print import PrintService

    export_settings = ExportConfig(
        export_resolution_mode=ExportResolutionMode.PRINT.value,
        export_print_size=2.54,
        export_dpi=100,
    )

    res, _ = PrintService.apply_layout(img, export_settings, border_size=0.254, border_color="#000000")

    # Total size should be 100x100
    assert res.shape == (100, 100, 3)
    # Border should be black (0.0)
    assert np.allclose(res[0, 0], 0.0)
    # Content should be white (1.0)
    assert np.allclose(res[50, 50], 1.0)


def test_export_stands_in_camera_matrix_for_an_active_input_icc() -> None:
    """An active Input ICC supplies its own primaries rotation, so the camera's own
    must come out as identity — but the decode still needs the white-balance fold
    (issue #991), which nulling cam_xyz outright would also have dropped."""
    from negpy.features.process.capture_color import camera_to_working_matrix

    service = ImageProcessor()
    file_path = "/fake/shot.raf"
    matrix = [[0.7, -0.1, -0.07], [-0.56, 1.34, 0.24], [-0.15, 0.22, 0.73]]
    wb = [1.9, 1.0, 1.6]
    service._cam_xyz_by_path[file_path] = (matrix, wb)

    img = np.full((4, 4, 3), 0.2, dtype=np.float32)
    service._prepare_export_source = lambda *a, **k: (img, "sRGB", "token")

    captured = {}

    def fake_run_pipeline(*args, **kwargs):
        captured.update(kwargs)
        return img, {}

    service.run_pipeline = fake_run_pipeline

    params = WorkspaceConfig()

    export_with_icc = ExportConfig(icc_input_path="/custom.icc")
    service._render_export_buffer(file_path, params, export_with_icc, "hash", prefer_gpu=False)
    assert captured["cam_xyz"] != matrix
    assert captured["camera_wb"] == wb
    np.testing.assert_allclose(
        camera_to_working_matrix(captured["cam_xyz"], captured["camera_wb"]), np.diag(np.array(wb) / wb[1]), atol=1e-5
    )

    captured.clear()
    export_without_icc = ExportConfig(icc_input_path=None)
    service._render_export_buffer(file_path, params, export_without_icc, "hash", prefer_gpu=False)
    assert captured["cam_xyz"] == matrix


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
        compression="zlib",
        predictor=True,
    )
    res = out_buf.getvalue()
    assert len(res) > 0

    # Verify we can read it back
    read_back = tifffile.imread(io.BytesIO(res))
    assert read_back.dtype == np.uint16
    assert read_back.shape == (10, 10, 3)
