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


def _sharpen_params(mode: str, amount: float = 0.5, flat: bool = False) -> WorkspaceConfig:
    from dataclasses import replace

    from negpy.features.exposure.models import RenderIntent

    params = WorkspaceConfig(export=ExportConfig(export_resolution_mode=mode))
    params = replace(params, lab=replace(params.lab, sharpen=amount))
    if flat:
        params = replace(params, exposure=replace(params.exposure, render_intent=RenderIntent.FLAT))
    return params


def test_a_resized_export_sharpens_after_the_resize() -> None:
    assert ImageProcessor._sharpens_after_resize(_sharpen_params(ExportResolutionMode.TARGET_PX.value))
    assert ImageProcessor._sharpens_after_resize(_sharpen_params(ExportResolutionMode.PRINT.value))


def test_auto_masking_sharpens_an_original_size_export_after_the_pipeline() -> None:
    from dataclasses import replace

    params = _sharpen_params(ExportResolutionMode.ORIGINAL.value)
    assert ImageProcessor._sharpens_after_resize(replace(params, lab=replace(params.lab, sharpen_auto_masking=True)))


def test_original_size_flat_and_unsharpened_exports_sharpen_in_the_pipeline() -> None:
    assert not ImageProcessor._sharpens_after_resize(_sharpen_params(ExportResolutionMode.ORIGINAL.value))
    assert not ImageProcessor._sharpens_after_resize(_sharpen_params(ExportResolutionMode.TARGET_PX.value, flat=True))
    assert not ImageProcessor._sharpens_after_resize(_sharpen_params(ExportResolutionMode.TARGET_PX.value, amount=0.0))


def test_resize_sharpening_raises_edge_contrast_and_leaves_the_mat() -> None:
    before = np.full((60, 80, 3), 0.9, dtype=np.float32)
    before[10:50, 10:40] = 0.3
    before[10:50, 40:70] = 0.6
    params = _sharpen_params(ExportResolutionMode.TARGET_PX.value)
    after = ImageProcessor._sharpen_resized_content(before.copy(), (10, 10, 60, 40), params)

    assert after[30, 39, 0] < before[30, 39, 0]
    assert after[30, 40, 0] > before[30, 40, 0]
    mat = np.ones(before.shape[:2], dtype=bool)
    mat[10:50, 10:70] = False
    assert np.array_equal(after[mat], before[mat])


def test_resize_sharpening_is_the_lab_sharpening_in_encoded_values_and_keeps_alpha() -> None:
    from negpy.features.lab import logic
    from negpy.features.lab.models import LabConfig, SharpenMethod
    from negpy.kernel.image.logic import working_oetf_decode, working_oetf_encode

    rng = np.random.default_rng(3)
    rgba = rng.random((64, 48, 4), dtype=np.float32)
    for method in SharpenMethod:
        lab = LabConfig(sharpen=0.5, sharpen_method=method, sharpen_radius=1.5)
        out = logic.apply_resize_sharpening(rgba, lab)
        expected = working_oetf_encode(logic.apply_lab_sharpening(working_oetf_decode(np.ascontiguousarray(rgba[..., :3])), lab))
        np.testing.assert_array_equal(out[..., :3], expected)
        np.testing.assert_array_equal(out[..., 3], rgba[..., 3])
