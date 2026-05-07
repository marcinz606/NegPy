import numpy as np
from negpy.services.export.print import PrintService
from negpy.domain.models import ExportConfig, ExportResolutionMode


def test_calculate_paper_px_original():
    # 30cm long edge at 300dpi = 3543.3 px
    w, h = PrintService.calculate_paper_px(30.0, 300, "Original", 3000, 2000)
    assert w == 3543
    assert h == 2362


def test_calculate_paper_px_fixed_ratio():
    w, h = PrintService.calculate_paper_px(30.0, 300, "1:1", 3000, 2000)
    assert w == 3543
    assert h == 3543


def test_apply_layout_padding():
    # 3:2 content on 1:1 paper
    img = np.zeros((200, 300, 3), dtype=np.float32)
    config = ExportConfig(
        export_print_size=2.54,
        export_dpi=300,
        paper_aspect_ratio="1:1",
        export_resolution_mode=ExportResolutionMode.PRINT.value,
    )

    result, _ = PrintService.apply_layout(img, config)

    assert result.shape == (300, 300, 3)
    # Centered padding: (300-200)//2 = 50px
    assert np.all(result[0:50, :, :] == 1.0)
    assert np.all(result[250:300, :, :] == 1.0)
    assert np.all(result[50:250, :, :] == 0.0)


def test_apply_layout_with_border():
    # 3:2 image
    img = np.zeros((200, 300, 3), dtype=np.float32)
    # 0.1 inch border = 30px at 300 DPI
    config = ExportConfig(
        export_print_size=2.54,
        export_dpi=300,
        paper_aspect_ratio="Original",
        export_resolution_mode=ExportResolutionMode.ORIGINAL.value,
    )

    result, _ = PrintService.apply_layout(img, config, border_size=0.1 * 2.54, border_color="#ffffff")
    # In 'Original' mode (no resampling), paper should be img_size + 2*border
    # 300 + 60 = 360, 200 + 60 = 260
    assert result.shape == (260, 360, 3)
    # All borders should be 30px
    assert np.all(result[0:30, :, :] == 1.0)
    assert np.all(result[230:260, :, :] == 1.0)
    assert np.all(result[:, 0:30, :] == 1.0)
    assert np.all(result[:, 330:360, :] == 1.0)
    # Content should be intact
    assert np.all(result[30:230, 30:330, :] == 0.0)


def test_apply_layout_target_px_original_aspect():
    # 3:2 image, target long edge = 1000px, paper aspect = Original
    img = np.zeros((400, 600, 3), dtype=np.float32)
    config = ExportConfig(
        export_resolution_mode=ExportResolutionMode.TARGET_PX.value,
        export_target_long_edge_px=1000,
        paper_aspect_ratio="Original",
    )
    result, _ = PrintService.apply_layout(img, config)
    # No border: paper long edge equals target_long_edge_px (within rounding)
    assert max(result.shape[:2]) == 1000
    # Aspect preserved (3:2)
    assert result.shape[:2] == (666, 1000)


def test_apply_layout_target_px_fixed_ratio():
    # 3:2 image scaled into 1:1 paper at 1000px
    img = np.zeros((400, 600, 3), dtype=np.float32)
    config = ExportConfig(
        export_resolution_mode=ExportResolutionMode.TARGET_PX.value,
        export_target_long_edge_px=1000,
        paper_aspect_ratio="1:1",
    )
    result, _ = PrintService.apply_layout(img, config)
    assert result.shape == (1000, 1000, 3)


def test_target_px_ignores_print_size_and_dpi():
    # Same target_px with wildly different print_size/dpi → identical output
    img = np.zeros((200, 300, 3), dtype=np.float32)
    a = ExportConfig(
        export_resolution_mode=ExportResolutionMode.TARGET_PX.value,
        export_target_long_edge_px=800,
        paper_aspect_ratio="Original",
        export_print_size=10.0,
        export_dpi=72,
    )
    b = ExportConfig(
        export_resolution_mode=ExportResolutionMode.TARGET_PX.value,
        export_target_long_edge_px=800,
        paper_aspect_ratio="Original",
        export_print_size=100.0,
        export_dpi=600,
    )
    out_a, _ = PrintService.apply_layout(img, a)
    out_b, _ = PrintService.apply_layout(img, b)
    assert out_a.shape == out_b.shape
    assert max(out_a.shape[:2]) == 800
