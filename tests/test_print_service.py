import numpy as np
from src.services.export.print import PrintService
from src.domain.models import ExportConfig


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
        export_border_size=0.0,
        export_border_color="#ffffff",
        use_original_res=False,
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
        export_border_size=0.1 * 2.54,  # 30px
        export_border_color="#ffffff",
        use_original_res=True,
    )

    result, _ = PrintService.apply_layout(img, config)
    # In 'Original' mode with use_original_res, paper should be img_size + 2*border
    # 300 + 60 = 360, 200 + 60 = 260
    assert result.shape == (260, 360, 3)
    # All borders should be 30px
    assert np.all(result[0:30, :, :] == 1.0)
    assert np.all(result[230:260, :, :] == 1.0)
    assert np.all(result[:, 0:30, :] == 1.0)
    assert np.all(result[:, 330:360, :] == 1.0)
    # Content should be intact
    assert np.all(result[30:230, 30:330, :] == 0.0)
