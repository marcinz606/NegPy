"""Export panel shows a red prompt when no monitor profile is detected."""

from types import SimpleNamespace

from PyQt6.QtWidgets import QComboBox, QLabel

from negpy.desktop.view.sidebar.export import ExportSidebar
from negpy.desktop.view.styles.theme import THEME


def _stub(detected_bytes):
    combo = QComboBox()
    combo.addItem("As detected")
    return SimpleNamespace(
        state=SimpleNamespace(monitor_icc_detected_bytes=detected_bytes),
        display_detected_label=QLabel(),
        display_combo=combo,
    )


def test_no_profile_shows_red_prompt() -> None:
    s = _stub(None)
    ExportSidebar._refresh_display_info(s)
    assert "select your monitor" in s.display_detected_label.text().lower()
    assert THEME.channel_red in s.display_detected_label.styleSheet()


def test_detected_profile_shows_muted_label() -> None:
    from PIL import ImageCms

    data = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    s = _stub(data)
    ExportSidebar._refresh_display_info(s)
    assert s.display_detected_label.text().startswith("Detected:")
    assert THEME.text_muted in s.display_detected_label.styleSheet()
    assert THEME.channel_red not in s.display_detected_label.styleSheet()
