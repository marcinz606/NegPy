"""Export panel shows a red prompt when no monitor profile is detected."""

from types import SimpleNamespace

from PyQt6.QtWidgets import QComboBox, QLabel, QSizePolicy

from negpy.desktop.view.sidebar.export import ExportSidebar
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.export_settings_form import constrain_combo


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


def test_constrain_combo_bounds_width_for_long_items() -> None:
    """A long item (e.g. a verbose monitor profile / ICC filename) must not
    blow up the combo's width and stretch the export panel (#325)."""
    long_item = "As detected (sRGB IEC61966-2.1 Color Space Profile, very verbose)"

    wide = QComboBox()
    wide.addItem(long_item)

    narrow = QComboBox()
    narrow.addItem(long_item)
    constrain_combo(narrow)

    assert narrow.sizeHint().width() < wide.sizeHint().width()
    assert narrow.sizeAdjustPolicy() == QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    assert narrow.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
