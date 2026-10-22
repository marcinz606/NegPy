from negpy.kernel.system.i18n import tr
from typing import Dict

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout

from negpy.desktop.view.styles.templates import hint_label, pin_button_box, section_subheader, wrap_tooltip
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.sliders import CompactSlider, align_slider_columns
from negpy.features.exposure.models import DEFAULT_TARGETS, TUNABLE_TARGETS
from negpy.desktop.view.widgets.dialog_geometry import remember_dialog_geometry
from negpy.desktop.view.widgets.floating_panel import float_over_app

# Fractions in the model, shown as percent.
_PERCENT_KEYS = frozenset({"anchor_meter_strength", "auto_grade_strength"})
# At its minimum the target is off.
_OFF_AT_MIN = frozenset({"shadow_hold_strength", "highlight_hold_density"})

# (heading, hint, ((key, slider label, tooltip), ...)).
_GROUPS = (
    (
        tr("Auto Density"),
        tr("How bright Auto Density makes each frame, and how closely it follows the meter."),
        (
            (
                "anchor_target_density",
                tr("Print Density Target"),
                tr("Raise to print the middle tones darker, or lower to print them brighter."),
            ),
            (
                "anchor_meter_strength",
                tr("Metering Strength"),
                tr("How closely Auto Density follows the meter, so a higher value prints dark and bright scenes more alike."),
            ),
            (
                "anchor_meter_band",
                tr("Metering Band"),
                tr("The largest brightness change Auto Density may make, so a lower value keeps a roll more even."),
            ),
        ),
    ),
    (
        tr("Auto Grade"),
        tr("The contrast Auto Grade gives each frame."),
        (
            (
                "auto_grade_target",
                tr("Contrast Target"),
                tr("The printed contrast Auto Grade aims for on every frame, higher for punchier prints."),
            ),
            (
                "auto_grade_strength",
                tr("Adaptation Strength"),
                tr("How much Auto Grade adapts the grade to each negative to give every frame the same detail range."),
            ),
        ),
    ),
    (
        tr("Shadows and Highlights"),
        tr("Where Auto Grade puts the darkest and brightest detail."),
        (
            (
                "shadow_reach_density",
                tr("Shadow Reach"),
                tr("How dark the darkest detail must print, so a flat negative still gets a black."),
            ),
            (
                "shadow_hold_strength",
                tr("Shadow Hold"),
                tr("Lifts the shadows of a contrasty frame when its darkest detail would go black."),
            ),
            (
                "highlight_hold_density",
                tr("Highlight Hold"),
                tr("Keeps detail in the brightest areas, such as a sunlit wall or sky, so they do not print pure white."),
            ),
        ),
    ),
)


def _shown(key: str, value: float) -> float:
    return value * 100.0 if key in _PERCENT_KEYS else value


def _stored(key: str, value: float) -> float:
    return value / 100.0 if key in _PERCENT_KEYS else value


def _make_slider(key: str, label: str, value: float) -> CompactSlider:
    lo, hi = TUNABLE_TARGETS[key]
    if key in _PERCENT_KEYS:
        slider = CompactSlider(label, lo * 100.0, hi * 100.0, value * 100.0, step=1.0, precision=1, unit="%")
    else:
        slider = CompactSlider(label, lo, hi, value)
    slider.set_default(_shown(key, DEFAULT_TARGETS[key]))
    if key in _OFF_AT_MIN:
        slider.spin.setSpecialValueText(tr("Off"))
    return slider


class ExposureTargetsDialog(QDialog):
    """Modeless editor for the app-global Auto Density / Auto Grade targets.

    Emits live previews as sliders move; the sidebar renders them and decides
    whether to persist or restore on close.
    """

    targets_previewed = pyqtSignal(dict)

    def __init__(self, current: Dict[str, float], parent=None, *, repo=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Auto Density and Grade Targets"))
        float_over_app(self)
        self.setMinimumWidth(340)

        self._sliders: Dict[str, CompactSlider] = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(THEME.space_xl, THEME.space_xl, THEME.space_xl, THEME.space_xl)
        root.setSpacing(THEME.space_sm)
        root.addWidget(hint_label(tr("These apply to every frame. The open frame previews them.")))

        for title, hint, entries in _GROUPS:
            root.addWidget(section_subheader(title))
            root.addWidget(hint_label(hint))
            for key, label, tooltip in entries:
                slider = _make_slider(key, label, float(current.get(key, DEFAULT_TARGETS[key])))
                slider.setToolTip(wrap_tooltip(tooltip))
                slider.valueChanged.connect(self._emit_preview)
                self._sliders[key] = slider
                root.addWidget(slider)
        align_slider_columns(self)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.RestoreDefaults | QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        pin_button_box(buttons)
        restore = buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults)
        if restore is not None:
            restore.clicked.connect(self._restore_defaults)
        root.addWidget(buttons)
        remember_dialog_geometry(self, repo, "exposure_targets")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # A top-level layout ignores the wrapped hints' height-for-width, so a restored or
        # dragged height would clip them.
        layout = self.layout()
        if layout is not None:
            self.setMinimumHeight(max(layout.totalHeightForWidth(self.width()), layout.totalMinimumSize().height()))

    def values(self) -> Dict[str, float]:
        return {key: _stored(key, float(slider.value())) for key, slider in self._sliders.items()}

    def _emit_preview(self) -> None:
        self.targets_previewed.emit(self.values())

    def _restore_defaults(self) -> None:
        for key, slider in self._sliders.items():
            slider.setValue(_shown(key, DEFAULT_TARGETS[key]))  # setValue doesn't re-emit; preview once below
        self._emit_preview()
