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
        "Auto Density",
        "How bright the metered midtone prints and how far the meter is trusted.",
        (
            (
                "anchor_target_density",
                "Print Density Target",
                "Density the metered reference tone prints at. Raise for darker, moodier prints; lower for brighter ones.",
            ),
            (
                "anchor_meter_strength",
                "Metering Strength",
                "How far the anchor moves from the assumed key toward what was measured. "
                "0% ignores the meter; 100% forces every frame to the metered midtone "
                "(low-key and high-key scenes lose their intended key).",
            ),
            (
                "anchor_meter_band",
                "Metering Band",
                "Hard safety clamp around the assumed key. Raise to allow bigger exposure "
                "swings between frames; lower to keep a roll consistent.",
            ),
        ),
    ),
    (
        "Auto Grade",
        "The contrast each frame prints at, from its textural density range.",
        (
            (
                "auto_grade_target",
                "Contrast Target",
                "Printed contrast aimed for across all frames. Raise for punchier prints, lower for softer ones.",
            ),
            (
                "auto_grade_strength",
                "Adaptation Strength",
                "How far the grade is pulled toward a normal negative's textural range. 0% prints "
                "every frame on one fixed paper; 100% prints every frame's detail range alike.",
            ),
        ),
    ),
    (
        "Shadows and Highlights",
        "Where Auto Grade puts the darkest and brightest textured tones.",
        (
            (
                "shadow_reach_density",
                "Shadow Reach",
                "Print density the darkest textured tones must reach. The grade only goes harder "
                "for it, so a flat negative still prints a black. Lower for softer shadows.",
            ),
            (
                "shadow_hold_strength",
                "Shadow Hold",
                "How far the shadows of a contrasty frame are lifted when its darkest textured tones "
                "print far past black. Raise for more open shadows; Off leaves them.",
            ),
            (
                "highlight_hold_density",
                "Highlight Hold",
                "Print density the brightest textured tones must keep, so a sunlit wall or sky holds "
                "tone instead of printing paper white. Off lets them reach paper white.",
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
        slider.spin.setSpecialValueText("Off")
    return slider


class ExposureTargetsDialog(QDialog):
    """Modeless editor for the app-global Auto Density / Auto Grade targets.

    Emits live previews as sliders move; the sidebar renders them and decides
    whether to persist or restore on close.
    """

    targets_previewed = pyqtSignal(dict)

    def __init__(self, current: Dict[str, float], parent=None, *, repo=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Density and Grade Targets")
        float_over_app(self)
        self.setMinimumWidth(340)

        self._sliders: Dict[str, CompactSlider] = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(THEME.space_xl, THEME.space_xl, THEME.space_xl, THEME.space_xl)
        root.setSpacing(THEME.space_sm)
        root.addWidget(hint_label("These apply to every frame. The open frame previews them."))

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
