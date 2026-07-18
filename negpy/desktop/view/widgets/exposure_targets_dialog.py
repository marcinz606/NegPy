from typing import Dict

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout

from negpy.desktop.view.styles.templates import hint_label, section_subheader
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.features.exposure.models import DEFAULT_TARGETS, TUNABLE_TARGETS

# (key, slider label, tooltip) grouped under a heading + explanatory blurb.
_GROUPS = (
    (
        "AUTO DENSITY",
        "Auto Density meters each frame's midtone and anchors the print exposure there. "
        "These set how bright that anchor prints and how far the meter is trusted.",
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
                "0 ignores the meter entirely; 1 forces every frame to the metered midtone "
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
        "AUTO GRADE",
        "Auto Grade normalizes contrast across a roll so dense negatives stop printing over-contrasty and flat ones stop printing muddy.",
        (
            (
                "auto_grade_target",
                "Contrast Target",
                "Printed contrast aimed for across all frames. Raise for punchier prints, lower for softer ones.",
            ),
            (
                "auto_grade_strength",
                "Adaptation Strength",
                "How strongly the grade follows each scene's own range. 0 is a fixed grade "
                "for every frame; 1 fully normalizes them to the same contrast.",
            ),
        ),
    ),
)


class ExposureTargetsDialog(QDialog):
    """Modeless editor for the app-global Auto Density / Auto Grade targets.

    Emits live previews as sliders move; the sidebar renders them and decides
    whether to persist or restore on close.
    """

    targets_previewed = pyqtSignal(dict)

    def __init__(self, current: Dict[str, float], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Density & Grade Targets")
        self.setMinimumWidth(340)

        self._sliders: Dict[str, CompactSlider] = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(4)

        for title, blurb, entries in _GROUPS:
            root.addWidget(section_subheader(title))
            root.addWidget(hint_label(blurb))
            for key, label, tooltip in entries:
                lo, hi = TUNABLE_TARGETS[key]
                slider = CompactSlider(label, lo, hi, float(current.get(key, DEFAULT_TARGETS[key])))
                slider.setToolTip(tooltip)
                slider.valueChanged.connect(self._emit_preview)
                self._sliders[key] = slider
                root.addWidget(slider)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.RestoreDefaults | QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        restore = buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults)
        if restore is not None:
            restore.clicked.connect(self._restore_defaults)
        root.addWidget(buttons)

    def values(self) -> Dict[str, float]:
        return {key: float(slider.value()) for key, slider in self._sliders.items()}

    def _emit_preview(self) -> None:
        self.targets_previewed.emit(self.values())

    def _restore_defaults(self) -> None:
        for key, slider in self._sliders.items():
            slider.setValue(DEFAULT_TARGETS[key])  # setValue doesn't re-emit; preview once below
        self._emit_preview()
