from negpy.kernel.system.i18n import tr
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QGridLayout, QPushButton, QVBoxLayout

from negpy.desktop.view.styles.templates import hint_label, labeled_toggle, pin_button_box
from negpy.desktop.view.widgets.dialog_geometry import remember_dialog_geometry
from negpy.features.metadata.models import (
    DESCRIPTION_FIELD_LABELS,
    DESCRIPTION_FIELD_ORDER,
    normalize_description_fields,
)


class DescriptionFieldsDialog(QDialog):
    """Pick which metadata values join into EXIF ImageDescription."""

    def __init__(self, selected: object, parent=None, *, repo=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Description Fields"))
        self.setMinimumWidth(320)

        root = QVBoxLayout(self)
        root.addWidget(
            hint_label(tr("The fields that are on are joined with • into the export ImageDescription. Empty values are skipped."))
        )

        enabled = set(normalize_description_fields(selected))
        self._checks: dict[str, QPushButton] = {}
        grid = QGridLayout()
        for i, key in enumerate(DESCRIPTION_FIELD_ORDER):
            label = DESCRIPTION_FIELD_LABELS[key]
            box = labeled_toggle("", label, key in enabled, tr("Join {value} into the ImageDescription", value=label))
            self._checks[key] = box
            grid.addWidget(box, i // 2, i % 2)
        root.addLayout(grid)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        pin_button_box(buttons)
        root.addWidget(buttons)
        remember_dialog_geometry(self, repo, "description_fields")

    def selected_fields(self) -> tuple[str, ...]:
        return normalize_description_fields(key for key, box in self._checks.items() if box.isChecked())
