from PyQt6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QVBoxLayout

from negpy.desktop.view.styles.templates import hint_label
from negpy.features.metadata.models import (
    DESCRIPTION_FIELD_LABELS,
    DESCRIPTION_FIELD_ORDER,
    normalize_description_fields,
)


class DescriptionFieldsDialog(QDialog):
    """Pick which metadata values join into EXIF ImageDescription."""

    def __init__(self, selected: object, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Description fields")
        self.setMinimumWidth(320)

        root = QVBoxLayout(self)
        root.addWidget(hint_label("Checked fields are joined with • into the export ImageDescription. Empty values are skipped."))

        enabled = set(normalize_description_fields(selected))
        self._checks: dict[str, QCheckBox] = {}
        for key in DESCRIPTION_FIELD_ORDER:
            box = QCheckBox(DESCRIPTION_FIELD_LABELS[key])
            box.setChecked(key in enabled)
            self._checks[key] = box
            root.addWidget(box)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def selected_fields(self) -> tuple[str, ...]:
        return normalize_description_fields(key for key, box in self._checks.items() if box.isChecked())
