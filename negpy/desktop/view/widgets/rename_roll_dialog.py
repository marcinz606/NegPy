from PyQt6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QLineEdit, QVBoxLayout

from negpy.desktop.view.styles.templates import field_label, hint_label, pin_button_box


class RenameRollDialog(QDialog):
    """Renames a folder roll: its display name always, and -- only if asked -- the
    actual folder on disk too. The checkbox starts unchecked every time; it is never
    remembered, since a disk rename is a bigger deal than a library-only one.
    """

    def __init__(self, current_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rename Roll")
        self.setMinimumWidth(340)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        root.addWidget(field_label("Name"))
        self.name_edit = QLineEdit(current_name)
        self.name_edit.selectAll()
        root.addWidget(self.name_edit)

        self.rename_folder_check = QCheckBox("Also rename the folder on disk")
        root.addWidget(self.rename_folder_check)
        root.addWidget(
            hint_label(
                "Renames the actual folder, in place. Inside a cloud-sync folder (Dropbox, "
                "iCloud, OneDrive), a sync client may see this as a delete and re-upload "
                "rather than a rename.",
                kind="warning",
            )
        )

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        pin_button_box(buttons)
        root.addWidget(buttons)

        self.name_edit.setFocus()

    def name(self) -> str:
        return self.name_edit.text().strip()

    def rename_folder(self) -> bool:
        return self.rename_folder_check.isChecked()
