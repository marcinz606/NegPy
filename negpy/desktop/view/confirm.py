from PyQt6.QtWidgets import QMessageBox


def confirm_unload(parent, *, clear_all: bool = False, count: int = 1) -> bool:
    """Ask the user to confirm removing image(s) from the session.

    Unloading only drops the frames from the current list — saved edits stay in the
    database keyed by content hash — but re-adding a large roll is tedious, and an
    accidental Clear All is destructive to the working set, so we gate it behind a
    prompt. Enter confirms (default button); Esc cancels.
    """
    if clear_all:
        title = "Clear All"
        text = "Remove all loaded images from the session?"
    elif count > 1:
        title = "Unload Selected"
        text = f"Unload the {count} selected images from the session?"
    else:
        title = "Unload"
        text = "Unload this image from the session?"

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(title)
    box.setText(text)
    box.setInformativeText("Your saved edits stay in the database — this only removes the frames from the list.")
    box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(QMessageBox.StandardButton.Yes)
    return box.exec() == QMessageBox.StandardButton.Yes
