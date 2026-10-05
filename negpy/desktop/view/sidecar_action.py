"""Load Edit from Sidecar, the one action behind every surface that offers it."""

import os

from PyQt6.QtWidgets import QFileDialog, QMessageBox

from negpy.services.assets.sidecar import sidecar_path_for

LABEL = "Load Edit from Sidecar…"


def load_edit_from_sidecar(parent, controller) -> None:
    """Pick a `.negpy` for the active frame, starting at its own sidecar beside the source."""
    state = controller.session.state
    idx = state.selected_file_idx
    if not 0 <= idx < len(state.uploaded_files):
        return
    asset = state.uploaded_files[idx]
    start = sidecar_path_for(asset["path"], int(asset.get("half") or 0))
    if not os.path.exists(start):
        start = os.path.dirname(start)
    path, _ = QFileDialog.getOpenFileName(parent, "Load Edit from Sidecar", start, "NegPy Sidecars (*.negpy)")
    if not path:
        return
    if controller.session.load_edit_from_sidecar(path):
        controller.set_status(f"Loaded edit from {os.path.basename(path)}", 4000)
    else:
        QMessageBox.warning(parent, "Sidecar", f"{os.path.basename(path)} is not a readable NegPy sidecar.")
