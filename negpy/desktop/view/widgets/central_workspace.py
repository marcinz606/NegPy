"""Central canvas/contact-sheet workspace switching.

The roll selector is owned by the scanner sidebar, which remains the single
source of truth for its model, selection, and callbacks.  This stack only
moves that same widget into the larger central workspace and decides which
workspace is visible.
"""

from PyQt6.QtWidgets import QStackedWidget, QVBoxLayout, QWidget


class CentralWorkspaceStack(QStackedWidget):
    """Switch between the normal image canvas and a whole-roll contact sheet."""

    def __init__(self, canvas: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("central_workspace")
        self.canvas = canvas
        self.roll_preview_page = QWidget(self)
        self.roll_preview_page.setObjectName("central_roll_preview_page")
        self._roll_preview_layout = QVBoxLayout(self.roll_preview_page)
        self._roll_preview_layout.setContentsMargins(0, 0, 0, 0)
        self._roll_preview_layout.setSpacing(0)
        self.roll_selector: QWidget | None = None

        self.addWidget(self.canvas)
        self.addWidget(self.roll_preview_page)
        self.setCurrentWidget(self.canvas)

    def mount_roll_selector(self, selector: QWidget) -> None:
        """Place the existing selector in the central page without cloning it."""

        if selector is self.roll_selector:
            return
        if self.roll_selector is not None:
            self._roll_preview_layout.removeWidget(self.roll_selector)
        self.roll_selector = selector
        self._roll_preview_layout.addWidget(selector, 1)

    def show_roll_preview(self) -> None:
        """Show the mounted contact sheet when it has been made available."""

        if self.roll_selector is None:
            return
        self.setCurrentWidget(self.roll_preview_page)
        self.roll_selector.setVisible(True)

    def show_loaded_roll_preview(self, _preview_token: str, _session: object) -> None:
        """Show only a preview that the sidebar successfully populated."""

        selector = self.roll_selector
        model = getattr(selector, "model", None)
        if model is None or model.rowCount() == 0:
            return
        self.show_roll_preview()

    def show_canvas(self) -> None:
        """Return to the normal asset canvas without discarding preview state."""

        self.setCurrentWidget(self.canvas)

    def show_canvas_after_roll_scan(self, completion: object) -> None:
        """Return once at least one completed frame is ready for import."""

        if getattr(completion, "rgb_paths", ()):
            self.show_canvas()

    def show_canvas_after_roll_error(self, failure: object) -> None:
        """Leave stale coordinates only when scanner recovery invalidated them."""

        if bool(getattr(failure, "recovery_required", False)):
            self.show_canvas()

    def is_showing_roll_preview(self) -> bool:
        return self.currentWidget() is self.roll_preview_page
