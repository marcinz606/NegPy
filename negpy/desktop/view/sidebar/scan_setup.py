from negpy.kernel.system.i18n import tr

from PyQt6.QtCore import pyqtSignal

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import hint_label
from negpy.desktop.view.widgets.segmented_choice import SegmentedChoice
from negpy.features.exposure.papers import resolve_paper
from negpy.features.process.models import SCAN_SETUP_PAPERS, ProcessMode, scan_setup_rig, scan_setup_values

_LIGHT_HINT = {
    "camera": {
        "white": tr("A lightbox or LED panel with white light."),
        "narrowband": tr("A Scanlight or RGB LED light with pure red, green and blue."),
    },
    "scanner": {
        "white": tr("A white lamp, as in Plustek, Epson and most flatbed scanners."),
        "narrowband": tr("RGB LED light, as in a Nikon Coolscan or Kodak Pakon."),
    },
}
_DECODE_EFFECT = {
    ("camera", "white"): tr("Decodes the RAW with the camera white balance."),
    ("camera", "narrowband"): tr("Decodes the RAW linear, with no white balance."),
    ("scanner", "white"): tr("Reads the scan linear, with no white balance."),
    ("scanner", "narrowband"): tr("Reads the scan linear, with no white balance."),
}
_CARDS_EFFECT = {
    ("camera", "white"): tr("Shows the Raw Decode and Optics cards."),
    ("camera", "narrowband"): tr("Shows the Raw Decode, Optics and Trichrome cards."),
    ("scanner", "white"): tr("Hides the Raw Decode and Optics cards."),
    ("scanner", "narrowband"): tr("Hides the Raw Decode and Optics cards."),
}


def detected_capture(state) -> str:
    """A camera RAW (it has a camera matrix) reads as "camera", anything else as "scanner"; with no frame open, "camera"."""
    if not state.current_file_hash:
        return "camera"
    return "camera" if state.preview_cam_xyz is not None else "scanner"


def resolved_rig(repo, state) -> tuple[str, str]:
    """(capture, light) of the scanning setup, Auto resolved against the open frame."""
    capture, light = scan_setup_rig(repo.get_global_setting("scan_setup"))
    return (detected_capture(state) if capture == "auto" else capture), light


class ScanSetupSidebar(BaseSidebar):
    """How the roll was scanned: it sets the RAW decode and the print stock, and decides which
    Roll-tab controls show. One setting for the app, as the rig is. Camera or Film Scanner is
    read from the open frame until the user picks one; a pick also picks the Scan tab's scanner."""

    capture_picked = pyqtSignal(str)

    def _init_ui(self) -> None:
        self.capture_btn = SegmentedChoice(
            (("fa5s.camera", tr("Camera")), ("fa5s.camera-retro", tr("Film Scanner"))),
            tr("What you scanned with, which NegPy reads from the file until you pick one."),
            data=("camera", "scanner"),
        )
        self.light_btn = SegmentedChoice(
            (("mdi6.lightbulb-outline", tr("White Light")), ("mdi6.led-on", tr("Narrowband Light"))),
            tr("The light behind the negative"),
            data=("white", "narrowband"),
        )
        self.hint = hint_label("")
        self.layout.addWidget(self.capture_btn)
        self.layout.addWidget(self.light_btn)
        self.layout.addWidget(self.hint)

    def _connect_signals(self) -> None:
        self.capture_btn.currentChanged.connect(lambda _i: self._apply(self.capture_btn.currentData()))
        self.capture_btn.currentChanged.connect(lambda _i: self.capture_picked.emit(self.capture_btn.currentData()))
        # A light change keeps a capture still read from the file.
        self.light_btn.currentChanged.connect(lambda _i: self._apply(self._saved()[0]))

    def _saved(self) -> tuple[str, str]:
        return scan_setup_rig(self.controller.session.repo.get_global_setting("scan_setup"))

    def _apply(self, capture: str) -> None:
        self.controller.apply_scan_setup(capture, self.light_btn.currentData())
        # With no frame open nothing re-syncs the panel, and the cards follow the setup.
        self.controller.config_updated.emit()

    def sync_ui(self) -> None:
        capture, light = self._saved()
        resolved = detected_capture(self.state) if capture == "auto" else capture
        for btn, value in ((self.capture_btn, resolved), (self.light_btn, light)):
            btn.blockSignals(True)
            btn.setCurrentIndex(btn.findData(value))
            btn.blockSignals(False)
        # A B&W negative prints the same under either light, so the choice hides and keeps its value.
        process = self.state.config.process
        bw = process.process_mode == ProcessMode.BW and not process.source_color
        self.light_btn.setVisible(not bw)
        detected = (
            (tr("A camera RAW, so a camera.") if resolved == "camera" else tr("Not a camera RAW, so a film scanner."))
            if capture == "auto" and self.state.current_file_hash
            else ""
        )
        # apply_scan_setup moves only a Color frame on one of these stocks.
        stock = ""
        if process.process_mode == ProcessMode.C41:
            paper = self.state.config.exposure.paper_profile
            stock = (
                tr("Prints on {stock}.", stock=resolve_paper(scan_setup_values(resolved, light)[1]).label)
                if paper in SCAN_SETUP_PAPERS
                else tr("Keeps the print stock you picked.")
            )
        rig = (resolved, light)
        lines = ("" if bw else _LIGHT_HINT[resolved][light], detected, _DECODE_EFFECT[rig], _CARDS_EFFECT[rig], stock)
        # A table, not <ul>: a wrapped line hangs under its own text, and the bullets stay flush left.
        rows = "".join(f"<tr><td>•&nbsp;</td><td>{t}</td></tr>" for t in lines if t)
        self.hint.setText('<table cellspacing="0" cellpadding="0">' + rows + "</table>")
