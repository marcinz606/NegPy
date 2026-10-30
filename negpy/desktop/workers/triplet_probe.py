from typing import Dict, Optional, Tuple

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.features.rgbscan.logic import FrameProbe, place_triplet, probe_frame
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


class TripletProbeWorker(QObject):
    """Reads three hand-picked files and places them as red, green and blue."""

    progress = pyqtSignal(int, int, str)  # current, total, label
    placed = pyqtSignal(object)  # {"paths", "triplet" (Triplet or None), "reason"}

    @pyqtSlot(object)
    def run(self, paths: Tuple[str, ...]) -> None:
        probes: Dict[str, Optional[FrameProbe]] = {}
        for i, path in enumerate(paths):
            self.progress.emit(i, len(paths), path)
            try:
                probes[path] = probe_frame(path)
            except Exception as e:
                # rawpy refuses a TIFF or a JPEG: such a file has no channel to read.
                logger.info("Trichrome merge could not read %s: %s", path, e)
                probes[path] = None
        try:
            triplet, reason = place_triplet(probes)
        except Exception as e:
            # `placed` must always fire: it ends the batch that holds the shared lane.
            logger.warning("Trichrome merge could not place %s: %s", paths, e)
            triplet, reason = None, "channels"
        self.placed.emit({"paths": tuple(paths), "triplet": triplet, "reason": reason})
