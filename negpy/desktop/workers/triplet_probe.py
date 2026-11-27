from concurrent.futures import ThreadPoolExecutor
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
        def _probe_one(path: str) -> Optional[FrameProbe]:
            try:
                return probe_frame(path)
            except Exception as e:
                # rawpy refuses a TIFF or a JPEG: such a file has no channel to read.
                logger.info("Trichrome merge could not read %s: %s", path, e)
                return None

        probes: Dict[str, Optional[FrameProbe]] = {}
        if len(paths) <= 1:
            for i, path in enumerate(paths):
                self.progress.emit(i, len(paths), path)
                probes[path] = _probe_one(path)
        else:
            with ThreadPoolExecutor(max_workers=len(paths)) as pool:
                futures = [(i, path, pool.submit(_probe_one, path)) for i, path in enumerate(paths)]
                for i, path, fut in futures:
                    self.progress.emit(i, len(paths), path)
                    probes[path] = fut.result()
        try:
            triplet, reason = place_triplet(probes)
        except Exception as e:
            # `placed` must always fire: it ends the batch that holds the shared lane.
            logger.warning("Trichrome merge could not place %s: %s", paths, e)
            triplet, reason = None, "channels"
        self.placed.emit({"paths": tuple(paths), "triplet": triplet, "reason": reason})
