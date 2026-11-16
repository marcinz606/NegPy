import os
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.domain.models import WorkspaceConfig
from negpy.features.leak.models import LeakShape
from negpy.features.leak.roll import FrameBox
from negpy.services.rendering.image_processor import ImageProcessor


@dataclass(frozen=True)
class LeakRollTask:
    """The frames to read, each with its edit. With *shapes*, the leak drawn on the current frame,
    whose picture box is *source*, is carried to each; without, each frame is searched for leaks."""

    files: Tuple[dict, ...]
    params_by_path: Dict[str, WorkspaceConfig]
    shapes: Tuple[LeakShape, ...] = ()
    source: Optional[FrameBox] = None


class LeakRollWorker(QObject):
    """Decodes each frame of a roll for its picture box, and either carries the current frame's leak
    to it or looks for one on it."""

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(dict)  # {"shapes": {hash: shapes}} carrying, {"found": [hash]} searching
    cancelled = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._processor = ImageProcessor()
        self._cancel = threading.Event()

    @pyqtSlot()
    def cancel(self) -> None:
        self._cancel.set()

    def arm(self) -> None:
        """Clears a Stop left from an earlier run; called on the GUI thread before dispatch."""
        self._cancel.clear()

    @pyqtSlot(object)
    def run(self, task: LeakRollTask) -> None:
        try:
            self._run(task)
        finally:
            self._processor.release_source_cache()

    def _run(self, task: LeakRollTask) -> None:
        from negpy.features.leak.auto import find_leaks
        from negpy.features.leak.roll import map_shapes, picture_box

        carried: Dict[str, Tuple[LeakShape, ...]] = {}
        found = []
        try:
            for i, f in enumerate(task.files):
                if self._cancel.is_set():
                    self.cancelled.emit()
                    return
                self.progress.emit(i + 1, len(task.files), os.path.basename(f["path"]))
                params = task.params_by_path[f["path"]]
                img = self._processor._load_source_f32(f["path"], params)[0][..., :3]
                if task.shapes and task.source is not None:
                    box = picture_box(img, params.geometry)
                    if box is not None:
                        carried[f["hash"]] = map_shapes(task.shapes, task.source, box)
                elif find_leaks(img, params):
                    found.append(f["hash"])
                del img
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
            return
        self.finished.emit({"shapes": carried} if task.shapes else {"found": found})


@dataclass(frozen=True)
class LeakAutoTask:
    """One frame's decoded preview and edit, to search for leaks. *gain_slices* and *orientation*
    place the flat field on the buffer, as the render's do."""

    file_hash: str
    img: np.ndarray
    config: WorkspaceConfig
    gain_slices: tuple = ()
    orientation: int = 1


class LeakAutoWorker(QObject):
    """Auto on the open frame, off the GUI thread."""

    finished = pyqtSignal(dict)  # {"hash": file hash, "found": [LeakShape], "shape": (h, w)}
    error = pyqtSignal(str)

    @pyqtSlot(object)
    def run(self, task: LeakAutoTask) -> None:
        from negpy.features.leak.auto import find_leaks
        from negpy.services.rendering.image_processor import precorrect_source

        try:
            img = precorrect_source(task.img, task.config, task.gain_slices, task.orientation)
            found = find_leaks(np.asarray(img, dtype=np.float32), task.config)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
            return
        self.finished.emit({"hash": task.file_hash, "found": found, "shape": task.img.shape[:2]})
