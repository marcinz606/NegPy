import gc
import threading
from dataclasses import dataclass
from typing import Dict, Tuple

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.domain.models import WorkspaceConfig
from negpy.features.stitch.logic import StitchCancelled, StitchError, register_parts
from negpy.services.rendering.image_processor import ImageProcessor


@dataclass(frozen=True)
class StitchTask:
    """Immutable request to register a set of overlapping scans.

    ``files`` are asset-dict copies in registration order; files[0] is the
    primary/reference part. Params carry each part's decode-relevant settings
    (flat-field, linear-raw) so registration sees the same buffers a later
    decode replays the transforms against.
    """

    files: Tuple[dict, ...]
    params_by_path: Dict[str, WorkspaceConfig]


class StitchWorker(QObject):
    """Registers the parts full-res and emits the resulting geometry.

    No file writes and no full-res blend: registration success is the gate;
    the composite itself is assembled at decode/preview time from the payload.
    """

    progress = pyqtSignal(int, int, str)  # current, total, label
    registered = pyqtSignal(object)  # {"files", "transforms", "canvas", "sizes"}
    cancelled = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._processor = ImageProcessor()
        self._cancel = threading.Event()

    @pyqtSlot()
    def cancel(self) -> None:
        self._cancel.set()

    @pyqtSlot(object)
    def run(self, task: StitchTask) -> None:
        self._cancel.clear()
        parts = []
        total = len(task.files) + 1
        try:
            for i, f in enumerate(task.files):
                if self._cancel.is_set():
                    self.cancelled.emit()
                    return
                self.progress.emit(i, total, f"Decoding {f['name']}")
                f32, _, _ = self._processor._decode_oriented_f32(f["path"], task.params_by_path[f["path"]])
                parts.append(f32)
            self.progress.emit(len(task.files), total, "Registering frames")
            transforms, canvas = register_parts(parts, is_cancelled=self._cancel.is_set)
            self.registered.emit(
                {
                    "files": list(task.files),
                    "transforms": tuple(tuple(float(v) for v in m.ravel()) for m in transforms),
                    "canvas": (int(canvas[0]), int(canvas[1])),
                    "sizes": tuple((p.shape[1], p.shape[0]) for p in parts),
                }
            )
        except StitchCancelled:
            self.cancelled.emit()
        except StitchError as e:
            self.error.emit(str(e))
        except Exception as e:
            self.error.emit(f"Stitch failed: {e}")
        finally:
            parts.clear()
            self._processor.cleanup(release_source_cache=True, collect=False)
            gc.collect()
