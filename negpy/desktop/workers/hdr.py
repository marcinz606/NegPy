import gc
import threading
from dataclasses import dataclass, replace
from typing import Dict, Tuple

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.domain.models import WorkspaceConfig
from negpy.features.flatfield.models import FlatFieldConfig
from negpy.features.hdr.logic import ExposureStats, choose_reference, measure_exposure, solve_ratios
from negpy.features.hdr.models import HdrConfig
from negpy.services.rendering.image_processor import ImageProcessor


@dataclass(frozen=True)
class HdrTask:
    """Immutable request to solve a bracket of one frame.

    ``files`` are asset-dict copies. Order does not matter: the reference and the exposure
    order are both measured. Params carry each frame's decode-relevant settings so the
    solve sees the same buffers a later decode merges.
    """

    files: Tuple[dict, ...]
    params_by_path: Dict[str, WorkspaceConfig]


class HdrWorker(QObject):
    """Picks the exposure reference and solves the exposure ratios, off the UI thread.

    No merged buffer is emitted and nothing is written: the ratios are the whole result,
    and the merge itself happens at decode. Solving once here rather than per decode is
    what makes the preview and the export merge identically.
    """

    progress = pyqtSignal(int, int, str)  # current, total, label
    solved = pyqtSignal(object)  # {"files", "reference", "ratios"}
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
    def run(self, task: HdrTask) -> None:
        self._cancel.clear()
        frames = []
        total = len(task.files) + 1
        try:
            for i, f in enumerate(task.files):
                if self._cancel.is_set():
                    self.cancelled.emit()
                    return
                self.progress.emit(i, total, f"Decoding {f['name']}")
                # Flat-field off for the solve: the decode pins the white level, so
                # saturation sits at exactly 1.0 and the reference/ratio thresholds mean
                # what they say. A gain map applied first moves that point. The merge at
                # decode time defers it for the same reason.
                params = replace(task.params_by_path[f["path"]], flatfield=FlatFieldConfig(), hdr=HdrConfig())
                f32, _, _ = self._processor._decode_oriented_f32(f["path"], params)
                frames.append(f32)

            shapes = {f.shape for f in frames}
            if len(shapes) > 1:
                self.error.emit("Bracket frames differ in size — they must be the same capture setup")
                return

            self.progress.emit(len(task.files), total, "Solving exposures")
            stats = [ExposureStats(f["path"], *measure_exposure(buf)) for f, buf in zip(task.files, frames)]
            reference = choose_reference(stats)
            ratios = solve_ratios(frames, reference)
            self.solved.emit(
                {
                    "files": list(task.files),
                    "reference": int(reference),
                    "ratios": tuple(float(r) for r in ratios),
                    "clipped": tuple(float(s.clipped) for s in stats),
                }
            )
        except Exception as e:
            self.error.emit(f"HDR merge failed: {e}")
        finally:
            frames.clear()
            self._processor.cleanup(release_source_cache=True, collect=False)
            gc.collect()
