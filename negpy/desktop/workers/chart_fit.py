import threading
from dataclasses import dataclass, replace
from typing import Optional

import negpy_core
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.papers import DEFAULT_PROFILE_KEY
from negpy.features.process.chart import ChartSpec, FitCancelled, fit_chart_matrix
from negpy.features.process.models import ProcessMode
from negpy.kernel.system.logging import get_logger
from negpy.services.rendering.image_processor import ImageProcessor

logger = get_logger(__name__)

# The fit renders the frame dozens of times; patches stay many pixels wide at this size.
FIT_LONG_EDGE = 768


@dataclass(frozen=True)
class ChartSource:
    """The open frame as the preview renders it."""

    buffer: np.ndarray
    config: WorkspaceConfig
    source_hash: str
    cam_xyz: Optional[list]
    camera_wb: Optional[list]
    gain_slices: tuple
    orientation: int


def chart_fit_config(config: WorkspaceConfig, matrix: np.ndarray) -> WorkspaceConfig:
    """The frame with *matrix* at full strength, metered on its own and printed on the Digital
    stock: a roll baseline or locked bounds were measured without the matrix, and a print stock's
    dyes would end up in the rig's matrix."""
    process = replace(
        config.process,
        crosstalk_matrix=tuple(float(v) for v in np.ravel(matrix)),
        crosstalk_strength=1.0,
        crosstalk_process=ProcessMode.C41,
        use_luma_average=False,
        use_color_average=False,
        use_cast_average=False,
        lock_bounds=False,
        local_floors=(0.0, 0.0, 0.0),
        local_ceils=(0.0, 0.0, 0.0),
    )
    return replace(config, process=process, exposure=replace(config.exposure, paper_profile=DEFAULT_PROFILE_KEY))


class ChartFitWorker(QThread):
    """Renders the frame for the chart dialog, then fits the matrix, on a CPU processor of its own."""

    previewed = pyqtSignal(object)  # working-encoded float render, identity matrix
    progress = pyqtSignal(float)
    fitted = pyqtSignal(object)  # ChartFit
    failed = pyqtSignal(str)

    def __init__(self, source: ChartSource, parent=None) -> None:
        super().__init__(parent)
        buf = np.asarray(source.buffer, dtype=np.float32)
        scale = FIT_LONG_EDGE / max(buf.shape[:2])
        if scale < 1.0:
            h, w = buf.shape[:2]
            buf = negpy_core.resize_area(np.ascontiguousarray(buf), (max(1, round(w * scale)), max(1, round(h * scale))))
        self._buffer = buf
        self._source = source
        self._processor = ImageProcessor(use_gpu=False)
        self._job: tuple = ("preview",)
        self._cancel = threading.Event()

    def render(self, matrix: np.ndarray) -> np.ndarray:
        s = self._source
        out, _ = self._processor.run_pipeline(
            self._buffer,
            chart_fit_config(s.config, matrix),
            s.source_hash,
            render_size_ref=float(max(self._buffer.shape[:2])),
            prefer_gpu=False,
            readback_metrics=False,
            wants_uv_grid=False,
            cam_xyz=s.cam_xyz,
            camera_wb=s.camera_wb,
            gain_slices=s.gain_slices,
            source_orientation=s.orientation,
        )
        return np.asarray(out)[:, :, :3]

    def start_preview(self) -> None:
        self._job = ("preview",)
        self.start()

    def start_fit(self, corners_px: np.ndarray, chart: ChartSpec) -> None:
        self._job = ("fit", corners_px, chart)
        self._cancel.clear()
        self.start()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            if self._job[0] == "preview":
                self.previewed.emit(self.render(np.eye(3)))
                return
            _, corners_px, chart = self._job
            self.fitted.emit(fit_chart_matrix(self.render, corners_px, chart, progress=self.progress.emit, cancelled=self._cancel.is_set))
        except FitCancelled:
            pass
        except Exception as exc:
            logger.exception("Chart fit failed")
            self.failed.emit(str(exc))
