from dataclasses import dataclass
from typing import List
import os
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from src.domain.models import WorkspaceConfig, ExportConfig, ExportFormat
from src.services.rendering.image_processor import ImageProcessor
from src.services.export.templating import render_export_filename


@dataclass(frozen=True)
class ExportTask:
    """Immutable data for a high-resolution export job."""

    file_info: dict
    params: WorkspaceConfig
    export_settings: ExportConfig
    gpu_enabled: bool = True


class ExportWorker(QObject):
    """
    Background batch export orchestrator.
    Maintains UI responsiveness during heavy processing.
    """

    progress = pyqtSignal(int, int, str)  # current, total, filename
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._processor = ImageProcessor()

    @pyqtSlot(list)
    def run_batch(self, tasks: List[ExportTask]) -> None:
        """Processes an ordered list of export tasks."""
        total = len(tasks)
        try:
            for i, task in enumerate(tasks):
                full_name = task.file_info["name"]
                name = os.path.splitext(full_name)[0]
                self.progress.emit(i + 1, total, name)

                bits, _ = self._processor.process_export(
                    task.file_info["path"],
                    task.params,
                    task.export_settings,
                    task.file_info["hash"],
                    prefer_gpu=task.gpu_enabled,
                )

                if bits:
                    out_dir = task.export_settings.export_path
                    os.makedirs(out_dir, exist_ok=True)

                    ext = (
                        "jpg"
                        if task.export_settings.export_fmt == ExportFormat.JPEG
                        else "tiff"
                    )

                    filename = render_export_filename(
                        task.file_info["path"], task.export_settings
                    )
                    path = os.path.join(out_dir, f"{filename}.{ext}")

                    with open(path, "wb") as f:
                        f.write(bits)

                # Aggressive VRAM evacuation between files
                self._processor.cleanup()

            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
