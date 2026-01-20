from dataclasses import dataclass
from typing import List
import os
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from src.domain.models import WorkspaceConfig, ExportConfig, ExportFormat
from src.services.rendering.image_processor import ImageProcessor


@dataclass(frozen=True)
class ExportTask:
    """
    Data for a single file export in a batch.
    """

    file_info: dict
    params: WorkspaceConfig
    export_settings: ExportConfig


class ExportWorker(QObject):
    """
    Handles batch export processing in a background thread.
    """

    progress = pyqtSignal(int, int, str)  # current, total, filename
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._processor = ImageProcessor()

    @pyqtSlot(list)
    def run_batch(self, tasks: List[ExportTask]) -> None:
        """
        Processes a list of export tasks.
        """
        total = len(tasks)
        try:
            for i, task in enumerate(tasks):
                filename = task.file_info["name"]
                self.progress.emit(i + 1, total, filename)

                result_bytes, _ = self._processor.process_export(
                    task.file_info["path"],
                    task.params,
                    task.export_settings,
                    task.file_info["hash"],
                )

                if result_bytes:
                    # Save to disk
                    # Logic for filename generation from pattern
                    # For simplicity, we'll use a basic join for now or port Pattern logic
                    out_dir = task.export_settings.export_path
                    os.makedirs(out_dir, exist_ok=True)

                    ext = (
                        "jpg"
                        if task.export_settings.export_fmt == ExportFormat.JPEG
                        else "tiff"
                    )
                    out_name = f"positive_{filename}.{ext}"
                    out_path = os.path.join(out_dir, out_name)

                    with open(out_path, "wb") as f:
                        f.write(result_bytes)

            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
