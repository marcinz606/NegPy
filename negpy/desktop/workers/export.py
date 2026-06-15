from dataclasses import dataclass
from typing import List, Optional, Any
import os
import tempfile
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from negpy.domain.models import WorkspaceConfig, ExportConfig, ExportFormat
from negpy.features.metadata.writer import embed_metadata
from negpy.features.metadata.models import MetadataConfig
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE
from negpy.services.rendering.image_processor import ImageProcessor
from negpy.services.export.templating import render_export_filename
from negpy.services.export.contact_sheet import ContactSheetService, CELL_PX


@dataclass(frozen=True)
class ExportTask:
    """Immutable data for a high-resolution export job."""

    file_info: dict
    params: WorkspaceConfig
    export_settings: ExportConfig
    gpu_enabled: bool = True
    bounds_override: Optional[Any] = None
    source_exif: Optional[dict] = None
    metadata_config: Optional[MetadataConfig] = None
    working_color_space: str = WORKING_COLOR_SPACE


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
                    bounds_override=task.bounds_override,
                    working_color_space=task.working_color_space,
                )

                if bits:
                    # Embed metadata if config is provided
                    if task.metadata_config is not None:
                        bits = embed_metadata(bits, task.metadata_config, task.source_exif)

                    out_dir = (
                        os.path.dirname(task.file_info["path"]) if task.export_settings.same_as_source else task.export_settings.export_path
                    )
                    os.makedirs(out_dir, exist_ok=True)

                    ext = "jpg" if task.export_settings.export_fmt == ExportFormat.JPEG else "tiff"

                    filename = render_export_filename(
                        task.file_info["path"], task.export_settings, border_size=task.params.finish.border_size
                    )
                    path = os.path.join(out_dir, f"{filename}.{ext}")

                    if not task.export_settings.overwrite:
                        counter = 2
                        while os.path.exists(path):
                            path = os.path.join(out_dir, f"{filename}_{counter}.{ext}")
                            counter += 1

                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(dir=out_dir, delete=False, suffix=".part") as tmp:
                            tmp_path = tmp.name
                            tmp.write(bits)
                        os.replace(tmp_path, path)
                    except Exception as write_err:
                        if tmp_path is not None and os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                        self.error.emit(str(write_err))
                        continue

                # Aggressive VRAM evacuation between files
                self._processor.cleanup()

            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    @pyqtSlot(list, str)
    def run_contact_sheet(self, tasks: List[ExportTask], out_dir: str) -> None:
        """Renders each task small and composites darkroom contact sheet(s) on black."""
        total = len(tasks)
        try:
            tiles = []
            for i, task in enumerate(tasks):
                name = os.path.splitext(task.file_info["name"])[0]
                self.progress.emit(i + 1, total, name)

                tile = self._processor.render_display_array(
                    task.file_info["path"],
                    task.params,
                    task.file_info["hash"],
                    target_long_px=CELL_PX * 2,
                    prefer_gpu=task.gpu_enabled,
                    working_color_space=task.working_color_space,
                )
                if tile is not None:
                    tiles.append(tile)
                self._processor.cleanup()

            sheets = ContactSheetService.build_sheets(tiles)
            os.makedirs(out_dir, exist_ok=True)

            for idx, sheet in enumerate(sheets):
                suffix = "" if idx == 0 else f"_{idx + 1}"
                path = os.path.join(out_dir, f"contact_sheet{suffix}.jpg")
                counter = 2
                while os.path.exists(path):
                    path = os.path.join(out_dir, f"contact_sheet{suffix}_{counter}.jpg")
                    counter += 1

                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(dir=out_dir, delete=False, suffix=".part") as tmp:
                        tmp_path = tmp.name
                        sheet.save(tmp, format="JPEG", quality=95, subsampling=0)
                    os.replace(tmp_path, path)
                except Exception as write_err:
                    if tmp_path is not None and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    self.error.emit(str(write_err))

            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
