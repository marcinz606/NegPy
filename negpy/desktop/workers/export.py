from dataclasses import dataclass
from typing import List, Optional, Any, Union
import os
import tempfile
import threading
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from negpy.domain.models import WorkspaceConfig, ExportConfig, ExportFormat, ExportPreset, ExportPresetOutputMode
from negpy.features.metadata.writer import embed_metadata
from negpy.features.metadata.models import MetadataConfig
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE
from negpy.services.rendering.image_processor import ImageProcessor
from negpy.services.export.templating import render_export_filename
from negpy.services.export.contact_sheet import ContactSheetService


@dataclass(frozen=True)
class ExportTask:
    """Immutable data for a high-resolution export job."""

    file_info: dict
    params: WorkspaceConfig
    export_settings: Union[ExportConfig, ExportPreset]
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
    cancelled = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._processor = ImageProcessor()
        self._cancel = threading.Event()

    @pyqtSlot()
    def cancel(self) -> None:
        """Requests the running batch stop after the current file (keeps partial output)."""
        self._cancel.set()

    @pyqtSlot(list)
    def run_batch(self, tasks: List[ExportTask]) -> None:
        """Processes an ordered list of export tasks."""
        self._cancel.clear()
        total = len(tasks)
        try:
            for i, task in enumerate(tasks):
                if self._cancel.is_set():
                    self.cancelled.emit()
                    return
                full_name = task.file_info["name"]
                name = os.path.splitext(full_name)[0]
                self.progress.emit(i + 1, total, name)

                bits, status = self._processor.process_export(
                    task.file_info["path"],
                    task.params,
                    task.export_settings,
                    task.file_info["hash"],
                    prefer_gpu=task.gpu_enabled,
                    bounds_override=task.bounds_override,
                    working_color_space=task.working_color_space,
                )

                if not bits:
                    # process_export returns (None, error) on failure; surface it
                    # rather than silently skipping the file.
                    self.error.emit(status)
                    continue

                if bits:
                    # Skipped for DNG (EXIF re-write strips DNG tags) and JXL
                    # (embed_metadata corrupts the .jxl stream).
                    if task.metadata_config is not None and task.export_settings.export_fmt not in (
                        ExportFormat.DNG,
                        ExportFormat.JXL,
                    ):
                        bits = embed_metadata(bits, task.metadata_config, task.source_exif)

                    source_dir = os.path.dirname(task.file_info["path"])
                    output_mode = task.export_settings.output_mode
                    if output_mode == ExportPresetOutputMode.SUBFOLDER_OF_SOURCE:
                        subfolder = task.export_settings.output_subfolder or ""
                        out_dir = os.path.join(source_dir, subfolder) if subfolder else source_dir
                    elif output_mode == ExportPresetOutputMode.ABSOLUTE:
                        out_dir = task.export_settings.output_path or source_dir
                    else:
                        out_dir = source_dir
                    os.makedirs(out_dir, exist_ok=True)

                    _EXT = {
                        ExportFormat.JPEG: "jpg",
                        ExportFormat.TIFF: "tiff",
                        ExportFormat.PNG: "png",
                        ExportFormat.DNG: "dng",
                        ExportFormat.JXL: "jxl",
                    }
                    ext = _EXT.get(task.export_settings.export_fmt, "jpg")

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

    @pyqtSlot(list, str, int, int, int, int)
    def run_contact_sheet(self, tasks: List[ExportTask], out_dir: str, cell_px: int, gap: int, margin: int, max_tiles: int) -> None:
        """Renders each task small and composites darkroom contact sheet(s) on black."""
        self._cancel.clear()
        total = len(tasks)
        try:
            tiles = []
            for i, task in enumerate(tasks):
                if self._cancel.is_set():
                    self.cancelled.emit()
                    return
                name = os.path.splitext(task.file_info["name"])[0]
                self.progress.emit(i + 1, total, name)

                tile = self._processor.render_display_array(
                    task.file_info["path"],
                    task.params,
                    task.file_info["hash"],
                    target_long_px=cell_px * 2,
                    prefer_gpu=task.gpu_enabled,
                    working_color_space=task.working_color_space,
                )
                if tile is not None:
                    tiles.append(tile)
                self._processor.cleanup()

            sheets = ContactSheetService.build_sheets(tiles, max_tiles=max_tiles, cell_px=cell_px, gap=gap, margin=margin)
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
