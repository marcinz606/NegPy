from dataclasses import dataclass
from typing import List, Optional, Any, Union
import gc
import os
import tempfile
import threading
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from negpy.domain.models import WorkspaceConfig, ExportConfig, ExportFormat, ExportPreset, ExportPresetOutputMode
from negpy.features.metadata.writer import embed_metadata, preserve_source_metadata
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


def _same_decode_source(a: ExportTask, b: ExportTask) -> bool:
    """True when the decoded f32 source cache is reusable for the next task
    (mirrors the _load_source_f32 cache key; the key still verifies on read)."""
    return (
        a.file_info["path"] == b.file_info["path"]
        and a.params.process.linear_raw == b.params.process.linear_raw
        and a.params.rgbscan == b.params.rgbscan
        and a.params.flatfield == b.params.flatfield
    )


_EXT = {
    ExportFormat.JPEG: "jpg",
    ExportFormat.TIFF: "tiff",
    ExportFormat.PNG: "png",
    ExportFormat.DNG: "dng",
    ExportFormat.JXL: "jxl",
    ExportFormat.WEBP: "webp",
}


def resolve_export_dir(task: ExportTask) -> str:
    """Destination folder for a task, per its output-mode rule."""
    source_dir = os.path.dirname(task.file_info["path"])
    output_mode = task.export_settings.output_mode
    if output_mode == ExportPresetOutputMode.SUBFOLDER_OF_SOURCE:
        subfolder = task.export_settings.output_subfolder or ""
        return os.path.join(source_dir, subfolder) if subfolder else source_dir
    if output_mode == ExportPresetOutputMode.ABSOLUTE:
        return task.export_settings.output_path or source_dir
    return source_dir


def resolve_export_naming(task: ExportTask) -> tuple[str, str, str]:
    """(out_dir, filename-stem, extension) for a task — the shared source of truth for
    both conflict detection and the actual write, so they can never disagree."""
    out_dir = resolve_export_dir(task)
    ext = _EXT.get(task.export_settings.export_fmt, "jpg")
    filename = render_export_filename(
        task.file_info["path"],
        task.export_settings,
        border_size=task.params.finish.border_size,
        half=int(task.file_info.get("half") or 0),
    )
    return out_dir, filename, ext


def resolve_export_target_path(task: ExportTask) -> str:
    """The path a task writes to before any overwrite/rename resolution."""
    out_dir, filename, ext = resolve_export_naming(task)
    return os.path.join(out_dir, f"{filename}.{ext}")


def find_export_conflicts(tasks: List[ExportTask]) -> List[str]:
    """Target paths for the batch that already exist on disk (would be overwritten)."""
    return [path for task in tasks if os.path.exists(path := resolve_export_target_path(task))]


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
                    half=int(task.file_info.get("half") or 0),
                    split_x=float(task.file_info.get("split_x") or 0.5),
                )

                if not bits:
                    # process_export returns (None, error) on failure; surface it
                    # rather than silently skipping the file.
                    self.error.emit(status)
                    continue

                if bits:
                    # Skipped for DNG (EXIF re-write strips DNG tags), JXL
                    # (embed_metadata corrupts the .jxl stream), and WebP
                    # (embed_metadata has no WebP branch).
                    if task.metadata_config is not None and task.export_settings.export_fmt not in (
                        ExportFormat.DNG,
                        ExportFormat.JXL,
                        ExportFormat.WEBP,
                    ):
                        if task.metadata_config.protect_original_metadata:
                            bits = preserve_source_metadata(
                                bits,
                                task.file_info["path"],
                                task.source_exif,
                            )
                        else:
                            bits = embed_metadata(bits, task.metadata_config, task.source_exif)

                    out_dir, filename, ext = resolve_export_naming(task)
                    os.makedirs(out_dir, exist_ok=True)
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

                # VRAM evacuated per file; decoded source kept for a same-file next
                # task (multi-format presets), gc deferred to once per batch.
                nxt = tasks[i + 1] if i + 1 < len(tasks) else None
                self._processor.cleanup(
                    release_source_cache=nxt is None or not _same_decode_source(task, nxt),
                    collect=False,
                )

            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            gc.collect()

    @pyqtSlot(list, str, int, int, int, int, bool, str, str)
    def run_contact_sheet(
        self,
        tasks: List[ExportTask],
        out_dir: str,
        cell_px: int,
        gap: int,
        margin: int,
        max_tiles: int,
        show_labels: bool,
        background_color: str,
        label_color: str,
    ) -> None:
        """Renders each task small and composites contact sheet(s)."""
        self._cancel.clear()
        total = len(tasks)
        try:
            tiles = []
            labels: list[str] = []
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
                    # half-size decode is visually identical at ~600px proof tiles
                    fast_decode=True,
                    half=int(task.file_info.get("half") or 0),
                    split_x=float(task.file_info.get("split_x") or 0.5),
                )
                if tile is not None:
                    tiles.append(tile)
                    labels.append(task.file_info["name"])

            sheets = ContactSheetService.build_sheets(
                tiles,
                labels=labels if show_labels else None,
                show_labels=show_labels,
                background_color=background_color,
                label_color=label_color,
                max_tiles=max_tiles,
                cell_px=cell_px,
                gap=gap,
                margin=margin,
            )
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
        finally:
            # Release GPU resources once per batch, not per tile (avoids pool rebuild each frame).
            self._processor.cleanup()
