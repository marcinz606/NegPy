from dataclasses import dataclass
from typing import List, Optional, Any, Union
import gc
import os
import tempfile
import threading
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from negpy.domain.models import ColorSpace, WorkspaceConfig, ExportConfig, ExportFormat, ExportPreset, ExportPresetOutputMode
from negpy.features.metadata.writer import embed_metadata, export_embed_plan, preserve_source_metadata
from negpy.features.metadata.models import MetadataConfig
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE, ColorSpaceRegistry
from negpy.services.rendering.image_processor import ImageProcessor
from negpy.features.hdr.models import hdr_frame_paths
from negpy.services.export.templating import render_export_filename
from negpy.services.export.contact_sheet import ContactSheetService


def _srgb_icc_bytes() -> Optional[bytes]:
    """Bundled sRGB profile for tagging contact sheets (tiles are display/sRGB)."""
    path = ColorSpaceRegistry.get_icc_path(ColorSpace.SRGB.value)
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None


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
    # The two halves' own edits, for a whole-frame scan that was worked on split.
    # Set means one file holding both frames; `params` is then only a naming/metadata carrier.
    diptych: Optional[tuple[WorkspaceConfig, WorkspaceConfig]] = None


@dataclass(frozen=True)
class LinearOutputTask:
    """One frame's linear-output job. ``options`` is the keyword payload for
    export_linear_output, resolved on the UI thread where the config lives."""

    file_info: dict
    out_path: str
    options: dict


def _same_decode_source(a: ExportTask, b: ExportTask) -> bool:
    """True when the decoded f32 source cache is reusable for the next task
    (mirrors the _load_source_f32 cache key; the key still verifies on read)."""
    return (
        a.file_info["path"] == b.file_info["path"]
        and a.params.process.linear_raw == b.params.process.linear_raw
        and a.params.process.sensor_matrix == b.params.process.sensor_matrix
        and a.params.rgbscan == b.params.rgbscan
        and a.params.flatfield == b.params.flatfield
    )


_EXT = {
    ExportFormat.JPEG: "jpg",
    ExportFormat.TIFF: "tiff",
    ExportFormat.PNG: "png",
    ExportFormat.JXL: "jxl",
    ExportFormat.WEBP: "webp",
}


def resolve_output_dir(source_path: str, settings: ExportPreset) -> str:
    """Destination folder for one source file, per its output-mode rule. Linear Output
    calls this too, so every intent answers the destination question the same way."""
    source_dir = os.path.dirname(source_path)
    output_mode = settings.output_mode
    if output_mode == ExportPresetOutputMode.SUBFOLDER_OF_SOURCE:
        subfolder = settings.output_subfolder or ""
        return os.path.join(source_dir, subfolder) if subfolder else source_dir
    if output_mode == ExportPresetOutputMode.ABSOLUTE:
        return settings.output_path or source_dir
    return source_dir


def resolve_export_dir(task: ExportTask) -> str:
    """Destination folder for a task, per its output-mode rule."""
    return resolve_output_dir(task.file_info["path"], task.export_settings)


def resolve_export_naming(task: ExportTask) -> tuple[str, str, str]:
    """(out_dir, filename-stem, extension) for a task — the shared source of truth for
    both conflict detection and the actual write, so they can never disagree."""
    out_dir = resolve_export_dir(task)
    ext = _EXT.get(task.export_settings.export_fmt, "jpg")
    frames = hdr_frame_paths(task.file_info)
    filename = render_export_filename(
        # A merge is named after its alphabetically first frame, not the reference frame its path
        # points at, because the reference is chosen from picture content.
        min(frames, key=lambda p: os.path.basename(p).lower()) if frames else task.file_info["path"],
        task.export_settings,
        border_size=task.params.finish.border_size,
        half=int(task.file_info.get("half") or 0),
        metadata=task.metadata_config,
        composite="DIPTYCH" if task.diptych else ("HDR" if frames else ""),
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

                # TIFF/PNG take the metadata at the first encode; the post-hoc
                # rewrite re-compresses the full-res file.
                embed_plan = None
                if task.metadata_config is not None and task.export_settings.export_fmt in (ExportFormat.TIFF, ExportFormat.PNG):
                    embed_plan = export_embed_plan(
                        task.metadata_config,
                        task.source_exif,
                        task.file_info["path"],
                    )

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
                    crop_rect=tuple(task.file_info["crop_rect"]) if task.file_info.get("crop_rect") else None,
                    gutter_thickness=float(task.file_info.get("gutter_thickness") or 0.0),
                    diptych=task.diptych,
                    embed_plan=embed_plan,
                )

                if not bits:
                    # process_export returns (None, error) on failure. Surface it rather than skipping the
                    # file silently.
                    self.error.emit(status)
                    continue

                if bits:
                    if task.metadata_config is not None and embed_plan is None:
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

                # The engine evicts stale pool textures itself per render; the decoded
                # source is kept until the next task needs a different file.
                nxt = tasks[i + 1] if i + 1 < len(tasks) else None
                if nxt is None or not _same_decode_source(task, nxt):
                    self._processor.release_source_cache()

            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self._processor.cleanup(release_source_cache=True, collect=False)
            gc.collect()

    @pyqtSlot(list)
    def run_linear_output(self, tasks: List[LinearOutputTask]) -> None:
        """Writes each frame's decoded linear buffer. Own slot rather than a branch in
        run_batch: linear output bypasses the render pipeline and the export settings."""
        from negpy.services.export.linear_output import export_linear_output

        self._cancel.clear()
        total = len(tasks)
        try:
            for i, task in enumerate(tasks):
                if self._cancel.is_set():
                    self.cancelled.emit()
                    return
                name = os.path.splitext(task.file_info["name"])[0]
                self.progress.emit(i + 1, total, name)
                try:
                    export_linear_output(task.file_info["path"], task.out_path, **task.options)
                except Exception as e:
                    self.error.emit(f"Linear Output failed for {name}: {e}")

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
                    crop_rect=tuple(task.file_info["crop_rect"]) if task.file_info.get("crop_rect") else None,
                    gutter_thickness=float(task.file_info.get("gutter_thickness") or 0.0),
                )
                if tile is not None:
                    tiles.append(tile)
                    labels.append(task.file_info["name"])
                else:
                    # A dropped tile silently shrinks the sheet, so report it or the run looks like a clean
                    # success with frames missing.
                    self.error.emit(f"{name}: could not be rendered for the contact sheet")

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

            sheet_icc = _srgb_icc_bytes()
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
                        sheet.save(tmp, format="JPEG", quality=95, subsampling=0, icc_profile=sheet_icc)
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
