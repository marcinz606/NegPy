from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Any, Union
import gc
import math
import os
import tempfile
import threading

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from negpy.domain.models import (
    ColorSpace,
    WorkspaceConfig,
    ExportConfig,
    ExportFormat,
    ExportPreset,
    ExportPresetOutputMode,
    ExportResolutionMode,
)
from negpy.features.metadata import resolution as resolution_source
from negpy.features.metadata.resolution import Resolution
from negpy.features.metadata.writer import embed_metadata, export_embed_plan, preserve_source_metadata
from negpy.features.metadata.fsdate import sync_export_filesystem_dates
from negpy.features.metadata.models import MetadataConfig
from negpy.infrastructure.display.color_spaces import WORKING_COLOR_SPACE, ColorSpaceRegistry
from negpy.services.rendering.image_processor import ImageProcessor
from negpy.features.hdr.models import hdr_frame_paths
from negpy.services.export.print import PrintService
from negpy.services.export.templating import render_export_filename
from negpy.services.export.contact_sheet import ContactSheetService
from negpy.services.export.contact_sheet_layout import (
    MM_PER_INCH,
    ContactSheetSettings,
    SheetFormat,
    best_dpi,
    film_geometry,
    plan_sheets,
)
from negpy.services.export.contact_sheet_roll import SheetFrame, SheetLook, tile_params, turns_for
from negpy.services.export.encoders import encode_jpeg


def _protects_metadata(task: "ExportTask") -> bool:
    return task.metadata_config is not None and task.metadata_config.protect_original_metadata


def _export_resolution(task: "ExportTask") -> Optional[Resolution]:
    """Resolution to write into the exported file, or None to make no claim at all.

    Protect original metadata returns the source's own record untouched — the exact
    rationals and unit, and nothing when it declares nothing. Otherwise Original
    resamples no pixels, so the source still describes them; Print and Pixels do, so
    the size the user asked for wins.

    The cached EXIF only covers files the user has selected, so a batch export of
    untouched frames has to reach the file itself.
    """
    source = resolution_source.read_source(task.file_info.get("path"), task.source_exif)
    if _protects_metadata(task):
        return source
    if task.export_settings.export_resolution_mode == ExportResolutionMode.ORIGINAL and source is not None:
        return source
    return Resolution.from_dpi(PrintService.resolution_tag_dpi(task.export_settings))


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
    # Subfolder of Source's base, overriding the source file's own directory, for a virtual
    # roll with no single folder of its own. None everywhere else.
    roll_export_root: Optional[str] = None


@dataclass(frozen=True)
class ContactSheetJob:
    """Built on the GUI thread; `frames` come in sheet order with the configs they print with."""

    frames: tuple[SheetFrame, ...]
    format: SheetFormat
    frame_size: str
    settings: ContactSheetSettings
    look: SheetLook
    out_dir: str
    gpu_enabled: bool = True
    working_color_space: str = WORKING_COLOR_SPACE
    jpeg_quality: int = 95
    jpeg_progressive: bool = False
    # Each frame's place on the roll; empty means 0, 1, 2 …
    numbers: tuple[int, ...] = ()
    # Frames that start a new strip.
    breaks: tuple[int, ...] = ()


def contact_sheet_paths(out_dir: str, count: int) -> list[str]:
    """One free suffix for every sheet of the set."""
    pages = [""] if count == 1 else [f"_{i + 1}of{count}" for i in range(count)]
    serial = 1
    while True:
        suffix = "" if serial == 1 else f"_{serial}"
        paths = [os.path.join(out_dir, f"contact_sheet{suffix}{page}.jpg") for page in pages]
        if not any(os.path.exists(path) for path in paths):
            return paths
        serial += 1


@dataclass(frozen=True)
class LinearOutputTask:
    """One frame's linear-output job. ``options`` is the keyword payload for
    export_linear_output, resolved on the UI thread where the config lives."""

    file_info: dict
    out_path: str
    options: dict


_EXT = {
    ExportFormat.JPEG: "jpg",
    ExportFormat.TIFF: "tiff",
    ExportFormat.PNG: "png",
    ExportFormat.JXL: "jxl",
    ExportFormat.WEBP: "webp",
}


def resolve_output_dir(source_path: str, settings: ExportPreset, roll_export_root: Optional[str] = None) -> str:
    """Destination folder for one source file, per its output-mode rule. Linear Output
    calls this too, so every intent answers the destination question the same way.

    `roll_export_root` replaces the source file's own directory as the Subfolder of
    Source base. Same as Source and Absolute already name an explicit destination and
    ignore it; only Subfolder of Source needs a stand-in when the file's directory is
    not "the roll's folder" (a virtual roll with no single folder of its own).
    """
    source_dir = os.path.dirname(source_path)
    output_mode = settings.output_mode
    if output_mode == ExportPresetOutputMode.SUBFOLDER_OF_SOURCE:
        base_dir = source_dir if roll_export_root is None else roll_export_root
        subfolder = settings.output_subfolder or ""
        return os.path.join(base_dir, subfolder) if subfolder else base_dir
    if output_mode == ExportPresetOutputMode.ABSOLUTE:
        return settings.output_path or source_dir
    return source_dir


def resolve_export_dir(task: ExportTask) -> str:
    """Destination folder for a task, per its output-mode rule."""
    return resolve_output_dir(task.file_info["path"], task.export_settings, task.roll_export_root)


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
    contact_sheet_written = pyqtSignal(str)  # the folder the sheets went to

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
        """Processes an ordered list of export tasks, pipelined: the prefetcher
        prepares the next source and the finisher encodes+writes the previous
        render while the current one renders. One finish in flight, so at most
        two full-res buffers are held."""
        self._cancel.clear()
        total = len(tasks)
        finisher = ThreadPoolExecutor(max_workers=1)
        prefetcher = ThreadPoolExecutor(max_workers=1)
        pending: Optional[Future] = None

        def _drain(fut: Future) -> None:
            err = fut.result()
            if err:
                self.error.emit(err)

        try:
            for i, task in enumerate(tasks):
                if self._cancel.is_set():
                    break
                full_name = task.file_info["name"]
                name = os.path.splitext(full_name)[0]
                self.progress.emit(i + 1, total, name)

                nxt = tasks[i + 1] if i + 1 < len(tasks) else None
                prefetch_next = nxt is not None and nxt.diptych is None
                # Not on the first task: its own prepare would queue behind this on the gate.
                if prefetch_next and i > 0:
                    self._submit_prefetch(prefetcher, nxt)

                # TIFF/PNG take the metadata at the first encode; the post-hoc
                # rewrite re-compresses the full-res file.
                embed_plan = None
                if task.metadata_config is not None and task.export_settings.export_fmt in (ExportFormat.TIFF, ExportFormat.PNG):
                    embed_plan = export_embed_plan(
                        task.metadata_config,
                        task.source_exif,
                        task.file_info["path"],
                        resolution=_export_resolution(task),
                    )

                buffer, status = self._processor.render_export(
                    task.file_info["path"],
                    task.params,
                    task.export_settings,
                    task.file_info["hash"],
                    prefer_gpu=task.gpu_enabled,
                    bounds_override=task.bounds_override,
                    half=int(task.file_info.get("half") or 0),
                    split_x=float(task.file_info.get("split_x") or 0.5),
                    crop_rect=tuple(task.file_info["crop_rect"]) if task.file_info.get("crop_rect") else None,
                    gutter_thickness=float(task.file_info.get("gutter_thickness") or 0.0),
                    split_axis=str(task.file_info.get("split_axis") or "x"),
                    diptych=task.diptych,
                )
                if prefetch_next and i == 0:
                    self._submit_prefetch(prefetcher, nxt)

                if buffer is None:
                    # render_export returns (None, error) on failure. Surface it rather
                    # than skipping the file silently.
                    self.error.emit(status)
                    continue

                if pending is not None:
                    _drain(pending)
                pending = finisher.submit(self._finish_task, task, buffer, status, embed_plan)

            if pending is not None:
                _drain(pending)
                pending = None
            if self._cancel.is_set():
                self.cancelled.emit()
            else:
                self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            prefetcher.shutdown(wait=True)
            finisher.shutdown(wait=True)
            self._processor.cleanup(release_source_cache=True, collect=False)
            gc.collect()

    def _submit_prefetch(self, prefetcher: ThreadPoolExecutor, nxt: ExportTask) -> None:
        prefetcher.submit(
            self._processor.prefetch_export_source,
            nxt.file_info["path"],
            nxt.params,
            nxt.file_info["hash"],
            half=int(nxt.file_info.get("half") or 0),
            split_x=float(nxt.file_info.get("split_x") or 0.5),
            crop_rect=tuple(nxt.file_info["crop_rect"]) if nxt.file_info.get("crop_rect") else None,
            gutter_thickness=float(nxt.file_info.get("gutter_thickness") or 0.0),
            split_axis=str(nxt.file_info.get("split_axis") or "x"),
        )

    def _finish_task(self, task: ExportTask, buffer: np.ndarray, color_space: str, embed_plan: Optional[tuple]) -> Optional[str]:
        """Encode + metadata + atomic write for one rendered frame, on the finisher
        thread. Returns an error message, or None on success."""
        resolution = _export_resolution(task)
        bits, status = self._processor.encode_export(
            buffer,
            task.export_settings,
            color_space,
            task.working_color_space,
            embed_plan=embed_plan,
            resolution=resolution,
        )
        if not bits:
            return status

        if task.metadata_config is not None and embed_plan is None:
            if task.metadata_config.protect_original_metadata:
                bits = preserve_source_metadata(
                    bits,
                    task.file_info["path"],
                    task.source_exif,
                    resolution=resolution,
                )
            else:
                bits = embed_metadata(bits, task.metadata_config, task.source_exif, resolution=resolution)

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
            # The write above stamps the filesystem dates with the export time; the EXIF
            # dates are already right. Tools that sort by file date read the former.
            sync_export_filesystem_dates(path, task.file_info["path"])
        except Exception as write_err:
            if tmp_path is not None and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return str(write_err)
        return None

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

    @pyqtSlot(object)
    def run_contact_sheet(self, job: "ContactSheetJob") -> None:
        """Sheets are written as `.part` files and moved into place together, so a run never leaves half a set.

        Every exit emits `finished` or `cancelled`: that releases the batch lane.
        """
        self._cancel.clear()
        parts: list[str] = []
        try:
            geometry = film_geometry(job.format, job.frame_size)
            settings = job.settings
            plan = plan_sheets(settings.paper_width, settings.paper_height, geometry, len(job.frames), settings.roll_label, job.breaks)
            if not plan.pages:
                self.error.emit(f"Contact sheet: {plan.reason}")
                self.finished.emit()
                return
            dpi = best_dpi(settings.paper_width, settings.paper_height, settings.dpi)
            px_per_mm = dpi / MM_PER_INCH
            window_long_px = int(math.ceil(max(geometry.frame_along, geometry.frame_across) * px_per_mm))
            target_long_px = int(window_long_px * 1.5)
            paths = contact_sheet_paths(job.out_dir, len(plan.pages))
            os.makedirs(job.out_dir, exist_ok=True)
            icc = _srgb_icc_bytes()
            total = len(job.frames) + len(plan.pages)
            step = 0

            for page_index, page in enumerate(plan.pages):
                tiles: list[Optional[np.ndarray]] = [None] * len(job.frames)
                turns = [0] * len(job.frames)
                indices = [i for strip in page.strips for i in range(strip.first, strip.first + strip.count)]
                for n, index in enumerate(indices):
                    if self._cancel.is_set():
                        self.cancelled.emit()
                        return
                    frame = job.frames[index]
                    info = frame.asset
                    step += 1
                    self.progress.emit(step, total, os.path.splitext(frame.name)[0])
                    next_path = job.frames[indices[n + 1]].asset.get("path") if n + 1 < len(indices) else None
                    tile = self._processor.render_display_array(
                        info["path"],
                        tile_params(frame.config),
                        info["hash"],
                        target_long_px=target_long_px,
                        prefer_gpu=job.gpu_enabled,
                        working_color_space=job.working_color_space,
                        fast_decode=True,
                        half=int(info.get("half") or 0),
                        split_x=float(info.get("split_x") or 0.5),
                        crop_rect=tuple(info["crop_rect"]) if info.get("crop_rect") else None,
                        gutter_thickness=float(info.get("gutter_thickness") or 0.0),
                        split_axis=str(info.get("split_axis") or "x"),
                        keep_source=next_path == info.get("path"),
                    )
                    if tile is None:
                        # The frame prints blank so the numbering stays in step.
                        self.error.emit(f"{frame.name}: could not be rendered for the contact sheet")
                        continue
                    turns[index] = turns_for(frame, geometry, tile.shape[:2])
                    tiles[index] = tile

                if self._cancel.is_set():
                    self.cancelled.emit()
                    return
                step += 1
                self.progress.emit(step, total, f"Sheet {page_index + 1} of {len(plan.pages)}")
                sheet = ContactSheetService.render_sheet(plan, page_index, tiles, turns, px_per_mm, job.look, numbers=job.numbers or None)
                del tiles
                if self._cancel.is_set():
                    self.cancelled.emit()
                    return
                with tempfile.NamedTemporaryFile(dir=job.out_dir, delete=False, suffix=".part") as tmp:
                    parts.append(tmp.name)
                    tmp.write(
                        encode_jpeg(
                            sheet,
                            icc=icc,
                            resolution=Resolution.from_dpi(dpi),
                            quality=job.jpeg_quality,
                            progressive=job.jpeg_progressive,
                        )
                    )
                del sheet

            for part, path in zip(parts, paths):
                os.replace(part, path)
            parts.clear()
            self.contact_sheet_written.emit(job.out_dir)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit()
        finally:
            for part in parts:
                try:
                    os.unlink(part)
                except OSError:
                    pass
            # Release GPU resources once per batch, not per tile (avoids pool rebuild each frame).
            self._processor.cleanup()
