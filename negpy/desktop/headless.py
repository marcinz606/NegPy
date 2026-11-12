"""``NegPy --render``: write positives of scanned negatives with no window, then exit.

Each scan renders with its saved NegPy edit, or with the standard settings and the film check a
first open runs. The command reads NegPy's databases and never writes them."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from typing import Callable, List, Optional

from negpy.desktop.launch import LaunchRequest
from negpy.domain.models import (
    ExportConfig,
    ExportPreset,
    WorkspaceConfig,
    flat_export_config,
    flat_master_config,
    preset_from_export_config,
)
from negpy.features.geometry.models import crops_frame, with_auto_crop
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger
from negpy.services.export.launch_recipe import LaunchRecipe, apply_recipe, recipe_intent

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1


class _ReadOnlyEdits:
    """Saved edits and export presets, read from the databases only when they exist: opening a
    missing SQLite file creates it, and this command never writes NegPy's data."""

    def __init__(self) -> None:
        self._repo = None
        if os.path.isfile(APP_CONFIG.edits_db_path) and os.path.isfile(APP_CONFIG.settings_db_path):
            from negpy.infrastructure.storage.repository import StorageRepository

            self._repo = StorageRepository(APP_CONFIG.edits_db_path, APP_CONFIG.settings_db_path)

    def edit_for(self, file_hash: str, path: str) -> Optional[WorkspaceConfig]:
        if self._repo is None:
            return None
        try:
            saved = self._repo.load_file_settings(file_hash)
            if saved is None:
                found = self._repo.load_file_settings_by_path(path)
                saved = found[1] if found else None
            return saved
        except Exception:
            logger.warning("Could not read the saved edit of %s", path, exc_info=True)
            return None

    def preset(self, name: str) -> Optional[ExportPreset]:
        if self._repo is None:
            return None
        wanted = name.strip().lower()
        try:
            return next((p for p in self._repo.load_export_presets() if p.name.strip().lower() == wanted), None)
        except Exception:
            logger.warning("Could not read the export presets", exc_info=True)
            return None


def expand_inputs(paths: tuple[str, ...]) -> tuple[List[str], List[str]]:
    """(scans to render, inputs not found). A folder gives its supported files, not its subfolders."""
    from negpy.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS

    supported = {ext.lower() for ext in SUPPORTED_RAW_EXTENSIONS}
    scans: List[str] = []
    missing: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            names = sorted(os.listdir(path), key=str.lower)
            scans += [
                os.path.join(path, n)
                for n in names
                if os.path.isfile(os.path.join(path, n)) and os.path.splitext(n)[1].lower() in supported
            ]
        elif os.path.isfile(path):
            scans.append(path)
        else:
            missing.append(path)
    return scans, missing


def _detect_source_color(path: str) -> Optional[bool]:
    from negpy.features.process.logic import detect_source_color
    from negpy.services.rendering.preview_manager import PreviewManager

    scan = PreviewManager().decode_for_detection(path)
    return detect_source_color(scan) if scan is not None else None


def render_config(
    saved: Optional[WorkspaceConfig],
    recipe: LaunchRecipe,
    detect: Callable[[], Optional[bool]],
) -> WorkspaceConfig:
    """The edit a preview renders with: the saved one or the defaults, with the film check of a
    first open and, for ``--auto-crop``, Auto Crop armed on a frame that has no crop of its own."""
    config = saved if saved is not None else WorkspaceConfig()
    if config.process.source_color is None:
        source_color = detect()
        if source_color is not None:
            config = replace(config, process=replace(config.process, source_color=source_color))
    if recipe.auto_crop and not crops_frame(config.geometry) and not config.geometry.crop_from_auto:
        config = with_auto_crop(config, True)
    return config


def run_render(request: LaunchRequest, emit: Callable[[str], None] = lambda m: print(m, file=sys.stderr)) -> int:
    """Render every scan of ``request`` and return the exit code: 0 when all were written."""
    from negpy.desktop.workers.export import ExportTask, ExportWorker
    from negpy.kernel.image.logic import calculate_file_hash

    recipe = request.recipe
    edits = _ReadOnlyEdits()
    preset = None
    if recipe.preset:
        preset = edits.preset(recipe.preset)
        if preset is None:
            emit(f"NegPy: export preset not found: {recipe.preset}")
            return EXIT_FAILED
    flat = recipe_intent(recipe, preset) == "flat"
    delivery = apply_recipe(ExportConfig(), recipe, preset)

    scans, missing = expand_inputs(request.paths)
    failures = [f"not found: {p}" for p in missing]
    tasks = []
    for path in scans:
        try:
            file_hash = calculate_file_hash(path)
            config = render_config(edits.edit_for(file_hash, path), recipe, lambda p=path: _detect_source_color(p))
            export = flat_export_config(delivery) if flat else delivery
            params = replace(flat_master_config(config) if flat else config, export=export)
            tasks.append(
                ExportTask(
                    file_info={"path": path, "name": os.path.basename(path), "hash": file_hash},
                    params=params,
                    export_settings=preset_from_export_config(export),
                    metadata_config=params.metadata,
                )
            )
        except Exception as error:
            logger.exception("Could not prepare %s", path)
            failures.append(f"{os.path.basename(path)}: {error}")

    if tasks:
        worker = ExportWorker()
        worker.error.connect(failures.append)
        worker.warning.connect(lambda message: emit(f"NegPy: {message}"))
        worker.run_batch(tasks)

    for failure in failures:
        logger.error("Render failed: %s", failure)
        emit(f"NegPy: {failure}")
    if not scans and not missing:
        emit("NegPy: no supported scans found")
        return EXIT_FAILED
    logger.info("Render finished: %d scans, %d failures", len(scans), len(failures))
    return EXIT_FAILED if failures else EXIT_OK
