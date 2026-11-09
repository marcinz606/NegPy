"""Estimate Roll: reads every slide of a roll and pools one set of Fade Restoration values."""

import threading
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Optional

import numpy as np

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.desktop.workers.chart_fit import ChartSource, fade_estimate_grid
from negpy.desktop.workers.render import _decode_asset_preview_with_meta, gain_slices_for_asset
from negpy.domain.models import WorkspaceConfig
from negpy.features.process.fade import (
    blend_gray_readings,
    estimate_red,
    gray_readings,
    measure_crossover,
    pick_neutrals,
    render_white_space,
    survival_faded,
    tone_bands_only,
)
from negpy.features.process.fade_auto_red import AUTO_RED_HINT_RATIO, auto_red, range_shift, survival_points
from negpy.features.process.fade_roll import CrossoverPick, FrameReading, pick_crossover, pool_red, pool_survivals
from negpy.kernel.system.i18n import tr
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FadeRollTask:
    """The frames to read, each with its own config. Every frame is read as its own Estimate reads it,
    at its best Gray Mix and without Remove Stain, so the open frame does not change the result."""

    frames: list[tuple[dict, WorkspaceConfig]]
    workspace_color_space: str


@dataclass
class FadeRollResult:
    """The pooled values for the Fade card, and what the pool left out."""

    changes: dict = field(default_factory=dict)
    read: int = 0
    counted: int = 0
    outliers: list[str] = field(default_factory=list)
    no_grays: list[str] = field(default_factory=list)
    #: Frames that did not decode or gave no grid.
    unread: list[str] = field(default_factory=list)
    #: Frames whose own range is short enough for Restore Range.
    short_range: int = 0
    #: The frame whose crossover table helps the most other frames, None with none.
    crossover: Optional[CrossoverPick] = None
    reason: str = ""


class _Frame(NamedTuple):
    """One frame with clear grays: its asset, Estimate grid, white space, gray reader and reading."""

    info: dict
    config: WorkspaceConfig
    grid: np.ndarray
    space: str
    reader: Callable
    reading: FrameReading


class FadeRollWorker(QObject):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(object)
    cancelled = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, preview_service) -> None:
        super().__init__()
        self._preview_service = preview_service
        self._cancel = threading.Event()

    @pyqtSlot()
    def cancel(self) -> None:
        self._cancel.set()

    def arm(self) -> None:
        self._cancel.clear()

    @pyqtSlot(FadeRollTask)
    def process(self, task: FadeRollTask) -> None:
        try:
            result = self._run(task)
        except Exception as e:
            logger.error(f"Estimate Roll failure: {e}")
            self.error.emit(str(e))
            return
        if result is None:
            self.cancelled.emit()
        else:
            self.finished.emit(result)

    def _read(self, info: dict, config: WorkspaceConfig, task: FadeRollTask):
        """(grid, white space) of one frame as the open frame's Estimate reads it."""
        raw, meta = _decode_asset_preview_with_meta(self._preview_service, info, config, task.workspace_color_space)
        cam_xyz, camera_wb = meta.get("cam_xyz"), meta.get("camera_wb")
        source = ChartSource(
            raw, config, info["hash"], cam_xyz, camera_wb, gain_slices_for_asset(info), int(meta.get("orientation", 1) or 1)
        )
        grid = fade_estimate_grid(source)
        process = config.process
        return grid, render_white_space(process, process.process_mode, cam_xyz, camera_wb)

    def _run(self, task: FadeRollTask) -> Optional[FadeRollResult]:
        result = FadeRollResult()
        frames = []
        total = len(task.frames)
        for n, (info, config) in enumerate(task.frames, 1):
            if self._cancel.is_set():
                return None
            self.progress.emit(n, total, info["name"])
            try:
                grid, space = self._read(info, config, task)
            except Exception as e:
                logger.error(f"Estimate Roll could not read {info['name']}: {e}")
                grid = None
            if grid is None:
                result.unread.append(info["name"])
                continue
            result.read += 1
            readings = gray_readings(grid)
            if tone_bands_only(readings):
                result.no_grays.append(info["name"])
                continue

            picked = pick_neutrals(grid, readings, camera_raw=bool(space), white_space=space)
            if picked is None:
                result.no_grays.append(info["name"])
                continue
            mix, est = picked
            refs, _ = blend_gray_readings(readings, mix)
            if est.red is None or est.reason or refs is None:
                result.no_grays.append(info["name"])
                continue

            def reader(_grid, readings=readings, mix=mix):
                return blend_gray_readings(readings, mix)

            frames.append(_Frame(info, config, grid, space, reader, FrameReading(info["hash"], est.green, est.blue)))

        pooled = pool_survivals([f.reading for f in frames])
        if pooled is None:
            result.reason = tr("Too few frames with clear grays to estimate the roll.")
            return result
        names = {f.info["hash"]: f.info["name"] for f in frames}
        result.outliers = [names[key] for key in pooled.outliers]
        kept = [f for f in frames if f.reading.key not in pooled.outliers]
        result.counted = len(kept)
        green, blue = round(pooled.green, 3), round(pooled.blue, 3)
        red, shift = estimate_red(green, blue), 0.0
        # Restore Range acts on the roll where most frames would suggest it on their own Estimate.
        ranged = False
        reds, shift_at = [], []
        for f in kept:
            if self._cancel.is_set():
                return None
            memo: dict = {}

            def points_at(r, f=f, memo=memo):
                if r not in memo:
                    memo[r] = survival_points(f.grid, f.config.process, (r, green, blue), False, bool(f.space), f.space, 0.0)
                return memo[r]

            found, _ = auto_red(points_at, red, 0.0)
            if found is not None:
                reds.append(found["fade_ratio_r"])
                shift_at.append(lambda r, points_at=points_at, at=red: range_shift(points_at(at), points_at(r)))
        result.short_range = sum(1 for r in reds if r <= red * AUTO_RED_HINT_RATIO)
        if 2 * result.short_range > len(kept):
            pooled_red = pool_red(reds, shift_at)
            if pooled_red is not None:
                (red, shift), ranged = pooled_red, True
        candidates = []
        for f in kept:
            if self._cancel.is_set():
                return None
            faded = survival_faded(f.grid, f.reader, (red, green, blue), camera_raw=bool(f.space), white_space=f.space, shift=shift)
            if faded is not None:
                candidates.append((f.reading.key, faded[0], faded[1], measure_crossover(*faded)))
        pick = pick_crossover(candidates)
        result.crossover = None if pick is None else pick._replace(key=names[pick.key])
        result.changes = dict(
            fade_ratio_r=red, fade_ratio_g=green, fade_ratio_b=blue, fade_density_shift=shift, fade_auto_red=ranged, fade_white_anchor=False
        )
        return result
