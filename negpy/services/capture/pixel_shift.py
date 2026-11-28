"""A camera that takes a Pixel Shift set per trigger and hands back one merged file.

It wraps the session's camera, so every capture mode sees one file per exposure as before.
"""

import glob
import os
import shutil
import sys
import threading
import traceback
from typing import Any, Callable, Optional, Protocol, Sequence

import numpy as np
import rawpy

from negpy.infrastructure.loaders.constants import PIXEL_SHIFT_FRAMES_FOLDER
from negpy.kernel.system.i18n import tr
from negpy.kernel.system.logging import get_logger
from negpy.kernel.system.memory import available_system_memory_bytes
from negpy.kernel.system.paths import safe_unlink
from negpy.services.capture.pixel_shift_dng import raw_color, write_linear_dng
from negpy.services.capture.pixel_shift_merge import Position, measure_wobble, merge_set
from negpy.services.capture.service import CaptureError
from negpy.services.rendering.prefetch_policy import fits_in_memory

logger = get_logger(__name__)

#: Sets shot for one exposure before one plain frame is kept instead.
MAX_TRIES = 3
#: Where the set's own frames go when they are kept, beside the merged file.
FRAMES_FOLDER = PIXEL_SHIFT_FRAMES_FOLDER


def _load_mosaic(path: str) -> np.ndarray:
    """A RAW's visible mosaic, copied off the decoder."""
    with rawpy.imread(path) as raw:
        return raw.raw_image_visible.copy()


def _load_colors(path: str) -> np.ndarray:
    """A RAW's visible CFA color indices."""
    with rawpy.imread(path) as raw:
        return raw.raw_colors_visible.copy()


def _load_black(path: str) -> float:
    """A RAW's black level."""
    with rawpy.imread(path) as raw:
        return float(np.mean(raw.black_level_per_channel))


def _remove(paths: list[str]) -> None:
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


#: Free disk space left over after the merged file and the merge's own file.
DISK_MARGIN_BYTES = 256 * 1024 * 1024


def merge_buffers(sizes: Sequence[int], path: str, available: Optional[int], free_disk: int) -> tuple[list[np.ndarray], bool]:
    """One uint16 buffer per size (in samples, the merged image first), and whether they are mapped from the file
    `path`: in memory when they fit, else in that file. Raises `CaptureError` when `free_disk` cannot hold the
    merged file and that file."""
    nbytes = 2 * sum(sizes)
    in_memory = fits_in_memory(nbytes, available)
    needed = 2 * sizes[0] + (0 if in_memory else nbytes) + DISK_MARGIN_BYTES
    if free_disk < needed:
        raise CaptureError(
            tr(
                "Not enough free disk space to merge the frames. Free {value} GB and scan again.",
                value=f"{max(0.1, (needed - free_disk) / 1e9):.1f}",
            )
        )
    if in_memory:
        return [np.empty(size, np.uint16) for size in sizes], False
    with open(path, "wb") as handle:
        handle.truncate(nbytes)
    logger.info("Pixel Shift merge works in a file of %.1f GB: free memory is %s", nbytes / 1e9, available)
    # shortcut: the file is sparse, so a disk that fills during the merge ends the process; reserve the blocks if that is seen.
    starts = [2 * sum(sizes[:i]) for i in range(len(sizes))]
    return [np.memmap(path, dtype=np.uint16, mode="r+", offset=start, shape=(size,)) for start, size in zip(starts, sizes)], True


#: The share of one exposure's time that the shots take; the check, the merge and the save take the rest.
_SHOOTING_SHARE = 0.8


class SetCamera(Protocol):
    """A camera that can take a whole set on one trigger."""

    def capture(self, out_path: str, shutter: Optional[str] = None, iso: Optional[str] = None, aperture: Optional[str] = None) -> str: ...

    def capture_set(
        self,
        out_path: str,
        count: int,
        shutter: Optional[str] = None,
        iso: Optional[str] = None,
        aperture: Optional[str] = None,
        on_shot: Optional[Callable[[int, int], None]] = None,
    ) -> list[str]: ...

    def close(self) -> None: ...


class PixelShiftCamera:
    """`Camera` whose `capture` shoots one frame per entry of `positions`, checks them for movement,
    shoots again when they moved, merges them and returns the merged DNG's path. `report` gets one line per step and
    `part` the share of the exposure that is done, 0 to 1."""

    def __init__(
        self,
        camera: SetCamera,
        positions: Sequence[Position],
        *,
        frames_folder: str,
        keep_frames: bool = False,
        fallback: bool = True,
        report: Callable[[str], None] = lambda _text: None,
        part: Callable[[float], None] = lambda _share: None,
        cancel: Optional[threading.Event] = None,
    ) -> None:
        self._camera = camera
        self._positions = tuple(positions)
        self._count = len(self._positions)
        self._frames_folder = frames_folder  # outside the caller's staging folder, which the caller deletes
        self._keep_frames = keep_frames
        self._fallback = fallback  # False where one plain frame cannot stand in: beside two merged channels
        self._report = report
        self._part = part
        self._cancel = cancel

    def capture(self, out_path: str, shutter: Optional[str] = None, iso: Optional[str] = None, aperture: Optional[str] = None) -> str:
        stem = os.path.splitext(out_path)[0]
        for attempt in range(1, MAX_TRIES + 1):
            self._check_cancel()
            frames = self._camera.capture_set(
                out_path,
                self._count,
                shutter=shutter,
                iso=iso,
                aperture=aperture,
                on_shot=self._on_shot,
            )
            rgb: Optional[np.ndarray] = None
            scratch = stem + ".merge"
            try:
                self._check_cancel()
                self._report(tr("Checking the frames for movement…"))
                self._part(_SHOOTING_SHARE)
                wobble = measure_wobble((_load_mosaic(p) for p in frames), self._positions, _load_black(frames[0]))
                logger.info(
                    "Pixel Shift set %d of %d: rms %.3f worst %.3f bands %d soft shot %d uneven shot %d steady %s",
                    attempt,
                    MAX_TRIES,
                    wobble.rms,
                    wobble.worst,
                    wobble.bands,
                    wobble.soft_shot,
                    wobble.uneven_shot,
                    wobble.steady,
                )
                self._check_cancel()
                if wobble.steady:
                    if not wobble.judged:
                        self._report(tr("The frame has too little detail to check for movement."))
                    self._report(tr("Merging {count} frames…", count=self._count))
                    self._part(0.9)
                    rgb = merge_set((_load_mosaic(p) for p in frames), _load_colors(frames[0]), self._positions, self._buffer(scratch))
                    self._part(0.95)  # a steady set is saved even after Stop: it cannot be shot again for free
                elif attempt < MAX_TRIES:
                    _remove(frames)
                    self._report(
                        tr("The set moved or a frame is blurred. Shooting again ({n} of {total})…", n=attempt + 1, total=MAX_TRIES)
                    )
                    continue
                elif self._fallback:
                    best = frames[wobble.sharpest_shot - 1]
                    kept = stem + os.path.splitext(best)[1]
                    os.replace(best, kept)
                    _remove(frames)
                    self._report(tr("Every set moved or was blurred. Kept one frame without Pixel Shift."))
                    return kept
                else:
                    raise CaptureError(tr("Every set moved or was blurred. Check the focus and the rig, then shoot the frame again."))
                return self._save(rgb, frames, stem + ".dng")
            except CaptureError:  # cancelled, moved in every set, or no disk space for the merge: the frames are of no use
                _remove(frames)
                raise
            except BaseException:
                self._move_out(frames)  # the only copy of a steady set that did not become a file
                raise
            finally:
                # The mapping closes before its file goes: an error's traceback holds the frames that hold it.
                rgb = None
                if (error := sys.exception()) is not None:
                    traceback.clear_frames(error.__traceback__)
                safe_unlink(scratch)
        raise AssertionError("unreachable")

    def _buffer(self, scratch: str) -> Callable[[Sequence[int]], Sequence[np.ndarray]]:
        free_disk = shutil.disk_usage(os.path.dirname(scratch) or ".").free
        return lambda sizes: merge_buffers(sizes, scratch, available_system_memory_bytes(), free_disk)[0]

    def _on_shot(self, n: int, count: int) -> None:
        self._report(tr("Shot {n} of {count}", n=n, count=count))
        self._part(_SHOOTING_SHARE * n / count)

    def _move_out(self, frames: list[str]) -> bool:
        """Move the set beside the merged file, in place of an earlier take's frames. False when it failed."""
        try:
            os.makedirs(self._frames_folder, exist_ok=True)
            stem = os.path.basename(frames[0]).rsplit("_S", 1)[0]
            _remove(glob.glob(os.path.join(glob.escape(self._frames_folder), glob.escape(stem) + "_S[0-9][0-9].*")))
            for path in frames:
                if os.path.exists(path):
                    os.replace(path, os.path.join(self._frames_folder, os.path.basename(path)))
            return True
        except OSError:  # the first error is the one to report
            logger.warning("could not move the Pixel Shift frames to %s", self._frames_folder, exc_info=True)
            return False

    def _save(self, rgb: np.ndarray, frames: list[str], dng_path: str) -> str:
        self._report(tr("Saving the merged frame…"))
        write_linear_dng(rgb, raw_color(frames[0]), dng_path, self._count)
        if self._keep_frames:
            if not self._move_out(frames):
                self._report(tr("Could not keep the source frames."))
        else:
            _remove(frames)
        return dng_path

    def _check_cancel(self) -> None:
        if self._cancel is not None and self._cancel.is_set():
            raise CaptureError(tr("capture cancelled"))

    def close(self) -> None:
        self._camera.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._camera, name)
