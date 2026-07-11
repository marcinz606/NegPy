"""Trichromatic R/G/B triplet capture orchestration.

Sets the Scanlight to one narrowband channel, fires the camera, repeats for red,
green, blue — then turns the light off. The three RAWs are handed to NegPy's
RGB-Scan mode, which aligns/merges/inverts them. Hardware is injected, so the
whole flow is unit-testable with fakes.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from typing import Callable, Optional

from negpy.infrastructure.capture.base import (
    CAPTURE_ORDER,
    Camera,
    CaptureResult,
    CaptureSettings,
    Channel,
    LightSource,
)
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

#: Placeholder suffix on the path handed to `Camera.capture`. The camera replaces it with
#: its own — `.ARW`, `.CR3`, `.NEF` — and returns where the file actually landed, so this
#: is only ever a stem separator.
_RAW_SUFFIX = ".raw"

ProgressCb = Callable[[float], None]


class CaptureError(RuntimeError):
    """Raised when a triplet capture cannot complete."""


def verify_raw_size(path: str, min_bytes: int, max_bytes: int) -> None:
    """Raise CaptureError if the captured RAW is missing or implausibly sized."""
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise CaptureError(f"captured file is missing: {path}") from exc
    if not min_bytes <= size <= max_bytes:
        lo, hi = min_bytes // 1024 // 1024, max_bytes // 1024 // 1024
        raise CaptureError(
            f"{os.path.basename(path)} is {size / 1024 / 1024:.1f} MB, outside the plausible "
            f"RAW range ({lo}-{hi} MB) — did the film move or the camera misfire?"
        )


def _capture_validated_single(
    camera: Camera,
    *,
    final_stem: str,
    shutter: Optional[str],
    min_raw_bytes: int,
    max_raw_bytes: int,
    cancel: Optional[threading.Event] = None,
) -> str:
    """Capture beside the destination and replace it only after size validation."""
    output_folder = os.path.dirname(final_stem)
    staging_dir = tempfile.mkdtemp(prefix=f".{os.path.basename(final_stem)}-", suffix=".capture", dir=output_folder)
    try:
        staged_stem = os.path.join(staging_dir, os.path.basename(final_stem))
        staged_path = camera.capture(staged_stem + _RAW_SUFFIX, shutter=shutter)
        verify_raw_size(staged_path, min_raw_bytes, max_raw_bytes)
        final_path = os.path.join(output_folder, os.path.basename(staged_path))
        if cancel is not None and cancel.is_set():
            raise CaptureError("capture cancelled")
        os.replace(staged_path, final_path)
        return final_path
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def capture_single(
    camera: Camera,
    *,
    roll_name: str,
    frame_number: int,
    output_folder: str,
    shutter: Optional[str] = None,
    min_raw_bytes: int = 8 * 1024 * 1024,
    max_raw_bytes: int = 200 * 1024 * 1024,
    cancel: Optional[threading.Event] = None,
) -> str:
    """One camera exposure with NO light control — for normal white-light scanning without the
    Scanlight (the operator's own high-CRI light stays on). One file, no R/G/B split; imported
    into NegPy as an ordinary single RAW (no triplet merge)."""
    os.makedirs(output_folder, exist_ok=True)
    stem = os.path.join(output_folder, f"{roll_name}_Frame{frame_number:03d}")
    logger.info("capturing single (white-light) frame → %s", stem)
    return _capture_validated_single(
        camera,
        final_stem=stem,
        shutter=shutter,
        min_raw_bytes=min_raw_bytes,
        max_raw_bytes=max_raw_bytes,
        cancel=cancel,
    )


def _promote_triplet(staged_paths: dict[Channel, str], output_folder: str) -> dict[Channel, str]:
    """Replace a frame's three channel files as one recoverable operation.

    All captures have already passed validation before this runs. Existing files are
    backed up inside the staging directory so an I/O failure during promotion can roll
    every channel back to the prior complete triplet.
    """
    final_paths = {channel: os.path.join(output_folder, os.path.basename(path)) for channel, path in staged_paths.items()}
    staging_dir = os.path.dirname(next(iter(staged_paths.values())))
    backups: dict[Channel, str] = {}
    promoted: list[Channel] = []

    try:
        for channel in CAPTURE_ORDER:
            final_path = final_paths[channel]
            if os.path.exists(final_path):
                backup_path = os.path.join(staging_dir, f".previous-{os.path.basename(final_path)}")
                try:
                    os.link(final_path, backup_path)
                except OSError:
                    shutil.copy2(final_path, backup_path)
                backups[channel] = backup_path

        for channel in CAPTURE_ORDER:
            os.replace(staged_paths[channel], final_paths[channel])
            promoted.append(channel)
    except OSError as exc:
        for channel in reversed(promoted):
            final_path = final_paths[channel]
            backup_path = backups.get(channel)
            try:
                if backup_path is not None and os.path.exists(backup_path):
                    os.replace(backup_path, final_path)
                else:
                    os.remove(final_path)
            except OSError:
                logger.exception("could not roll back channel %s after promotion failure", channel.letter)
        raise CaptureError(f"could not promote completed triplet: {exc}") from exc

    return final_paths


class CaptureService:
    """Orchestrates one R/G/B triplet capture from an injected light + camera."""

    def __init__(
        self,
        light: LightSource,
        camera: Camera,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._light = light
        self._camera = camera
        self._sleep = sleep

    def set_light(self, r: int, g: int, b: int) -> None:
        """Live light control for the RGB sliders (no capture)."""
        self._light.set_color(r=r, g=g, b=b)

    def capture_triplet(
        self,
        settings: CaptureSettings,
        progress: Optional[ProgressCb] = None,
        cancel: Optional[threading.Event] = None,
        on_channel: Optional[Callable[[str], None]] = None,
    ) -> CaptureResult:
        """Capture red, green, blue exposures of the current frame.

        Each channel: light that channel only → settle → capture → size-check.
        The light is always turned off when done (even on error). Raises
        `CaptureError` on cancel or an implausibly sized file.
        """
        os.makedirs(settings.output_folder, exist_ok=True)
        frame_name = f"{settings.roll_name}_Frame{settings.frame_number:03d}"
        staging_dir = tempfile.mkdtemp(prefix=f".{frame_name}-", suffix=".capture", dir=settings.output_folder)
        level = {Channel.RED: settings.levels[0], Channel.GREEN: settings.levels[1], Channel.BLUE: settings.levels[2]}
        staged_paths: dict[Channel, str] = {}
        paths: dict[Channel, str] = {}

        try:
            for i, ch in enumerate(CAPTURE_ORDER):
                if cancel is not None and cancel.is_set():
                    raise CaptureError("capture cancelled")
                if on_channel is not None:
                    on_channel(ch.letter)

                r, g, b = ch.rgb(level[ch])
                _t0 = time.perf_counter()
                self._light.set_color(r=r, g=g, b=b)
                self._sleep(settings.settle_s)
                _t1 = time.perf_counter()

                stem = os.path.join(
                    staging_dir,
                    f"{frame_name}_{ch.letter}",
                )
                shutter = settings.shutters[i] if settings.shutters is not None else None
                out_path = self._camera.capture(stem + _RAW_SUFFIX, shutter=shutter)
                self._verify_size(out_path, settings)
                staged_paths[ch] = out_path
                # Per-channel timing so the scan-speed bottleneck (settle vs. shutter+RAW download)
                # is visible in the log: settle is a fixed wait, capture+download is transport-bound.
                logger.info(
                    "channel %s: settle %.0f ms, capture+download %.0f ms → %s",
                    ch.letter,
                    (_t1 - _t0) * 1000,
                    (time.perf_counter() - _t1) * 1000,
                    os.path.basename(out_path),
                )

                if progress is not None:
                    progress((i + 1) / len(CAPTURE_ORDER))

            if cancel is not None and cancel.is_set():
                raise CaptureError("capture cancelled")
            paths = _promote_triplet(staged_paths, settings.output_folder)
        finally:
            try:
                self._light.off()
            except Exception:
                logger.exception("failed to turn the Scanlight off after capture")
            shutil.rmtree(staging_dir, ignore_errors=True)

        return CaptureResult(
            frame_number=settings.frame_number,
            red_path=paths[Channel.RED],
            green_path=paths[Channel.GREEN],
            blue_path=paths[Channel.BLUE],
        )

    def capture_white(
        self,
        *,
        roll_name: str,
        frame_number: int,
        output_folder: str,
        w_level: int,
        shutter: Optional[str] = None,
        settle_s: float = 0.4,
        min_raw_bytes: int = 8 * 1024 * 1024,
        max_raw_bytes: int = 200 * 1024 * 1024,
        cancel: Optional[threading.Event] = None,
    ) -> str:
        """Single white-light exposure for slide / E-6 film (one file, no R/G/B split)."""
        os.makedirs(output_folder, exist_ok=True)
        try:
            self._light.set_color(w=w_level)
            self._sleep(settle_s)
            stem = os.path.join(output_folder, f"{roll_name}_Frame{frame_number:03d}")
            logger.info("capturing white frame → %s", stem)
            return _capture_validated_single(
                self._camera,
                final_stem=stem,
                shutter=shutter,
                min_raw_bytes=min_raw_bytes,
                max_raw_bytes=max_raw_bytes,
                cancel=cancel,
            )
        finally:
            try:
                self._light.off()
            except Exception:
                logger.exception("failed to turn the Scanlight off after capture")

    def _verify_size(self, path: str, settings: CaptureSettings) -> None:
        verify_raw_size(path, settings.min_raw_bytes, settings.max_raw_bytes)
