"""Background worker for Scanlight + camera triplet capture. Mirrors ScanWorker.

Owns the live Scanlight serial connection (so the RGB sliders stay responsive)
and one libgphoto2 session, held open across captures and shared with live view.
"""

from negpy.kernel.system.i18n import mark, tr

import math
import os
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.infrastructure.capture.base import CAPTURE_ORDER, CaptureSettings, Channel
from negpy.infrastructure.capture.gphoto import (
    CameraCapabilities,
    CameraClaimedError,
    CameraUnavailable,
    GphotoCamera,
    LiveViewUnsupported,
    list_cameras,
)
from negpy.infrastructure.capture.lights import Light, open_light
from negpy.infrastructure.capture.settings import WhiteCaptureMode
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger
from negpy.services.capture.calibration import (
    CLIP_CEILING,
    REFERENCE_LEVELS,
    REFERENCE_SHUTTER,
    CalibrationExposureError,
    CalibrationService,
    Roi,
    normalize_start_point,
    scan_ladder,
    true_seconds,
)
from negpy.services.capture.service import CaptureError, CaptureService, capture_single

logger = get_logger(__name__)


# A strip that never reports its end stops here.
_STRIP_FRAME_LIMIT = 60
# The bare light is several stops brighter than the film base the preset exposes for.
_FLAT_START_FACTOR = 1 / 8
_FLAT_ATTEMPTS = 4
_FLAT_LOW, _FLAT_HIGH, _FLAT_TARGET = 0.45, 0.92, 0.75  # brightest channel, fraction of the clip level
_FLAT_CLIPPED_STEP = 1 / 4  # a clipped reference hides its level, so step down two stops


def _current_shutter(camera) -> str:
    """The body's shutter label, or "" when the camera does not report it."""
    read = getattr(camera, "read_settings", None)
    info = (read() if read else {}).get("shutter") or {}
    options, cur = info.get("options") or [], info.get("cur", -1)
    return next((str(o.get("label", "")) for o in options if o.get("raw") == cur), "")


def _flat_peak(path: str) -> float:
    """The brightest channel's high percentile over the lit area, as a fraction of the clip level."""
    import numpy as np

    from negpy.infrastructure.capture.raw_demosaic import linear_demosaic

    rgb = linear_demosaic(path, half_size=True).astype(np.float32)
    brightest = rgb.max(axis=2)
    lit = brightest > 0.35 * np.percentile(brightest, 95)
    return float(max(np.percentile(rgb[..., c][lit], 99.5) for c in range(3)) / CLIP_CEILING)


@dataclass(frozen=True)
class CaptureRequest:
    roll_name: str
    frame_number: int
    output_folder: str
    levels: tuple[int, int, int]
    settle_s: float = 0.4
    inter_exposure_delay_s: float = 0.0
    port: str = ""
    shutters: tuple[str, str, str] = ("", "", "")
    white_mode: bool = False
    w_level: int = 255
    shutter_w: str = ""
    white_process_mode: WhiteCaptureMode = WhiteCaptureMode.AUTO
    is_retake: bool = False  # a retake overwrites an existing frame → keep its files on abort
    rgb_mode: bool = True  # True = Scanlight R/G/B triplet; False = one plain white-light shot (no Scanlight)
    iso: str = ""  # RGB preset's baked ISO/aperture — the triplet forces them; "" = leave as set
    aperture: str = ""
    as_roll: bool = False
    single_capture: bool = False  # RGB preset shot as one exposure with all three LEDs lit
    sensor_profile: str = ""  # the single-capture preset's sensor profile, for the frame's roll
    channels: tuple[str, ...] = ("R", "G", "B")  # the visible exposures to shoot; a subset shoots alone, into its own files
    shutter_ir: str = ""  # the infrared exposure's own shutter; "" = the body's current
    infrared: bool = False  # end the frame with an infrared exposure at shutter_ir
    capture_delay_s: float = 0.0  # seconds to hold the first exposure; Stop ends the wait
    end_light: tuple[int, int, int, int] = (0, 0, 0, 0)  # r, g, b, w left lit after a successful frame; zeros = off
    advance_film: bool = False  # after a Scan, move the film to the next frame on a light that has a film transport
    whole_strip: bool = False  # scan and advance frame after frame until the strip ends or Stop
    strip_flat: bool = False  # with whole_strip: after the eject, shoot the empty dock for the roll's Flat Field profile
    shutter_ladder: tuple[str, ...] = ()  # the body's shutter labels, for the flat's own exposure


@dataclass(frozen=True)
class ExposureProbeRequest:
    """One RAW probe at the body's current exposure, metered by `meter_frame`."""

    shutter: str  # the body's current shutter label, the probe's
    shutter_candidates: tuple[str, ...] = ()  # this body's writable ladder; empty = the built-in one
    medium: Optional[str] = None  # "negative" or "positive" from the preset; None lets the classifier decide
    session: int = 0  # the live-view session that asked, echoed back so a stopped one's report is dropped
    wb_decode: bool = False  # the import applies the as-shot white balance, so the target must hold after its gain


@dataclass(frozen=True)
class LiveViewRequest:
    """No parameters: libgphoto2 finds the camera itself, over USB."""


@dataclass(frozen=True)
class CalibrationRequest:
    roi: Roi
    output_folder: str
    port: str = ""
    settle_s: float = 0.4
    target_fraction: float = 0.9
    shutter_candidates: tuple[str, ...] = ()  # this body's writable shutter ladder (from the live-view JSON)
    # Phase-1 start point, already normalized to the live ISO/aperture by the sidebar. The
    # defaults keep the fixed reference (ISO 100 / f8) when the caller supplies none.
    start_levels: tuple[int, int, int] = REFERENCE_LEVELS
    start_shutter: str = REFERENCE_SHUTTER
    single_capture: bool = False  # solve for R, G and B lit together in one exposure


# The central half of the frame, the region a sensor profile built from files measures.
_SENSOR_ROI = Roi(0.25, 0.25, 0.5, 0.5)


@dataclass(frozen=True)
class SensorResponseRequest:
    """Measure the sensor's response to each Scanlight LED on the bare light. `cancel` stops
    this run alone, so abandoning it never reaches a scan queued on the same worker."""

    port: str = ""
    settle_s: float = 0.4
    cancel: threading.Event = field(default_factory=threading.Event)


class _AnySet:
    """Several cancel events read as one."""

    def __init__(self, *events: threading.Event) -> None:
        self._events = events

    def is_set(self) -> bool:
        return any(e.is_set() for e in self._events)


def _setting_label(info) -> str:
    """The current label of one `read_settings` entry, "" when it has none."""
    if not isinstance(info, dict):
        return ""
    return next((str(o.get("label", "")) for o in info.get("options", []) if o.get("raw") == info.get("cur")), "")


def _shutters_or_none(shutters: tuple[str, str, str]):
    """UI shutter strings → capture-settings form (None = leave the camera as-is)."""
    if not any(s.strip() for s in shutters):
        return None
    return tuple(s.strip() or None for s in shutters)


class CaptureWorker(QObject):
    """Drives the Scanlight + camera off the UI thread."""

    light_set = pyqtSignal(int, int, int, int)  # r, g, b, w actually applied
    progress = pyqtSignal(float)  # 0.0..1.0
    countdown = pyqtSignal(int)  # whole seconds left of the Capture Delay; 0 when the shot starts
    channel = pyqtSignal(str)  # "R"/"G"/"B" as each triplet channel starts
    #: One set_camera_setting call ran to completion: verified, rejected, or skipped without a
    #: session. Always fires, so the sidebar's pending-write gate on Scan cannot stick.
    camera_setting_applied = pyqtSignal(str)
    finished = pyqtSignal(list)  # [red_path, green_path, blue_path]
    clipped = pyqtSignal(str, float)  # a single-file capture's path, share of its picture at saturation
    cancelled = pyqtSignal()
    error = pyqtSignal(str)
    status = pyqtSignal(str)
    flat_captured = pyqtSignal(str)  # the bare-light reference shot after a strip
    strip_finished = pyqtSignal(int, bool)  # frames scanned, the strip ran to its end
    scan_button_pressed = pyqtSignal()  # the light's own Scan button
    live_view_started = pyqtSignal(str)  # jpeg path being refreshed
    #: This body advertises no CAPTURE_PREVIEW, so no stream was started (issue #621). Carries
    #: the operator-facing reason; scanning itself is unaffected.
    live_view_unsupported = pyqtSignal(str)
    #: The preview thread died after its retries and the session was dropped (issue #617). It
    #: carries the last gphoto error, so the UI can stop the spinner and offer a retry.
    live_view_failed = pyqtSignal(str)
    focus_magnifier_unavailable = pyqtSignal(str)  # operator-facing reason
    calibration_progress = pyqtSignal(float, str)
    calibration_finished = pyqtSignal(object)  # CalibrationResult
    calibration_exposure = pyqtSignal(str)  # "over"/"under" — target unreachable, run aborted, no preset
    exposure_probe_finished = pyqtSignal(object)  # (session, MeterReading)
    # (session, reason, camera_lost): camera_lost when the capture failed and the session was closed
    exposure_probe_failed = pyqtSignal(object)
    sensor_response_progress = pyqtSignal(float, str)
    sensor_response_measured = pyqtSignal(object)  # 3x3 array: sensor channel rows, LED columns
    sensor_response_failed = pyqtSignal(str)
    presence_polled = pyqtSignal(bool, bool)  # a usable camera is on the bus, the Scanlight answers
    poll_status = pyqtSignal(dict)  # {usb_ok, usb_model, light_ok, light_detail}
    light_temp_polled = pyqtSignal(object)  # Scanlight LED temperature °C, or None (light-only, safe mid-scan)

    def __init__(self) -> None:
        super().__init__()
        self._light: Optional[Light] = None
        self._light_port = ""
        self._idle_light = (0, 0, 0, 0)  # the last live-control color, put back after a sensor measurement
        self._cancel = threading.Event()
        # One libgphoto2 session serves live view, settings and stills: the body allows a
        # single PTP claim, and GphotoCamera serialises the three internally. Held open across
        # frames, closed on error and at shutdown.
        self._camera: Optional[GphotoCamera] = None
        self._model = ""
        # True after an open failed because another program holds the USB claim; drives the
        # "in use by another app" camera status. Cleared by the next open attempt that does
        # not hit the claim (success, or a different failure).
        self._claimed_elsewhere = False
        # One identify probe per bus appearance (read the body's real model name): tried once,
        # so a body that fails to open for non-claim reasons is not hammered every poll tick.
        self._identify_attempted = False
        # Abilities of the body last opened; None until one has been. Outlives the session so
        # the poll keeps reporting them after the identify probe closes it again.
        self._caps: Optional[CameraCapabilities] = None

    # ----- camera session (one per body, held open) -----

    def _acquire_camera(self) -> GphotoCamera:
        """Return the held session, opening it on first use. Every open attempt re-decides
        the "claimed by another app" verdict — while we hold a session the poll must never
        probe (a probe IS a claim, and would steal the body mid-live-view), so open attempts
        are the verdict's only eyes."""
        if self._camera is None:
            # The callback runs on the camera's own preview thread; the signal hop is what
            # moves the report onto the Qt side.
            self._camera = GphotoCamera(
                on_preview_died=self.live_view_failed.emit,
                # A stream that fails on a body that still answers is the "no preview" state,
                # not a dead camera. Same UI as a body that never advertised one.
                on_preview_unusable=self.live_view_unsupported.emit,
                on_magnifier_unusable=self.focus_magnifier_unavailable.emit,
            )
        if not self._camera.is_open():
            try:
                self._camera.open()
            except CameraClaimedError:
                # Another program holds the USB claim (on macOS usually the system's own camera
                # daemon, which needs no visible window). Flip the dot on the False-to-True
                # transition immediately: the next timer poll is seconds away, and "Scan did
                # nothing" against a green dot is the confusion this state exists for. Only on the
                # transition, because the poll's own re-probe lands here too and an unconditional
                # push would recurse.
                first = not self._claimed_elsewhere
                self._claimed_elsewhere = True
                if first:
                    self.poll_connection(self._light_port)
                raise
            except Exception:
                self._claimed_elsewhere = False  # failed differently — not (or no longer) a claim problem
                raise
            self._claimed_elsewhere = False
            self._model = self._camera.model
            # Cached so the poll keeps reporting them once the session is released again, which
            # the last camera window closing does, and the UI gates on the result.
            # Optional, because the `Camera` protocol does not require it: a backend that
            # cannot report abilities must leave the UI ungated instead of breaking the session.
            self._caps = getattr(self._camera, "capabilities", None)
        return self._camera

    def _holds_camera(self) -> bool:
        """True while we own the PTP claim — probing again would be refused by the body."""
        return self._camera is not None and self._camera.is_open()

    def _close_camera(self) -> None:
        if self._camera is not None:
            try:
                self._camera.close()
            except Exception:
                logger.exception("error closing camera session")
            self._camera = None

    # ----- light -----

    def _ensure_light(self, port: str) -> Light:
        # Reuse the held connection only while it is the same port and still alive. An
        # unplugged Scanlight leaves a dead handle, and without the health check we would reuse
        # it and never re-detect the device after it is plugged back in.
        if self._light is not None and self._light_port == port and self._light.is_connected():
            return self._light
        if self._light is not None:
            try:
                self._light.close()
            except Exception:
                logger.exception("error closing previous Scanlight connection")
            self._light = None
        self._light = open_light(port or None)
        self._light_port = port
        if hasattr(self._light, "on_scan_button"):
            self._light.on_scan_button = self.scan_button_pressed.emit
        return self._light

    @pyqtSlot(int, int, int, int, str)
    def set_light(self, r: int, g: int, b: int, w: int, port: str) -> None:
        """Live light control — RGB for framing/preview, or white (w) for focus."""
        try:
            self._ensure_light(port).set_color(r=r, g=g, b=b, w=w)
            self._idle_light = (r, g, b, w)
            self.light_set.emit(r, g, b, w)
        except Exception as e:
            msg = str(e)
            logger.warning("set_light failed: %s", msg)
            # A missing Scanlight is a persistent condition the Light LED already shows, so keep
            # it out of the shared status line, where it would linger and go stale.
            if not ("auto-discover" in msg or "No serial ports" in msg):
                self.error.emit(tr("Scanlight: {e}", e=e))

    # ----- capture -----

    @pyqtSlot(CaptureRequest)
    def run_capture(self, req: CaptureRequest) -> bool:
        """Capture one frame, or a whole strip. True when the frame is on disk; a failure or Stop
        has emitted `error` or `cancelled`."""
        if req.whole_strip:
            self._run_strip(req)
            return False
        _t0 = time.perf_counter()
        try:
            if not self._holds_camera():
                self.status.emit(tr("Connecting to camera…"))
            # The same session that streams live view takes the shot: no reconnect, and the
            # preview pauses for the second or two the body needs.
            camera = self._acquire_camera()
            logger.info("run_capture setup %.0f ms", (time.perf_counter() - _t0) * 1000)

            if not req.rgb_mode:
                # Normal white-light scanning, with no Scanlight: one plain camera shot imported
                # as an ordinary single RAW, with no LED control and no triplet. The operator's
                # own light stays on, and the shutter is whatever the live-view steppers set.
                self._wait_before_capture(req.capture_delay_s)
                self.status.emit(tr("Capturing…"))
                path = capture_single(
                    camera,
                    roll_name=req.roll_name,
                    frame_number=req.frame_number,
                    output_folder=req.output_folder,
                    shutter=(req.shutters[0] or None),
                    cancel=self._cancel,
                )
                self._finish_frame(req, [path])
                self._report_clipping(path)
                return True

            # RGB / white-preset captures drive the Scanlight, so it must be connected.
            light = self._ensure_light(req.port)
            service = CaptureService(light, camera)
            # Only the frame's last exposure leaves the light lit; an earlier one hands straight to the next.
            visible_end = None if req.infrared else req.end_light

            def _capture_ir() -> str:
                """The infrared exposure at its own shutter, with the body's shutter put back after,
                so a frame shot at the body's shutter never inherits the infrared one."""
                shutter = req.shutter_ir.strip() or None
                readable = shutter is not None and hasattr(camera, "read_settings") and hasattr(camera, "set_shutter")
                before = camera.read_settings().get("shutter", {}).get("cur", -1) if readable else -1
                try:
                    return service.capture_channel(
                        roll_name=req.roll_name,
                        frame_number=req.frame_number,
                        output_folder=req.output_folder,
                        channel=Channel.IR,
                        level=255,
                        shutter=shutter,
                        iso=req.iso or None,
                        aperture=req.aperture or None,
                        settle_s=req.settle_s,
                        cancel=self._cancel,
                        end_light=req.end_light,
                    )
                finally:
                    if isinstance(before, int) and before >= 0:
                        camera.set_shutter(before)

            def _with_ir(paths: list) -> list:
                """Append the frame's infrared exposure when the preset asks for one. The visible
                exposures are on disk by now, so a failed infrared one ends the frame without it
                instead of failing the frame, which would delete them as a partial capture."""
                if not req.infrared:
                    return paths
                self.channel.emit("IR")
                self.status.emit(tr("Capturing IR channel…"))
                try:
                    paths.append(_capture_ir())
                except Exception as exc:  # noqa: BLE001
                    logger.exception("infrared exposure failed")
                    self.status.emit(tr("Infrared exposure failed, frame kept without it: {error}", error=exc))
                self.progress.emit(1.0)
                return paths

            self._wait_before_capture(req.capture_delay_s)  # after the light check, so a dead light fails at once
            if req.white_mode:  # one broadband white exposure (slide / B&W), no RGB triplet
                self.status.emit(tr("Capturing white frame…"))
                path = service.capture_white(
                    roll_name=req.roll_name,
                    frame_number=req.frame_number,
                    output_folder=req.output_folder,
                    w_level=req.w_level,
                    shutter=req.shutter_w or None,
                    settle_s=req.settle_s,
                    cancel=self._cancel,
                    end_light=visible_end,
                )
                self._finish_frame(req, _with_ir([path]))
                self._report_clipping(path)
                return True

            settings = CaptureSettings(
                roll_name=req.roll_name,
                frame_number=req.frame_number,
                output_folder=req.output_folder,
                levels=req.levels,
                settle_s=req.settle_s,
                inter_exposure_delay_s=req.inter_exposure_delay_s,
                shutters=_shutters_or_none(req.shutters),
                iso=req.iso or None,  # force the preset's exposure on the body before each shot
                aperture=req.aperture or None,
            )
            if req.single_capture:  # one narrowband exposure, imported as an ordinary single RAW
                self.status.emit(tr("Capturing…"))
                path = service.capture_rgb_single(settings, cancel=self._cancel, end_light=visible_end)
                self._finish_frame(req, _with_ir([path]))
                self._report_clipping(path)
                return True

            if tuple(req.channels) != ("R", "G", "B"):  # a subset of the visible exposures, each shot alone
                paths = []
                last = [ch for ch in CAPTURE_ORDER if ch.letter in req.channels][-1:]
                for i, ch in enumerate(CAPTURE_ORDER):
                    if ch.letter not in req.channels:
                        continue
                    self.channel.emit(ch.letter)
                    self.status.emit(tr("Capturing {channel} channel…", channel=ch.letter))
                    paths.append(
                        service.capture_channel(
                            roll_name=req.roll_name,
                            frame_number=req.frame_number,
                            output_folder=req.output_folder,
                            channel=ch,
                            level=req.levels[i] or 255,
                            shutter=req.shutters[i].strip() or None,
                            iso=req.iso or None,
                            aperture=req.aperture or None,
                            settle_s=req.settle_s,
                            cancel=self._cancel,
                            end_light=visible_end if [ch] == last else None,
                        )
                    )
                self._finish_frame(req, _with_ir(paths))
                return True

            self.status.emit(tr("Capturing R / G / B…"))
            _names = {"R": mark("red"), "G": mark("green"), "B": mark("blue")}

            def _on_channel(letter: str) -> None:
                self.channel.emit(letter)
                self.status.emit(tr("Capturing {value} channel…", value=tr(_names.get(letter, letter))))

            result = service.capture_triplet(
                settings,
                progress=lambda f: self.progress.emit(f * (0.75 if req.infrared else 1.0)),
                cancel=self._cancel,
                on_channel=_on_channel,
                end_light=visible_end,
            )

            self._finish_frame(req, _with_ir(result.paths))
            return True
        except Exception as e:
            self._cleanup_partial_frame(req)  # keep the folder a clean multiple of 3 for NegPy's grouper
            if self._cancel.is_set():
                # Deliberate cancellation is not a camera failure. The staged capture was
                # discarded before promotion, so keep the healthy live-view session.
                self.cancelled.emit()
                return False
            self._close_camera()  # discard a possibly-broken held session
            logger.exception("capture failed")
            self.error.emit(str(e))
            return False

    def _finish_frame(self, req: CaptureRequest, paths: list) -> None:
        """Report the frame, then move the film when asked. The import starts at once; the next
        capture waits on this thread until the film is in place. A Retake never moves the film."""
        self.finished.emit(paths)
        if req.advance_film and not req.is_retake:
            self._advance_film(req.port)

    def _advance_film(self, port: str, eject: bool = True) -> Optional[bool]:
        """Move the light's film one frame. True at the end of the strip, False when the next frame
        is in place, None when the light has no film transport or the move failed. With `eject`,
        the strip leaves the dock at its end."""
        try:
            light = self._ensure_light(port)
            advance = getattr(light, "advance_frame", None)
            if advance is None:
                return None
            self.status.emit(tr("Moving the film to the next frame…"))
            at_end = advance() is None
        except Exception as exc:  # noqa: BLE001
            logger.exception("film advance failed")
            self.status.emit(tr("The film did not move: {error}. Move it with the dock's buttons.", error=exc))
            return None
        if not at_end:
            self.status.emit(tr("The next frame is in place."))
            return False
        if not eject:
            return True
        # The last frame stops in the window, short of the exit.
        if self._eject(light) is not False:
            self.status.emit(tr("End of the strip. Insert the next strip."))
        return True

    def _eject(self, light) -> Optional[bool]:
        """True when the strip is out, False when the eject failed, None for a light that has no eject."""
        eject = getattr(light, "eject", None)
        if eject is None:
            return None
        try:
            self.status.emit(tr("Ejecting the film…"))
            eject()
        except Exception as exc:  # noqa: BLE001
            logger.exception("eject failed")
            self.status.emit(tr("The film did not eject: {error}. Use the dock's buttons.", error=exc))
            return False
        return True

    @pyqtSlot(str)
    def eject_film(self, port: str) -> None:
        """The Eject button: move the strip out of a light with a film transport."""
        try:
            light = self._ensure_light(port)
        except Exception as exc:  # noqa: BLE001
            self.status.emit(tr("The film did not eject: {error}. Use the dock's buttons.", error=exc))
            return
        if self._eject(light):
            self.status.emit(tr("The film is out. Insert the next strip."))

    def _run_strip(self, req: CaptureRequest) -> None:
        """Scan frame after frame, moving the film after each, until the strip ends, a frame fails
        or Stop. At the end the strip leaves the dock; with `strip_flat`, the empty dock is shot next.
        The dock cannot advance after it moves the film out of the window and back, so the flat
        never comes first."""
        scanned, at_end = 0, False
        for frame_number in range(req.frame_number, req.frame_number + _STRIP_FRAME_LIMIT):
            if self._cancel.is_set():
                self.cancelled.emit()
                return
            one = replace(req, frame_number=frame_number, whole_strip=False, strip_flat=False, advance_film=False)
            if not self.run_capture(one):
                return
            scanned += 1
            moved = self._advance_film(req.port, eject=False)
            if moved is None:
                break
            if moved:
                at_end = True
                break
        flat_error = ""
        if at_end and self._eject(self._ensure_light(req.port)) and req.strip_flat and not self._cancel.is_set():
            try:
                self.flat_captured.emit(self._capture_flat(req))
            except Exception as exc:  # noqa: BLE001
                if not self._cancel.is_set():
                    logger.exception("flat field capture failed")
                    flat_error = tr("Flat Field: {error}", error=exc)
        self.strip_finished.emit(scanned, at_end)
        if flat_error:
            self.status.emit(flat_error)

    def _capture_flat(self, req: CaptureRequest) -> str:
        """Shoot the empty dock at the frame's LED levels and a shutter that puts the brightest
        channel in range. The file goes beside the Flat Field profiles, never into the roll."""
        from negpy.services.capture.meter import shutter_at_most

        camera = self._acquire_camera()
        light = self._ensure_light(req.port)
        ladder = scan_ladder(req.shutter_ladder)
        start = next((label for label in (req.shutters[0], _current_shutter(camera)) if label), "")
        if not ladder or not start:
            raise CaptureError(tr("The camera reports no shutter speeds."))
        shutter = shutter_at_most(true_seconds(start, ladder) * _FLAT_START_FACTOR, ladder) or ladder[0]
        folder = os.path.join(APP_CONFIG.flatfield_dir, "references")
        service = CaptureService(light, camera)
        self.status.emit(tr("Shooting the Flat Field on the empty dock…"))
        for _attempt in range(_FLAT_ATTEMPTS):
            settings = CaptureSettings(
                roll_name=f"{req.roll_name}_FlatField",
                frame_number=0,
                output_folder=folder,
                levels=req.levels,
                settle_s=req.settle_s,
                shutters=(shutter, shutter, shutter),
                iso=req.iso or None,
                aperture=req.aperture or None,
            )
            path = service.capture_rgb_single(settings, cancel=self._cancel)
            peak = _flat_peak(path)
            if _FLAT_LOW <= peak <= _FLAT_HIGH:
                return path
            factor = _FLAT_CLIPPED_STEP if peak > _FLAT_HIGH else _FLAT_TARGET / max(peak, 1e-3)
            better = shutter_at_most(true_seconds(shutter, ladder) * factor, ladder)
            if better is None or better == shutter:
                break
            shutter = better
        if peak <= _FLAT_HIGH:
            return path  # darker than ideal, but unclipped: the gain is a ratio
        raise CaptureError(tr("The Flat Field clips at the fastest shutter. Close the aperture or lower the ISO."))

    def _report_clipping(self, path: str) -> None:
        """Emit `clipped` when a single-file capture clips as Meter judges a probe. Runs after
        `finished`, so the import does not wait; a failed check never fails the frame."""
        from negpy.infrastructure.capture.raw_demosaic import linear_demosaic
        from negpy.services.capture.meter import picture_clipping

        try:
            share = picture_clipping(linear_demosaic(path, half_size=True))
        except Exception:  # noqa: BLE001
            logger.exception("clipping check failed: %s", path)
            return
        if share is not None:
            self.clipped.emit(path, share)

    def _wait_before_capture(self, delay_s: float) -> None:
        """Hold the first exposure for `delay_s`; Stop ends the wait as a cancellation."""
        if delay_s <= 0:
            return
        deadline = time.monotonic() + delay_s
        while (left := deadline - time.monotonic()) > 0:
            self.countdown.emit(math.ceil(left))
            if self._cancel.wait(min(left, 1.0)):
                raise CaptureError("canceled before the first exposure")
        self.countdown.emit(0)

    def _cleanup_partial_frame(self, req: CaptureRequest) -> None:
        """After a cancelled/failed capture, delete this frame's exposures so the output
        folder stays a clean multiple of three. NegPy groups triplets by consecutive
        sort order, so 1–2 leftover exposures would shift every following frame's grouping.

        Skipped for white-light captures (a single frame is already complete, not a triplet)
        and for retakes (the frame existed complete; overwriting leaves three files in place).
        """
        if req.white_mode or req.single_capture or req.is_retake or not req.rgb_mode:
            return  # single-file captures (white / single RGB / normal) aren't triplets → nothing to keep aligned
        prefix = f"{req.roll_name}_Frame{req.frame_number:03d}_"  # trailing _ → only channel files
        try:
            names = os.listdir(req.output_folder)
        except OSError:
            return
        for name in names:
            if name.startswith(prefix):  # any RAW suffix — the camera chose it
                path = os.path.join(req.output_folder, name)
                try:
                    os.remove(path)
                    logger.info("removed partial capture file %s", path)
                except OSError:
                    logger.exception("could not remove partial capture file %s", path)

    def cancel(self) -> None:
        self._cancel.set()

    def arm(self) -> None:
        """Clears a Stop left from an earlier run. Called on the GUI thread before dispatch,
        so a Stop pressed before the queued run starts is kept."""
        self._cancel.clear()

    @pyqtSlot(str)
    def poll_light_temp(self, port: str) -> None:
        """Read the Scanlight's cached LED temperature (telemetry, no serial request, no
        camera) and emit it. Safe to call even during live-view/scan — the light is a
        separate device from the camera's single SDK session."""
        temp = None
        try:
            temp = self._ensure_light(port).last_temp_c
        except Exception:
            pass
        self.light_temp_polled.emit(temp)

    @pyqtSlot(str)
    def poll_presence(self, port: str) -> None:
        """Whether a camera and the Scanlight are there to capture with. Enumerates the bus and
        never opens the body, so a caller that owns no camera window leaves no claim behind."""
        try:
            camera = bool(list_cameras()) and not self._claimed_elsewhere
        except Exception:
            camera = False
        try:
            self._ensure_light(port).describe()
            light = True
        except Exception:
            light = False
        self.presence_polled.emit(camera, light)

    @pyqtSlot(str)
    def poll_connection(self, port: str) -> None:
        """Lightweight presence check for the auto-connect UI: is a body enumerated, and is
        the light up? Off the UI thread; called on a timer. Enumerating does not claim the
        camera, so this is safe to run while live view streams."""
        status = {
            "usb_ok": False,
            "usb_model": "",
            "light_ok": False,
            "light_detail": "not connected",
            "light_has_white": True,
            "light_has_ir": False,
            "light_moves_film": False,
            "usb_claimed_elsewhere": False,
            # Advertised abilities of the last body opened. Optimistic until a session has been
            # opened: never grey a control out on an uninspected body.
            "camera_preview": True,
            "camera_config": True,
        }
        try:
            # Always ask the bus, never our own handle: unplugging the camera leaves the handle
            # behind and it reports "connected" forever. Enumerating does not claim the device,
            # so this is safe even mid-stream.
            found = list_cameras()
            if found:
                identify = not self._model and not self._identify_attempted
                if (self._claimed_elsewhere or identify) and not self._holds_camera():
                    # One probe open (open, read, close), for two jobs. Safe in both states
                    # because we hold no session ourselves:
                    # * First sight of a body: read its real model name. libgphoto2's database
                    #   labels newer bodies "USB PTP Class Camera" (#431), and only the device
                    #   knows its name. Tried ONCE per bus appearance, so a body that fails to
                    #   open for other reasons is not hammered every tick. This also detects a
                    #   foreign claim before the user's first live-view or scan attempt.
                    # * Claimed verdict standing: retry until the other app lets go. Only an
                    #   open can clear the verdict, and Scan and Calibrate are gated while it
                    #   stands, so without this the dot stayed red after Preview closed.
                    self._identify_attempted = True
                    try:
                        self._acquire_camera()
                        # The claim is kept, never handed back. A freed PTP interface is taken by
                        # the OS camera daemon within seconds and held for minutes, so a probe that
                        # opened and closed left the user's first live view or scan refused. The
                        # poll runs only while the camera panel is visible, so holding follows the
                        # user; the last camera window closing still releases it.
                    except Exception:
                        pass  # verdict already updated by _acquire_camera's except paths
                # A body on the bus that refused our last open, because another app holds the
                # claim, is not usable. Surface that instead of a healthy "connected" dot.
                status["usb_ok"], status["usb_model"] = True, self._model or found[0]["model"]
                status["usb_claimed_elsewhere"] = self._claimed_elsewhere
                # Nothing republishes the body's settings while no preview streams, so the poll
                # does: the card's exposure steppers fill before Live View ever opens.
                publish = getattr(self._camera, "publish_settings", None) if self._holds_camera() else None
                if publish is not None:
                    publish()
                if self._caps is not None:
                    status["camera_preview"], status["camera_config"] = self._caps.preview, self._caps.config
            elif self._claimed_elsewhere:
                # A claimed body routinely DROPS OFF gphoto's enumeration while the other app
                # holds it, because macOS's daemons answer the bus in our stead. That flapping
                # is part of the claimed state, not an unplug: clearing the verdict here made
                # the dot snap back to green one poll tick after the failed open.
                status["usb_ok"], status["usb_claimed_elsewhere"] = True, True
                status["usb_model"] = self._model or "USB PTP Class Camera"
            else:
                # A genuinely absent body, with no claim verdict standing: forget its name and
                # allow a fresh identify, since the next body to appear may be a different one.
                self._model = ""
                self._identify_attempted = False
                self._caps = None  # the next body may be a different one
                if self._camera is not None:
                    logger.info("camera disappeared from the bus — dropping the session")
                    self._close_camera()
        except CameraUnavailable as e:
            status["usb_model"] = str(e)
        except Exception:
            logger.exception("camera poll failed")
        try:
            light = self._ensure_light(port)
            detail, has_white, has_ir = light.describe()
            status["light_ok"], status["light_detail"] = True, detail
            status["light_has_white"], status["light_has_ir"] = has_white, has_ir
            status["light_moves_film"] = hasattr(light, "advance_frame")
        except Exception:
            pass
        self.poll_status.emit(status)

    # ----- live view -----

    @pyqtSlot(LiveViewRequest)
    def start_live_view(self, req: LiveViewRequest) -> None:
        try:
            camera = self._acquire_camera()
            camera.start()  # a no-op when the preview thread is already up
            self.live_view_started.emit(camera.jpeg_path)
        except LiveViewUnsupported as e:
            # Not a failure: the body scans fine, it only has no preview to show. Reported on
            # its own signal, so the scan window opens without a stream instead of erroring.
            logger.info("live view unavailable: %s", e)
            self.live_view_unsupported.emit(str(e))
        except Exception as e:
            logger.exception("start_live_view failed")
            self.error.emit(tr("Live view: {e}", e=e))

    @pyqtSlot()
    def stop_live_view(self) -> None:
        """Stop the preview thread but keep the session — a scan still needs it."""
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                logger.exception("error stopping live view")

    @pyqtSlot()
    def close_camera_session(self) -> None:
        """Release the held PTP session (called once neither the scan window nor the
        preset-calibration pop-up is open). Some bodies (Fuji in particular) get stuck
        in a tethered-capture state on the camera side until the session is cleanly
        exited — leaving it open past the last window that uses it makes the *next*
        connection attempt hang rather than reconnect."""
        self._close_camera()

    def _camera_control_failed(self, action: str, exc: Exception) -> None:
        """Turn a failed Qt camera-control callback into a recoverable disconnect."""
        logger.exception("%s failed", action)
        self._close_camera()
        self.error.emit(tr("Camera control failed: {exc}. Reconnect and try again.", exc=exc))

    @pyqtSlot(bool)
    def set_focus_magnifier(self, on: bool) -> None:
        """Toggle the camera's hardware focus magnifier."""
        try:
            if self._holds_camera():
                self._camera.set_focus_magnifier(on)
        except Exception as exc:  # noqa: BLE001 — exceptions cannot cross a Qt slot
            self._camera_control_failed("set_focus_magnifier", exc)

    @pyqtSlot(int, int)
    def set_focus_magnifier_pos(self, x: int, y: int) -> None:
        """Aim the magnifier at (x, y) on the 640x480 preview grid."""
        try:
            if self._holds_camera():
                self._camera.set_focus_magnifier_at(x, y)
        except Exception as exc:  # noqa: BLE001 — exceptions cannot cross a Qt slot
            self._camera_control_failed("set_focus_magnifier_pos", exc)

    @pyqtSlot(str, int, str)
    def set_camera_setting(self, which: str, raw: int, label: str = "") -> None:
        """Change a live camera setting (iso/shutter/wb/aperture); `raw` is a choice index and
        `label` the text the panel shows for it, which the body must still list there. Takes the
        session like a scan does: the card's steppers write with no Live View window open, after
        the window's close released it."""
        try:
            if self._claimed_elsewhere and not self._holds_camera():
                return  # the camera dot already says another app holds it; the poll's retry clears that
            cam = self._acquire_camera()
            {
                "iso": cam.set_iso,
                "shutter": cam.set_shutter,
                "aperture": cam.set_aperture,
            }.get(which, lambda _r, _l=None: None)(*((raw, label) if label else (raw,)))
        except Exception as exc:  # noqa: BLE001 — exceptions cannot cross a Qt slot
            self._camera_control_failed("set_camera_setting", exc)
        finally:
            self.camera_setting_applied.emit(which)

    # ----- calibration -----

    @pyqtSlot(CalibrationRequest)
    def run_calibration(self, req: CalibrationRequest) -> None:
        try:
            import tempfile

            from negpy.infrastructure.capture.raw_demosaic import linear_demosaic

            light = self._ensure_light(req.port)
            if not self._holds_camera():
                self.status.emit(tr("Connecting to camera…"))
            camera = self._acquire_camera()  # the same session live view streams on
            # Half-res decode: calibration only meters a uniform base patch, so it is much
            # faster. The raw-Bayer clip check stays full-res, and the scans still decode
            # full-res.
            service = CalibrationService(light, camera, lambda p: linear_demosaic(p, half_size=True), settle_s=req.settle_s)
            # Calibration fires several throwaway RAWs. Keep them outside the user's roll: the
            # camera replaces `.raw` with its own suffix, so deleting one guessed filename left
            # files behind and could overwrite a user file of that name. A directory boundary
            # cleans every suffix on every success, error and cancel path.
            with tempfile.TemporaryDirectory(prefix="negpy-calibration-") as scratch_dir:
                scratch = os.path.join(scratch_dir, "capture.raw")
                result = service.calibrate(
                    req.roi,
                    scratch,
                    start_levels=req.start_levels,
                    start_shutter=req.start_shutter,  # normalized to the live ISO/aperture by the sidebar
                    target_fraction=req.target_fraction,
                    candidates=req.shutter_candidates,  # empty → calibrate falls back to the built-in ladder
                    progress=self.calibration_progress.emit,
                    cancel=self._cancel,
                    single_capture=req.single_capture,
                )
            self.calibration_finished.emit(result)
        except CalibrationExposureError as e:
            # An expected outcome, not a broken session: the target is unreachable at these
            # aperture and hardware limits. Keep the camera claim, since the user adjusts the
            # aperture and retries at once and a reconnect would cost seconds for nothing.
            logger.info("calibration aborted: %s", e)
            self.calibration_exposure.emit(e.status)
        except Exception as e:
            self._close_camera()  # discard a possibly-broken held session
            if self._cancel.is_set():
                self.cancelled.emit()
                return
            logger.exception("calibration failed")
            self.error.emit(tr("Calibration: {e}", e=e))

    @pyqtSlot(ExposureProbeRequest)
    def run_exposure_probe(self, req: ExposureProbeRequest) -> None:
        """One probe under the light as lit, at the body's current exposure, into a scratch
        directory the way calibration shoots; the reading says which shutter to set."""
        # Every outcome reports exactly once, tagged with the session that asked: a reading, a
        # failure that kept the camera, or a capture failure that closed it.
        import tempfile

        from negpy.infrastructure.capture.raw_demosaic import as_shot_wb, linear_demosaic
        from negpy.services.capture.meter import meter_frame

        try:
            scratch = tempfile.TemporaryDirectory(prefix="negpy-meter-", ignore_cleanup_errors=True)
        except OSError as e:
            self.exposure_probe_failed.emit((req.session, tr("no scratch directory: {error}", error=e), False))
            return
        with scratch as scratch_dir:
            try:
                if not self._holds_camera():
                    self.status.emit(tr("Connecting to camera…"))
                camera = self._acquire_camera()
                # Pinned to the label the reading is computed against, not whatever a queued
                # stepper write leaves on the body.
                written = camera.capture(os.path.join(scratch_dir, "probe.raw"), shutter=req.shutter)
            except Exception as e:
                logger.exception("meter probe failed")
                try:
                    self._close_camera()  # discard a possibly-broken held session
                finally:
                    self.exposure_probe_failed.emit((req.session, str(e), True))
                return
            try:
                img = linear_demosaic(written, half_size=True)
                wb = as_shot_wb(written) if req.wb_decode else None
                reading = meter_frame(img, req.shutter, req.shutter_candidates, medium=req.medium, wb=wb)
            except Exception as e:  # the camera session stands; only the decode or the maths failed
                logger.exception("meter failed")
                self.exposure_probe_failed.emit((req.session, str(e), False))
                return
        self.exposure_probe_finished.emit((req.session, reading))

    @pyqtSlot(SensorResponseRequest)
    def measure_sensor_response(self, req: SensorResponseRequest) -> None:
        """Shoot each LED alone on the bare light and report the sensor's response. Needs no
        live view: the frame is uniform, so the patch is fixed and the ladder is read here."""
        if req.cancel.is_set():
            return
        self._cancel.clear()
        held = self._holds_camera()
        on_camera = False
        try:
            import tempfile

            from negpy.infrastructure.capture.raw_demosaic import linear_demosaic

            self.sensor_response_progress.emit(0.0, "Connecting…")
            light = self._ensure_light(req.port)
            on_camera = True
            camera = self._acquire_camera()
            settings = camera.read_settings()
            shutter = settings.get("shutter") or {}
            candidates = scan_ladder(o.get("label", "") for o in shutter.get("options", []))
            if not candidates or not shutter.get("writable", True):
                # The built-in ladder is one vendor's vocabulary; another body ignores it (#768).
                raise RuntimeError("This camera is not reporting settable shutter speeds. Set it to Manual (M), then try again.")
            aperture = settings.get("aperture") or {}
            start_levels, start_shutter = normalize_start_point(
                _setting_label(settings.get("iso")),
                _setting_label(aperture) if aperture.get("writable", False) else "",
                candidates=candidates,
            )
            service = CalibrationService(light, camera, lambda p: linear_demosaic(p, half_size=True), settle_s=req.settle_s)
            with tempfile.TemporaryDirectory(prefix="negpy-sensor-") as scratch_dir:
                response = service.measure_sensor_response(
                    _SENSOR_ROI,
                    os.path.join(scratch_dir, "capture.raw"),
                    start_levels=start_levels,
                    start_shutter=start_shutter,
                    candidates=candidates,
                    progress=self.sensor_response_progress.emit,
                    cancel=_AnySet(req.cancel, self._cancel),
                )
            self.sensor_response_measured.emit(response)
        except CalibrationExposureError as e:
            logger.info("sensor measurement aborted: %s", e)
            self.sensor_response_failed.emit(
                f"The {e.channel} exposure clips even at the fastest shutter. Close the aperture or lower the ISO, then try again."
            )
        except Exception as e:
            if req.cancel.is_set() or self._cancel.is_set():
                return
            logger.exception("sensor measurement failed")
            if held and on_camera:
                # A session someone else opened is dropped with the failure, so its windows hear of it.
                held = False
                self.live_view_failed.emit(str(e))
            self.sensor_response_failed.emit(str(e))
        finally:
            if not held:
                self._close_camera()
            if any(self._idle_light) and self._light is not None:
                try:
                    self._light.set_color(*self._idle_light)
                except Exception:
                    logger.exception("failed to restore the Scanlight after the sensor measurement")

    def shutdown(self) -> None:
        """Stop any capture and release the light (called on app teardown)."""
        self._cancel.set()
        self.stop_live_view()
        self._close_camera()  # release the held keep-alive session
        if self._light is not None:
            try:
                self._light.off()
            except Exception:
                logger.exception("error turning Scanlight off on shutdown")
            try:
                self._light.close()
            except Exception:
                logger.exception("error closing Scanlight on shutdown")
            self._light = None
