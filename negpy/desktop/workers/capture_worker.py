"""Background worker for Scanlight + camera triplet capture. Mirrors ScanWorker.

Owns the live Scanlight serial connection (so the RGB sliders stay responsive)
and one libgphoto2 session, held open across captures and shared with live view.
"""

import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.infrastructure.capture.base import CaptureSettings
from negpy.infrastructure.capture.gphoto import CameraClaimedError, CameraUnavailable, GphotoCamera, list_cameras
from negpy.infrastructure.capture.protocol import describe_hardware, has_white_channel
from negpy.infrastructure.capture.scanlight import Scanlight
from negpy.kernel.system.logging import get_logger
from negpy.services.capture.calibration import REFERENCE_LEVELS, REFERENCE_SHUTTER, CalibrationExposureError, CalibrationService, Roi
from negpy.services.capture.service import CaptureService, capture_single

logger = get_logger(__name__)


@dataclass(frozen=True)
class CaptureRequest:
    roll_name: str
    frame_number: int
    output_folder: str
    levels: tuple[int, int, int]
    settle_s: float = 0.4
    port: str = ""
    shutters: tuple[str, str, str] = ("", "", "")
    white_mode: bool = False
    w_level: int = 255
    shutter_w: str = ""
    white_process_mode: str = "auto"  # "auto" | "E-6" | "B&W"
    is_retake: bool = False  # a retake overwrites an existing frame → keep its files on abort
    rgb_mode: bool = True  # True = Scanlight R/G/B triplet; False = one plain white-light shot (no Scanlight)
    iso: str = ""  # RGB preset's baked ISO/aperture — the triplet forces them; "" = leave as set
    aperture: str = ""


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
    # Phase-1 start point, already normalized to the live ISO/aperture by the sidebar. Defaults keep
    # the fixed reference (ISO 100 / f8) when the caller doesn't supply one.
    start_levels: tuple[int, int, int] = REFERENCE_LEVELS
    start_shutter: str = REFERENCE_SHUTTER


def _shutters_or_none(shutters: tuple[str, str, str]):
    """UI shutter strings → capture-settings form (None = leave the camera as-is)."""
    if not any(s.strip() for s in shutters):
        return None
    return tuple(s.strip() or None for s in shutters)


class CaptureWorker(QObject):
    """Drives the Scanlight + camera off the UI thread."""

    light_set = pyqtSignal(int, int, int, int)  # r, g, b, w actually applied
    progress = pyqtSignal(float)  # 0.0..1.0
    channel = pyqtSignal(str)  # "R"/"G"/"B" as each triplet channel starts
    #: One set_camera_setting call ran to completion — verified, rejected, or skipped without a
    #: session. Always fires, so the sidebar's pending-write gate on Scan can never stick.
    camera_setting_applied = pyqtSignal(str)
    finished = pyqtSignal(list)  # [red_path, green_path, blue_path]
    cancelled = pyqtSignal()
    error = pyqtSignal(str)
    status = pyqtSignal(str)
    live_view_started = pyqtSignal(str)  # jpeg path being refreshed
    calibration_progress = pyqtSignal(float, str)
    calibration_finished = pyqtSignal(object)  # CalibrationResult
    calibration_exposure = pyqtSignal(str)  # "over"/"under" — target unreachable, run aborted, no preset
    poll_status = pyqtSignal(dict)  # {usb_ok, usb_model, light_ok, light_detail}
    light_temp_polled = pyqtSignal(object)  # Scanlight LED temperature °C, or None (light-only, safe mid-scan)

    def __init__(self) -> None:
        super().__init__()
        self._light: Optional[Scanlight] = None
        self._light_port = ""
        self._cancel = threading.Event()
        # One libgphoto2 session serves live view, settings and stills alike — the body
        # allows a single PTP claim, and GphotoCamera serialises the three internally.
        # Held open across frames; closed on error and at shutdown.
        self._camera: Optional[GphotoCamera] = None
        self._model = ""
        # True after an open failed because another program holds the USB claim; drives the
        # "in use by another app" camera status. Cleared by the next open attempt that does
        # not hit the claim (success, or a different failure).
        self._claimed_elsewhere = False
        # One identify probe per bus appearance (read the body's real model name): tried once,
        # so a body that fails to open for non-claim reasons is not hammered every poll tick.
        self._identify_attempted = False

    # ----- camera session (one per body, held open) -----

    def _acquire_camera(self) -> GphotoCamera:
        """Return the held session, opening it on first use. Every open attempt re-decides
        the "claimed by another app" verdict — while we hold a session the poll must never
        probe (a probe IS a claim, and would steal the body mid-live-view), so open attempts
        are the verdict's only eyes."""
        if self._camera is None:
            self._camera = GphotoCamera()
        if not self._camera.is_open():
            try:
                self._camera.open()
            except CameraClaimedError:
                # Another program holds the USB claim (macOS: Preview/Photos/Image Capture).
                # On the False→True transition, flip the dot NOW — the next timer poll is up
                # to 3 s away, and "Scan did nothing" against a green dot is exactly the
                # confusion this state exists for. Only on the transition: the poll's own
                # re-probe lands here too, and an unconditional push would recurse.
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

    def _ensure_light(self, port: str) -> Scanlight:
        # Reuse the held connection only while it's the same port AND still alive — a Scanlight
        # that was unplugged leaves a dead handle; without the health check we'd keep reusing it
        # and never re-detect the device after it's plugged back in.
        if self._light is not None and self._light_port == port and self._light.is_connected():
            return self._light
        if self._light is not None:
            try:
                self._light.close()
            except Exception:
                logger.exception("error closing previous Scanlight connection")
            self._light = None
        self._light = Scanlight(port or None)
        self._light_port = port
        return self._light

    @pyqtSlot(int, int, int, int, str)
    def set_light(self, r: int, g: int, b: int, w: int, port: str) -> None:
        """Live light control — RGB for framing/preview, or white (w) for focus."""
        try:
            self._ensure_light(port).set_color(r=r, g=g, b=b, w=w)
            self.light_set.emit(r, g, b, w)
        except Exception as e:
            msg = str(e)
            logger.warning("set_light failed: %s", msg)
            # A missing Scanlight is a persistent condition already shown by the Light LED
            # (red, "Scanlight: not connected") — don't push it into the shared status line,
            # where it would linger and go stale (e.g. "camera-only" after the camera drops).
            if not ("auto-discover" in msg or "No serial ports" in msg):
                self.error.emit(f"Scanlight: {e}")

    # ----- capture -----

    @pyqtSlot(CaptureRequest)
    def run_capture(self, req: CaptureRequest) -> None:
        self._cancel.clear()
        _t0 = time.perf_counter()
        try:
            if not self._holds_camera():
                self.status.emit("Connecting to camera…")
            # The same session that streams live view takes the shot: no reconnect, and
            # the preview simply pauses for the ~2 s the body needs.
            camera = self._acquire_camera()
            logger.info("run_capture setup %.0f ms", (time.perf_counter() - _t0) * 1000)

            if not req.rgb_mode:
                # Normal white-light scanning (no Scanlight): one plain camera shot, imported as
                # an ordinary single RAW — no LED control, no triplet. The operator's own light
                # stays on; the shutter is whatever the live-view steppers set.
                self.status.emit("Capturing…")
                path = capture_single(
                    camera,
                    roll_name=req.roll_name,
                    frame_number=req.frame_number,
                    output_folder=req.output_folder,
                    shutter=(req.shutters[0] or None),
                    cancel=self._cancel,
                )
                self.finished.emit([path])
                return

            # RGB / white-preset captures drive the Scanlight, so it must be connected.
            light = self._ensure_light(req.port)
            service = CaptureService(light, camera)
            if req.white_mode:  # one broadband white exposure (slide / B&W), no RGB triplet
                self.status.emit("Capturing white frame…")
                path = service.capture_white(
                    roll_name=req.roll_name,
                    frame_number=req.frame_number,
                    output_folder=req.output_folder,
                    w_level=req.w_level,
                    shutter=req.shutter_w or None,
                    settle_s=req.settle_s,
                    cancel=self._cancel,
                )
                self.finished.emit([path])
                return

            settings = CaptureSettings(
                roll_name=req.roll_name,
                frame_number=req.frame_number,
                output_folder=req.output_folder,
                levels=req.levels,
                settle_s=req.settle_s,
                shutters=_shutters_or_none(req.shutters),
                iso=req.iso or None,  # force the preset's exposure on the body before each shot
                aperture=req.aperture or None,
            )
            self.status.emit("Capturing R / G / B…")
            _names = {"R": "red", "G": "green", "B": "blue"}

            def _on_channel(letter: str) -> None:
                self.channel.emit(letter)
                self.status.emit(f"Capturing {_names.get(letter, letter)} channel…")

            result = service.capture_triplet(
                settings,
                progress=self.progress.emit,
                cancel=self._cancel,
                on_channel=_on_channel,
            )

            self.finished.emit(result.paths)
        except Exception as e:
            self._cleanup_partial_frame(req)  # keep the folder a clean multiple of 3 for NegPy's grouper
            if self._cancel.is_set():
                # Deliberate cancellation is not a camera failure. The staged capture was
                # discarded before promotion, so preserve the healthy live-view/PTP session.
                self.cancelled.emit()
                return
            self._close_camera()  # discard a possibly-broken held session
            logger.exception("capture failed")
            self.error.emit(str(e))

    def _cleanup_partial_frame(self, req: CaptureRequest) -> None:
        """After a cancelled/failed capture, delete this frame's exposures so the output
        folder stays a clean multiple of three. NegPy groups triplets by consecutive
        sort order, so 1–2 leftover exposures would shift every following frame's grouping.

        Skipped for white-light captures (a single frame is already complete, not a triplet)
        and for retakes (the frame existed complete; overwriting leaves three files in place).
        """
        if req.white_mode or req.is_retake or not req.rgb_mode:
            return  # single-file captures (white / normal) aren't triplets → nothing to keep aligned
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
            "usb_claimed_elsewhere": False,
        }
        try:
            # Always ask the bus, never our own handle: unplugging the camera leaves the
            # handle behind, and it would keep reporting "connected" forever. Enumerating
            # does not claim the device, so this is safe even mid-stream.
            found = list_cameras()
            if found:
                identify = not self._model and not self._identify_attempted
                if (self._claimed_elsewhere or identify) and not self._holds_camera():
                    # One probe open (open → read → close immediately), for two jobs — safe in
                    # both states because we hold no session ourselves:
                    # * First sight of a body: read its real model name. libgphoto2's database
                    #   labels post-database bodies "USB PTP Class Camera" (the a7C II is not
                    #   in it, #431); only the device knows its name. Tried ONCE per bus
                    #   appearance — a body that fails to open for other reasons must not be
                    #   hammered every tick. This also detects a foreign claim proactively,
                    #   before the user's first live-view/scan attempt.
                    # * Claimed verdict standing: retry until the other app lets go. Only an
                    #   open can clear the verdict, and Scan/Calibrate are gated while it
                    #   stands — without this the dot stayed red after Preview closed.
                    self._identify_attempted = True
                    try:
                        self._acquire_camera()
                        self._close_camera()
                    except Exception:
                        pass  # verdict already updated by _acquire_camera's except paths
                # A body on the bus that refused our last open (another app holds the claim)
                # is not usable — surface that instead of a healthy "connected" dot.
                status["usb_ok"], status["usb_model"] = True, self._model or found[0]["model"]
                status["usb_claimed_elsewhere"] = self._claimed_elsewhere
            elif self._claimed_elsewhere:
                # A claimed body routinely DROPS OFF gphoto's enumeration while the other app
                # holds it (macOS's daemons answer the bus in our stead) — that flapping is
                # part of the claimed state, not an unplug. Clearing the verdict here made the
                # dot snap back to green within one poll tick of the failed open (rig find).
                status["usb_ok"], status["usb_claimed_elsewhere"] = True, True
                status["usb_model"] = self._model or "USB PTP Class Camera"
            else:
                # A genuinely absent body (no claim verdict standing): forget its name and
                # allow a fresh identify — the next body to appear may be a different one.
                self._model = ""
                self._identify_attempted = False
                if self._camera is not None:
                    logger.info("camera disappeared from the bus — dropping the session")
                    self._close_camera()
        except CameraUnavailable as e:
            status["usb_model"] = str(e)
        except Exception:
            logger.exception("camera poll failed")
        try:
            fw, hw = self._ensure_light(port).get_fw_version()
            status["light_ok"], status["light_detail"] = True, f"{describe_hardware(hw)} (fw{fw})"
            status["light_has_white"] = has_white_channel(hw)
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
        except Exception as e:
            logger.exception("start_live_view failed")
            self.error.emit(f"Live view: {e}")

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
        self.error.emit(f"Camera control failed: {exc}. Reconnect and try again.")

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

    @pyqtSlot(str, int)
    def set_camera_setting(self, which: str, raw: int) -> None:
        """Change a live camera setting (iso/shutter/wb/aperture); `raw` is a choice index."""
        try:
            if self._holds_camera():
                cam = self._camera
                {
                    "iso": cam.set_iso,
                    "shutter": cam.set_shutter,
                    "aperture": cam.set_aperture,
                }.get(which, lambda _r: None)(raw)
        except Exception as exc:  # noqa: BLE001 — exceptions cannot cross a Qt slot
            self._camera_control_failed("set_camera_setting", exc)
        finally:
            self.camera_setting_applied.emit(which)

    # ----- calibration -----

    @pyqtSlot(CalibrationRequest)
    def run_calibration(self, req: CalibrationRequest) -> None:
        self._cancel.clear()
        try:
            import tempfile

            from negpy.infrastructure.capture.raw_demosaic import linear_demosaic

            light = self._ensure_light(req.port)
            if not self._holds_camera():
                self.status.emit("Connecting to camera…")
            camera = self._acquire_camera()  # the same session live view streams on
            # Half-res decode: calibration only meters a uniform base patch, so it's ~4× faster
            # (the raw-Bayer clip check stays full-res, and the actual scans still decode full-res).
            service = CalibrationService(light, camera, lambda p: linear_demosaic(p, half_size=True), settle_s=req.settle_s)
            # Calibration fires several throwaway RAWs. Keep them outside the user's roll:
            # the camera replaces `.raw` with its own suffix, so deleting one guessed filename
            # is insufficient and used to leave `_negpy_calibration.ARW` behind (or overwrite a
            # user-owned file with that name). The directory boundary cleans every suffix and
            # every success/error/cancel path.
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
                )
            self.calibration_finished.emit(result)
        except CalibrationExposureError as e:
            # An expected outcome (the target is unreachable at these aperture/hardware limits),
            # not a broken session: keep the camera claim — the user adjusts the aperture and
            # retries immediately, and a reconnect would cost ~4 s for nothing.
            logger.info("calibration aborted: %s", e)
            self.calibration_exposure.emit(e.status)
        except Exception as e:
            self._close_camera()  # discard a possibly-broken held session
            if self._cancel.is_set():
                self.cancelled.emit()
                return
            logger.exception("calibration failed")
            self.error.emit(f"Calibration: {e}")

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
