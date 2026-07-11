"""Camera capture through libgphoto2 — one session, no proprietary SDK, no helper binary.

`GphotoCamera` is both the capture device (the `Camera` protocol) and the live-view
source: libgphoto2 keeps a single PTP session per body, so previews, property writes and
stills all ride the same handle behind one lock. It writes the preview JPEG and a
settings JSON to the same paths the previous helper used, so the UI polls them unchanged.

Four behaviours of the library shape this module; each is guarded below:

* Reading a value off a choice widget that has **no** choices dereferences a NULL and
  kills the process with SIGSEGV — no `except` can catch it. See `_safe_value`.
* Property writes are asynchronous. The body needs ~1-2 s before it reports a new value
  back, so a write is confirmed by polling, never assumed. See `_set_verified`.
* After a still, the event queue must be drained or the *next* capture fails with a bare
  `[-1] unspecified error`. See `_drain_events`.
* Choice strings are run through gettext, so on a German desktop `focusmode` reads
  'Manuell'. The message locale is pinned before the library loads. See `_pin_locale`.

Vendors name the same control differently and expose different subsets of it, so every
property is looked up rather than assumed — see `_PROPERTIES` and `_MAGNIFIERS`. Only Sony
bodies have been tested.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from negpy.infrastructure.loaders.constants import (
    SUPPORTED_JPEG_EXTENSIONS,
    SUPPORTED_RAW_EXTENSIONS,
    SUPPORTED_TIFF_EXTENSIONS,
)
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

#: libgphoto2 property names behind the settings the scan window exposes, in the order
#: they are tried. `iso` and `shutterspeed` are the same everywhere (libgphoto2 normalises
#: them onto the generic PTP properties), but the aperture is not: the generic name is
#: `f-number` (Sony, Panasonic), while Canon, Nikon, Fujifilm, Olympus and Sigma expose
#: `aperture`. No body offers both.
#:
#: There is deliberately no white balance here: NegPy decodes with a fixed neutral WB
#: (`use_camera_wb=False`), so the camera's setting only tints the preview, never a scan.
_PROPERTIES: dict[str, tuple[str, ...]] = {
    "iso": ("iso",),
    "shutter": ("shutterspeed",),
    "aperture": ("f-number", "aperture"),
}


@dataclass(frozen=True)
class _Magnifier:
    """How one vendor exposes its live-view magnifier.

    Sony packs the zoom ratio and the target position into a single `"ratio,x,y"` string;
    Canon and Nikon split them across two properties. Only the packed form carries a
    coordinate space we know (640x480), so aiming is Sony-only — elsewhere the magnifier
    zooms wherever the body already points.
    """

    ratio: str
    #: Sony lists [Off, 1, 6.9, 13.7] where the first step only repositions; the others
    #: list [1, 5, 10] where the first entry *is* off.
    skip_first_step: bool = False
    packed: bool = False


#: Tried in order; the first whose ratio property exists wins. Most bodies have none, and
#: that is normal — the feature disables itself rather than failing.
#: Canon and Nikon keep the target point in a second property (`eoszoomposition`,
#: `liveviewzoomarea`) whose coordinate space is unknown here, so those bodies magnify
#: wherever they already look — the centre, unless something moved it. Fujifilm's
#: `zoompos` is the *lens* zoom and read-only; Olympus and Panasonic expose nothing.
_MAGNIFIERS = (
    _Magnifier(ratio="focusmagnifier", skip_first_step=True, packed=True),  # PTP_VENDOR_SONY
    _Magnifier(ratio="eoszoom"),  # PTP_VENDOR_CANON
    _Magnifier(ratio="liveviewimagezoomratio"),  # PTP_VENDOR_NIKON
)

#: Where the camera should put the file it just took. Tethered capture wants it in memory,
#: not on a card — Canon and Nikon default to the card, and fail outright without one.
_CAPTURE_TARGET = "capturetarget"

#: The body reports the magnifier position on a 640x480 grid, clamps it so the magnified
#: box stays inside the frame, and treats (0, 0) as "switch off" — so keep x/y away from
#: the edges rather than trusting the clamp.
_GRID_W, _GRID_H = 640, 480
_GRID_MARGIN = 8

#: A property write is asynchronous; poll this long for the body to report it back.
_WRITE_SETTLE_S = 3.0
#: The magnifier itself takes ~1-2 s to engage — the same wait covers it.
_EVENT_DRAIN_S = 1.0

_PREVIEW_INTERVAL_S = 0.05  # the body tops out near 24 fps; this leaves headroom
_SETTINGS_INTERVAL_S = 2.0
#: Consecutive preview failures before the session is treated as gone. One dropped frame
#: is normal; three in a row means the camera was unplugged, powered off, or taken away.
_MAX_PREVIEW_FAILURES = 3

# General NegPy imports also accept JPEG/TIFF, but camera scanning promises a linear RAW
# source. A body left in JPEG mode can easily produce an 8+ MB file that passes the size
# guard while permanently discarding highlight/color information.
_CAMERA_RAW_EXTENSIONS = frozenset(SUPPORTED_RAW_EXTENSIONS - SUPPORTED_JPEG_EXTENSIONS - SUPPORTED_TIFF_EXTENSIONS)


class CameraUnavailable(RuntimeError):
    """python-gphoto2 is not installed, or no camera answered."""


class GphotoError(RuntimeError):
    """A camera operation failed."""


def _pin_locale() -> None:
    """Force libgphoto2's choice strings to English.

    The library translates them through gettext, so `focusmode` comes back as 'Manuell'
    on a German desktop and 'Manual' on an English one. Only the *message* locale is
    pinned — touching `LC_ALL` would also change number formatting for the rest of the app.
    """
    os.environ["LANGUAGE"] = "C"


def _gp() -> Any:
    """Import python-gphoto2 lazily, so NegPy runs fine without it."""
    _pin_locale()
    try:
        import gphoto2  # noqa: PLC0415 — optional dependency, imported on demand
    except ImportError as exc:  # pragma: no cover — depends on the install
        raise CameraUnavailable(
            "Camera scanning needs python-gphoto2. Install it with `pip install gphoto2` "
            "(macOS and Linux only — libgphoto2 has no Windows build)."
        ) from exc
    return gphoto2


def default_jpeg_path() -> str:
    """Where the live-view thread publishes the newest preview frame."""
    return os.path.join(tempfile.gettempdir(), "negpy_scanlight_live.jpg")


def default_settings_path() -> str:
    """Where the live-view thread publishes current camera settings (JSON)."""
    return os.path.join(tempfile.gettempdir(), "negpy_scanlight_settings.json")


def _safe_value(gp: Any, widget: Any) -> Optional[str]:
    """A widget's value, or None when reading it would crash the process.

    A choice widget with zero choices — `f-number` on a lens with no electronic
    aperture, which is exactly the kind of lens used for film scanning — makes
    libgphoto2 hand back a NULL string that the binding then dereferences. That is a
    SIGSEGV, not an exception, so the only defence is not to ask.
    """
    kind = widget.get_type()
    if kind in (gp.GP_WIDGET_RADIO, gp.GP_WIDGET_MENU) and widget.count_choices() == 0:
        return None
    return widget.get_value()


def _choices(gp: Any, widget: Any) -> list[str]:
    kind = widget.get_type()
    if kind not in (gp.GP_WIDGET_RADIO, gp.GP_WIDGET_MENU):
        return []
    return [widget.get_choice(i) for i in range(widget.count_choices())]


def list_cameras() -> list[dict]:
    """Every camera libgphoto2 can see, as `{"model", "port"}` dicts."""
    gp = _gp()
    found = gp.Camera.autodetect()
    return [{"model": found.get_name(i), "port": found.get_value(i)} for i in range(found.count())]


def _model_name(camera: Any) -> str:
    for line in str(camera.get_summary()).splitlines():
        if line.lower().startswith("model"):
            return line.split(":", 1)[-1].strip()
    return "camera"


class GphotoCamera:
    """One libgphoto2 session: live view, camera settings, focus magnifier and stills.

    Implements the `Camera` protocol (`capture`, `close`) plus the live-view surface the
    scan window drives. `gp_module` is injectable so the tests never touch hardware.
    """

    def __init__(
        self,
        *,
        jpeg_path: Optional[str] = None,
        settings_path: Optional[str] = None,
        gp_module: Optional[Any] = None,
    ) -> None:
        self._gp = gp_module or _gp()
        self._jpeg_path = jpeg_path or default_jpeg_path()
        self._settings_path = settings_path or default_settings_path()
        self._camera: Any = None
        self._model = ""
        # Cleared when the body stops answering — unplugging it leaves the handle behind,
        # and a stale handle reports itself open forever.
        self._alive = False
        self._lock = threading.RLock()
        self._preview: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # Raised while a still is in flight, so the preview thread doesn't queue another
        # ~40 ms frame grab ahead of the next channel of a triplet.
        self._busy = threading.Event()
        self._reset_body_state()

    def _reset_body_state(self) -> None:
        """Forget controls whose names and semantics belong to one camera body."""
        self._magnifier: Optional[_Magnifier] = None
        self._magnifier_ratios: Optional[tuple[str, str]] = None
        self._magnifier_off = ""
        self._magnifier_probed = False
        self._aim_warned = False
        self._names: dict[str, Optional[str]] = {}  # settings key → this body's property name
        self._position = (_GRID_W // 2, _GRID_H // 2)

    # ----- session ---------------------------------------------------------------

    @property
    def jpeg_path(self) -> str:
        return self._jpeg_path

    @property
    def settings_path(self) -> str:
        return self._settings_path

    @property
    def model(self) -> str:
        return self._model

    def open(self) -> None:
        if self.is_open():
            return
        self.close()  # drop a handle left behind by a body that went away
        camera = self._gp.Camera()
        try:
            camera.init()
        except self._gp.GPhoto2Error as exc:
            # A camera sits on the bus but won't open: on macOS the ImageCapture daemons
            # hand it to Preview, Photos or Image Capture the moment one of them is open,
            # and only one program may claim it. Say so — the enumeration that drives the
            # status dot cannot see this, so the error line is the user's only clue.
            raise GphotoError(f"could not open the camera: {exc}. Close Preview, Photos and Image Capture, then retry.") from exc
        self._camera = camera
        self._model = _model_name(camera)
        self._alive = True
        logger.info("gphoto2 session open: %s", self._model)
        self._prefer_memory_capture()

    def is_open(self) -> bool:
        return self._camera is not None and self._alive

    def close(self) -> None:
        self.stop()
        with self._lock:
            self._alive = False
            if self._camera is not None:
                try:
                    self._camera.exit()
                except Exception as exc:  # noqa: BLE001 — teardown must not raise
                    logger.warning("gphoto2 exit: %s", exc)
                self._camera = None
            self._reset_body_state()

    def _require(self) -> Any:
        if self._camera is None:
            self.open()
        return self._camera

    # ----- properties ------------------------------------------------------------

    def _set_verified(
        self,
        name: str,
        value: str,
        settle_s: float = _WRITE_SETTLE_S,
        match: Optional[Any] = None,
    ) -> bool:
        """Write a property and poll until the body reports it back. Writes are async.

        `match` decides when the read-back counts as the value landing; it defaults to
        equality. The magnifier needs its own test, because the body echoes a position
        alongside the ratio and clamps that position to keep the box inside the frame.
        """
        camera = self._require()
        accepts = match or (lambda got: got == value)

        try:
            current = _safe_value(self._gp, camera.get_single_config(name))
            if current is not None and accepts(current) and match is None:
                return True  # already there — skip the round trip (a scan re-sends the shutter)

            widget = camera.get_single_config(name)
            widget.set_value(value)
            camera.set_single_config(name, widget)
        except self._gp.GPhoto2Error as exc:
            # A value this body does not offer, e.g. a shutter label from the fallback
            # ladder. Report it; a scan degrades to the current exposure rather than dying.
            logger.warning("gphoto2: could not set %s to %r: %s", name, value, exc)
            return False

        deadline = time.monotonic() + settle_s
        while time.monotonic() < deadline:
            try:
                got = _safe_value(self._gp, camera.get_single_config(name))
            except self._gp.GPhoto2Error as exc:
                logger.warning("gphoto2: could not verify %s at %r: %s", name, value, exc)
                return False
            if got is not None and accepts(got):
                return True
            time.sleep(0.1)
        logger.warning("gphoto2: %s did not settle on %r", name, value)
        return False

    def _prefer_memory_capture(self) -> None:
        """Ask the body to hand a still straight to us instead of writing it to a card.

        Sony already does; Canon and Nikon default to the card and refuse to shoot without
        one. The option is named differently everywhere ('sdram', 'Internal RAM'), so match
        on the word rather than a fixed label, and leave the body alone if nothing matches.
        """
        with self._lock:
            camera = self._require()
            try:
                widget = camera.get_single_config(_CAPTURE_TARGET)
            except self._gp.GPhoto2Error:
                return
            for choice in _choices(self._gp, widget):
                if re.search(r"\bram\b|sdram", choice, re.IGNORECASE):
                    if self._set_verified(_CAPTURE_TARGET, choice, settle_s=1.0):
                        logger.info("gphoto2: capture target set to %r", choice)
                    return
            logger.info("gphoto2: this body offers no in-memory capture target; it will use its card")

    def _property(self, key: str) -> Optional[str]:
        """This body's name for a setting, or None when it offers none of the candidates.

        Vendors name the same control differently, so ask the camera instead of assuming.
        Resolved once per session — the answer cannot change while a body stays plugged in.
        """
        if key in self._names:
            return self._names[key]
        camera = self._require()
        for name in _PROPERTIES[key]:
            try:
                camera.get_single_config(name)
            except self._gp.GPhoto2Error:
                continue
            self._names[key] = name
            return name
        logger.info("gphoto2: this body has no %s control", key)
        self._names[key] = None
        return None

    def _set_choice(self, key: str, index: int) -> None:
        with self._lock:
            name = self._property(key)
            if name is None:
                return
            camera = self._require()
            widget = camera.get_single_config(name)
            choices = _choices(self._gp, widget)
            if not 0 <= index < len(choices):
                logger.warning("gphoto2: %s has no choice %d", name, index)
                return
            self._set_verified(name, choices[index])

    def set_iso(self, raw: int) -> None:
        self._set_choice("iso", int(raw))

    def set_shutter(self, raw: int) -> None:
        self._set_choice("shutter", int(raw))

    def set_aperture(self, raw: int) -> None:
        self._set_choice("aperture", int(raw))

    def read_settings(self) -> dict:
        """Current value + options for each settable property, in the UI's schema.

        A property the body cannot offer (aperture on a manual lens) is simply absent —
        the scan window then greys its stepper out.
        """
        with self._lock:
            camera = self._require()
            out: dict[str, dict] = {}
            for key in _PROPERTIES:
                name = self._property(key)
                if name is None:
                    continue
                widget = camera.get_single_config(name)
                choices = _choices(self._gp, widget)
                if not choices:
                    continue  # nothing to offer, and reading its value would segfault
                current = _safe_value(self._gp, widget)
                out[key] = {
                    "cur": choices.index(current) if current in choices else -1,
                    "writable": not widget.get_readonly(),
                    "options": [{"label": label, "raw": i} for i, label in enumerate(choices)],
                }
            return out

    # ----- focus magnifier -------------------------------------------------------

    def _probe_magnifier(self) -> Optional[_Magnifier]:
        """Find this body's live-view magnifier, or None. Probed once per session.

        Absence is the common case — only Sony, Canon and Nikon expose one at all — and it
        must never raise: these calls arrive from a Qt slot, where an exception aborts the
        process.
        """
        if self._magnifier_probed:
            return self._magnifier
        self._magnifier_probed = True
        camera = self._require()
        for spec in _MAGNIFIERS:
            try:
                widget = camera.get_single_config(spec.ratio)
            except self._gp.GPhoto2Error:
                continue
            choices = _choices(self._gp, widget)
            if len(choices) < 2:
                continue  # a magnifier with nothing to select is no magnifier
            steps = choices[1:]  # choices[0] is the off/1x entry
            if spec.skip_first_step and len(steps) >= 2:
                steps = steps[1:]  # Sony's first step repositions without magnifying
            self._magnifier = spec
            self._magnifier_off = choices[0]
            self._magnifier_ratios = (steps[0], steps[-1])
            logger.info("gphoto2: focus magnifier via %r, steps %s", spec.ratio, self._magnifier_ratios)
            return spec
        logger.info("gphoto2: this body has no focus magnifier")
        return None

    def _write_magnifier(self, ratio: str) -> None:
        """Set the zoom ratio, carrying the aim point along on bodies that pack them together."""
        spec, (x, y) = self._magnifier, self._position
        assert spec is not None  # noqa: S101 — callers probe first
        if not spec.packed:
            self._set_verified(spec.ratio, ratio, match=lambda got: got == ratio)
            return
        value = f"{ratio},{x},{y}"
        # Verify the ratio alone: the body echoes a position back, and clamps it so the
        # magnified box stays inside the frame.
        self._set_verified(spec.ratio, value, match=lambda got: got.split(",", 1)[0] == ratio)

    def set_focus_magnifier(self, on: bool) -> None:
        with self._lock:
            if self._probe_magnifier() is None or self._magnifier_ratios is None:
                return
            self._write_magnifier(self._magnifier_ratios[0] if on else self._magnifier_off)

    def set_focus_magnifier_at(self, x: int, y: int) -> None:
        """Magnify at a point on the 640x480 preview grid.

        Only Sony's property carries a coordinate space we know, so only there does the
        point aim the magnifier; elsewhere it simply zooms wherever the body already looks.
        The origin is never sent (the body reads (0, 0) as "switch off"), and a point near
        an edge is pulled back so the magnified box still fits in the frame.
        """
        x = max(_GRID_MARGIN, min(int(x), _GRID_W - _GRID_MARGIN))
        y = max(_GRID_MARGIN, min(int(y), _GRID_H - _GRID_MARGIN))
        self._position = (x, y)
        with self._lock:
            spec = self._probe_magnifier()
            if spec is None or self._magnifier_ratios is None:
                return
            if not spec.packed and not self._aim_warned:
                self._aim_warned = True
                logger.info("gphoto2: %r cannot be aimed; magnifying at the body's own position", spec.ratio)
            self._write_magnifier(self._magnifier_ratios[0])

    # ----- capture ---------------------------------------------------------------

    def _drain_events(self, budget_s: float = _EVENT_DRAIN_S) -> None:
        """Consume the events a still leaves behind.

        Skipping this makes the *next* `capture()` fail with a bare `[-1]`.
        """
        camera = self._require()
        deadline = time.monotonic() + budget_s
        while time.monotonic() < deadline:
            kind, _data = camera.wait_for_event(50)
            if kind == self._gp.GP_EVENT_TIMEOUT:
                return

    def capture(self, out_path: str, shutter: Optional[str] = None) -> str:
        """Take one still, write the RAW next to `out_path`, and return where it landed.

        The suffix comes from the camera — a Canon writes `.CR3`, a Nikon `.NEF` — so the
        returned path may differ from the one asked for. Callers must use the return value.
        Blocks the preview meanwhile.
        """
        self._busy.set()
        try:
            with self._lock:
                camera = self._require()
                if shutter:
                    name = self._property("shutter")
                    if name is None or not self._set_verified(name, shutter):
                        raise GphotoError(f"could not set shutter to {shutter!r}: camera rejected it or it did not settle")
                try:
                    path = camera.capture(self._gp.GP_CAPTURE_IMAGE)
                    suffix = os.path.splitext(path.name)[1]
                    if suffix.lower() not in _CAMERA_RAW_EXTENSIONS:
                        shown = suffix or "(no extension)"
                        raise GphotoError(f"camera returned {shown}, not a RAW file; set the camera to RAW-only image quality and retry")
                    camera_file = camera.file_get(path.folder, path.name, self._gp.GP_FILE_TYPE_NORMAL)
                    data = bytes(memoryview(camera_file.get_data_and_size()))
                except self._gp.GPhoto2Error as exc:
                    raise GphotoError(f"capture failed: {exc}") from exc
                finally:
                    self._drain_events()
        finally:
            self._busy.clear()

        suffix = os.path.splitext(path.name)[1]
        if suffix:
            out_path = os.path.splitext(out_path)[0] + suffix
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp = f"{out_path}.part"
        with open(tmp, "wb") as handle:
            handle.write(data)
        os.replace(tmp, out_path)
        logger.info("gphoto2 captured %s (%.1f MB)", os.path.basename(out_path), len(data) / 1e6)
        return out_path

    # ----- live view -------------------------------------------------------------

    def start(self) -> None:
        if self.is_running():
            return
        self.open()
        self._stop.clear()
        self._preview = threading.Thread(target=self._preview_loop, name="gphoto-liveview", daemon=True)
        self._preview.start()

    def is_running(self) -> bool:
        return self._preview is not None and self._preview.is_alive()

    def stop(self) -> None:
        self._stop.set()
        thread, self._preview = self._preview, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)

    def _preview_loop(self) -> None:
        next_settings = 0.0
        failures = 0
        while not self._stop.is_set():
            if self._busy.is_set():  # stand aside for a capture rather than race it for the lock
                self._stop.wait(0.02)
                continue
            try:
                with self._lock:
                    if self._camera is None:
                        return
                    frame = self._camera.capture_preview()
                    data = bytes(memoryview(frame.get_data_and_size()))
                self._publish_frame(data)
                failures = 0
                if time.monotonic() >= next_settings:
                    self._publish_settings()
                    next_settings = time.monotonic() + _SETTINGS_INTERVAL_S
            except Exception as exc:  # noqa: BLE001 — a dropped frame must not kill the thread
                failures += 1
                logger.warning("gphoto2 live view (%d/%d): %s", failures, _MAX_PREVIEW_FAILURES, exc)
                if failures >= _MAX_PREVIEW_FAILURES:
                    # The body stopped answering — unplugged, powered off, or claimed by
                    # another app. Mark the handle dead so nothing reuses it; don't call
                    # close() from here, it would join this very thread.
                    logger.warning("gphoto2: camera stopped answering, dropping the session")
                    self._alive = False
                    return
                self._stop.wait(0.5)
                continue
            self._stop.wait(_PREVIEW_INTERVAL_S)

    def _publish_frame(self, data: bytes) -> None:
        tmp = f"{self._jpeg_path}.part"
        with open(tmp, "wb") as handle:
            handle.write(data)
        os.replace(tmp, self._jpeg_path)  # atomic: the UI only ever sees a whole frame

    def _publish_settings(self) -> None:
        try:
            payload = self.read_settings()
        except Exception as exc:  # noqa: BLE001
            logger.warning("gphoto2 settings: %s", exc)
            return
        tmp = f"{self._settings_path}.part"
        with open(tmp, "w") as handle:
            json.dump(payload, handle)
        os.replace(tmp, self._settings_path)
