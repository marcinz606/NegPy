"""Camera capture through libgphoto2 — one session, no proprietary SDK, no helper binary.

`GphotoCamera` is both the capture device (the `Camera` protocol) and the live-view
source: libgphoto2 keeps a single PTP session per body, so previews, property writes and
stills all ride the same handle behind one lock. It writes the preview JPEG and a
settings JSON to the same paths the previous helper used, so the UI polls them unchanged.

Five behaviours of the library shape this module; each is guarded below:

* Reading a value off a choice widget that has **no** choices dereferences a NULL and
  kills the process with SIGSEGV — no `except` can catch it. See `_safe_value`.
* Property writes are asynchronous. The body needs ~1-2 s before it reports a new value
  back, so a write is confirmed by polling, never assumed. See `_set_verified`.
* After a still, the event queue must be drained or the *next* capture fails with a bare
  `[-1] unspecified error`. See `_drain_events`; after a successful still it runs on a
  small worker so the wait overlaps the next channel's LED settle (`_finish_shot_async`).
* Choice strings are run through gettext, so on a German desktop `focusmode` reads
  'Manuell'. The message locale is pinned before the library loads. See `_pin_locale`.
* A downloaded shot must also be *deleted* off the body. Bodies without a capture-target
  setting (Fujifilm) keep every tethered object and refuse further captures once their
  buffer fills — the X-T5 dies at exactly shot #13, ~1 GB. See the delete in `capture`.

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
from typing import Any, Callable, Optional

from negpy.infrastructure.loaders.constants import (
    SUPPORTED_JPEG_EXTENSIONS,
    SUPPORTED_RAW_EXTENSIONS,
    SUPPORTED_TIFF_EXTENSIONS,
)
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

#: libgphoto2 property names behind the settings the scan window exposes, in the order they
#: are tried. `iso` and `shutterspeed` are the same everywhere, but the aperture is not: the
#: generic name is `f-number` (Sony, Panasonic) while Canon, Nikon, Fujifilm, Olympus and
#: Sigma expose `aperture`. No body offers both.
#:
#: There is no white balance here: NegPy decodes with a fixed neutral WB, so the camera's
#: setting only tints the preview, never a scan.
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


#: Tried in order: the first whose ratio property exists wins. Most bodies have none, which
#: is normal, and the feature disables itself instead of failing.
#: Canon and Nikon keep the target point in a second property whose coordinate space is
#: unknown here, so those bodies magnify wherever they already look. Fujifilm's `zoompos` is
#: the *lens* zoom and read-only, and Olympus and Panasonic expose nothing.
_MAGNIFIERS = (
    _Magnifier(ratio="focusmagnifier", skip_first_step=True, packed=True),  # PTP_VENDOR_SONY
    _Magnifier(ratio="eoszoom"),  # PTP_VENDOR_CANON
    _Magnifier(ratio="liveviewimagezoomratio"),  # PTP_VENDOR_NIKON
)

#: Where the camera should put the file it just took. Tethered capture wants it in memory,
#: not on a card: Canon and Nikon default to the card and fail outright without one.
_CAPTURE_TARGET = "capturetarget"

#: The body reports the magnifier position on a 640x480 grid, clamps it so the magnified box
#: stays inside the frame, and treats (0, 0) as "switch off". Keep x/y away from the edges
#: instead of trusting the clamp.
_GRID_W, _GRID_H = 640, 480
_GRID_MARGIN = 8

#: A property write is asynchronous; poll this long for the body to report it back.
_WRITE_SETTLE_S = 3.0
#: The magnifier takes a second or two to engage, which the same wait covers.
_EVENT_DRAIN_S = 1.0
#: The quiet window that ends an event drain. Fujifilm bodies deliver post-shot events with
#: long pauses, and events left unread block their next operation camera-side. Cutting the
#: drain at the first short quiet window is how the X-T5's live view died after every still
#: (issue #658), so the window is per-body: patient on Fujifilm, snappy elsewhere.
_DRAIN_SILENCE_MS = 50
_FUJI_DRAIN_SILENCE_MS = 300
_FUJI_DRAIN_BUDGET_S = 2.0

_PREVIEW_INTERVAL_S = 0.05  # the body tops out near 24 fps; this leaves headroom
_SETTINGS_INTERVAL_S = 2.0
#: Consecutive preview failures before the session is treated as gone. One dropped frame
#: is normal; three in a row means the camera was unplugged, powered off, or taken away.
_MAX_PREVIEW_FAILURES = 3

# General NegPy imports also accept JPEG and TIFF, but camera scanning promises a linear RAW
# source. A body left in JPEG mode can produce a large file that passes the size guard while
# permanently discarding highlight and color information.
_CAMERA_RAW_EXTENSIONS = frozenset(SUPPORTED_RAW_EXTENSIONS - SUPPORTED_JPEG_EXTENSIONS - SUPPORTED_TIFF_EXTENSIONS)


#: The catch-all entry unlisted PTP bodies fall back to. It claims every operation, so "not
#: in the database" must never be reported as unsupported on its own.
_GENERIC_DRIVER = "USB PTP Class Camera"


@dataclass(frozen=True)
class CameraCapabilities:
    """What libgphoto2's driver entry says this body can do.

    Read once per session from `Camera.get_abilities()`. Only trustworthy for bodies the
    database actually knows: an unlisted body matches the generic `USB PTP Class Camera`
    entry, which advertises everything (a7C II matches it and works fine), so a missing
    bit is evidence and a present one is not. Absence is therefore the only thing acted on.
    """

    driver_model: str  # the abilities entry that matched, not the device's own model name
    preview: bool = True
    config: bool = True

    @property
    def mtp_mode(self) -> bool:
        """Did we match a vendor's MTP-mode entry? Most bodies appear twice in the database,
        once per USB mode, and the MTP variant drops live view while the Control/PC-Remote
        one keeps it — so a missing preview bit there means "wrong mode", not "cannot"."""
        return "mtp" in self.driver_model.lower()

    @property
    def generic(self) -> bool:
        """Matched the catch-all PTP entry, i.e. this body is not in libgphoto2's database."""
        return self.driver_model == _GENERIC_DRIVER


class CameraUnavailable(RuntimeError):
    """python-gphoto2 is not installed, or no camera answered."""


class GphotoError(RuntimeError):
    """A camera operation failed."""


class LiveViewUnsupported(GphotoError):
    """This body's driver entry has no CAPTURE_PREVIEW, so live view must not be attempted.

    Typed because it is a normal state, not a fault: an a6000 scans perfectly well without a
    preview. Calling `capture_preview()` anyway is actively harmful — the body refuses it at
    the device level and the PTP session wedges until the camera is power-cycled (issue #621).
    """


class CameraClaimedError(GphotoError):
    """The body is on the bus but another program holds its USB claim (gphoto -53).

    On macOS the ImageCapture daemons hand the camera to Preview, Photos or Image Capture
    the moment one of them opens; only one program may claim it. Typed so the UI can show
    a persistent "in use by another app" state on the camera dot — enumeration alone
    cannot see this (the bus listing still succeeds while the claim fails)."""


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
        on_preview_died: Optional[Callable[[str], None]] = None,
        on_preview_unusable: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._gp = gp_module or _gp()
        self._jpeg_path = jpeg_path or default_jpeg_path()
        self._settings_path = settings_path or default_settings_path()
        # Called from the preview thread when the stream dies after retries, never on a normal
        # stop(). Without it a body that stops answering (issue #617) leaves the UI spinning on
        # "loading live view" forever.
        self._on_preview_died = on_preview_died
        # Called instead of the above when the stream fails but the body still answers. The
        # session stays open and scanning continues; only the preview is given up.
        self._on_preview_unusable = on_preview_unusable
        self._camera: Any = None
        self._model = ""
        # Cleared when the body stops answering. Unplugging it leaves the handle behind, and a
        # stale handle reports itself open forever.
        self._alive = False
        self._lock = threading.RLock()
        self._preview: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # Raised while a still is in flight, so the preview thread does not queue another frame
        # grab ahead of the next channel of a triplet.
        self._busy = threading.Event()
        # The post-shot event drain of the last successful still (see _finish_shot_async).
        self._post_shot: Optional[threading.Thread] = None
        # Raised by a completed still, consumed by the preview loop before its next frame. A
        # flag rather than the loop watching `_busy`: a capture can begin and end between two
        # poll ticks, and a missed drain blocks the next preview on Fujifilm.
        self._drain_owed = threading.Event()
        self._reset_body_state()

    def _reset_body_state(self) -> None:
        """Forget controls whose names and semantics belong to one camera body."""
        self._capabilities = CameraCapabilities(driver_model="")
        # Set once a stream has failed on a body that is otherwise fine, so nothing restarts it.
        # Retrying walks a Fujifilm off the USB bus entirely (issue #658).
        self._preview_broken = False
        self._drain_silence_ms = _DRAIN_SILENCE_MS
        self._drain_budget_s = _EVENT_DRAIN_S
        self._drain_owed.clear()  # a fresh body owes nothing until it has taken a shot
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
            # A camera sits on the bus but will not open. On macOS the ImageCapture daemons
            # hand it to Preview, Photos or Image Capture as soon as one of them is open, and
            # only one program may claim it. Raise the claim case typed (-53,
            # GP_ERROR_IO_USB_CLAIM) so the connection poll can pin an "in use by another app"
            # state on the camera dot, which enumeration alone cannot see.
            message = f"could not open the camera: {exc}. Close Preview, Photos and Image Capture, then retry."
            if getattr(exc, "code", None) == -53:
                raise CameraClaimedError(message) from exc
            raise GphotoError(message) from exc
        self._camera = camera
        self._model = _model_name(camera)
        self._alive = True
        self._capabilities = self._read_capabilities(camera)
        if "fuji" in self._capabilities.driver_model.lower():
            self._drain_silence_ms = _FUJI_DRAIN_SILENCE_MS
            self._drain_budget_s = _FUJI_DRAIN_BUDGET_S
        logger.info(
            "gphoto2 session open: %s (driver %r, live view %s, settings %s)",
            self._model,
            self._capabilities.driver_model,
            "yes" if self._capabilities.preview else "NO",
            "yes" if self._capabilities.config else "NO",
        )
        self._prefer_memory_capture()

    def _read_capabilities(self, camera: Any) -> CameraCapabilities:
        """What the matched driver entry advertises. Any failure means "assume it all works":
        a body that would have run fine must never be blocked by a failed introspection."""
        try:
            abilities = camera.get_abilities()
            operations = int(abilities.operations)
            return CameraCapabilities(
                driver_model=str(abilities.model),
                preview=bool(operations & self._gp.GP_OPERATION_CAPTURE_PREVIEW),
                config=bool(operations & self._gp.GP_OPERATION_CONFIG),
            )
        except Exception as exc:  # noqa: BLE001 — introspection is advisory, never load-bearing
            logger.warning("gphoto2: could not read camera abilities (%s); assuming full support", exc)
            return CameraCapabilities(driver_model="")

    @property
    def capabilities(self) -> CameraCapabilities:
        """The open session's advertised abilities; optimistic defaults while closed."""
        return self._capabilities

    def is_open(self) -> bool:
        return self._camera is not None and self._alive

    def close(self) -> None:
        self.stop()
        prev = self._post_shot
        if prev is not None and prev.is_alive():
            prev.join(timeout=3.0)  # let a post-shot drain finish before the handle goes away
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
            offered = _choices(self._gp, widget)
            if offered and value not in offered:
                # Writing a label the body never published is how a Sony-flavoured fallback
                # ladder reached a Nikon: libgphoto2 takes the string, the camera ignores it,
                # and the only symptom is a read-back that never settles (issue #768). Name
                # what the body does offer instead, at the cost of one log line.
                logger.warning(
                    "gphoto2: %s does not offer %r; this body publishes %d values (%s%s)",
                    name,
                    value,
                    len(offered),
                    ", ".join(offered[:8]),
                    ", …" if len(offered) > 8 else "",
                )
                return False
            widget.set_value(value)
            camera.set_single_config(name, widget)
        except self._gp.GPhoto2Error as exc:
            # A value this body does not offer, for example a shutter label from the fallback
            # ladder. Report it: a scan degrades to the current exposure instead of dying.
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
                # Absence is normal, since the X-T5 has no capture-target setting at all, but
                # say so: the silent skip cost a diagnosis round in issue #658.
                logger.info("gphoto2: this body has no capture-target setting; leaving it alone")
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
                options = [{"label": label, "raw": i} for i, label in enumerate(choices)]
                cur = choices.index(current) if current in choices else -1
                if key == "iso":
                    # A scan wants a fixed, single-shot ISO. Sony also lists "Auto ISO" and the
                    # Multi Frame Noise Reduction pseudo-ISOs, which put the body in a mode the
                    # scan cannot use, so keep only the plain numeric ISOs. Each keeps its
                    # original raw index.
                    fixed = [o for o in options if o["label"].isdigit()]
                    if fixed:
                        options = fixed
                        if cur not in {o["raw"] for o in fixed}:
                            # The body is on Auto or MFNR. Instead of faking a fixed value in the
                            # stepper, switch the body to its lowest real ISO and report that.
                            # Fire-and-forget: the next settings read confirms it.
                            lowest = min(fixed, key=lambda o: int(o["label"]))
                            try:
                                widget.set_value(choices[lowest["raw"]])
                                camera.set_single_config(name, widget)
                                cur = lowest["raw"]
                            except self._gp.GPhoto2Error as exc:
                                logger.warning("gphoto2: could not switch %s off Auto/MFNR: %s", name, exc)
                out[key] = {
                    "cur": cur,
                    "writable": not widget.get_readonly(),
                    "options": options,
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
        """Set the zoom ratio (carrying the aim point on bodies that pack them together) and
        return without waiting for the read-back.

        The body takes ~1-2 s to engage the magnifier, and polling the property for that whole
        time holds the single PTP claim — which freezes the live-view preview until it lands, and
        that freeze *is* the click-to-zoom lag. Fire-and-forget instead: send the write, release
        the lock, and the zoom shows up in the still-streaming preview as the body engages (the
        reads no longer compete with that engage either). Nothing downstream depends on the
        magnifier, so the confirmation isn't worth the freeze."""
        spec, (x, y) = self._magnifier, self._position
        assert spec is not None  # noqa: S101 — callers probe first
        value = ratio if not spec.packed else f"{ratio},{x},{y}"
        camera = self._require()
        try:
            widget = camera.get_single_config(spec.ratio)
            widget.set_value(value)
            camera.set_single_config(spec.ratio, widget)
        except self._gp.GPhoto2Error as exc:
            logger.warning("gphoto2: could not set magnifier %r to %r: %s", spec.ratio, value, exc)

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

    def _drain_events(self, budget_s: Optional[float] = None) -> None:
        """Consume the events a still leaves behind.

        Skipping this makes the *next* `capture()` fail with a bare `[-1]`. The quiet window
        that ends the drain is per-body (see `_FUJI_DRAIN_SILENCE_MS`): on Fujifilm, events
        left unread past the first pause are what killed live view after every still.
        """
        camera = self._require()
        deadline = time.monotonic() + (self._drain_budget_s if budget_s is None else budget_s)
        while time.monotonic() < deadline:
            kind, _data = camera.wait_for_event(self._drain_silence_ms)
            if kind == self._gp.GP_EVENT_TIMEOUT:
                return

    def capture(self, out_path: str, shutter: Optional[str] = None, iso: Optional[str] = None, aperture: Optional[str] = None) -> str:
        """Take one still, write the RAW next to `out_path`, and return where it landed.

        The suffix comes from the camera — a Canon writes `.CR3`, a Nikon `.NEF` — so the
        returned path may differ from the one asked for. Callers must use the return value.
        Blocks the preview meanwhile. `iso`/`aperture` lock the body to the preset's exposure
        (a scan re-asserts them so a drifted setting can't falsify it); `_set_verified` skips the
        write when the body is already there, so it costs a read unless something actually moved.
        """
        prev = self._post_shot
        if prev is not None and prev.is_alive():
            prev.join()  # the previous shot's drain normally finished during the caller's LED settle
        self._busy.set()
        drained_async = False
        try:
            with self._lock:
                camera = self._require()
                t0 = time.perf_counter()
                if shutter:
                    name = self._property("shutter")
                    if name is None or not self._set_verified(name, shutter):
                        raise GphotoError(f"could not set shutter to {shutter!r}: camera rejected it or it did not settle")
                for prop, value in (("iso", iso), ("aperture", aperture)):
                    # Not fatal like the shutter: a warning plus the current setting beats
                    # aborting a scan, and the preset's value was settable at calibration.
                    if value:
                        name = self._property(prop)
                        if name is None or not self._set_verified(name, value):
                            logger.warning("gphoto2: could not lock %s to %r for the scan; using the current setting", prop, value)
                t_assert = time.perf_counter()
                try:
                    path = camera.capture(self._gp.GP_CAPTURE_IMAGE)
                    t_shot = time.perf_counter()
                    suffix = os.path.splitext(path.name)[1]
                    if suffix.lower() not in _CAMERA_RAW_EXTENSIONS:
                        shown = suffix or "(no extension)"
                        raise GphotoError(f"camera returned {shown}, not a RAW file; set the camera to RAW-only image quality and retry")
                    camera_file = camera.file_get(path.folder, path.name, self._gp.GP_FILE_TYPE_NORMAL)
                    data = bytes(memoryview(camera_file.get_data_and_size()))
                    t_download = time.perf_counter()
                    try:
                        # Delete the downloaded shot from the body. On a self-recycling RAM
                        # target this is redundant at worst, but a body without a capture target
                        # keeps every tethered object until it refuses to capture at all
                        # (issue #658).
                        camera.file_delete(path.folder, path.name)
                    except self._gp.GPhoto2Error as exc:
                        logger.debug("gphoto2: could not delete %s off the camera: %s", path.name, exc)
                except BaseException:
                    # A failed shot drains inline, still under the lock: the error must leave a
                    # clean queue behind before the preview or a retry touches the session.
                    self._drain_events()
                    raise
        except self._gp.GPhoto2Error as exc:
            raise GphotoError(f"capture failed: {exc}") from exc
        else:
            self._drain_owed.set()  # the preview must clear this shot's stragglers before resuming
            self._finish_shot_async()
            drained_async = True
        finally:
            if not drained_async:
                self._busy.clear()

        suffix = os.path.splitext(path.name)[1]
        if suffix:
            out_path = os.path.splitext(out_path)[0] + suffix
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp = f"{out_path}.part"
        with open(tmp, "wb") as handle:
            handle.write(data)
        os.replace(tmp, out_path)
        logger.info(
            "gphoto2 captured %s (%.1f MB): assert %.0f ms, shot %.0f ms, download %.0f ms (%.0f MB/s)",
            os.path.basename(out_path),
            len(data) / 1e6,
            (t_assert - t0) * 1000,
            (t_shot - t_assert) * 1000,
            (t_download - t_shot) * 1000,
            len(data) / 1e6 / max(t_download - t_shot, 1e-9),
        )
        return out_path

    def _finish_shot_async(self) -> None:
        """Drain the shot's leftover events off the caller's thread.

        The ~150 ms of wait_for_event after every still would otherwise sit between two
        channels of a triplet; on a worker it runs while the service is already switching
        the light and sleeping its LED settle. Vendor-neutral by construction — the order
        on the wire stays exactly sequential (shot → download → drain → next operation),
        because `_busy` keeps the preview parked until the queue is clean and the next
        `capture()` joins this thread before claiming the body.
        """

        def _run() -> None:
            start = time.perf_counter()
            try:
                with self._lock:
                    if self._camera is not None and self._alive:
                        self._drain_events()
                logger.info("gphoto2 post-shot drain %.0f ms (overlapped)", (time.perf_counter() - start) * 1000)
            except Exception as exc:  # noqa: BLE001 — a teardown race must not kill the worker
                logger.warning("gphoto2 post-shot drain: %s", exc)
            finally:
                self._busy.clear()

        thread = threading.Thread(target=_run, name="gphoto-drain", daemon=True)
        self._post_shot = thread
        thread.start()

    # ----- live view -------------------------------------------------------------

    def start(self) -> None:
        if self.is_running():
            return
        self.open()
        if self._preview_broken:
            # A stream already failed on this body while it kept answering. Restarting it turns
            # one bad preview into a reconnect loop, so refuse until the session is closed. A
            # reconnect or power cycle clears the flag through _reset_body_state.
            raise LiveViewUnsupported(self._live_view_refusal())
        if not self._capabilities.preview:
            # Refuse instead of letting _preview_loop find out. The body rejects
            # capture_preview at the device level, and that refusal wedges the PTP session
            # until the camera is power-cycled (issue #621). Retrying deepens the wedge, so
            # catch it before the first call.
            # Publish the settings once on the way out, which is normally the preview loop's
            # job. Without them the scan window's steppers stay empty and the preset's exposure
            # cannot resolve to this body's choice indices.
            self._publish_settings()
            raise LiveViewUnsupported(self._live_view_refusal())
        self._stop.clear()
        self._preview = threading.Thread(target=self._preview_loop, name="gphoto-liveview", daemon=True)
        self._preview.start()

    def _live_view_refusal(self) -> str:
        """Why live view is off, phrased for the operator.

        The distinction matters: most bodies list a Control/PC-Remote entry *with* preview and
        an MTP one without, so a missing bit under MTP is a wrong USB mode the user can fix,
        not a limit of the camera. Only a body that lacks it in its tethering mode (the a6000)
        genuinely cannot preview — and it still scans.
        """
        if self._preview_broken:
            # The body advertised the capability, then failed to deliver it. Say so plainly
            # instead of blaming the connection, and make clear that scanning still works.
            return (
                f"{self._model or 'this camera'} accepted the connection but its live view does not work "
                "(a known gap in libgphoto2's Fujifilm support). Scanning still works, just without a preview."
            )
        if self._capabilities.mtp_mode:
            return (
                f"{self._model or 'this camera'} is connected in MTP mode, which has no live view. "
                "Switch the camera's USB setting to PC Remote (or Tether/Control) and reconnect."
            )
        return f"{self._model or 'this camera'} does not support live view. Scanning still works, just without a preview."

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
        # Drain after a still, never before one. Fujifilm queues post-shot events past the
        # overlapped drain's quiet window, and events left unread block capture_preview
        # (issue #658). Draining with nothing pending is worse than wasteful: on a freshly
        # opened Canon EOS, wait_for_event segfaults the process (issue #745) and no `except`
        # can catch it. `_drain_owed` is raised only by a completed still.
        saw_capture = False
        while not self._stop.is_set():
            if self._busy.is_set():  # stand aside for a capture rather than race it for the lock
                self._stop.wait(0.02)
                continue
            try:
                with self._lock:
                    if self._camera is None:
                        return
                    if self._drain_owed.is_set():
                        self._drain_owed.clear()
                        saw_capture = True
                        self._drain_events()
                    frame = self._camera.capture_preview()
                    data = bytes(memoryview(frame.get_data_and_size()))
                self._publish_frame(data)
                failures = 0
                if time.monotonic() >= next_settings:
                    self._publish_settings()
                    next_settings = time.monotonic() + _SETTINGS_INTERVAL_S
            except Exception as exc:  # noqa: BLE001 — a dropped frame must not kill the thread
                failures += 1
                if saw_capture:
                    self._drain_owed.set()  # unread events can block a retry, but only a capture leaves any
                logger.warning("gphoto2 live view (%d/%d): %s", failures, _MAX_PREVIEW_FAILURES, exc)
                if failures >= _MAX_PREVIEW_FAILURES:
                    # Repeated failures used to mean the body was gone, and the whole session was
                    # dropped. On Fujifilm bodies that verdict is wrong: capture_preview returns
                    # [-1] forever while the camera keeps shooting fine (issue #658). Tearing the
                    # session down kills a working scan, and the reopen-and-retry cycle walks the
                    # body off the USB bus. Ask the camera before deciding.
                    if self._camera_answers():
                        logger.warning("gphoto2: live view failed but the camera still answers — continuing without a preview")
                        self._preview_broken = True
                        self._report_preview_stopped(self._on_preview_unusable, self._live_view_refusal())
                        return
                    logger.warning("gphoto2: camera stopped answering, dropping the session")
                    self._alive = False  # nothing may reuse the handle; close() would join this thread
                    self._report_preview_stopped(self._on_preview_died, str(exc))
                    return
                self._stop.wait(0.5)
                continue
            self._stop.wait(_PREVIEW_INTERVAL_S)

    def _camera_answers(self) -> bool:
        """Does the body still respond to a plain property read?

        The one question that separates "this camera is gone" from "only its preview is
        broken". A config read is the cheapest vendor-neutral liveness check there is, and it
        touches nothing the preview owns. A body that advertises no CONFIG cannot be asked
        this way, so it keeps the old verdict — rare, and erring toward the safer teardown.
        """
        if not self._capabilities.config:
            return False
        try:
            with self._lock:
                camera = self._camera
                if camera is None or not self._alive:
                    return False
                name = self._property("iso")
                if name is None:
                    return False
                camera.get_single_config(name)
            return True
        except Exception:  # noqa: BLE001 — any failure here means "cannot confirm it is alive"
            return False

    def _report_preview_stopped(self, callback: Optional[Callable[[str], None]], reason: str) -> None:
        if callback is None:
            return
        try:
            callback(reason)
        except Exception:  # noqa: BLE001 — the report must not break teardown
            logger.exception("preview-stopped callback failed")

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
