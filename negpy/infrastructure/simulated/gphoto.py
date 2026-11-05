"""A python-gphoto2 stand-in: one body that exposes the simulated film under the simulated Scanlight.

A still lit white or by the triplet's first channel advances the film; the other two channels stay on it.
A Bulb exposure, opened over USB or by the light's shutter pulse, adds up the light it saw and is one frame.
"""

import functools
import io
import time
from typing import Optional

import numpy as np
from PIL import Image
import tifffile

from negpy.infrastructure.capture.base import CAPTURE_ORDER
from negpy.infrastructure.simulated.images import film
from negpy.infrastructure.simulated.scanlight import current_color, current_ir, light_dose, take_pulse

MODEL = "Simulated Camera"
_SENSOR_HW = (2000, 3000)  # a smaller frame fails the capture path's minimum RAW size
_PREVIEW_HW = (480, 640)
_BLACK, _WHITE = 512, 16383
# Clear-base signal per LED count per second at ISO 100, as a fraction of full scale.
# The calibration's reference start point must land below target.
_RESPONSE = np.array([0.012, 0.055, 0.13], np.float32)
# Every photosite sees 850 nm alike, dimmed by the body's IR-cut filter.
_IR_RESPONSE = 0.02
_CROSSTALK = 0.03
_WHITE_SHARE = 0.5
_SHUTTERS = (
    "1/250", "1/200", "1/160", "1/125", "1/100", "1/80", "1/60", "1/50", "1/40", "1/30", "1/25", "1/20", "1/15",
    "1/13", "1/10", "1/8", "1/6", "1/5", "1/4", "1/3", "0.4", "1/2", "0.6", "0.8", "1", "1.3", "1.6", "2", "Bulb",
)  # fmt: skip
# DNGVersion, UniqueCameraModel, CFARepeatPatternDim, CFAPattern (RGGB), BlackLevel, WhiteLevel, ColorMatrix1, AsShotNeutral.
_DNG_TAGS = [
    (50706, "B", 4, (1, 4, 0, 0), True),
    (50708, "s", 0, MODEL, True),
    (33421, "H", 2, (2, 2), True),
    (33422, "B", 4, (0, 1, 1, 2), True),
    (50714, "H", 1, (_BLACK,), True),
    (50717, "H", 1, (_WHITE,), True),
    (50721, "2i", 9, (1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1), True),
    (50728, "2I", 3, (1, 1, 1, 1, 1, 1), True),
]
_CFA_PHOTOMETRIC = 32803
# Enough texture that every frame of a roll passes trichrome grouping's same-frame floor.
_FILM_SHAPES = 24
_FIRST_CHANNEL = "RGB".index(CAPTURE_ORDER[0].letter)


def _seconds(label: str) -> float:
    num, _, den = label.partition("/")
    try:
        return float(num) / float(den) if den else float(num)
    except ValueError:
        return 1.0  # Bulb: a plain capture at Bulb exposes for a second


class GPhoto2Error(Exception):
    def __init__(self, message: str, code: int = -1) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code


class _Widget:
    def __init__(self, name: str, value: Optional[str], choices: list[str], readonly: bool = False) -> None:
        self.name, self.value, self.choices, self.readonly = name, value, choices, readonly
        self.pending = value

    def get_name(self) -> str:
        return self.name

    def get_type(self) -> int:
        return SimGphoto.GP_WIDGET_RADIO

    def get_readonly(self) -> bool:
        return self.readonly

    def count_choices(self) -> int:
        return len(self.choices)

    def get_choice(self, i: int) -> str:
        return self.choices[i]

    def get_value(self) -> Optional[str]:
        return self.value

    def set_value(self, value: str) -> None:
        self.pending = value


class _Data:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def get_data_and_size(self) -> bytes:
        return self._data


class _Path:
    folder = "/"

    def __init__(self, name: str) -> None:
        self.name = name


class _Abilities:
    model = MODEL
    operations = 1 | 8 | 16  # CAPTURE_IMAGE | CAPTURE_PREVIEW | CONFIG


class _CameraList:
    def __init__(self, items: list[tuple[str, str]]) -> None:
        self._items = items

    def count(self) -> int:
        return len(self._items)

    def get_name(self, i: int) -> str:
        return self._items[i][0]

    def get_value(self, i: int) -> str:
        return self._items[i][1]


class _Camera:
    def __init__(self, gp: "SimGphoto") -> None:
        self._gp = gp

    def init(self) -> None:
        pass

    def exit(self) -> None:
        pass

    def get_summary(self) -> str:
        return f"Model: {MODEL}\nSerial: SIM0001"

    def get_abilities(self) -> _Abilities:
        return _Abilities()

    def get_single_config(self, name: str) -> _Widget:
        try:
            return self._gp.props[name]
        except KeyError:
            raise GPhoto2Error(f"no property {name}", -2) from None

    def set_single_config(self, name: str, widget: _Widget) -> None:
        widget.value = widget.pending
        if name == "bulb":
            self._gp.bulb_switch(bool(int(widget.value or 0)))

    def capture(self, _kind: int) -> _Path:
        self._gp.pending_raw = self._gp.still_dng()
        self._gp.shot_events = 1
        return _Path("capt0001.DNG")

    def file_get(self, _folder: str, _name: str, _kind: int) -> _Data:
        return _Data(self._gp.pending_raw)

    def file_delete(self, _folder: str, _name: str) -> None:
        pass

    def capture_preview(self) -> _Data:
        return _Data(self._gp.preview_jpeg())

    def wait_for_event(self, _ms: int) -> tuple[int, object]:
        self._gp.collect_pulse()
        if self._gp.events:
            return self._gp.events.pop(0)
        if self._gp.shot_events:
            self._gp.shot_events -= 1
            return SimGphoto.GP_EVENT_CAPTURE_COMPLETE, None
        return SimGphoto.GP_EVENT_TIMEOUT, None


class SimGphoto:
    GP_WIDGET_TEXT, GP_WIDGET_RADIO, GP_WIDGET_MENU = 2, 5, 6
    GP_CAPTURE_IMAGE, GP_FILE_TYPE_NORMAL = 0, 1
    GP_EVENT_TIMEOUT, GP_EVENT_FILE_ADDED, GP_EVENT_CAPTURE_COMPLETE = 0, 2, 3
    GP_OPERATION_CAPTURE_PREVIEW, GP_OPERATION_CONFIG = 8, 16
    GPhoto2Error = GPhoto2Error

    def __init__(self) -> None:
        self.props = {
            "iso": _Widget("iso", "100", ["Auto ISO", "100", "200", "400", "800"]),
            "shutterspeed": _Widget("shutterspeed", "1/4", list(_SHUTTERS)),
            # A manual lens: no electronic aperture.
            "f-number": _Widget("f-number", None, [], readonly=True),
            "capturetarget": _Widget("capturetarget", "card", ["card", "sdram"]),
            "focusmagnifier": _Widget("focusmagnifier", "Off,320,240", ["Off", "1", "6.9", "13.7"]),
            "bulb": _Widget("bulb", "0", []),
        }
        self.pending_raw = b""
        self.shot_events = 0
        self.events: list[tuple[int, object]] = []
        self._bulb_open: Optional[float] = None
        self._rng = np.random.default_rng(0)
        self._frame = 0
        self.Camera = lambda: _Camera(self)
        self.Camera.autodetect = lambda: _CameraList([(MODEL, "usb:sim")])

    def _signal(self, transmittance: np.ndarray, ir_transmittance: np.ndarray, dose: Optional[tuple] = None) -> np.ndarray:
        """Each sensor channel's exposure, as a fraction of full scale, through `transmittance`.
        `dose` is a Bulb exposure's level-seconds per channel, in place of the light as lit now."""
        if dose is None:
            (r, g, b, w), ir, seconds = current_color(), current_ir(), _seconds(self.props["shutterspeed"].value or "1")
        else:
            (r, g, b, w, ir), seconds = dose, 1.0
        led = np.array([r, g, b], np.float32)
        lit = led + _CROSSTALK * (led.sum() - led) + _WHITE_SHARE * w
        iso = self.props["iso"].value or ""
        gain = seconds * (int(iso) / 100 if iso.isdigit() else 1.0)
        signal = transmittance * (_RESPONSE * lit * gain)
        if ir:
            signal = signal + (ir_transmittance * (_IR_RESPONSE * ir * gain))[..., None]
        return signal

    def bulb_switch(self, on: bool) -> None:
        """The USB bulb button: pressed opens the shutter, released closes it and adds the file."""
        if self.props["shutterspeed"].value != "Bulb":
            return
        if on:
            self._bulb_open = time.monotonic()
        elif self._bulb_open is not None:
            self._finish_bulb(self._bulb_open, time.monotonic())
            self._bulb_open = None

    def collect_pulse(self) -> None:
        """A shutter pulse from the light's 3.5mm jack exposes a body in Bulb for the pulse."""
        pulse = take_pulse()
        if pulse is not None and self.props["shutterspeed"].value == "Bulb":
            self._finish_bulb(*pulse)

    def _finish_bulb(self, t0: float, t1: float) -> None:
        self.pending_raw = self.still_dng(light_dose(t0, t1))
        self.events.append((self.GP_EVENT_FILE_ADDED, _Path("capt0001.DNG")))
        self.events.append((self.GP_EVENT_CAPTURE_COMPLETE, None))

    def _advance_film(self) -> None:
        r, g, b, w = current_color()
        if current_ir() and not any((r, g, b, w)):
            return  # an infrared exposure stays on the frame it belongs to
        if w > max(r, g, b) or int(np.argmax((r, g, b))) == _FIRST_CHANNEL:
            self._frame += 1

    def still_dng(self, dose: Optional[tuple] = None) -> bytes:
        if dose is None:
            self._advance_film()
        else:
            self._frame += 1  # a Bulb exposure is a whole frame
        quad = self._signal(*_film(_SENSOR_HW[0] // 2, _SENSOR_HW[1] // 2, self._frame), dose)
        cfa = np.empty(_SENSOR_HW, np.float32)
        cfa[0::2, 0::2], cfa[0::2, 1::2], cfa[1::2, 0::2], cfa[1::2, 1::2] = quad[..., 0], quad[..., 1], quad[..., 1], quad[..., 2]
        counts = cfa * (_WHITE - _BLACK)
        counts += self._rng.normal(0.0, 1.0, _SENSOR_HW).astype(np.float32) * np.sqrt(counts + 4.0)
        raw = np.clip(counts + _BLACK, 0, _WHITE).astype(np.uint16)
        out = io.BytesIO()
        tifffile.imwrite(out, raw, photometric=_CFA_PHOTOMETRIC, subfiletype=0, extratags=_DNG_TAGS, metadata=None)
        return out.getvalue()

    def preview_jpeg(self) -> bytes:
        display = np.clip(self._signal(*_film(*_PREVIEW_HW, self._frame)), 0.0, 1.0) ** (1 / 2.2)
        out = io.BytesIO()
        Image.fromarray((display * 255).astype(np.uint8)).save(out, "JPEG", quality=95)
        return out.getvalue()


@functools.lru_cache(maxsize=4)
def _film(h: int, w: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    return film(h, w, seed, _FILM_SHAPES)


@functools.cache
def module() -> SimGphoto:
    return SimGphoto()
