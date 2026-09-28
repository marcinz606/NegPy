"""A python-gphoto2 module stand-in: one camera body with live view, settings and stills.

Every frame is exposed from the simulated film, the simulated Scanlight's current color and
the body's shutter and ISO, so calibration and the triplet channels respond to the light. A
still is a Bayer DNG, which the capture path decodes like any camera RAW.
"""

import functools
import io
from typing import Optional

import cv2
import numpy as np
import tifffile

from negpy.infrastructure.simulated.images import film
from negpy.infrastructure.simulated.scanlight import current_color

MODEL = "Simulated Camera"
_SENSOR_HW = (2000, 3000)  # a smaller frame fails the capture path's minimum RAW size
_PREVIEW_HW = (480, 640)
_BLACK, _WHITE = 512, 16383
# Clear-base signal, as a fraction of full scale per LED count per second at ISO 100, of the
# sensor channel each LED lights. The calibration's reference start point lands below target.
_RESPONSE = np.array([0.012, 0.055, 0.13], np.float32)
_CROSSTALK = 0.03  # what each narrowband LED leaks into the other two sensor channels
_WHITE_SHARE = 0.5
_SHUTTERS = (
    "1/250", "1/200", "1/160", "1/125", "1/100", "1/80", "1/60", "1/50", "1/40", "1/30", "1/25", "1/20", "1/15",
    "1/13", "1/10", "1/8", "1/6", "1/5", "1/4", "1/3", "0.4", "1/2", "0.6", "0.8", "1", "1.3", "1.6", "2",
)  # fmt: skip
_DNG_TAGS = [
    (50706, "B", 4, (1, 4, 0, 0), True),  # DNGVersion
    (50708, "s", 0, MODEL, True),  # UniqueCameraModel
    (33421, "H", 2, (2, 2), True),  # CFARepeatPatternDim
    (33422, "B", 4, (0, 1, 1, 2), True),  # CFAPattern: RGGB
    (50714, "H", 1, (_BLACK,), True),  # BlackLevel
    (50717, "H", 1, (_WHITE,), True),  # WhiteLevel
    (50721, "2i", 9, (1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1), True),  # ColorMatrix1: identity
    (50728, "2I", 3, (1, 1, 1, 1, 1, 1), True),  # AsShotNeutral
]
_CFA_PHOTOMETRIC = 32803


def _seconds(label: str) -> float:
    num, _, den = label.partition("/")
    return float(num) / float(den) if den else float(num)


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

    def wait_for_event(self, _ms: int) -> tuple[int, None]:
        if self._gp.shot_events:
            self._gp.shot_events -= 1
            return SimGphoto.GP_EVENT_CAPTURE_COMPLETE, None
        return SimGphoto.GP_EVENT_TIMEOUT, None


class SimGphoto:
    GP_WIDGET_TEXT, GP_WIDGET_RADIO, GP_WIDGET_MENU = 2, 5, 6
    GP_CAPTURE_IMAGE, GP_FILE_TYPE_NORMAL = 0, 1
    GP_EVENT_TIMEOUT, GP_EVENT_CAPTURE_COMPLETE = 0, 3
    GP_OPERATION_CAPTURE_PREVIEW, GP_OPERATION_CONFIG = 8, 16
    GPhoto2Error = GPhoto2Error

    def __init__(self) -> None:
        self.props = {
            "iso": _Widget("iso", "100", ["Auto ISO", "100", "200", "400", "800"]),
            "shutterspeed": _Widget("shutterspeed", "1/4", list(_SHUTTERS)),
            # A manual scanning lens: no electronic aperture, so no choices.
            "f-number": _Widget("f-number", None, [], readonly=True),
            "capturetarget": _Widget("capturetarget", "card", ["card", "sdram"]),
            "focusmagnifier": _Widget("focusmagnifier", "Off,320,240", ["Off", "1", "6.9", "13.7"]),
        }
        self.pending_raw = b""
        self.shot_events = 0
        self._rng = np.random.default_rng(0)
        self.Camera = lambda: _Camera(self)
        self.Camera.autodetect = lambda: _CameraList([(MODEL, "usb:sim")])

    def _signal(self, transmittance: np.ndarray) -> np.ndarray:
        """Each sensor channel's exposure, as a fraction of full scale, through `transmittance`."""
        r, g, b, w = current_color()
        led = np.array([r, g, b], np.float32)
        lit = led + _CROSSTALK * (led.sum() - led) + _WHITE_SHARE * w
        iso = self.props["iso"].value or ""
        gain = _seconds(self.props["shutterspeed"].value or "1") * (int(iso) / 100 if iso.isdigit() else 1.0)
        return transmittance * (_RESPONSE * lit * gain)

    def still_dng(self) -> bytes:
        quad = self._signal(_film(_SENSOR_HW[0] // 2, _SENSOR_HW[1] // 2))
        cfa = np.empty(_SENSOR_HW, np.float32)
        cfa[0::2, 0::2], cfa[0::2, 1::2], cfa[1::2, 0::2], cfa[1::2, 1::2] = quad[..., 0], quad[..., 1], quad[..., 1], quad[..., 2]
        counts = cfa * (_WHITE - _BLACK)
        counts += self._rng.normal(0.0, 1.0, _SENSOR_HW).astype(np.float32) * np.sqrt(counts + 4.0)
        raw = np.clip(counts + _BLACK, 0, _WHITE).astype(np.uint16)
        out = io.BytesIO()
        tifffile.imwrite(out, raw, photometric=_CFA_PHOTOMETRIC, subfiletype=0, extratags=_DNG_TAGS, metadata=None)
        return out.getvalue()

    def preview_jpeg(self) -> bytes:
        display = np.clip(self._signal(_film(*_PREVIEW_HW)), 0.0, 1.0) ** (1 / 2.2)
        _ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor((display * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        return jpeg.tobytes()


@functools.cache
def _film(h: int, w: int) -> np.ndarray:
    return film(h, w)[0]


@functools.cache
def module() -> SimGphoto:
    return SimGphoto()
