"""A python-gphoto2 module stand-in: one camera body with live view, settings and stills.

A still is a real RAW file read from disk (`NEGPY_SIM_RAW`, a file or a folder whose RAWs are
used in turn), because the capture path checks its size and decodes it.
"""

import functools
import itertools
import os
from pathlib import Path
from typing import Iterator, Optional

import cv2

from negpy.infrastructure.loaders.constants import SUPPORTED_JPEG_EXTENSIONS, SUPPORTED_RAW_EXTENSIONS, SUPPORTED_TIFF_EXTENSIONS
from negpy.infrastructure.simulated.images import negative

MODEL = "Simulated Camera"
_RAW_EXTENSIONS = SUPPORTED_RAW_EXTENSIONS - SUPPORTED_JPEG_EXTENSIONS - SUPPORTED_TIFF_EXTENSIONS
_DEFAULT_RAW_DIR = "samples"


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


def _raw_files() -> list[Path]:
    root = Path(os.environ.get("NEGPY_SIM_RAW") or _DEFAULT_RAW_DIR)
    if root.is_file():
        return [root]
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _RAW_EXTENSIONS)


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
        raw = self._gp.next_raw()
        self._gp.pending_raw = raw
        self._gp.shot_events = 1
        return _Path(f"capt0001{raw.suffix}")

    def file_get(self, _folder: str, _name: str, _kind: int) -> _Data:
        return _Data(self._gp.pending_raw.read_bytes())

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
            "shutterspeed": _Widget("shutterspeed", "1/60", ["1/250", "1/125", "1/60", "1/30", "1/15", "1/8", "1/4", "1/2", "1"]),
            # A manual scanning lens: no electronic aperture, so no choices.
            "f-number": _Widget("f-number", None, [], readonly=True),
            "capturetarget": _Widget("capturetarget", "card", ["card", "sdram"]),
            "focusmagnifier": _Widget("focusmagnifier", "Off,320,240", ["Off", "1", "6.9", "13.7"]),
        }
        self.pending_raw = Path()
        self.shot_events = 0
        self._raws: Optional[Iterator[Path]] = None
        self._preview: Optional[bytes] = None
        self.Camera = lambda: _Camera(self)
        self.Camera.autodetect = lambda: _CameraList([(MODEL, "usb:sim")])

    def next_raw(self) -> Path:
        if self._raws is None:
            files = _raw_files()
            if not files:
                raise GPhoto2Error("simulated camera has no RAW to return: set NEGPY_SIM_RAW to a RAW file or a folder of RAWs")
            self._raws = itertools.cycle(files)
        return next(self._raws)

    def preview_jpeg(self) -> bytes:
        if self._preview is None:
            rgb, _ir = negative(480, 640)
            _ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor((rgb >> 8).astype("uint8"), cv2.COLOR_RGB2BGR))
            self._preview = jpeg.tobytes()
        return self._preview


@functools.cache
def module() -> SimGphoto:
    return SimGphoto()
