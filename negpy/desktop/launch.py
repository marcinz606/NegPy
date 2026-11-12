"""Files named on the command line, and the handoff of those files to a NegPy that is already open."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from typing import Optional, Sequence

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtNetwork import QLocalServer, QLocalSocket

from negpy.kernel.system.logging import get_logger
from negpy.services.export.launch_recipe import INTENTS, LaunchRecipe, parse_color_space, parse_format, parse_ratio

logger = get_logger(__name__)

_CONNECT_TIMEOUT_MS = 1000
_REPLY_TIMEOUT_MS = 5000
_ACK = b"ok"


@dataclass(frozen=True)
class LaunchRequest:
    paths: tuple[str, ...] = ()
    recipe: LaunchRecipe = field(default_factory=LaunchRecipe)
    render: bool = False  # --render: write previews without a window, then exit


class LaunchArgsError(ValueError):
    """A command-line flag NegPy cannot use. The launch stops before it opens or hands off anything."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # type: ignore[override]
        raise LaunchArgsError(message)


def _positive(kind):
    def convert(text: str):
        value = kind(text)
        if value <= 0:
            raise ValueError(text)
        return value

    convert.__name__ = kind.__name__
    return convert


def _quality(text: str) -> int:
    value = int(text)
    if not 1 <= value <= 100:
        raise ValueError(text)
    return value


def _choice(parse):
    def convert(text: str) -> str:
        try:
            return parse(text)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from None

    return convert


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="NegPy", add_help=False, allow_abbrev=False)
    parser.add_argument("--preset")
    parser.add_argument("--intent", choices=INTENTS)
    parser.add_argument("--format", dest="export_fmt", type=_choice(parse_format))
    parser.add_argument("--bit-depth", dest="export_bit_depth", type=int, choices=(8, 16))
    parser.add_argument("--quality", type=_quality)
    color = parser.add_mutually_exclusive_group()
    color.add_argument("--color-space", type=_choice(parse_color_space))
    color.add_argument("--profile")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--original-size", action="store_true")
    size.add_argument("--long-edge", type=_positive(int))
    size.add_argument("--print-size", type=_positive(float))
    parser.add_argument("--dpi", type=_positive(int))
    parser.add_argument("--ratio", dest="paper_aspect_ratio", type=_choice(parse_ratio))
    parser.add_argument("--filename", dest="filename_pattern")
    parser.add_argument("--output-dir")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.add_argument("--quit-after-export", action="store_true")
    parser.add_argument("--auto-crop", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("paths", nargs="*")
    return parser


def parse_launch_args(argv: Sequence[str]) -> LaunchRequest:
    """Files and folders to open, and the export recipe for them. Single-dash arguments Qt or the OS
    add (``-style``, macOS ``-psn_``) are ignored; an unknown ``--`` flag raises ``LaunchArgsError``.
    Paths resolve against the caller's working directory, because a running instance has its own."""
    known, unknown = _build_parser().parse_known_args(list(argv))
    flags = [arg for arg in unknown if arg.startswith("--")]
    if flags:
        raise LaunchArgsError(f"unrecognized arguments: {' '.join(flags)}")
    if known.dpi and known.long_edge:
        raise LaunchArgsError("--dpi applies to --print-size, not to --long-edge")
    if known.profile and not os.path.isfile(known.profile):
        raise LaunchArgsError(f"ICC profile not found: {known.profile}")
    values = vars(known)
    render = values.pop("render")
    paths = tuple(os.path.abspath(p) for p in values.pop("paths") if p and not p.startswith("-"))
    if render:
        if not paths:
            raise LaunchArgsError("--render needs at least one file or folder")
        if known.intent == "linear":
            raise LaunchArgsError("--render makes print or flat files; --intent linear is not available")
        if known.quit_after_export:
            raise LaunchArgsError("--quit-after-export does not apply to --render, which always exits")
    for key in ("output_dir", "profile"):
        if values[key]:
            values[key] = os.path.abspath(os.path.expanduser(values[key]))
    return LaunchRequest(paths=paths, recipe=LaunchRecipe(**values), render=render)


def server_name(user_dir: str) -> str:
    """One name per data folder: two NegPy instances on separate data folders never share databases."""
    key = os.path.normcase(os.path.abspath(user_dir))
    user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    return "negpy-" + hashlib.sha1(f"{user}|{key}".encode("utf-8")).hexdigest()[:16]


def encode_request(request: LaunchRequest) -> bytes:
    return json.dumps({"paths": list(request.paths), "recipe": asdict(request.recipe)}).encode("utf-8")


def decode_request(data: bytes) -> Optional[LaunchRequest]:
    try:
        payload = json.loads(data.decode("utf-8"))
        paths = tuple(str(p) for p in payload.get("paths") or ())
        known = {f.name for f in fields(LaunchRecipe)}
        recipe = LaunchRecipe(**{k: v for k, v in (payload.get("recipe") or {}).items() if k in known})
        return LaunchRequest(paths=paths, recipe=recipe)
    except (ValueError, AttributeError, TypeError):
        logger.warning("Ignored a malformed launch request")
        return None


def hand_off(name: str, request: LaunchRequest) -> bool:
    """Send the request to the NegPy listening on ``name``. True when it took the request; the
    caller then exits. Blocking socket calls need no QCoreApplication."""
    socket = QLocalSocket()
    socket.connectToServer(name)
    if not socket.waitForConnected(_CONNECT_TIMEOUT_MS):
        return False
    if sys.platform == "win32":
        # Windows lets only the foreground process give the foreground away.
        try:
            import ctypes

            ctypes.windll.user32.AllowSetForegroundWindow(-1)  # ASFW_ANY
        except (AttributeError, OSError):
            pass
    data = encode_request(request)
    socket.write(len(data).to_bytes(4, "big") + data)
    socket.waitForBytesWritten(_REPLY_TIMEOUT_MS)
    ok = socket.waitForReadyRead(_REPLY_TIMEOUT_MS) and bytes(socket.readAll().data()).startswith(_ACK)
    socket.disconnectFromServer()
    return ok


class LaunchServer(QObject):
    """Listens for a second NegPy launch and emits its request on the GUI thread."""

    received = pyqtSignal(object)  # LaunchRequest

    def __init__(self, name: str, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._server = QLocalServer(self)
        self._server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        self._server.newConnection.connect(self._on_connection)
        if not self._server.listen(name):
            # A crashed instance can leave its socket file behind on Linux and macOS.
            QLocalServer.removeServer(name)
            if not self._server.listen(name):
                logger.warning("Launch handoff unavailable: %s", self._server.errorString())

    def is_listening(self) -> bool:
        return self._server.isListening()

    def close(self) -> None:
        self._server.close()

    def _on_connection(self) -> None:
        while self._server.hasPendingConnections():
            socket = self._server.nextPendingConnection()
            if socket is None:
                return
            socket.setProperty("negpy_buffer", b"")
            socket.readyRead.connect(self._on_ready_read)
            socket.disconnected.connect(socket.deleteLater)

    def _on_ready_read(self) -> None:
        socket = self.sender()
        if not isinstance(socket, QLocalSocket):
            return
        buffer = bytes(socket.property("negpy_buffer") or b"") + bytes(socket.readAll().data())
        socket.setProperty("negpy_buffer", buffer)
        if len(buffer) < 4 or len(buffer) < 4 + int.from_bytes(buffer[:4], "big"):
            return
        request = decode_request(buffer[4 : 4 + int.from_bytes(buffer[:4], "big")])
        socket.readyRead.disconnect(self._on_ready_read)
        socket.write(_ACK)
        if request is not None:
            self.received.emit(request)
