"""A Lento Dock serial port stand-in: prints BOOT on open, answers the continuous-light commands
and moves a strip of `frames` frames."""

import queue

_IDENT = "IDENT serial=035SIM/0001/26 model=35 hwrev=v1.0"
_VER = "VER fw=1.0.224 model=35"


class SimDockSerial:
    port = "sim-dock"

    def __init__(self, *, saved_mode: str = "AUTO", saved_expmode: str = "BRGB", frames: int = 6) -> None:
        self.is_open = True
        self.lines: list[str] = []  # every command received, in order
        self.mode = saved_mode
        self.expmode = saved_expmode
        self.idle = "AUTO"
        self.hold = 0
        self.duty = (0, 0, 0)
        self.frames = frames  # frames on the strip; the first one is in the window
        self.frame = 0  # the frame in the window, counted from the first
        self.st = "READY" if frames else "IDLE"
        self._out: queue.Queue[bytes] = queue.Queue()
        self._buf = b""
        self._say("BOOT fw=1.0.224 model=35 reason=nrst film=0 entrance=0 cal=persisted")
        self._say(self._state())

    @property
    def lit(self) -> tuple[int, int, int]:
        """The duty the panel shows: continuous mode, idle light on, held."""
        if self.expmode == "CONT" and self.idle != "OFF" and self.hold:
            return self.duty
        return (0, 0, 0)

    def _say(self, line: str) -> None:
        self._out.put((line + "\r\n").encode("ascii"))

    @property
    def film(self) -> int:
        """The film sensor, which reads no film while the last frame is still in the window."""
        return int(self.frames > 0 and self.frame < self.frames - 1)

    def _state(self) -> str:
        return (
            f"STATE link=1 seq=1 mode={self.mode} st={self.st} film={self.film} frame={self.frame - 1}/{self.frame} "
            f"busy=0 expmode={self.expmode} shutter=JACK"
        )

    def press_scan(self) -> None:
        """The S button in host mode: the dock makes its own exposure, then reports it."""
        self.st = "EXPOSE"
        self._say(self._state())
        self._say("SHOT done ms=3000")
        self.st = "READY"
        self._say(self._state())

    def _frame_next(self) -> None:
        if self.mode != "SEMI" or self.frames == 0:
            self._say("ERR STATE")
            return
        self._say("OK")
        self.st = "SEEK"
        self._say(self._state())
        if self.frame >= self.frames - 1:
            self.frames = self.frame = 0
            self.st = "IDLE"
            self._say("END_OF_ROLL")
        else:
            self.frame += 1
            self.st = "READY"
            self._say(f"FRAMEAT idx={self.frame - 1} count={self.frame}")
        self._say(self._state())

    def write(self, data: bytes) -> None:
        self._buf += data
        while b"\n" in self._buf:
            raw, self._buf = self._buf.split(b"\n", 1)
            line = raw.decode("ascii").strip()
            if line:
                self.lines.append(line)
                self._answer(line)

    def _answer(self, line: str) -> None:
        cmd, _, arg = line.partition(" ")
        if cmd == "IDENT":
            self._say(_IDENT)
        elif cmd == "VER":
            self._say(_VER)
        elif cmd in ("STATE?", "PING"):
            self._say(self._state())
        elif cmd == "MODE" and arg in ("SEMI", "AUTO", "CONFIG"):
            self.mode = arg
            self._say(f"MODE={arg}")
        elif cmd == "EXPMODE" and arg in ("FIX", "BWHITE", "BRGB", "SEQ", "BRGBPF", "CONT"):
            self.expmode = arg
            self._say(f"EXPMODE={arg}")
        elif cmd == "IDLE" and arg:
            self.idle = arg
            self._say(f"IDLE={arg}")
        elif cmd == "CONTHOLD" and arg in ("0", "1"):
            self.hold = int(arg)
            self._say(f"CONTHOLD {arg}")
        elif line == "FRAME NEXT":
            self._frame_next()
        elif line == "EJECT":
            self._say("OK")
            self.frames = self.frame = 0
            self.st = "IDLE"
            self._say("EJECTED")
            self._say(self._state())
        elif cmd == "CONTRGB":
            parts = arg.split()
            if len(parts) == 3 and all(p.isdigit() and int(p) <= 1000 for p in parts):
                self.duty = (int(parts[0]), int(parts[1]), int(parts[2]))
                self._say(f"CONTRGB {arg}")
            else:
                self._say("ERR")
        else:
            self._say("ERR ?")

    def read(self, _n: int) -> bytes:
        try:
            return self._out.get(timeout=0.02)
        except queue.Empty:
            return b""

    def close(self) -> None:
        self.is_open = False
