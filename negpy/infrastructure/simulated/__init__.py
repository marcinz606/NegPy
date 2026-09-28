"""Stand-ins for the camera, the Scanlight and film scanners, for development without hardware.

Each fake replaces the lowest layer only (the gphoto2 module, the serial port, a scanner
backend), so the real drivers, workers and panels run on top of it. `make run-sim` sets the flag.
"""

import os


def enabled() -> bool:
    return os.environ.get("NEGPY_SIMULATE_HARDWARE") == "1"
