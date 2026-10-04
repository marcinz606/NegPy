"""Stand-ins for the camera, the Scanlight and film scanners; `make run-sim` sets the flag.

Each replaces only the lowest layer, so the real drivers, workers and panels run on top of it.
"""

import os


def enabled() -> bool:
    return os.environ.get("NEGPY_SIMULATE_HARDWARE") == "1"
