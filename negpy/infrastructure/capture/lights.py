"""The light source the Scan panel drives: a Scanlight, or a Lento Dock.

Both drivers share the Scanlight's interface plus `describe()`. A Scanlight on the bus is
used first; the dock is looked for only when no Scanlight is found, so a rig with both uses
the Scanlight.
"""

from __future__ import annotations

import os
from typing import Optional, Union

from negpy.infrastructure import simulated
from negpy.infrastructure.capture import lento_dock, scanlight
from negpy.infrastructure.capture.lento_dock import LentoDock
from negpy.infrastructure.capture.scanlight import Scanlight

Light = Union[Scanlight, LentoDock]


def open_light(port: Optional[str] = None) -> Light:
    """Connect to the light on `port`, or find one. Raises the Scanlight's discovery error
    when neither light is found."""
    if simulated.enabled():
        if os.environ.get("NEGPY_SIMULATE_LIGHT") == "lento":
            from negpy.infrastructure.simulated.lento_dock import SimDockSerial

            return LentoDock(serial_obj=SimDockSerial())
        return Scanlight()
    if port:
        return lento_dock.connect(port) if lento_dock.is_dock_port(port) else Scanlight(port)
    try:
        found = scanlight.discover_port()
    except RuntimeError as no_scanlight:
        try:
            return lento_dock.connect()
        except RuntimeError:
            raise no_scanlight from None
    return Scanlight(found)
