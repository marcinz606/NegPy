"""The light source the Scan panel drives: a Scanlight, a CuteNew RGB AIR or a Lento Dock.

The drivers share the Scanlight's interface plus `describe()`. An RGB AIR is found by its USB
IDs, before the Scanlight's discovery, which would take its macOS `usbmodem` port. The dock is
looked for only when no Scanlight is found, because probing a port resets the device on it.
"""

from __future__ import annotations

import os
from typing import Optional, Union

from negpy.infrastructure import simulated
from negpy.infrastructure.capture import lento_dock, rgbair, scanlight
from negpy.infrastructure.capture.lento_dock import LentoDock
from negpy.infrastructure.capture.rgbair import RgbAir
from negpy.infrastructure.capture.scanlight import Scanlight

Light = Union[Scanlight, RgbAir, LentoDock]


def open_light(port: Optional[str] = None) -> Light:
    """Connect to the light on `port`, or find one. Raises the Scanlight's discovery error
    when no light is found."""
    if simulated.enabled():
        if os.environ.get("NEGPY_SIMULATE_LIGHT") == "lento":
            from negpy.infrastructure.simulated.lento_dock import SimDockSerial

            return LentoDock(serial_obj=SimDockSerial())
        if os.environ.get("NEGPY_SIMULATE_LIGHT") == "rgbair":
            from negpy.infrastructure.simulated.rgbair import SimRgbAirSerial

            return RgbAir(serial_obj=SimRgbAirSerial())
        return Scanlight()
    if port:
        if rgbair.is_rgbair_port(port):
            return RgbAir(port)
        return lento_dock.connect(port) if lento_dock.is_dock_port(port) else Scanlight(port)
    air = rgbair.find_port()
    if air:
        return RgbAir(air)
    try:
        found = scanlight.discover_port()
    except RuntimeError as no_scanlight:
        try:
            return lento_dock.connect()
        except RuntimeError:
            raise no_scanlight from None
    return Scanlight(found)
