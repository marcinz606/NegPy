"""Persisted Scanlight-capture panel settings (stored as a global setting dict)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScanlightSettings:
    """Sticky settings for the Scanlight capture sidebar.

    Persisted via the session repo under the `scanlight_settings` key, mirroring
    `ScannerSettings`. `port` empty = auto-discover the Scanlight serial port; the
    camera carries no settings at all, libgphoto2 finds it on the USB bus.
    """

    r_level: int = 255
    g_level: int = 255
    b_level: int = 255
    shutter_r: str = ""
    shutter_g: str = ""
    shutter_b: str = ""
    white_mode: bool = False
    w_level: int = 255
    shutter_w: str = ""
    white_process_mode: str = "auto"
    roll_name: str = "Roll001"
    output_folder: str = ""
    port: str = ""  # Scanlight serial port ("" = autodetect); the camera needs no address

    @classmethod
    def defaults(cls) -> "ScanlightSettings":
        return cls()
