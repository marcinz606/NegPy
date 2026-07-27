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
    w_level: int = 0  # RGB scanning uses no white; a white-light preset raises it to 255
    shutter_w: str = ""
    iso: str = ""  # RGB preset's calibrated ISO/aperture, forced on the body at scan time
    aperture: str = ""  # "" for a manual-aperture lens (set by hand on the ring)
    white_process_mode: str = "auto"
    roll_name: str = "Roll001"
    output_folder: str = ""
    port: str = ""  # Scanlight serial port ("" = autodetect); the camera needs no address
    # How the shutter is triggered. "usb" = libgphoto2 tethered capture (the default).
    # "fuji_ble" = fire the shutter over Bluetooth and pick the RAW up out of the folder the
    # camera's WiFi auto-save writes to — for bodies (Fujifilm GFX) that stick in a gphoto2
    # tethered-capture state. The pairing is stored flat below; the drop folder is watched.
    camera_backend: str = "usb"
    fuji_drop_folder: str = ""
    fuji_address: str = ""
    fuji_name: str = ""
    fuji_flavor: str = ""  # "basic" | "secure"
    fuji_token: str = ""  # hex, Basic pairing only
    fuji_serial: str = ""  # hex, Secure pairing only

    @classmethod
    def defaults(cls) -> "ScanlightSettings":
        return cls()

    @property
    def fuji_paired(self) -> bool:
        """A Fujifilm BLE pairing is stored (enough to connect and fire)."""
        return bool(self.fuji_address and self.fuji_flavor)
