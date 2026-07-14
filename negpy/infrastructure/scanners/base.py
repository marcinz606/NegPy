import threading
from dataclasses import dataclass
from typing import Callable, Protocol

from negpy.infrastructure.scanners.params import ScanMode, ScanParams
from negpy.infrastructure.scanners.result import ScanResult


@dataclass(frozen=True)
class ScannerCapabilities:
    ir_channel: bool
    supported_dpi: tuple[int, ...]
    supported_depths: tuple[int, ...]
    sources: tuple[ScanMode, ...]
    max_area_mm: tuple[float, float]  # (width, height)
    # Device exposes a samples-per-scan option (hardware multi-sampling, e.g.
    # Nikon Coolscan). UI-gating only — SaneBackend.scan() fails loud on its
    # own if samples_per_scan > 1 is requested against a device without it.
    multi_sample: bool = False
    # Maximum transport position advertised by a roll adapter. This is a
    # mechanical capacity bound (e.g. SA-30 reports 40), never an inferred
    # count of exposures on the inserted strip or roll.
    adapter_frame_capacity: int | None = None
    # Device exposes hardware auto-exposure metering (SANE `ae` option).
    # UI-gating only — SaneBackend.scan() fails loud on its own if
    # auto_exposure is requested against a device without it.
    auto_exposure: bool = False
    # Device exposes both registration-window options (`subframe` + `br_y`)
    # needed to position a ScanParams.registered_geometry request. UI-gating
    # only — SaneBackend.scan() fails loud on its own if registered_geometry
    # is requested against a device missing either option.
    registered_geometry: bool = False
    # Device exposes an active, settable medium-eject action. The UI uses this
    # only to decide whether to show an Eject button; SaneBackend.eject()
    # repeats the capability check before touching the transport.
    can_eject: bool = False
    # Device exposes a frame-position control even if its current capacity is
    # unavailable. Coolscan feeders report a 1..0 range while parked, so this
    # preserves the distinction between a parked transport and no transport
    # control at all without inventing an adapter capacity. Appended to keep
    # the positional order of older capability fields backward compatible.
    adapter_frame_control: bool = False


@dataclass(frozen=True)
class ScannerDevice:
    id: str  # SANE device name e.g. "plustek:libusb:001:008"
    vendor: str
    model: str
    capabilities: ScannerCapabilities


class ScannerBackend(Protocol):
    def list_devices(self) -> list[ScannerDevice]: ...
    def scan(
        self,
        device_id: str,
        params: ScanParams,
        progress: Callable[[float], None],
        cancel: threading.Event,
    ) -> ScanResult: ...
