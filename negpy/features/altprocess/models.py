from dataclasses import dataclass
from enum import StrEnum


class AltProcess(StrEnum):
    NONE = "none"
    LITH = "lith"
    CYANOTYPE = "cyanotype"
    SABATTIER = "sabattier"


class Sensitizer(StrEnum):
    """Cyanotype sensitizer. Classic = Herschel's ammonium ferric citrate,
    New = Ware's ammonium ferric oxalate."""

    CLASSIC = "classic"
    NEW = "new"


@dataclass(frozen=True)
class AltProcessConfig:
    """
    The Alternative Processes panel. One config for every process because they
    are mutually exclusive — you cannot lith-develop a cyanotype — so the state
    is one enum rather than booleans that could all be set.

    Lith takes its color from the Exposure panel's paper profile; cyanotype is
    on rag paper and takes its color from the sensitizer; a Sabattier print is
    neutral silver.
    """

    alt_process: AltProcess = AltProcess.NONE
    lith_exposure: float = 2.0
    lith_snatch: float = 0.55
    lith_abruptness: float = 0.6
    cyano_sensitizer: Sensitizer = Sensitizer.CLASSIC
    cyano_exposure: float = 0.0
    cyano_scale: float = 1.4
    cyano_bleach: float = 0.0
    cyano_tannin: float = 0.0
    # Sabattier: how far the light tones reverse (1 flattens them to the fold, above 1 they
    # reverse), where the fold sits (a fraction of the paper's Dmax) and the tray's agitation
    # after the flash (0 still, wide Mackie lines; 1 constant, none).
    sabattier_strength: float = 1.3
    sabattier_reexposure: float = 0.45
    sabattier_agitation: float = 0.7
