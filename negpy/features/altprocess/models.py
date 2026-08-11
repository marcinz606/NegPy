from dataclasses import dataclass
from enum import StrEnum


class AltProcess(StrEnum):
    NONE = "none"
    LITH = "lith"
    CYANOTYPE = "cyanotype"


class Sensitizer(StrEnum):
    """Cyanotype sensitiser. Classic = Herschel's ammonium ferric citrate,
    New = Ware's ammonium ferric oxalate."""

    CLASSIC = "classic"
    NEW = "new"


@dataclass(frozen=True)
class AltProcessConfig:
    """
    The Alternative Processes panel. One config for both processes because they
    are mutually exclusive — you cannot lith-develop a cyanotype — so the state
    is one enum rather than two booleans that could both be set.

    Lith takes its colour from the Exposure panel's paper profile; cyanotype is
    on rag paper and takes its colour from the sensitiser.
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
