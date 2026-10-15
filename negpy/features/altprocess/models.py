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
    """The Alternative Processes panel. The processes are mutually exclusive, so the state
    is one enum. Lith takes its color from the paper profile, cyanotype from the sensitizer;
    a Sabattier print is neutral silver."""

    alt_process: AltProcess = AltProcess.NONE
    lith_exposure: float = 2.0
    lith_snatch: float = 0.55
    lith_abruptness: float = 0.6
    cyano_sensitizer: Sensitizer = Sensitizer.CLASSIC
    cyano_exposure: float = 0.0
    cyano_scale: float = 1.4
    cyano_bleach: float = 0.0
    cyano_tannin: float = 0.0
    # Sabattier: strength 1 flattens the light tones to the fold, above 1 reverses them; the fold
    # sits at reexposure x paper Dmax; agitation 0 draws wide Mackie lines, 1 none.
    sabattier_strength: float = 1.3
    sabattier_reexposure: float = 0.45
    sabattier_agitation: float = 0.7
