from dataclasses import dataclass


@dataclass(frozen=True)
class FilmMetadataConfig:
    """
    Analog-specific recording data.
    """

    film_stock: str = ""
    iso: int = 400
    developer: str = ""
    dilution: str = ""
    scan_hardware: str = ""
    notes: str = ""
