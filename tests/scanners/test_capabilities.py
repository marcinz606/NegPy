"""Tests for SANE source name normalization to ScanMode."""

from negpy.infrastructure.scanners.params import ScanMode
from negpy.infrastructure.scanners.sane_backend import _SOURCE_MAP


def _normalize(source: str) -> ScanMode | None:
    s_stripped = source.strip().lower()
    if "(" in s_stripped:
        s_base = s_stripped.split("(")[0].strip()
    else:
        s_base = s_stripped
    return _SOURCE_MAP.get(s_base)


class TestSourceMap:
    # Plustek sources
    def test_plustek_negative(self) -> None:
        assert _normalize("Negative") == ScanMode.NEGATIVE

    def test_plustek_positive(self) -> None:
        assert _normalize("Positive") == ScanMode.POSITIVE

    def test_plustek_transparency(self) -> None:
        assert _normalize("Transparency") == ScanMode.TRANSPARENCY

    # Epson sources
    def test_epson_transparency_unit(self) -> None:
        assert _normalize("Transparency Unit") == ScanMode.TRANSPARENCY

    def test_epson_tpu(self) -> None:
        assert _normalize("TPU") == ScanMode.TRANSPARENCY

    def test_epson_film(self) -> None:
        assert _normalize("Film") == ScanMode.TRANSPARENCY

    def test_epson_negative_film(self) -> None:
        assert _normalize("Negative Film") == ScanMode.NEGATIVE

    def test_epson_positive_film(self) -> None:
        assert _normalize("Positive Film") == ScanMode.POSITIVE

    def test_epson_slide(self) -> None:
        assert _normalize("Slide") == ScanMode.POSITIVE

    # Canon sources
    def test_canon_film(self) -> None:
        assert _normalize("Film") == ScanMode.TRANSPARENCY

    def test_canon_negative(self) -> None:
        assert _normalize("Negative") == ScanMode.NEGATIVE

    def test_canon_slide(self) -> None:
        assert _normalize("Slide") == ScanMode.POSITIVE

    # Case insensitivity
    def test_case_insensitive(self) -> None:
        assert _normalize("negative") == ScanMode.NEGATIVE
        assert _normalize("NEGATIVE") == ScanMode.NEGATIVE
        assert _normalize("nEgAtIvE") == ScanMode.NEGATIVE

    # Strips whitespace
    def test_strips_whitespace(self) -> None:
        assert _normalize("  Negative  ") == ScanMode.NEGATIVE

    # Unknown sources excluded
    def test_unknown_excluded(self) -> None:
        assert _normalize("Flatbed") is None
        assert _normalize("Reflective") is None
        assert _normalize("ADF") is None
        assert _normalize("Color") is None
        assert _normalize("Gray") is None

    # Sources with parentheticals (IR variants etc.)
    def test_parenthetical_stripped(self) -> None:
        assert _normalize("Transparency (IR)") == ScanMode.TRANSPARENCY
        assert _normalize("Negative (Color)") == ScanMode.NEGATIVE
