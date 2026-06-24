"""Tests for SANE source name normalization to ScanMode."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from negpy.infrastructure.scanners.params import ScanMode
from negpy.infrastructure.scanners.sane_backend import (
    _SOURCE_MAP,
    _caps_from_options,
    _split_rgbi,
)


@dataclass
class FakeOption:
    """Stand-in for python-sane's Option (only the fields _caps_from_options reads)."""

    constraint: Any = None
    desc: str = ""


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


def _pieusb_opt() -> dict[str, FakeOption]:
    """Real option map from a Reflecta ProScan 7200 / Pacific Image (issues #293, #262).

    Keyed by py_name (hyphens → underscores), as python-sane exposes dev.opt. Note: no
    `source` option, RGBI mode, depth includes 1-bit lineart, resolution is a range.
    """
    return {
        "mode": FakeOption(constraint=["Lineart", "Halftone", "Gray", "Color", "RGBI"]),
        "depth": FakeOption(constraint=[1, 8, 16]),
        "resolution": FakeOption(constraint=(25.0, 3600.0, 1.0)),
        "br_x": FakeOption(constraint=(0.0, 37.676666259765625, 0.0)),
        "br_y": FakeOption(constraint=(0.0, 24.299331665039062, 0.0)),
        "clean_image": FakeOption(desc="Detect and remove dust and scratch artifacts"),
        "correct_infrared": FakeOption(desc="Correct infrared for red crosstalk"),
        "invert": FakeOption(desc="Correct for generic negative film"),
    }


class TestCapsFromOptions:
    # ── pieusb dedicated film scanners (issues #293, #262) ──────────────

    def test_pieusb_detected_as_film(self) -> None:
        caps = _caps_from_options(_pieusb_opt(), "pieusb:libusb:001:011")
        assert caps.sources  # non-empty → no longer skipped
        assert ScanMode.NEGATIVE in caps.sources

    def test_pieusb_ir_from_rgbi_mode(self) -> None:
        caps = _caps_from_options(_pieusb_opt(), "pieusb:libusb:001:011")
        assert caps.ir_channel is True

    def test_pieusb_lineart_depth_dropped(self) -> None:
        caps = _caps_from_options(_pieusb_opt(), "pieusb:libusb:001:011")
        assert caps.supported_depths == (8, 16)

    def test_pieusb_resolution_range_intersected(self) -> None:
        # Range (25, 3600) must intersect canonical stops, not be read as three values.
        caps = _caps_from_options(_pieusb_opt(), "pieusb:libusb:001:011")
        assert caps.supported_dpi == (75, 150, 300, 600, 1200, 2400, 3600)

    def test_pieusb_max_area_from_geometry(self) -> None:
        caps = _caps_from_options(_pieusb_opt(), "pieusb:libusb:001:011")
        assert caps.max_area_mm[0] == 37.676666259765625
        assert caps.max_area_mm[1] == 24.299331665039062

    def test_film_inferred_without_pieusb_id(self) -> None:
        # RGBI / negative-film signals alone classify it as film (id-agnostic).
        caps = _caps_from_options(_pieusb_opt(), "othervendor:libusb:001:001")
        assert caps.sources

    # ── plain flatbed: no source, no film signals → still skipped ───────

    def test_flatbed_without_source_skipped(self) -> None:
        opt = {
            "mode": FakeOption(constraint=["Color", "Gray", "Lineart"]),
            "depth": FakeOption(constraint=[8, 16]),
            "resolution": FakeOption(constraint=[75, 150, 300, 600]),
            "invert": FakeOption(desc="Invert image"),  # generic, not negative-film
        }
        caps = _caps_from_options(opt, "genesys:libusb:001:002")
        assert caps.sources == ()
        assert caps.ir_channel is False

    # ── explicit source path (Plustek) unchanged ───────────────────────

    def test_plustek_explicit_sources(self) -> None:
        opt = {
            "source": FakeOption(constraint=["Negative", "Positive", "Transparency"]),
            "resolution": FakeOption(constraint=[300, 600, 1200, 2400, 3600]),
            "depth": FakeOption(constraint=[8, 16]),
        }
        caps = _caps_from_options(opt, "plustek:libusb:001:008")
        assert caps.sources == (ScanMode.NEGATIVE, ScanMode.POSITIVE, ScanMode.TRANSPARENCY)
        assert caps.supported_dpi == (300, 600, 1200, 2400, 3600)

    def test_ir_from_dedicated_option(self) -> None:
        opt = {
            "source": FakeOption(constraint=["Transparency"]),
            "ir": FakeOption(),
        }
        caps = _caps_from_options(opt, "plustek:libusb:001:008")
        assert caps.ir_channel is True


class TestSplitRgbi:
    def test_splits_four_channels(self) -> None:
        arr = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
        rgb, ir = _split_rgbi(arr)
        assert rgb.shape == (2, 3, 3)
        assert ir.shape == (2, 3)
        assert np.array_equal(rgb, arr[:, :, :3])
        assert np.array_equal(ir, arr[:, :, 3])
