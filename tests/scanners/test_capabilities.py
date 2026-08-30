"""Tests for SANE source name normalization to ScanMode and capability detection."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from negpy.infrastructure.scanners.params import ScanMode
from negpy.infrastructure.scanners.sane_backend import (
    _SOURCE_MAP,
    _caps_from_options,
    _detect_auto_exposure,
    _split_rgbi,
)


@dataclass
class FakeOption:
    """Stand-in for python-sane's Option (only the fields _caps_from_options reads)."""

    constraint: Any = None
    desc: str = ""
    unit: Any = None


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
        "br_x": FakeOption(constraint=(0.0, 37.676666259765625, 0.0), unit=3),
        "br_y": FakeOption(constraint=(0.0, 24.299331665039062, 0.0), unit=3),
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

    def test_coolscan_pixel_geometry_is_converted_to_millimeters(self) -> None:
        opt = {
            "resolution": FakeOption(constraint=[1000, 2000, 4000]),
            "br_x": FakeOption(constraint=(0, 3945, 1), unit=1),
            "br_y": FakeOption(constraint=(0, 5958, 1), unit=1),
        }

        caps = _caps_from_options(opt, "coolscan3:usb:libusb:001:007")

        np.testing.assert_allclose(caps.max_area_mm, (25.0571, 37.83965))

    def test_pixel_geometry_without_dpi_uses_35mm_fallback(self) -> None:
        opt = {
            "br_x": FakeOption(constraint=(0, 3945, 1), unit=1),
            "br_y": FakeOption(constraint=(0, 5958, 1), unit=1),
        }

        caps = _caps_from_options(opt, "coolscan3:usb:libusb:001:007")

        assert caps.max_area_mm == (36.0, 25.0)

    def test_film_inferred_without_pieusb_id(self) -> None:
        # RGBI / negative-film signals alone classify it as film (id-agnostic).
        caps = _caps_from_options(_pieusb_opt(), "othervendor:libusb:001:001")
        assert caps.sources

    def test_roll_adapter_frame_range_is_reported_as_capacity(self) -> None:
        caps = _caps_from_options(
            {
                "frame": FakeOption(constraint=(1, 40, 1)),
                "infrared": FakeOption(),
            },
            "coolscan3:usb:libusb:001:007",
        )

        assert caps.adapter_frame_capacity == 40

    def test_parked_adapter_keeps_frame_control_without_inventing_capacity(self) -> None:
        caps = _caps_from_options(
            {
                "frame": FakeOption(constraint=(1, 0, 1)),
                "infrared": FakeOption(),
            },
            "coolscan3:usb:libusb:001:007",
        )

        assert caps.adapter_frame_control is True
        assert caps.adapter_frame_capacity is None

    def test_usable_eject_option_is_reported_to_the_ui(self) -> None:
        caps = _caps_from_options(
            {
                "frame": FakeOption(constraint=(1, 40, 1)),
                "eject": FakeOption(),
            },
            "coolscan3:usb:libusb:001:007",
        )

        assert caps.can_eject is True

    def test_missing_eject_option_defaults_false(self) -> None:
        caps = _caps_from_options(
            {"source": FakeOption(constraint=["Negative"])},
            "plustek:libusb:001:008",
        )

        assert caps.can_eject is False

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
        assert caps.adapter_frame_capacity is None

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

    def test_ir_from_source_name(self) -> None:
        # genesys (e.g. Epson flatbeds with a transparency adapter) exposes IR
        # only as a second `source` value, not a dedicated option or RGBI mode.
        opt = {
            "mode": FakeOption(constraint=["Color", "Gray"]),
            "source": FakeOption(constraint=["Transparency Adapter", "Transparency Adapter Infrared"]),
        }
        caps = _caps_from_options(opt, "genesys:libusb:001:003")
        assert caps.ir_channel is True


class TestAutoExposureCapability:
    """Hardware auto-exposure is a presence-only UI gate, mirroring _detect_ir."""

    def test_auto_exposure_true_with_ae_option(self) -> None:
        assert _detect_auto_exposure({"ae": FakeOption()}) is True

    def test_auto_exposure_false_without_ae_option(self) -> None:
        assert _detect_auto_exposure({"infrared": FakeOption()}) is False

    def test_caps_from_options_wires_auto_exposure(self) -> None:
        caps = _caps_from_options(
            {
                "frame": FakeOption(constraint=(1, 40, 1)),
                "infrared": FakeOption(),
                "ae": FakeOption(),
            },
            "coolscan3:usb:libusb:001:007",
        )
        assert caps.auto_exposure is True

    def test_caps_from_options_defaults_auto_exposure_false(self) -> None:
        caps = _caps_from_options(
            {
                "source": FakeOption(constraint=["Negative", "Positive", "Transparency"]),
                "resolution": FakeOption(constraint=[300, 600, 1200, 2400, 3600]),
                "depth": FakeOption(constraint=[8, 16]),
            },
            "plustek:libusb:001:008",
        )
        assert caps.auto_exposure is False


class TestAutofocusCapability:
    def test_caps_from_options_wires_autofocus(self) -> None:
        caps = _caps_from_options(
            {
                "frame": FakeOption(constraint=(1, 40, 1)),
                "autofocus": FakeOption(),
            },
            "coolscan3:usb:libusb:001:007",
        )
        assert caps.autofocus is True

    def test_caps_from_options_defaults_autofocus_false(self) -> None:
        caps = _caps_from_options(
            {
                "source": FakeOption(constraint=["Negative", "Positive", "Transparency"]),
                "resolution": FakeOption(constraint=[300, 600, 1200, 2400, 3600]),
                "depth": FakeOption(constraint=[8, 16]),
            },
            "plustek:libusb:001:008",
        )
        assert caps.autofocus is False


class TestSplitRgbi:
    def test_splits_four_channels(self) -> None:
        arr = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
        rgb, ir = _split_rgbi(arr)
        assert rgb.shape == (2, 3, 3)
        assert ir.shape == (2, 3)
        assert np.array_equal(rgb, arr[:, :, :3])
        assert np.array_equal(ir, arr[:, :, 3])


class TestResampleRowsToDpi:
    """`_resample_rows_to_dpi`: upsample only the row (y) axis to represent a higher DPI,
    leaving columns (x) untouched — the mechanism behind _resolve_resampled_resolutions."""

    def test_upsamples_rows_by_the_dpi_ratio(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _resample_rows_to_dpi

        arr = np.zeros((4, 10, 3), dtype=np.uint8)
        out = _resample_rows_to_dpi(arr, native_dpi=2400, target_dpi=3200)

        assert out.shape[0] == round(4 * 3200 / 2400)  # 5
        assert out.shape[1] == 10  # columns untouched
        assert out.shape[2] == 3

    def test_preserves_dtype(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _resample_rows_to_dpi

        arr = np.full((4, 4, 3), 1000, dtype=np.uint16)
        out = _resample_rows_to_dpi(arr, native_dpi=2400, target_dpi=3200)

        assert out.dtype == np.uint16

    def test_works_on_2d_grayscale(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _resample_rows_to_dpi

        arr = np.zeros((4, 10), dtype=np.uint8)
        out = _resample_rows_to_dpi(arr, native_dpi=2400, target_dpi=3200)

        assert out.shape == (5, 10)

    def test_exact_multiple_matches_expected_row_count(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _resample_rows_to_dpi

        arr = np.zeros((100, 10, 3), dtype=np.uint8)
        out = _resample_rows_to_dpi(arr, native_dpi=1600, target_dpi=3200)

        assert out.shape[0] == 200


class TestScanExposureTimeCapability:
    """Detection of the SANE `scan-exposure-time` option (e.g. genesys)."""

    def test_range_tuple_detected(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _detect_scan_exposure_time

        opt = {"scan_exposure_time": FakeOption(constraint=(11000, 65535, 1))}
        assert _detect_scan_exposure_time(opt) == (11000, 65535)

    def test_hyphenated_key_detected(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _detect_scan_exposure_time

        opt = {"scan-exposure-time": FakeOption(constraint=(11000, 65535, 1))}
        assert _detect_scan_exposure_time(opt) == (11000, 65535)

    def test_absent_returns_none(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _detect_scan_exposure_time

        assert _detect_scan_exposure_time({"source": FakeOption()}) is None

    def test_caps_from_options_wires_exposure_time(self) -> None:
        caps = _caps_from_options(
            {
                "source": FakeOption(constraint=["Negative", "Positive", "Transparency"]),
                "resolution": FakeOption(constraint=[300, 600, 1200, 2400, 3600]),
                "depth": FakeOption(constraint=[8, 16]),
                "scan_exposure_time": FakeOption(constraint=(11000, 65535, 1)),
            },
            "genesys:libusb:003:005",
        )
        assert caps.exposure_time_us == (11000, 65535)

    def test_caps_from_options_defaults_exposure_time_none(self) -> None:
        caps = _caps_from_options(
            {
                "source": FakeOption(constraint=["Negative", "Positive", "Transparency"]),
                "resolution": FakeOption(constraint=[300, 600, 1200, 2400, 3600]),
                "depth": FakeOption(constraint=[8, 16]),
            },
            "plustek:libusb:001:008",
        )
        assert caps.exposure_time_us is None


class TestResolveTransparencySource:
    """`_resolve_transparency_source`: which `source` constraint value to switch to before a
    scan, on a flatbed+TPU device that defaults to reflective flatbed."""

    def test_epson_style_returns_transparency_unit(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _resolve_transparency_source

        opt = {"source": FakeOption(constraint=["Flatbed", "Transparency Unit"])}
        assert _resolve_transparency_source(opt) == "Transparency Unit"

    def test_no_transparency_choice_returns_none(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _resolve_transparency_source

        opt = {"source": FakeOption(constraint=["Flatbed", "ADF"])}
        assert _resolve_transparency_source(opt) is None

    def test_no_source_option_returns_none(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _resolve_transparency_source

        assert _resolve_transparency_source({}) is None

    def test_non_list_constraint_returns_none(self) -> None:
        """Range-typed `source` would be unusual, but must not raise."""
        from negpy.infrastructure.scanners.sane_backend import _resolve_transparency_source

        opt = {"source": FakeOption(constraint=(0, 1, 1))}
        assert _resolve_transparency_source(opt) is None


class TestResolveFilmType:
    """`_resolve_film_type`: which SANE `film_type` value matches NegPy's own
    negative/positive concept (film_reads_positive), for devices that separate the two."""

    def test_negative_stock_matches_negative_film(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _resolve_film_type

        opt = {"film_type": FakeOption(constraint=["Positive Film", "Negative Film"])}
        assert _resolve_film_type(opt, "negative") == "Negative Film"

    def test_reversal_stock_matches_positive_film(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _resolve_film_type

        opt = {"film_type": FakeOption(constraint=["Positive Film", "Negative Film"])}
        assert _resolve_film_type(opt, "positive") == "Positive Film"
        assert _resolve_film_type(opt, "kodachrome") == "Positive Film"

    def test_mono_negative_matches_negative_film(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _resolve_film_type

        opt = {"film_type": FakeOption(constraint=["Positive Film", "Negative Film"])}
        assert _resolve_film_type(opt, "mono") == "Negative Film"

    def test_no_film_type_option_returns_none(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _resolve_film_type

        assert _resolve_film_type({}, "negative") is None


# Real values from `scanimage --help` on the project's reference Epson V500
# (epkowa/interpreter backend): `--resolution` and `--x-resolution`/`--y-resolution` report
# genuinely different, asymmetric native ladders.
_V500_X_RESOLUTIONS = [100, 200, 400, 600, 800, 1200, 1600, 3200, 6400]
_V500_Y_RESOLUTIONS = [80, 200, 320, 400, 600, 800, 1200, 1600, 2400, 3200, 4800, 6400]
_V500_PLAIN_RESOLUTIONS = [200, 400, 800, 1600]


class TestResolveSquareResolutions:
    """`_resolve_square_resolutions`: DPI values settable identically on both axes, so pixels
    stay square, when a device exposes `x_resolution`/`y_resolution` independently."""

    def test_intersects_asymmetric_axis_ladders(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _resolve_square_resolutions

        opt = {
            "x_resolution": FakeOption(constraint=_V500_X_RESOLUTIONS),
            "y_resolution": FakeOption(constraint=_V500_Y_RESOLUTIONS),
        }
        assert _resolve_square_resolutions(opt) == (200, 400, 600, 800, 1200, 1600, 3200, 6400)

    def test_missing_either_axis_returns_empty(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _resolve_square_resolutions

        assert _resolve_square_resolutions({"x_resolution": FakeOption(constraint=[100, 200])}) == ()
        assert _resolve_square_resolutions({"y_resolution": FakeOption(constraint=[100, 200])}) == ()
        assert _resolve_square_resolutions({}) == ()

    def test_non_list_constraint_returns_empty(self) -> None:
        """A plain (min, max, step) range would be unusual for this option, but must not raise."""
        from negpy.infrastructure.scanners.sane_backend import _resolve_square_resolutions

        opt = {
            "x_resolution": FakeOption(constraint=(50, 6400, 1)),
            "y_resolution": FakeOption(constraint=[100, 200]),
        }
        assert _resolve_square_resolutions(opt) == ()

    def test_no_overlap_returns_empty(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _resolve_square_resolutions

        opt = {
            "x_resolution": FakeOption(constraint=[100, 300]),
            "y_resolution": FakeOption(constraint=[200, 400]),
        }
        assert _resolve_square_resolutions(opt) == ()


class TestDetectDpiPrefersRicherAxisLadder:
    """_detect_dpi: prefer the x/y-derived ladder over `resolution` alone whenever it reaches
    higher — this is the actual bug found on the V500, where `resolution` alone caps at
    1600dpi despite the device supporting 6400dpi per axis."""

    def test_v500_reports_up_to_6400_not_1600(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _detect_dpi

        opt = {
            "resolution": FakeOption(constraint=_V500_PLAIN_RESOLUTIONS),
            "x_resolution": FakeOption(constraint=_V500_X_RESOLUTIONS),
            "y_resolution": FakeOption(constraint=_V500_Y_RESOLUTIONS),
        }
        dpis = _detect_dpi(opt)
        assert max(dpis) == 6400
        # 100 is reachable too: x has a native 100 step, and y's minimum (80) is below it, so
        # it's an upsample away — see _resolve_resampled_resolutions.
        assert dpis == (100, 200, 400, 600, 800, 1200, 1600, 3200, 6400)

    def test_falls_back_to_plain_resolution_when_axis_ladder_is_not_richer(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _detect_dpi

        opt = {
            "resolution": FakeOption(constraint=[100, 200, 300, 400, 600, 1200]),
            "x_resolution": FakeOption(constraint=[100, 200]),
            "y_resolution": FakeOption(constraint=[100, 200]),
        }
        assert _detect_dpi(opt) == (100, 200, 300, 400, 600, 1200)

    def test_falls_back_to_plain_resolution_when_no_axis_options(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _detect_dpi

        opt = {"resolution": FakeOption(constraint=[150, 300, 600, 1200])}
        assert _detect_dpi(opt) == (150, 300, 600, 1200)

    def test_uses_axis_ladder_when_resolution_option_is_absent(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _detect_dpi

        opt = {
            "x_resolution": FakeOption(constraint=_V500_X_RESOLUTIONS),
            "y_resolution": FakeOption(constraint=_V500_Y_RESOLUTIONS),
        }
        assert max(_detect_dpi(opt)) == 6400

    def test_no_resolution_info_at_all_returns_empty(self) -> None:
        """Must stay empty, not CANONICAL_DPI_STOPS: pixel-geometry devices with no DPI info
        fall back to a 35mm default area, which relies on this being falsy."""
        from negpy.infrastructure.scanners.sane_backend import _detect_dpi

        assert _detect_dpi({}) == ()


# Real values from the reference V500 under the Transparency Unit specifically (not the plain
# `scanimage --help` dump, which reads the Flatbed default): the y ladder actually loses its
# top two steps (3200, 6400) under TPU, gaining a 9600 instead of them.
_V500_TPU_X_RESOLUTIONS = [100, 200, 300, 400, 600, 800, 1200, 1600, 3200, 6400]
_V500_TPU_Y_RESOLUTIONS = [120, 200, 320, 400, 600, 800, 1200, 1600, 2400, 4800, 9600]


class TestResolveResampledResolutions:
    """`_resolve_resampled_resolutions`: target DPI -> native y to request, reaching values
    neither axis alone (nor their exact intersection) offers by upsampling y in software.

    This is the actual case that motivated the function: under the reference V500's
    Transparency Unit, y's native ladder drops 3200 and 6400 entirely, so the exact
    intersection with x tops out at 1600 — but x itself has native 3200/6400 steps, and y's
    2400 is a legitimate upsample source for a 3200 target.
    """

    def test_3200_target_upsamples_from_native_y_2400(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _resolve_resampled_resolutions

        opt = {
            "x_resolution": FakeOption(constraint=_V500_TPU_X_RESOLUTIONS),
            "y_resolution": FakeOption(constraint=_V500_TPU_Y_RESOLUTIONS),
        }
        resolved = _resolve_resampled_resolutions(opt)
        assert resolved[3200] == 2400

    def test_exact_matches_need_no_resample(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _resolve_resampled_resolutions

        opt = {
            "x_resolution": FakeOption(constraint=_V500_TPU_X_RESOLUTIONS),
            "y_resolution": FakeOption(constraint=_V500_TPU_Y_RESOLUTIONS),
        }
        resolved = _resolve_resampled_resolutions(opt)
        for value in (200, 400, 600, 800, 1200, 1600):
            assert resolved[value] == value

    def test_6400_target_upsamples_from_native_y_4800(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _resolve_resampled_resolutions

        opt = {
            "x_resolution": FakeOption(constraint=_V500_TPU_X_RESOLUTIONS),
            "y_resolution": FakeOption(constraint=_V500_TPU_Y_RESOLUTIONS),
        }
        assert _resolve_resampled_resolutions(opt)[6400] == 4800

    def test_never_downsamples(self) -> None:
        """A target below y's own minimum has nothing valid to upsample from and is omitted,
        not clamped to the nearest-above value (that would be a downsample of y)."""
        from negpy.infrastructure.scanners.sane_backend import _resolve_resampled_resolutions

        opt = {
            "x_resolution": FakeOption(constraint=[50, 100, 200]),
            "y_resolution": FakeOption(constraint=[120, 200]),
        }
        resolved = _resolve_resampled_resolutions(opt)
        assert 50 not in resolved  # below y's minimum (120): nothing to upsample from
        assert 100 not in resolved  # same: 100 < 120
        assert resolved[200] == 200

    def test_missing_either_axis_returns_empty(self) -> None:
        from negpy.infrastructure.scanners.sane_backend import _resolve_resampled_resolutions

        assert _resolve_resampled_resolutions({}) == {}
        assert _resolve_resampled_resolutions({"x_resolution": FakeOption(constraint=[100])}) == {}

    def test_v500_flatbed_ladder_matches_square_intersection_plus_extras(self) -> None:
        """Under Flatbed (where y keeps its full ladder, per _V500_Y_RESOLUTIONS), resampling
        should recover everything the exact intersection did, plus low-end values (100) that
        square-intersection alone missed because 100 isn't in y's list, just below it (80)."""
        from negpy.infrastructure.scanners.sane_backend import (
            _resolve_resampled_resolutions,
            _resolve_square_resolutions,
        )

        opt = {
            "x_resolution": FakeOption(constraint=_V500_X_RESOLUTIONS),
            "y_resolution": FakeOption(constraint=_V500_Y_RESOLUTIONS),
        }
        square = set(_resolve_square_resolutions(opt))
        resampled = set(_resolve_resampled_resolutions(opt).keys())
        assert square.issubset(resampled)
        assert 100 in resampled and 100 not in square
