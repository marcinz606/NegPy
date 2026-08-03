"""Parity tests for pieusb's auto-exposure against the SANE backend it ports.

The reference is sane-backends' pieusb: getGain/getGainSetting
(pieusb_specific.c:2420, 2440), updateGain2 (2528), the dg derivation in
sanei_pieusb_set_gain_offset's "from preview" branch (1912) and the 1%/99% loop
in sanei_pieusb_analyze_preview (2394). The expected numbers below are the C's,
not this implementation's.
"""

import numpy
import pytest

pieusb_calibration = pytest.importorskip("pieusb.calibration")

GAINS = pieusb_calibration.GAINS
gain_increase = pieusb_calibration.gain_increase
get_gain = pieusb_calibration.get_gain
get_gain_setting = pieusb_calibration.get_gain_setting
percentile_bounds = pieusb_calibration.percentile_bounds
update_gain = pieusb_calibration.update_gain


def test_gain_table_matches_the_firmware_values():
    assert len(GAINS) == 13
    assert GAINS[0] == 1.000
    assert GAINS[12] == 4.627


@pytest.mark.parametrize(
    "setting,expected",
    [
        (0, 1.000),
        (5, 1.075),
        (19, 1.3398),  # the default gain
        (30, 1.653),
        (59, 4.4292),  # last setting before the extrapolated branch
        (60, 4.627),
        (63, 5.2204),  # extrapolated past the table
    ],
)
def test_get_gain_interpolates_the_table(setting, expected):
    assert get_gain(setting) == pytest.approx(expected, abs=1e-4)


def test_get_gain_clamps_below_zero():
    assert get_gain(-5) == GAINS[0]


def test_get_gain_setting_inverts_get_gain():
    # 60-62 are excluded: getGain extrapolates from (setting - 55) while
    # getGainSetting's matching branch starts at 60, so the C's own pair does not
    # round-trip there. Reproducing that skew is deliberate.
    for setting in list(range(60)) + [63]:
        assert get_gain_setting(get_gain(setting)) == setting


def test_get_gain_setting_is_bounded():
    assert get_gain_setting(0.5) == 0
    assert get_gain_setting(1.0) == 0
    assert get_gain_setting(1000.0) == 63


def test_update_gain_splits_the_boost_between_gain_and_exposure():
    # Defaults (gain 19, exposure 2937) with the maximum dg the C allows.
    assert update_gain(19, 2937, 3.0) == (43, 5087)


@pytest.mark.parametrize("dg", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
def test_update_gain_delivers_exactly_dg(dg):
    """Gain takes sqrt(dg); exposure takes whatever quantisation left over."""
    setting, exposure = update_gain(19, 2937, dg)
    achieved = (get_gain(setting) / get_gain(19)) * (exposure / 2937)
    assert achieved == pytest.approx(dg, rel=2e-3)


def test_update_gain_can_reduce():
    setting, exposure = update_gain(40, 2937, 0.5)
    assert setting < 40
    assert exposure < 2937


def test_gain_increase_takes_the_smallest_channel_ratio():
    # Green saturates lowest, so green sets the ceiling for all three.
    dg = gain_increase((100, 200, 100), (58981, 52428, 58981))
    expected = (52428 / 65536) / (200 / 256)
    assert dg == pytest.approx(expected)


def test_gain_increase_is_capped_at_three():
    assert gain_increase((5, 5, 5), (58981, 52428, 58981)) == 3.00


def test_gain_increase_can_ask_to_pull_back():
    # A preview already brighter than the saturation reference wants dg < 1.
    assert gain_increase((250, 250, 250), (32768, 32768, 32768)) < 1.0


def test_gain_increase_skips_empty_channels():
    dg = gain_increase((0, 200, 0), (58981, 52428, 58981))
    assert dg == pytest.approx((52428 / 65536) / (200 / 256))


def test_gain_increase_is_a_noop_when_nothing_was_measured():
    assert gain_increase((0, 0, 0), (58981, 52428, 58981)) == 1.0


def test_percentile_bounds_shifts_16_bit_into_256_bins():
    # 40000 >> 8 = 156, and the bound is the last bin still under 99% (see
    # test_percentile_bounds_is_the_last_bin_below_the_threshold), so 155.
    plane = numpy.full((10, 10), 40000, dtype="<u2")
    _lower, upper = percentile_bounds(plane)
    assert upper == (40000 >> 8) - 1


def test_percentile_bounds_bins_8_bit_directly():
    # Divergence from the C, which would shift 8-bit samples into bin 0.
    plane = numpy.full((10, 10), 200, dtype="u1")
    _lower, upper = percentile_bounds(plane)
    assert upper == 199


def test_percentile_bounds_ignores_the_top_one_percent():
    # 0.5% specular highlights at full scale must not drag the bound up to 255.
    plane = numpy.zeros(1000, dtype="<u2").reshape(10, 100)
    plane[:] = 10 * 256
    plane.ravel()[:5] = 65535
    _lower, upper = percentile_bounds(plane)
    assert upper == 9


def _metering_scanner(mode="rgb", preview_plane_value=10 * 256, planes=3, cancelled=False):
    """A Scanner with just enough state for _meter_from_preview, no USB."""
    scanner_mod = pytest.importorskip("pieusb.scanner")
    from pieusb.option import generate_options
    from pieusb.types import ScanResult

    inquiry = _fake_inquiry()
    scanner = object.__new__(scanner_mod.Scanner)
    scanner.info = _FakeInfo(inquiry)
    scanner.params = generate_options(inquiry)
    scanner.params["mode"].value = mode
    scanner._get_gain_offset = lambda: {"saturation_level": (58981, 52428, 58981)}

    seen = {}

    def fake_scan_pass(started, emit):
        # Capture the options as the preview pass would see them. Stubs
        # _scan_pass, the sequence both passes share, not _run_scan, which is the
        # orchestrator that calls _meter_from_preview in the first place —
        # stubbing that would remove the code under test.
        seen["resolution"] = scanner.params["resolution"].value
        seen["sharpen"] = scanner.params["sharpen"].value
        seen["emit"] = emit
        rgb = numpy.full((8, 8, planes), preview_plane_value, dtype="<u2")
        return ScanResult(rgb=None if cancelled else rgb, cancelled=cancelled)

    scanner._scan_pass = fake_scan_pass
    return scanner, seen


class _FakeInfo:
    def __init__(self, inquiry):
        self.inquiry = inquiry


def _fake_inquiry():
    import dataclasses

    from pieusb.inquiry import Filter, InquiryResponse

    values = {f.name: 0 for f in dataclasses.fields(InquiryResponse)}
    values.update(
        filters=(Filter.RED, Filter.GREEN, Filter.BLUE),
        color_depths=(8, 16),
        color_formats=(),
        image_formats=(),
        scan_capabilities=(),
        optional_devices=(),
        max_resolution_x=10000,
        max_resolution_y=10000,
        max_scan_w=5000,
        max_scan_h=5000,
        minimum_exposure=100,
        maximum_exposure=1000,
        preview_scan_resolution=225,
        model_str="fake",
        vendor="v",
        revision="",
        production="",
        timestamp="",
        signature="",
        slide_transport=False,
    )
    return InquiryResponse(**values)


def test_metering_pass_runs_at_the_preview_resolution_with_quality_options_off():
    scanner, seen = _metering_scanner()
    scanner.params["resolution"].value = 3600
    scanner.params["sharpen"].value = True

    assert scanner._meter_from_preview(0.0) is None

    assert seen["resolution"] == 225
    assert seen["sharpen"] is False
    # ...and the caller's own settings are back afterwards.
    assert scanner.params["resolution"].value == 3600
    assert scanner.params["sharpen"].value is True


def test_metering_pass_reports_its_progress_as_metering():
    """The pass is a full scan, so its updates must not read as the real one's."""
    from pieusb.types import ScanPhase, UpdateData

    scanner, seen = _metering_scanner()
    reported = []
    scanner.on_update = reported.append

    scanner._meter_from_preview(0.0)

    seen["emit"](UpdateData(phase=ScanPhase.SCANNING, progress=0.5))
    assert reported == [UpdateData(phase=ScanPhase.METERING, progress=0.5)]


def test_metering_rewrites_gain_and_exposure_for_all_three_channels():
    scanner, _ = _metering_scanner()
    scanner._meter_from_preview(0.0)

    for suffix in ("r", "g", "b"):
        assert scanner.params[f"gain_{suffix}"].value != 19
        assert scanner.params[f"exp_time_{suffix}"].value != 2937
    # Infrared is never metered, matching updateGain2's 0..2 loop.
    assert scanner.params["gain_i"].value == 19
    assert scanner.params["exp_time_i"].value == 2937


def test_metering_applies_one_uniform_factor_to_every_channel():
    scanner, _ = _metering_scanner()
    scanner._meter_from_preview(0.0)

    gains = {scanner.params[f"gain_{s}"].value for s in ("r", "g", "b")}
    assert len(gains) == 1, "per-channel gains would rebalance the colour"


def test_metering_a_gray_pass_only_touches_the_green_channel():
    scanner, _ = _metering_scanner(mode="gray", planes=1)
    scanner._meter_from_preview(0.0)

    assert scanner.params["gain_g"].value != 19
    assert scanner.params["gain_r"].value == 19
    assert scanner.params["gain_b"].value == 19


def test_a_cancelled_metering_pass_is_returned_instead_of_scanning():
    scanner, _ = _metering_scanner(cancelled=True)
    result = scanner._meter_from_preview(0.0)

    assert result is not None and result.cancelled
    assert scanner.params["gain_r"].value == 19


def test_metering_leaves_settings_alone_when_the_preview_is_black():
    scanner, _ = _metering_scanner(preview_plane_value=0)
    assert scanner._meter_from_preview(0.0) is None
    assert scanner.params["gain_r"].value == 19
    assert scanner.params["exp_time_r"].value == 2937


class _RecordingDevice:
    """Captures every set_options() write instead of talking to USB."""

    def __init__(self):
        self.writes = []

    def command(self, code, out_data=None, cdb_length=0):
        self.writes.append(out_data)
        return b""


def _sent_payloads(table):
    """set_options()'s writes, keyed by the write code in their first 2 bytes."""
    import struct

    from pieusb.option import set_options

    dev = _RecordingDevice()
    set_options(dev, table)
    grouped = {}
    for payload in dev.writes:
        code = struct.unpack_from("<H", payload, 0)[0]
        grouped.setdefault(code, []).append(payload)
    return grouped


def _relative_exposures(grouped):
    import struct

    from pieusb.transport import SCSI_EXPOSURE

    return [struct.unpack_from("<H", p, 6)[0] for p in grouped[SCSI_EXPOSURE]]


def test_relative_and_absolute_exposure_are_separate_options():
    from pieusb.option import generate_options

    table = generate_options(_fake_inquiry())
    # Both exist, with the two defaults SANE uses for the two commands.
    assert [table[f"exp_rel_{c}"].value for c in "rgb"] == [100, 100, 100]
    assert [table[f"exp_time_{c}"].value for c in "rgbi"] == [2937] * 4
    # Relative exposure has no infrared entry: the C struct holds three colours.
    with pytest.raises(KeyError):
        table["exp_rel_i"]


def test_relative_exposure_reaches_only_the_scsi_exposure_write():
    from pieusb.option import generate_options

    table = generate_options(_fake_inquiry())
    table["exp_rel_r"].value = 55
    table["exp_rel_g"].value = 60
    table["exp_rel_b"].value = 65

    grouped = _sent_payloads(table)
    assert _relative_exposures(grouped) == [55, 60, 65]


def test_absolute_exposure_does_not_leak_into_the_relative_write():
    """The conflation regression: one option used to feed both commands."""
    from pieusb.option import generate_options

    table = generate_options(_fake_inquiry())
    for channel in "rgb":
        table[f"exp_time_{channel}"].value = 5087

    # Moving the absolute time leaves the relative write at its 100% default.
    assert _relative_exposures(_sent_payloads(table)) == [100, 100, 100]


def test_absolute_exposure_reaches_the_gain_offset_payload():
    import struct

    from pieusb.option import generate_options, set_options

    table = generate_options(_fake_inquiry())
    table["exp_time_r"].value = 5000
    table["exp_time_g"].value = 5001
    table["exp_time_b"].value = 5002
    table["exp_time_i"].value = 5003

    dev = _RecordingDevice()
    set_options(dev, table)

    # The gain/offset payload is the only 29-byte write. R/G/B exposure times
    # lead it; infrared's sits at byte 18, after the 12 offset/gain/light bytes.
    payload = next(p for p in dev.writes if len(p) == 29)
    assert struct.unpack_from("<3H", payload, 0) == (5000, 5001, 5002)
    assert struct.unpack_from("<H", payload, 18)[0] == 5003


def test_moving_the_relative_exposure_warns(caplog):
    from pieusb.option import generate_options

    table = generate_options(_fake_inquiry())
    table["mode"].value = "rgb"
    with caplog.at_level("WARNING"):
        table.validate()
    assert not [r for r in caplog.records if "relative exposure" in r.message]

    table["exp_rel_g"].value = 120
    caplog.clear()
    with caplog.at_level("WARNING"):
        table.validate()
    warning = next(r for r in caplog.records if "relative exposure" in r.message)
    assert "exp_rel_g=120" in warning.message
    assert "exp_time_" in warning.message


def test_metering_never_touches_the_relative_exposure():
    scanner, _ = _metering_scanner()
    scanner._meter_from_preview(0.0)

    assert [scanner.params[f"exp_rel_{c}"].value for c in "rgb"] == [100, 100, 100]


def test_percentile_bounds_is_the_last_bin_below_the_threshold():
    # 99% of the pixels sit in bin 10 and 1% in bin 200. The cumulative share is
    # already >= 0.99 at bin 10, so the C's loop stops before it: bound 9, not 10.
    plane = numpy.zeros((100, 100), dtype="<u2")
    plane[:] = 10 * 256
    plane.ravel()[:100] = 200 * 256
    _lower, upper = percentile_bounds(plane)
    assert upper == 9
