"""NkscanBackend: capability projection, frame resolution, option plumbing, error typing."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from negpy.infrastructure.scanners.base import TransientScanError
from negpy.infrastructure.scanners.nkscan_backend import _crop_frame, _offset_units, _shift_frame, _stack_rgb
from negpy.infrastructure.scanners.params import ScanMode, ScanParams
from tests.scanners import fake_nkscan
from tests.scanners.fake_nkscan import DEVICE_ID, FRAMES, FakeCapabilities, make_backend

_PARAMS = ScanParams(dpi=1000, depth=16, capture_ir=False)


def _scan(backend, params=_PARAMS, progress=None):
    return backend.scan(DEVICE_ID, params, progress or (lambda *_: None), threading.Event())


# ── capabilities ──────────────────────────────────────────────────────────


def test_capabilities_project_the_dpi_ladder_and_the_optical_stop() -> None:
    backend, _ = make_backend()
    caps = backend.list_devices()[0].capabilities

    assert caps.supported_dpi == (600, 1200, 2400, 3600, 4000)
    assert caps.supported_depths == (16,)
    assert caps.sources == (ScanMode.NEGATIVE, ScanMode.POSITIVE)


def test_capabilities_announce_the_nkscan_only_controls() -> None:
    backend, _ = make_backend()
    caps = backend.list_devices()[0].capabilities

    assert caps.hw_clean and caps.roll_discovery and caps.superfine
    assert caps.max_samples == 16
    assert "135" in caps.film_formats
    assert caps.ir_channel and caps.can_eject
    # Neither is controllable through the bindings, so neither gets a control.
    assert not caps.autofocus and not caps.auto_exposure
    assert caps.exposure_time_us is None
    # The frame count is unknown until a strip is measured.
    assert caps.adapter_frame_capacity is None


def test_a_continuous_range_outside_every_stop_still_offers_a_ladder() -> None:
    backend, _ = make_backend(caps=FakeCapabilities(x_dpi_range=(20, 40), optical_dpi=40))
    assert backend.list_devices()[0].capabilities.supported_dpi[0] == 40


def test_devices_are_named_from_the_unit() -> None:
    backend, _ = make_backend()
    device = backend.list_devices()[0]
    assert device.id == DEVICE_ID
    assert (device.vendor, device.model) == ("Nikon", "LS-50")


def test_list_devices_caches_and_refresh_re_probes() -> None:
    backend, module = make_backend()
    backend.list_devices()
    probes = len(module.opened)
    backend.list_devices()
    assert len(module.opened) == probes

    backend.refresh_devices()
    assert len(module.opened) > probes


def test_a_held_device_is_not_re_probed() -> None:
    """Probing opens the unit, and nkscan reserves it: a held one would refuse."""
    backend, module = make_backend()
    backend.list_devices()
    with backend.open_session(DEVICE_ID):
        probes = len(module.opened)
        assert [d.id for d in backend.refresh_devices()] == [DEVICE_ID]
        assert len(module.opened) == probes + 1  # the session itself, not a probe


# ── frames ────────────────────────────────────────────────────────────────


def test_a_scan_with_no_frame_takes_the_first_detected_one() -> None:
    backend, module = make_backend()
    _scan(backend)
    assert module.opened[-1].scans[0]["frame"] == FRAMES[0]


def test_a_frame_index_resolves_against_the_detected_rects() -> None:
    backend, module = make_backend()
    _scan(backend, ScanParams(dpi=1000, depth=16, capture_ir=False, frame=3))
    assert module.opened[-1].scans[0]["frame"] == FRAMES[2]


def test_a_frame_past_the_detected_count_fails_rather_than_scanning_something_else() -> None:
    backend, _ = make_backend()
    with pytest.raises(RuntimeError, match="Frame 9 was not detected"):
        _scan(backend, ScanParams(dpi=1000, depth=16, capture_ir=False, frame=9))


def test_discovery_runs_once_and_is_reused_across_scans() -> None:
    backend, module = make_backend()
    _scan(backend)
    _scan(backend, ScanParams(dpi=1000, depth=16, capture_ir=False, frame=2))
    assert sum(len(s.discoveries) for s in module.opened) == 1


def test_an_eject_forgets_the_rects_because_the_film_has_moved() -> None:
    backend, _ = make_backend(with_eject=True)
    _scan(backend)
    assert backend.frames(DEVICE_ID)

    assert backend.eject(DEVICE_ID) is True
    assert backend.frames(DEVICE_ID) == []


def test_no_detected_frames_is_a_plain_failure() -> None:
    backend, _ = make_backend(frames=())
    with pytest.raises(RuntimeError, match="No frames were detected"):
        _scan(backend)


def test_the_film_format_reaches_discovery() -> None:
    backend, module = make_backend()
    _scan(backend, ScanParams(dpi=1000, depth=16, capture_ir=False, film_format="66"))
    assert module.opened[-1].discoveries == ["66"]


def test_an_unknown_film_format_is_refused_before_the_unit_moves() -> None:
    backend, module = make_backend()
    with pytest.raises(RuntimeError, match="Unknown film format"):
        _scan(backend, ScanParams(dpi=1000, depth=16, capture_ir=False, film_format="120"))
    assert module.opened == []


# ── geometry ──────────────────────────────────────────────────────────────


def test_offset_millimetres_become_stage_addresses() -> None:
    assert _offset_units(25.4, 4000) == 4000
    assert _offset_units(0.0, 4000) == 0


def test_a_frame_slides_along_the_feed_axis_only() -> None:
    assert _shift_frame((100, 10, 1100, 810), 50) == (150, 10, 1150, 810)


def test_a_frame_cannot_slide_before_the_stage_range() -> None:
    assert _shift_frame((100, 10, 1100, 810), -400) == (0, 10, 1000, 810)


def test_a_window_crops_inside_the_frame() -> None:
    assert _crop_frame((100, 10, 1100, 810), (0.0, 0.5, 0.5, 1.0)) == (600, 10, 1100, 410)


def test_a_degenerate_window_keeps_a_pixel() -> None:
    top, left, bottom, right = _crop_frame((100, 10, 1100, 810), (0.5, 0.5, 0.5, 0.5))
    assert bottom > top and right > left


def test_the_offset_and_the_window_both_reach_the_scan() -> None:
    backend, module = make_backend()
    _scan(
        backend,
        ScanParams(dpi=1000, depth=16, capture_ir=False, frame=1, frame_offset_mm=25.4, window=(0.0, 0.0, 1.0, 0.5)),
    )
    top, _left, bottom, _right = module.opened[-1].scans[0]["frame"]
    assert top == FRAMES[0][0] + 4000
    assert bottom - top == (FRAMES[0][2] - FRAMES[0][0]) // 2


# ── result ────────────────────────────────────────────────────────────────


def test_the_planes_come_back_as_one_rgb_array() -> None:
    backend, _ = make_backend()
    result = _scan(backend)
    assert result.rgb.shape == (8, 6, 3)
    assert result.rgb.dtype == np.uint16
    assert result.ir is None and result.ir_valid_mask is None


def test_an_ir_scan_carries_the_plane_and_an_all_valid_mask() -> None:
    backend, _ = make_backend()
    result = _scan(backend, ScanParams(dpi=1000, depth=16, capture_ir=True))
    assert result.ir is not None and result.ir.shape == (8, 6)
    assert result.ir_valid_mask is not None and result.ir_valid_mask.all()


def test_a_single_plane_unit_still_yields_three_channels() -> None:
    mono = {"default": np.full((4, 3), 7, np.uint16)}
    rgb = _stack_rgb(mono)
    assert rgb.shape == (4, 3, 3)
    assert (rgb[..., 0] == rgb[..., 2]).all()


def test_a_scan_with_no_planes_at_all_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="no image planes"):
        _stack_rgb({})


# ── options ───────────────────────────────────────────────────────────────


def test_ice_samples_and_superfine_reach_the_scan() -> None:
    backend, module = make_backend()
    _scan(backend, ScanParams(dpi=2400, depth=16, capture_ir=True, clean=True, samples=4, superfine=True))
    asked = module.opened[-1].scans[0]
    assert asked["clean"] and asked["infrared"] and asked["superfine"]
    assert (asked["samples"], asked["dpi"]) == (4, 2400)
    # White balance stays locked: a negative is normalized downstream, not by the scanner.
    assert asked["lock_white_balance"] is True


def test_samples_outside_the_bound_are_refused() -> None:
    backend, module = make_backend()
    with pytest.raises(RuntimeError, match="Samples must be 1..16"):
        _scan(backend, ScanParams(dpi=1000, depth=16, capture_ir=False, samples=32))
    assert module.opened == []


def test_hardware_auto_exposure_is_refused_rather_than_ignored() -> None:
    backend, _ = make_backend()
    with pytest.raises(RuntimeError, match="meters every frame"):
        _scan(backend, ScanParams(dpi=1000, depth=16, capture_ir=False, auto_exposure=True))


def test_the_default_autofocus_request_is_honoured_silently() -> None:
    """nkscan focuses every frame itself, so the default True is the truth, not an unmet option."""
    backend, _ = make_backend()
    assert _PARAMS.autofocus is True
    assert _scan(backend).rgb.shape[2] == 3


# ── lifetime and errors ───────────────────────────────────────────────────


def test_a_scan_stages_the_unit_and_closes_it_again() -> None:
    backend, module = make_backend()
    _scan(backend)
    session = module.opened[-1]
    assert session.staged == 1 and session.closed


def test_film_is_loaded_when_the_holder_is_empty() -> None:
    backend, module = make_backend(media_loaded_at_open=False)
    _scan(backend)
    assert module.opened[-1].loads == 1


def test_a_held_device_refuses_a_stateless_scan() -> None:
    backend, _ = make_backend()
    with backend.open_session(DEVICE_ID):
        with pytest.raises(RuntimeError, match="held by an open session"):
            _scan(backend)


def test_a_second_session_on_a_held_device_is_refused() -> None:
    backend, _ = make_backend()
    with backend.open_session(DEVICE_ID):
        with pytest.raises(RuntimeError, match="already held"):
            backend.open_session(DEVICE_ID)


def test_a_session_scan_reuses_the_one_hold() -> None:
    backend, module = make_backend()
    with backend.open_session(DEVICE_ID) as session:
        session.scan(_PARAMS, lambda *_: None, threading.Event())
        session.scan(_PARAMS, lambda *_: None, threading.Event())
    assert len(module.opened) == 2  # one probe, one session
    assert len(module.opened[-1].scans) == 2


def test_a_closed_session_refuses_everything() -> None:
    backend, _ = make_backend()
    session = backend.open_session(DEVICE_ID)
    session.close()
    with pytest.raises(RuntimeError, match="is closed"):
        session.scan(_PARAMS, lambda *_: None, threading.Event())
    with pytest.raises(RuntimeError, match="is closed"):
        session.eject()


def test_a_failure_to_stage_still_releases_the_unit() -> None:
    backend, module = make_backend()
    module.frames = FRAMES
    module.discover_error = fake_nkscan.MediaError("no film in the holder")
    with pytest.raises(RuntimeError, match="no film"):
        _scan(backend)
    assert module.opened[-1].closed


def test_a_transport_glitch_is_typed_transient() -> None:
    backend, _ = make_backend(scan_error=fake_nkscan.TransportError("the link dropped"))
    with pytest.raises(TransientScanError):
        _scan(backend)


def test_a_busy_unit_is_typed_transient() -> None:
    backend, _ = make_backend(scan_error=fake_nkscan.DeviceBusy("another process has it"))
    with pytest.raises(TransientScanError):
        _scan(backend)


def test_a_media_fault_is_not_transient() -> None:
    backend, _ = make_backend(scan_error=fake_nkscan.MediaError("the holder jammed"))
    with pytest.raises(RuntimeError) as excinfo:
        _scan(backend)
    assert not isinstance(excinfo.value, TransientScanError)


def test_an_unsupported_operation_names_it() -> None:
    backend, _ = make_backend(scan_error=fake_nkscan.UnsupportedError("nope", op="clean", reason="mono film"))
    with pytest.raises(RuntimeError, match="clean: mono film"):
        _scan(backend)


def test_a_cancel_from_the_progress_callback_reads_as_cancelled() -> None:
    backend, _ = make_backend(scan_error=fake_nkscan.ScanCancelled("stopped"))
    with pytest.raises(RuntimeError, match="[Cc]ancel"):
        _scan(backend)


# ── progress ──────────────────────────────────────────────────────────────


def test_progress_reports_a_fraction_and_a_phase_name() -> None:
    backend, _ = make_backend(progress_steps=4)
    seen: list[tuple[float, str]] = []
    _scan(backend, progress=lambda fraction, phase="Scanning": seen.append((fraction, phase)))

    assert ("Detecting frames" in [p for _f, p in seen]) and ("Scanning" in [p for _f, p in seen])
    assert [f for f, p in seen if p == "Scanning"] == [0.25, 0.5, 0.75, 1.0]


def test_a_cancel_mid_read_stops_the_pass() -> None:
    backend, module = make_backend(progress_steps=4)
    cancel = threading.Event()

    def progress(_fraction: float, phase: str = "Scanning") -> None:
        if phase == "Scanning":
            cancel.set()

    with pytest.raises(RuntimeError, match="[Cc]ancel"):
        backend.scan(DEVICE_ID, _PARAMS, progress, cancel)
    assert len(module.opened[-1].scans) == 1


# ── units that read one line at a time ────────────────────────────────────


def test_a_unit_with_no_fast_read_still_scans() -> None:
    """The LS-50 offers only line ordering, and the bindings cannot be asked in advance."""
    backend, module = make_backend(line_ordering_only=True)
    result = _scan(backend)

    assert result.rgb.shape == (8, 6, 3)
    assert module.opened[-1].scans[-1]["superfine"] is True


def test_the_refusal_is_learned_once_per_device() -> None:
    backend, module = make_backend(line_ordering_only=True)
    _scan(backend)
    _scan(backend)

    # One wasted validation on the first scan, then straight to the read the unit accepts.
    asked = [s["superfine"] for session_ in module.opened for s in session_.scans]
    assert asked == [True, True]


def test_a_different_unsupported_operation_still_fails() -> None:
    backend, _ = make_backend(scan_error=fake_nkscan.UnsupportedError("nope", op="clean", reason="mono film"))
    with pytest.raises(RuntimeError, match="clean: mono film"):
        _scan(backend)


# ── how many frames the film carries ──────────────────────────────────────


def test_detect_frames_measures_the_loaded_film() -> None:
    backend, module = make_backend()
    assert backend.detect_frames(DEVICE_ID) == len(FRAMES)
    assert module.opened[-1].discoveries == [None]


def test_detect_frames_reuses_what_a_preview_already_measured() -> None:
    backend, module = make_backend()
    backend.detect_frames(DEVICE_ID)
    measurements = sum(len(s.discoveries) for s in module.opened)

    assert backend.detect_frames(DEVICE_ID) == len(FRAMES)
    assert sum(len(s.discoveries) for s in module.opened) == measurements


def test_detect_frames_uses_a_held_session_rather_than_opening_a_second() -> None:
    backend, module = make_backend()
    with backend.open_session(DEVICE_ID):
        held = len(module.opened)
        assert backend.detect_frames(DEVICE_ID, film_format="66") == len(FRAMES)
        assert len(module.opened) == held
        assert module.opened[-1].discoveries == ["66"]


def test_bare_film_detects_nothing() -> None:
    backend, _ = make_backend(frames=())
    assert backend.detect_frames(DEVICE_ID) == 0


def test_a_unit_that_offers_one_read_mode_offers_no_superfine_control() -> None:
    backend, _ = make_backend(caps=FakeCapabilities(multiline_read=False))
    caps = backend.list_devices()[0].capabilities
    assert caps.superfine is False


def test_a_unit_that_ignores_repeated_reads_offers_no_samples_control() -> None:
    backend, _ = make_backend(caps=FakeCapabilities(multi_reading=False))
    caps = backend.list_devices()[0].capabilities
    assert caps.max_samples == 1


def test_a_unit_that_says_it_has_no_fast_read_is_never_asked_for_one() -> None:
    """The refusal is free but the round trip is not, and the pages already answered."""
    backend, module = make_backend(caps=FakeCapabilities(multiline_read=False), line_ordering_only=True)
    backend.list_devices()
    _scan(backend)

    assert [s["superfine"] for s in module.opened[-1].scans] == [True]


def test_an_nkscan_too_old_to_say_assumes_both() -> None:
    """The bits landed after the first bindings; a wheel without them must not lose controls."""

    class OldCapabilities:
        vendor, product, revision, model = "Nikon", "LS-9000 ED", "1.00", "LS-9000"
        x_dpi_range = y_dpi_range = (500, 4000)
        optical_dpi = 4000

    backend, _ = make_backend(caps=OldCapabilities())
    caps = backend.list_devices()[0].capabilities
    assert caps.superfine is True and caps.max_samples == 16


# ── what is on the film ───────────────────────────────────────────────────


def test_reversal_film_is_measured_the_other_way_round() -> None:
    """Unexposed slide film develops to maximum density, a negative to its base."""
    backend, module = make_backend()
    session = backend.open_session(DEVICE_ID)
    backend.discover_frames(module.opened[-1], DEVICE_ID, film_format=None, film_type="positive")
    assert module.opened[-1].polarities == [True]

    backend.forget_frames(DEVICE_ID)
    backend.discover_frames(module.opened[-1], DEVICE_ID, film_format=None, film_type="mono")
    assert module.opened[-1].polarities == [True, False]
    session.close()


def test_ir_on_black_and_white_is_refused_before_the_unit_moves() -> None:
    backend, module = make_backend()
    with pytest.raises(RuntimeError, match="B&W negative blocks infrared"):
        _scan(backend, ScanParams(dpi=1000, depth=16, capture_ir=True, film_type="mono"))
    assert module.opened == []


def test_ice_on_kodachrome_is_refused_too() -> None:
    backend, _ = make_backend()
    with pytest.raises(RuntimeError, match="Kodachrome blocks infrared"):
        _scan(backend, ScanParams(dpi=1000, depth=16, capture_ir=False, clean=True, film_type="kodachrome"))


def test_black_and_white_still_scans_without_ir() -> None:
    backend, module = make_backend()
    _scan(backend, ScanParams(dpi=1000, depth=16, capture_ir=False, film_type="mono"))
    assert module.opened[-1].scans[0]["infrared"] is False


def test_an_unknown_film_type_is_refused() -> None:
    backend, _ = make_backend()
    with pytest.raises(RuntimeError, match="Unknown film type"):
        _scan(backend, ScanParams(dpi=1000, depth=16, capture_ir=False, film_type="tintype"))
