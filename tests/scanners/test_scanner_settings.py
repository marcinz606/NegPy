import json
from dataclasses import asdict

from negpy.infrastructure.scanners.settings import (
    ScannerSettings,
    format_frame_spec,
    parse_frame_spec,
    resolve_batch_selection,
)


def test_scan_window_default_is_none():
    assert ScannerSettings.defaults().scan_window is None


def test_scan_window_json_roundtrip_yields_tuple():
    saved = asdict(ScannerSettings(scan_window=(0.1, 0.2, 0.8, 0.9)))
    saved["scan_window"] = list(saved["scan_window"])  # JSON turns tuples into lists
    restored = ScannerSettings(**saved)
    assert restored.scan_window == (0.1, 0.2, 0.8, 0.9)
    assert isinstance(restored.scan_window, tuple)


def test_backend_default_matches_registry():
    from negpy.infrastructure.scanners.registry import DEFAULT_BACKEND_ID

    assert ScannerSettings.defaults().backend == DEFAULT_BACKEND_ID


def test_backend_survives_json_roundtrip():
    from dataclasses import replace

    original = replace(ScannerSettings.defaults(), backend="mock")
    restored = ScannerSettings(**json.loads(json.dumps(asdict(original), default=str)))
    assert restored.backend == "mock"
    assert restored == original


def test_per_frame_defaults_are_empty():
    d = ScannerSettings.defaults()
    assert d.frame_windows == {}
    assert d.selected_frames == ()


def test_frame_windows_and_selection_survive_json_roundtrip():
    original = ScannerSettings(
        frame_windows={3: (0.1, 0.1, 0.9, 0.9)},
        selected_frames=(1, 3),
    )
    # Exact repository path: json turns int keys → str, tuples → lists.
    restored = ScannerSettings(**json.loads(json.dumps(asdict(original), default=str)))
    assert restored.frame_windows == {3: (0.1, 0.1, 0.9, 0.9)}
    assert restored.selected_frames == (1, 3)
    assert restored == original


def test_resolve_batch_selection_uses_dialog_selection_sorted():
    settings = ScannerSettings(
        selected_frames=(4, 1, 2),
        frame_windows={1: (0.0, 0.0, 1.0, 1.0), 4: (0.1, 0.1, 0.5, 0.5)},
    )
    frames, windows, base = resolve_batch_selection(settings, capacity=6)
    assert frames == (1, 2, 4)
    assert windows == {1: (0.0, 0.0, 1.0, 1.0), 4: (0.1, 0.1, 0.5, 0.5)}
    assert base is None


def test_resolve_batch_selection_omits_selected_frame_without_a_window():
    settings = ScannerSettings(selected_frames=(1, 2), frame_windows={2: (0.1, 0.1, 0.5, 0.5)})
    frames, windows, base = resolve_batch_selection(settings, capacity=6)
    assert frames == (1, 2)
    assert windows == {2: (0.1, 0.1, 0.5, 0.5)}
    assert base is None


def test_resolve_batch_selection_falls_back_to_every_slot_the_feeder_holds():
    settings = ScannerSettings(scan_window=(0.2, 0.2, 0.8, 0.8))
    frames, windows, base = resolve_batch_selection(settings, capacity=4)
    assert frames == (1, 2, 3, 4)
    assert windows == {}
    assert base == (0.2, 0.2, 0.8, 0.8)


def test_a_measured_strip_with_no_selection_means_every_frame() -> None:
    """Its frame count is unknown until the film is measured, so a range would mean frame 1."""
    frames, windows, window = resolve_batch_selection(ScannerSettings(), whole_strip=True)
    assert frames == () and windows == {} and window is None


def test_a_selection_still_wins_on_a_measured_strip() -> None:
    settings = ScannerSettings(selected_frames=(2, 5))
    frames, _windows, _window = resolve_batch_selection(settings, whole_strip=True)
    assert frames == (2, 5)


# ── the frame list an operator types ──────────────────────────────────────


def test_parse_frame_spec_reads_ranges_lists_and_both():
    assert parse_frame_spec("1-6") == (1, 2, 3, 4, 5, 6)
    assert parse_frame_spec("1,2,5") == (1, 2, 5)
    assert parse_frame_spec(" 5, 1-3 ") == (1, 2, 3, 5)


def test_parse_frame_spec_reads_no_frames_as_every_frame():
    assert parse_frame_spec("") == ()
    assert parse_frame_spec("  ") == ()


def test_parse_frame_spec_refuses_what_it_cannot_read():
    import pytest

    for text in ("x", "2-", "-3", "0", "5-2", "1,,x"):
        with pytest.raises(ValueError):
            parse_frame_spec(text)


def test_format_frame_spec_collapses_runs():
    assert format_frame_spec(()) == ""
    assert format_frame_spec((1, 2, 3, 6)) == "1-3,6"
    assert format_frame_spec((4, 1, 2)) == "1-2,4"


def test_a_frame_spec_round_trips():
    for text in ("1-6", "1,3,5", "1-3,7-9"):
        assert format_frame_spec(parse_frame_spec(text)) == text


def test_a_saved_frame_range_becomes_a_selection():
    restored = ScannerSettings.from_dict({"frame_from": 2, "frame_to": 4, "dpi": 4000})
    assert restored.selected_frames == (2, 3, 4)
    assert restored.dpi == 4000


def test_an_unset_saved_frame_range_selects_nothing():
    assert ScannerSettings.from_dict({"frame_from": 1, "frame_to": 1}).selected_frames == ()


def test_a_key_this_version_dropped_keeps_the_rest_of_the_blob():
    restored = ScannerSettings.from_dict({"gone_in_this_version": True, "output_folder": "/scans"})
    assert restored.output_folder == "/scans"
