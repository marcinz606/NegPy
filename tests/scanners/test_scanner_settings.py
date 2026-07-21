import json
from dataclasses import asdict

from negpy.infrastructure.scanners.settings import ScannerSettings, resolve_batch_selection


def test_scan_window_default_is_none():
    assert ScannerSettings.defaults().scan_window is None


def test_scan_window_json_roundtrip_yields_tuple():
    saved = asdict(ScannerSettings(scan_window=(0.1, 0.2, 0.8, 0.9)))
    saved["scan_window"] = list(saved["scan_window"])  # JSON turns tuples into lists
    restored = ScannerSettings(**saved)
    assert restored.scan_window == (0.1, 0.2, 0.8, 0.9)
    assert isinstance(restored.scan_window, tuple)


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
    frames, windows, base = resolve_batch_selection(settings, 2, 3)
    assert frames == (1, 2, 4)
    assert windows == {1: (0.0, 0.0, 1.0, 1.0), 4: (0.1, 0.1, 0.5, 0.5)}
    assert base is None


def test_resolve_batch_selection_omits_selected_frame_without_a_window():
    settings = ScannerSettings(selected_frames=(1, 2), frame_windows={2: (0.1, 0.1, 0.5, 0.5)})
    frames, windows, base = resolve_batch_selection(settings, 1, 1)
    assert frames == (1, 2)
    assert windows == {2: (0.1, 0.1, 0.5, 0.5)}
    assert base is None


def test_resolve_batch_selection_falls_back_to_spinbox_range():
    settings = ScannerSettings(scan_window=(0.2, 0.2, 0.8, 0.8))
    frames, windows, base = resolve_batch_selection(settings, 2, 5)
    assert frames == (2, 3, 4, 5)
    assert windows == {}
    assert base == (0.2, 0.2, 0.8, 0.8)
