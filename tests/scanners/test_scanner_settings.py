from dataclasses import asdict

from negpy.infrastructure.scanners.settings import ScannerSettings


def test_scan_window_default_is_none():
    assert ScannerSettings.defaults().scan_window is None


def test_scan_window_json_roundtrip_yields_tuple():
    saved = asdict(ScannerSettings(scan_window=(0.1, 0.2, 0.8, 0.9)))
    saved["scan_window"] = list(saved["scan_window"])  # JSON turns tuples into lists
    restored = ScannerSettings(**saved)
    assert restored.scan_window == (0.1, 0.2, 0.8, 0.9)
    assert isinstance(restored.scan_window, tuple)
