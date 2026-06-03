import numpy as np

from negpy.features.process.logic import detect_process_mode
from negpy.features.process.models import ProcessMode


def _bw_scan() -> np.ndarray:
    """Monochrome: all channels equal."""
    gray = np.linspace(0.1, 0.9, 128 * 128, dtype=np.float32).reshape(128, 128)
    return np.stack([gray, gray, gray], axis=-1)


def _tinted_bw_scan() -> np.ndarray:
    """Monochrome with a colour tint: a single grey scaled per channel."""
    gray = np.linspace(0.1, 0.9, 128 * 128, dtype=np.float32).reshape(128, 128)
    return np.stack([gray * 0.8, gray * 0.95, gray * 1.0], axis=-1)


def _c41_scan() -> np.ndarray:
    """Strong orange mask (red >> blue) over a colour scene (decorrelated channels)."""
    rng = np.random.default_rng(0)
    r = np.clip(0.6 + rng.uniform(-0.15, 0.15, (128, 128)), 0, 1).astype(np.float32)
    g = np.clip(0.4 + rng.uniform(-0.15, 0.15, (128, 128)), 0, 1).astype(np.float32)
    b = np.clip(0.2 + rng.uniform(-0.15, 0.15, (128, 128)), 0, 1).astype(np.float32)
    return np.stack([r, g, b], axis=-1)


def _e6_scan() -> np.ndarray:
    """Saturated positive with balanced channel means (no orange cast)."""
    rng = np.random.default_rng(1)
    return rng.uniform(0.0, 1.0, (128, 128, 3)).astype(np.float32)


def test_detects_bw():
    assert detect_process_mode(_bw_scan()) == ProcessMode.BW


def test_detects_tinted_bw():
    assert detect_process_mode(_tinted_bw_scan()) == ProcessMode.BW


def test_detects_c41():
    assert detect_process_mode(_c41_scan()) == ProcessMode.C41


def test_detects_e6():
    assert detect_process_mode(_e6_scan()) == ProcessMode.E6


def test_invalid_input_falls_back_to_c41():
    assert detect_process_mode(None) == ProcessMode.C41
    assert detect_process_mode(np.zeros((4, 4), dtype=np.float32)) == ProcessMode.C41
