import numpy as np

from negpy.features.process.logic import detect_process_and_source, detect_process_mode, detect_source_color
from negpy.features.process.models import ProcessMode


def _bw_scan() -> np.ndarray:
    """Monochrome: all channels equal."""
    gray = np.linspace(0.1, 0.9, 128 * 128, dtype=np.float32).reshape(128, 128)
    return np.stack([gray, gray, gray], axis=-1)


def _tinted_bw_scan() -> np.ndarray:
    """Monochrome with a color tint: a single grey scaled per channel."""
    gray = np.linspace(0.1, 0.9, 128 * 128, dtype=np.float32).reshape(128, 128)
    return np.stack([gray * 0.8, gray * 0.95, gray * 1.0], axis=-1)


def _patches(rng, low: float, high: float) -> np.ndarray:
    """128x128 of 8x8 patches, each its own value: a scene's hues cover areas, unlike noise."""
    return np.kron(rng.uniform(low, high, (16, 16)), np.ones((8, 8)))


def _c41_scan() -> np.ndarray:
    """Strong orange mask (red >> blue) over a color scene (decorrelated channels)."""
    rng = np.random.default_rng(0)
    r = np.clip(0.6 + _patches(rng, -0.15, 0.15), 0, 1).astype(np.float32)
    g = np.clip(0.4 + _patches(rng, -0.15, 0.15), 0, 1).astype(np.float32)
    b = np.clip(0.2 + _patches(rng, -0.15, 0.15), 0, 1).astype(np.float32)
    return np.stack([r, g, b], axis=-1)


def _phoenix_scan() -> np.ndarray:
    """Harman Phoenix C41: purple base (R≈B>>G), decorrelated channels."""
    rng = np.random.default_rng(2)
    r = np.clip(0.50 + _patches(rng, -0.10, 0.10), 0, 1).astype(np.float32)
    g = np.clip(0.30 + _patches(rng, -0.10, 0.10), 0, 1).astype(np.float32)
    b = np.clip(0.48 + _patches(rng, -0.10, 0.10), 0, 1).astype(np.float32)
    return np.stack([r, g, b], axis=-1)


def _e6_scan() -> np.ndarray:
    """Saturated positive with balanced channel means (no orange cast)."""
    rng = np.random.default_rng(1)
    return np.stack([_patches(rng, 0.0, 1.0) for _ in range(3)], axis=-1).astype(np.float32)


def test_detects_bw():
    assert detect_process_mode(_bw_scan()) == ProcessMode.BW


def test_detects_tinted_bw():
    assert detect_process_mode(_tinted_bw_scan()) == ProcessMode.BW


def test_detects_c41():
    assert detect_process_mode(_c41_scan()) == ProcessMode.C41


def test_detects_phoenix_c41():
    assert detect_process_mode(_phoenix_scan()) == ProcessMode.C41


def test_detects_e6():
    assert detect_process_mode(_e6_scan()) == ProcessMode.E6


def test_invalid_input_falls_back_to_c41():
    assert detect_process_mode(None) == ProcessMode.C41
    assert detect_process_mode(np.zeros((4, 4), dtype=np.float32)) == ProcessMode.C41


def test_color_film_is_a_color_source():
    for scan in (_c41_scan(), _phoenix_scan(), _e6_scan()):
        assert detect_source_color(scan) is True


def test_bw_film_is_not_a_color_source():
    assert detect_source_color(_bw_scan()) is False
    assert detect_source_color(_tinted_bw_scan()) is False


def test_a_gray_scene_on_masked_film_reads_bw_but_stays_a_color_source():
    gray = np.linspace(0.1, 0.5, 128 * 128, dtype=np.float32).reshape(128, 128)
    scan = np.stack([gray * 1.0, gray * 0.6, gray * 0.4], axis=-1)
    assert detect_process_mode(scan) == ProcessMode.BW
    assert detect_source_color(scan) is True


def _camera_bw_scan(clip: bool) -> np.ndarray:
    """A B&W negative as a camera sees it without white balance: silver a little denser in blue,
    a weak red channel, sensor noise and, with `clip`, the thin end past sensor white."""
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:192, 0:256]
    density = 0.1 + 1.9 * (0.5 + 0.5 * np.sin(xx / 23.0) * np.cos(yy / 31.0))
    gain = np.array([0.45, 0.9, 0.68]) * (2.5 if clip else 1.0)
    k = np.array([0.95, 1.0, 1.1])
    scan = gain * 10.0 ** (-density[..., None] * k) + rng.normal(0.0, 0.002, (192, 256, 3))
    return np.clip(scan, 0.0, 1.0).astype(np.float32)


def test_a_camera_capture_of_bw_film_is_not_a_color_source():
    for clip in (False, True):
        mode, color = detect_process_and_source(_camera_bw_scan(clip))
        assert (mode, color) == (ProcessMode.BW, False)


def test_a_frame_too_dark_for_a_chroma_spread_falls_back_to_correlation():
    gray = np.linspace(0.0, 0.002, 128 * 128, dtype=np.float32).reshape(128, 128)
    assert detect_process_mode(np.stack([gray, gray, gray], axis=-1)) == ProcessMode.BW
