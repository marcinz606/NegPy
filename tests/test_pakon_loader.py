import numpy as np

from negpy.infrastructure.loaders.pakon_loader import PakonLoader

# Smallest valid Pakon spec: 1000x1500 ("F135 Plus Low Res").
H, W = 1000, 1500


def _planar_ramp() -> np.ndarray:
    """Planar (3, H, W) buffer with a per-column ramp so column 0 is distinct."""
    col = np.arange(W, dtype="<u2")
    plane = np.broadcast_to(col, (H, W))
    return np.stack([plane, plane + 100, plane + 200], axis=0).astype("<u2")


def _load_col0(path) -> np.ndarray:
    wrapper, _ = PakonLoader().load(str(path))
    img = wrapper.data  # NonStandardFileWrapper exposes .data
    return img[:, 0, :]  # (H, 3) leftmost column


def test_skips_16_byte_header(tmp_path):
    p = tmp_path / "headered.raw"
    with open(p, "wb") as f:
        np.array([16, W, H, 48], dtype="<u4").tofile(f)
        _planar_ramp().tofile(f)

    col0 = _load_col0(p)
    # Leaked header would shift column/channel alignment and break this.
    expected = np.array([0, 100, 200], dtype="<u2").astype(np.float32) / 65535.0
    np.testing.assert_allclose(col0, np.broadcast_to(expected, (H, 3)), atol=1e-6)


def test_headerless_still_decodes(tmp_path):
    p = tmp_path / "headerless.raw"
    with open(p, "wb") as f:
        _planar_ramp().tofile(f)  # exactly 9_000_000 bytes, no header

    col0 = _load_col0(p)
    expected = np.array([0, 100, 200], dtype="<u2").astype(np.float32) / 65535.0
    np.testing.assert_allclose(col0, np.broadcast_to(expected, (H, 3)), atol=1e-6)
