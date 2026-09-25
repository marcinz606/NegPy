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


def _write_headered(path, planes: np.ndarray, header=None) -> None:
    _, h, w = planes.shape
    with open(path, "wb") as f:
        np.array(header or [16, w, h, 48], dtype="<u4").tofile(f)
        planes.astype("<u2").tofile(f)


def test_header_dimensions_open_a_size_outside_the_table(tmp_path):
    # Base 8 on an F135 writes 2250x1500; scaled down, keeping a non-square, non-table size.
    h, w = 150, 225
    planes = np.stack([np.full((h, w), v, dtype="<u2") for v in (1000, 2000, 3000)])
    planes[0, 0, :] = np.arange(w)
    p = tmp_path / "base8.raw"
    _write_headered(p, planes)

    assert PakonLoader.can_handle(str(p))
    wrapper, _ = PakonLoader().load(str(p))
    img = wrapper.data
    assert img.shape == (h, w, 3)
    np.testing.assert_allclose(img[0, :, 0] * 65535.0, np.arange(w), atol=0.5)
    np.testing.assert_allclose(img[1, 0] * 65535.0, [1000, 2000, 3000], atol=0.5)


def test_header_skips_the_layout_guess(tmp_path):
    # A period-3 texture at the start of the red plane makes the guess read planar data as interleaved.
    h, w = 40, 60
    planes = np.stack([np.full((h, w), v, dtype="<u2") for v in (40000, 20000, 8000)])
    planes[0, :2] = np.tile([0, 30000, 60000], w // 3 * 2).reshape(2, w)
    assert PakonLoader._looks_interleaved(planes.reshape(-1))
    p = tmp_path / "texture.raw"
    _write_headered(p, planes)

    wrapper, _ = PakonLoader().load(str(p))
    np.testing.assert_allclose(wrapper.data[-1, -1] * 65535.0, [40000, 20000, 8000], atol=0.5)
    preview = np.asarray(PakonLoader().load_bounded_preview(str(p), 60))
    assert preview[-1, -1, 0] > preview[-1, -1, 1] > preview[-1, -1, 2]


def test_header_that_disagrees_with_file_size_is_ignored(tmp_path):
    planes = np.zeros((3, 40, 60), dtype="<u2")
    p = tmp_path / "wrong.raw"
    _write_headered(p, planes, header=[16, 61, 40, 48])

    assert PakonLoader.read_header(str(p)) is None
    assert not PakonLoader.can_handle(str(p))


def test_portrait_header_beats_the_size_table(tmp_path):
    # TLX writes a frame rotated 90 degrees as 2000x3000: the same file size as the F135 table entry, rows swapped.
    h, w = 3000, 2000
    rows = np.broadcast_to(np.arange(h, dtype="<u2")[:, None], (h, w))
    planes = np.stack([rows, rows + 1, rows + 2])
    p = tmp_path / "portrait.raw"
    _write_headered(p, planes)

    wrapper, _ = PakonLoader().load(str(p))
    img = wrapper.data
    assert img.shape == (h, w, 3)
    np.testing.assert_allclose(img[:, -1, 0] * 65535.0, np.arange(h), atol=0.5)
    assert PakonLoader().load_bounded_preview(str(p), 300).size == (200, 300)
