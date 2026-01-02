import numpy as np
from src.backend.raw_handlers import PlanarRawWrapper, load_pakon_raw, load_special_raw


def test_planar_raw_wrapper():
    # Create fake 10x10 RGB data
    data = np.random.rand(10, 10, 3).astype(np.float32)
    wrapper = PlanarRawWrapper(data)

    with wrapper as raw:
        processed = raw.postprocess()
        assert processed.shape == (10, 10, 3)
        assert processed.dtype == np.uint16
        # Verify normalization/scaling
        assert np.allclose(processed.astype(np.float32) / 65535.0, data, atol=1e-4)


def test_load_pakon_raw_invalid_size(tmp_path):
    # Create file with "weird" size
    f = tmp_path / "not_pakon.raw"
    f.write_bytes(b"\x00" * 500)
    assert load_pakon_raw(str(f)) is None


def test_load_pakon_raw_valid_lowres(tmp_path):
    # F135 Plus Low Res is 9,000,000 bytes
    f = tmp_path / "lowres.raw"
    # 16-bit planar: 1500x1000 * 3 channels * 2 bytes = 9MB
    f.write_bytes(b"\x00" * 9000000)

    result = load_pakon_raw(str(f))
    assert result is not None
    with result as raw:
        data = raw.postprocess()
        assert data.shape == (1000, 1500, 3)


def test_load_special_raw_fallback(tmp_path):
    # Verify the dispatcher finds the pakon loader
    f = tmp_path / "special.raw"
    f.write_bytes(b"\x00" * 36000000)  # High res signature

    result = load_special_raw(str(f))
    assert isinstance(result, PlanarRawWrapper)
