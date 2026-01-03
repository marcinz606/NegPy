import numpy as np
from src.backend.io import loader_factory, NonStandardFileWrapper


def test_non_standard_file_wrapper():
    # Create fake 10x10 RGB data
    data = np.random.rand(10, 10, 3).astype(np.float32)
    wrapper = NonStandardFileWrapper(data)

    with wrapper as raw:
        processed = raw.postprocess(output_bps=16)
        assert processed.shape == (10, 10, 3)
        assert processed.dtype == np.uint16
        # Verify normalization/scaling
        assert np.allclose(processed.astype(np.float32) / 65535.0, data, atol=1e-4)


def test_pakon_loader_detection(tmp_path):
    # F135 Plus Low Res is 9,000,000 bytes
    f = tmp_path / "lowres.raw"
    f.write_bytes(b"\x00" * 9000000)

    # Factory should pick PakonLoader
    raw = loader_factory.get_loader(str(f))
    assert isinstance(raw, NonStandardFileWrapper)
    with raw:
        data = raw.postprocess(output_bps=16)
        assert data.shape == (1000, 1500, 3)


def test_tiff_loader_detection(tmp_path):
    import imageio.v3 as iio

    f = tmp_path / "test.tiff"
    data = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    iio.imwrite(str(f), data)

    raw = loader_factory.get_loader(str(f))
    assert isinstance(raw, NonStandardFileWrapper)
    with raw:
        processed = raw.postprocess(output_bps=8)
        assert processed.shape == (10, 10, 3)
        assert processed.dtype == np.uint8
