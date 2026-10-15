import numpy as np
import pytest
import rawpy
import tifffile

from negpy.infrastructure.loaders.helpers import dng_bounded_preview
from negpy.infrastructure.loaders.rawpy_loader import RawpyLoader, _find_linearraw_page
from negpy.infrastructure.loaders.tiff_loader import NonStandardFileWrapper
from negpy.services.export.linear_output import _is_linearraw_dng, linear_output_source_type

_IFD0_TAGS = [
    (50706, "B", 4, (1, 4, 0, 0), True),  # DNGVersion
    (50708, "s", 0, "Sim", True),  # UniqueCameraModel
    (50721, "2i", 9, (1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1), True),  # ColorMatrix1
    (50728, "2I", 3, (1, 1, 1, 1, 1, 1), True),  # AsShotNeutral
]
_CFA_TAGS = [
    (33421, "H", 2, (2, 2), True),  # CFARepeatPatternDim
    (33422, "B", 4, (0, 1, 1, 2), True),  # CFAPattern RGGB
    (50714, "H", 1, (512,), True),  # BlackLevel
    (50717, "H", 1, (16383,), True),  # WhiteLevel
]


def _write_adobe_style_dng(path, preview_samples: int = 3, preview_subfiletype: int = 1) -> None:
    """Adobe DNG Converter layout: thumbnail IFD0, CFA main SubIFD, reduced LinearRaw JPEG XL SubIFD."""
    with tifffile.TiffWriter(path) as tw:
        tw.write(np.zeros((8, 12, 3), np.uint8), photometric="rgb", subfiletype=1, subifds=2, extratags=_IFD0_TAGS)
        tw.write(np.full((64, 96), 4000, np.uint16), photometric=32803, subfiletype=0, extratags=_CFA_TAGS)
        preview = np.full((32, 48, preview_samples), 60000, np.uint16)
        compression = 52546 if preview_samples == 3 else None
        tw.write(preview, photometric=34892, subfiletype=preview_subfiletype, planarconfig="contig", compression=compression)


def _linearraw_page(path, samples: int):
    with tifffile.TiffFile(path) as tif:
        page = _find_linearraw_page(tif, samples=samples)
        return None if page is None else tuple(page.shape)


def test_reduced_linearraw_preview_beside_a_cfa_is_not_the_main_image(tmp_path):
    path = tmp_path / "adobe.dng"
    _write_adobe_style_dng(path)

    assert _linearraw_page(path, 3) is None
    assert not _is_linearraw_dng(str(path))
    assert linear_output_source_type(str(path)) == "camera"


def test_adobe_style_preview_load_decodes_the_cfa(tmp_path):
    path = tmp_path / "adobe.dng"
    _write_adobe_style_dng(path)

    raw, metadata = RawpyLoader().load(str(path), preview_max_edge=64)
    with raw:
        assert isinstance(raw, rawpy.RawPy)
        assert not isinstance(raw, NonStandardFileWrapper)
        assert raw.raw_image.shape == (64, 96)


def test_main_image_linearraw_subifd_is_still_found(tmp_path):
    path = tmp_path / "linear.dng"
    _write_adobe_style_dng(path, preview_subfiletype=0)

    assert _linearraw_page(path, 3) == (32, 48, 3)
    assert _is_linearraw_dng(str(path))


@pytest.mark.parametrize("subfiletype, expected", [(1, None), (0, (32, 48, 4))])
def test_four_sample_linearraw_follows_the_main_image_flag(tmp_path, subfiletype, expected):
    path = tmp_path / "rgbi.dng"
    _write_adobe_style_dng(path, preview_samples=4, preview_subfiletype=subfiletype)

    assert _linearraw_page(path, 4) == expected


def test_bounded_thumbnail_skips_a_reduced_linearraw_preview(tmp_path):
    path = tmp_path / "adobe.dng"
    _write_adobe_style_dng(path)

    assert dng_bounded_preview(str(path), 32) == (False, None)
