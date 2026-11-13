import struct

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


_EXIF_PLACEHOLDER = 34666


def _write_scanner_dng(path, exposure: bool = False) -> None:
    """VueScan/SilverFast layout: thumbnail IFD0, LinearRaw main SubIFD. *exposure* adds an EXIF IFD
    with ExposureTime, as a camera's LinearRaw DNG (DxO, Lightroom Enhance) keeps."""
    extra = _IFD0_TAGS + ([(_EXIF_PLACEHOLDER, "I", 1, 0, True)] if exposure else [])
    with tifffile.TiffWriter(path) as tw:
        tw.write(np.zeros((8, 12, 3), np.uint8), photometric="rgb", subfiletype=1, subifds=1, extratags=extra)
        tw.write(np.full((16, 24, 3), 20000, np.uint16), photometric=34892, subfiletype=0, planarconfig="contig")
        if exposure:
            tw.write(np.zeros((1, 1), np.uint8), photometric="minisblack", extratags=[(33434, "2I", 1, (1, 125), True)])
    if exposure:
        # tifffile refuses to write ExifTag, so a placeholder entry becomes it, pointed at the spare IFD.
        with tifffile.TiffFile(path) as tif:
            tag, target, order = tif.pages[0].tags[_EXIF_PLACEHOLDER], tif.pages[1].offset, tif.byteorder
        with open(path, "r+b") as f:
            f.seek(tag.offset)
            f.write(struct.pack(order + "H", 34665))
            f.seek(tag.valueoffset)
            f.write(struct.pack(order + "I", target))


@pytest.mark.parametrize(
    "write, scanned",
    [
        (lambda p: _write_scanner_dng(p), True),
        (lambda p: _write_scanner_dng(p, exposure=True), False),
        (lambda p: _write_adobe_style_dng(p), False),
        (lambda p: _write_adobe_style_dng(p, preview_subfiletype=0), False),
    ],
    ids=["scanner", "camera-linearraw", "camera-cfa", "camera-cfa-with-linearraw"],
)
def test_a_scanner_dng_is_a_linearraw_main_image_without_camera_exposure(tmp_path, write, scanned):
    from negpy.infrastructure.loaders.helpers import is_scanned_dng
    from negpy.infrastructure.loaders.source_kind import is_scanned_positive_source

    path = tmp_path / "frame.dng"
    write(path)
    is_scanned_positive_source.cache_clear()
    assert is_scanned_dng(str(path)) is scanned
    assert is_scanned_positive_source(str(path)) is scanned
    is_scanned_positive_source.cache_clear()


def test_a_scanner_dng_slide_starts_positive(tmp_path):
    from dataclasses import replace

    from negpy.domain.models import WorkspaceConfig
    from negpy.features.process.models import ProcessConfig, ProcessMode
    from negpy.infrastructure.loaders.source_kind import is_scanned_positive_source, with_scanner_positive_default

    scan, camera = tmp_path / "scan.dng", tmp_path / "camera.dng"
    _write_scanner_dng(scan)
    _write_adobe_style_dng(camera)
    slide = replace(WorkspaceConfig(), process=ProcessConfig(process_mode=ProcessMode.E6))
    is_scanned_positive_source.cache_clear()
    assert with_scanner_positive_default(slide, str(scan)).process.positive_source is True
    assert with_scanner_positive_default(slide, str(camera)).process.positive_source is False
    is_scanned_positive_source.cache_clear()


class _Libraw(rawpy.RawPy):
    camera_whitebalance = [0.0, 1.0, 0.0, 0.0]


class _LibrawAsShot(rawpy.RawPy):
    camera_whitebalance = [2.0, 1.0, 1.5, 0.0]


@pytest.mark.parametrize(
    "raw, wanted, expected",
    [
        (_Libraw, True, False),
        (_LibrawAsShot, True, True),
        (_LibrawAsShot, False, False),
        (NonStandardFileWrapper, True, True),
    ],
    ids=["libraw-no-as-shot", "libraw-as-shot", "not-wanted", "wrapper"],
)
def test_a_file_without_as_shot_white_balance_decodes_neutral(raw, wanted, expected):
    """libraw answers camera WB on a file with none with its auto white balance."""
    from negpy.infrastructure.loaders.helpers import decodes_camera_wb

    assert decodes_camera_wb(raw(), wanted) is expected


def test_a_scanner_dng_never_reads_its_white_balance_tag(tmp_path):
    """A scanner DNG decodes 1,1,1 and folds nothing, whatever AsShotNeutral it carries."""
    from negpy.infrastructure.loaders.helpers import camera_wb_multipliers, decodes_camera_wb

    scan, camera = tmp_path / "scan.dng", tmp_path / "camera.dng"
    _write_scanner_dng(scan)
    _write_scanner_dng(camera, exposure=True)
    raw = _LibrawAsShot()
    assert camera_wb_multipliers(raw, str(scan)) is None
    assert decodes_camera_wb(raw, True, str(scan)) is False
    assert camera_wb_multipliers(raw, str(camera)) == [2.0, 1.0, 1.5]
    assert decodes_camera_wb(raw, True, str(camera)) is True


def test_exposure_tags_in_the_ifd_itself_mark_a_camera(tmp_path):
    from negpy.infrastructure.loaders.helpers import is_scanned_dng

    path = tmp_path / "camera.dng"
    with tifffile.TiffWriter(path) as tw:
        tw.write(
            np.zeros((8, 12, 3), np.uint8),
            photometric="rgb",
            subfiletype=1,
            subifds=1,
            extratags=_IFD0_TAGS + [(33434, "2I", 1, (1, 125), True)],
        )
        tw.write(np.full((16, 24, 3), 20000, np.uint16), photometric=34892, subfiletype=0, planarconfig="contig")
    assert is_scanned_dng(str(path)) is False


def test_a_scanner_dng_read_while_still_written_is_read_again(tmp_path):
    from negpy.infrastructure.loaders.source_kind import is_scanned_positive_source

    whole = tmp_path / "whole.dng"
    _write_scanner_dng(whole)
    path = tmp_path / "frame.dng"
    path.write_bytes(whole.read_bytes()[:64])
    is_scanned_positive_source.cache_clear()
    assert is_scanned_positive_source(str(path)) is False
    path.write_bytes(whole.read_bytes())
    assert is_scanned_positive_source(str(path)) is True
    is_scanned_positive_source.cache_clear()


def test_a_wrapper_of_a_scanner_dng_takes_no_white_balance(tmp_path):
    from negpy.infrastructure.loaders.helpers import decodes_camera_wb

    scan = tmp_path / "scan.dng"
    _write_scanner_dng(scan)
    assert decodes_camera_wb(NonStandardFileWrapper(wb_gains=(2.0, 1.0, 1.5)), True, str(scan)) is False
