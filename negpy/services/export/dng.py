"""DNG 1.4 LinearRaw writer for the rendered frame.

IFD0 holds a small RGB thumbnail plus the DNG-wide tags; one SubIFD holds the
full-resolution image. The integer variant (default) writes uncompressed
16-bit, which every LinearRaw reader decodes, LibRaw included. The float
variant writes float16 with Deflate and the floating-point predictor, the
pairing Adobe's own float DNGs use — Adobe readers only: LibRaw builds
without the DNG SDK cannot decode floating-point DNG at all, so NegPy reads
the float variant back through its own tifffile path.
"""

import io
import struct
from fractions import Fraction
from typing import Any, Dict, Optional

import cv2
import numpy as np
import tifffile

from negpy.features.metadata.resolution import Resolution
from negpy.kernel.image.logic import _WORKING_TO_XYZ, working_oetf_encode

# The working space's primaries operator, shared with the pipeline's Lab conversions.
_ADOBE_RGB_TO_XYZ_D65 = np.asarray(_WORKING_TO_XYZ, dtype=np.float64)

# ColorMatrix1 maps XYZ *under CalibrationIlluminant1* (D65) to "camera" space —
# here the working space — per the DNG spec; the reader owns any adaptation to its
# connection space. With AsShotNeutral = (1,1,1) the solved as-shot illuminant is
# D65 itself, so As Shot reads near 6500 K and lands at the file's rendered balance.
_COLOR_MATRIX_1 = np.linalg.inv(_ADOBE_RGB_TO_XYZ_D65)

_CALIBRATION_ILLUMINANT_D65 = 21
_THUMBNAIL_LONG_EDGE = 256

# Tag codes from the DNG 1.4 specification.
_TAG_DNG_VERSION = 50706
_TAG_DNG_BACKWARD_VERSION = 50707
_TAG_UNIQUE_CAMERA_MODEL = 50708
_TAG_BLACK_LEVEL = 50714
_TAG_WHITE_LEVEL = 50717
_TAG_DEFAULT_CROP_ORIGIN = 50719
_TAG_DEFAULT_CROP_SIZE = 50720
_TAG_COLOR_MATRIX_1 = 50721
_TAG_AS_SHOT_NEUTRAL = 50728
_TAG_BASELINE_EXPOSURE = 50730
_TAG_CALIBRATION_ILLUMINANT_1 = 50778
_TAG_ACTIVE_AREA = 50829
_TAG_ORIENTATION = 274
_TAG_PROFILE_NAME = 50936
_TAG_PROFILE_TONE_CURVE = 50940
_TAG_PROFILE_EMBED_POLICY = 50941

UNIQUE_CAMERA_MODEL = "NegPy Linear RGB"


def _srational(value: float, denominator: int = 1000000) -> tuple[int, int]:
    frac = Fraction(value).limit_denominator(denominator)
    return frac.numerator, frac.denominator


def _matrix_rationals(matrix: np.ndarray) -> tuple[int, ...]:
    flat: list[int] = []
    for value in matrix.reshape(-1):
        flat.extend(_srational(float(value)))
    return tuple(flat)


def _thumbnail(linear: np.ndarray) -> np.ndarray:
    h, w = linear.shape[:2]
    scale = _THUMBNAIL_LONG_EDGE / max(h, w)
    tw, th = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(linear, (tw, th), interpolation=cv2.INTER_AREA)
    encoded = np.asarray(working_oetf_encode(np.clip(small, 0.0, 1.0)))
    return (encoded * 255.0 + 0.5).astype(np.uint8)


def encode_dng(
    linear: np.ndarray,
    *,
    float_variant: bool = False,
    resolution: Optional[Resolution] = None,
    meta_kwargs: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Encode a scene-linear float32 RGB buffer in [0, 1] as DNG bytes.

    ``meta_kwargs`` is ``tiff_metadata_kwargs(...)`` output: its description,
    software and extratags ride on IFD0 beside the DNG tags.
    """
    arr = np.ascontiguousarray(linear, dtype=np.float32)
    h, w = arr.shape[:2]

    ifd0_tags: list[tuple] = [
        (_TAG_DNG_VERSION, 1, 4, bytes([1, 4, 0, 0]), False),
        (_TAG_DNG_BACKWARD_VERSION, 1, 4, bytes([1, 4, 0, 0]) if float_variant else bytes([1, 1, 0, 0]), False),
        (_TAG_UNIQUE_CAMERA_MODEL, 2, 0, UNIQUE_CAMERA_MODEL, False),
        (_TAG_COLOR_MATRIX_1, 10, 9, _matrix_rationals(_COLOR_MATRIX_1), False),
        (_TAG_AS_SHOT_NEUTRAL, 5, 3, (1, 1, 1, 1, 1, 1), False),
        (_TAG_BASELINE_EXPOSURE, 10, 1, (0, 1), False),
        (_TAG_CALIBRATION_ILLUMINANT_1, 3, 1, _CALIBRATION_ILLUMINANT_D65, False),
        # An embedded profile with an identity tone curve: Adobe readers then default
        # to "Embedded" instead of Adobe Color's base curve, so the file opens looking
        # like the render. Their other profiles stay selectable.
        (_TAG_PROFILE_NAME, 2, 0, "NegPy Linear", False),
        (_TAG_PROFILE_TONE_CURVE, 11, 4, (0.0, 0.0, 1.0, 1.0), False),
        (_TAG_PROFILE_EMBED_POLICY, 4, 1, 0, False),
    ]
    ifd0_tags.append((_TAG_ORIENTATION, 3, 1, 1, False))
    description = ""
    software = "NegPy"
    if meta_kwargs:
        description = meta_kwargs.get("description") or ""
        software = meta_kwargs.get("software") or software
        # Geometry is baked into the pixels, so a source Orientation (protect mode
        # keeps it) must not ride along: a raw editor would rotate the upright render.
        ifd0_tags.extend(t for t in (meta_kwargs.get("extratags") or []) if t[0] != _TAG_ORIENTATION)

    sub_tags: list[tuple] = [
        (_TAG_DEFAULT_CROP_ORIGIN, 4, 2, (0, 0), False),
        (_TAG_DEFAULT_CROP_SIZE, 4, 2, (w, h), False),
        (_TAG_ACTIVE_AREA, 4, 4, (0, 0, h, w), False),
    ]
    if float_variant:
        main = arr.astype(np.float16)
        compression: Optional[str] = "zlib"
        # Predictor 3 (floating-point): byte-plane split + differencing ahead of
        # Deflate, the pairing Adobe's own float DNGs use.
        predictor: Optional[int] = 3
    else:
        main = (np.clip(arr, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
        compression = None
        predictor = None
        sub_tags.append((_TAG_BLACK_LEVEL, 5, 3, (0, 1, 0, 1, 0, 1), False))
        sub_tags.append((_TAG_WHITE_LEVEL, 4, 3, (65535, 65535, 65535), False))

    page_kwargs: Dict[str, Any] = {}
    if resolution is not None:
        page_kwargs["resolution"] = (resolution.x, resolution.y)
        page_kwargs["resolutionunit"] = resolution.unit

    buf = io.BytesIO()
    with tifffile.TiffWriter(buf, byteorder="<") as writer:
        writer.write(
            _thumbnail(arr),
            photometric="rgb",
            compression=None,
            subifds=1,
            subfiletype=1,
            description=description,
            software=software,
            metadata=None,
            extratags=ifd0_tags,
            **page_kwargs,
        )
        # Written as RGB: tifffile then infers three color samples with no
        # ExtraSamples. Under photometric LINEAR_RAW it instead writes one color
        # sample plus two UNSPECIFIED extras, which contradicts the 3x3
        # ColorMatrix1. The tag is patched to LinearRaw below.
        writer.write(
            main,
            photometric="rgb",
            compression=compression,
            predictor=predictor,
            subfiletype=0,
            software=software,
            metadata=None,
            extratags=sub_tags,
            **page_kwargs,
        )
    return _patch_photometric_linear_raw(buf)


def _subifd_page(tf: "tifffile.TiffFile") -> Any:
    pages = tf.pages[0].pages
    if not pages:
        raise ValueError("DNG is missing its SubIFD page")
    return pages[0]


def _patch_photometric_linear_raw(buf: io.BytesIO) -> bytes:
    """Rewrite the SubIFD's PhotometricInterpretation from RGB to LinearRaw in place.
    Works on the writer's own BytesIO so a multi-hundred-MB export is not copied
    three extra times."""
    buf.seek(0)
    with tifffile.TiffFile(buf) as tf:
        offset = _subifd_page(tf).tags[262].valueoffset
    view = buf.getbuffer()
    struct.pack_into("<H", view, offset, 34892)
    del view
    buf.seek(0)
    with tifffile.TiffFile(buf) as tf:
        page = _subifd_page(tf)
        if int(page.photometric) != 34892 or page.samplesperpixel != 3 or page.tags.get(338) is not None:
            raise ValueError("DNG SubIFD layout is wrong after the LinearRaw patch")
    return buf.getvalue()
