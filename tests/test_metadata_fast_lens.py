"""A lens faster than f/1.0 keeps the rest of the export's EXIF."""

import piexif

from negpy.features.metadata.payload import ExifWriteFlags, MetadataPayload
from negpy.features.metadata.writer import _build_custom_exif


def test_a_lens_faster_than_f1_still_dumps_its_exif():
    payload = MetadataPayload(
        lens_model="Noct 50mm", max_aperture=0.95, camera_make="Nikon", exif_flags=ExifWriteFlags(camera=True, lens=True)
    )
    exif = _build_custom_exif(payload)

    assert piexif.dump(exif)
    assert piexif.ExifIFD.MaxApertureValue not in exif["Exif"]
    assert exif["Exif"][piexif.ExifIFD.FNumber]
