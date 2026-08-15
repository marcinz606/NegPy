"""Build Adobe XMP packets from resolved metadata payloads."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from negpy.features.metadata.payload import MetadataPayload

_XMP_BEGIN = '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>'
_XMP_END = '<?xpacket end="w"?>'

_NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "negpy": "https://negpy.app/ns/1.0/",
}

for prefix, uri in _NS.items():
    ET.register_namespace(prefix, uri)


def _to_rational(value: float) -> str:
    num = round(value * 1000.0)
    return f"{num}/1000"


def _sub(parent: ET.Element, prefix: str, tag: str, text: str) -> None:
    ET.SubElement(parent, f"{{{_NS[prefix]}}}{tag}").text = text


def build_xmp_xml(payload: MetadataPayload, *, standalone: bool = True) -> str:
    """Build a standards-compliant XMP packet string."""
    root = ET.Element(f"{{{_NS['x']}}}xmpmeta")
    rdf = ET.SubElement(root, f"{{{_NS['rdf']}}}RDF")
    desc = ET.SubElement(rdf, f"{{{_NS['rdf']}}}Description")
    desc.set(f"{{{_NS['rdf']}}}about", "")

    for prefix in ("dc", "negpy"):
        desc.set(f"xmlns:{prefix}", _NS[prefix])

    # negpy namespace: the original film capture. A structured mirror, with standard EXIF
    # when flagged.
    if payload.camera_make:
        _sub(desc, "negpy", "CaptureCameraMake", payload.camera_make)
    if payload.camera_model:
        _sub(desc, "negpy", "CaptureCameraModel", payload.camera_model)
    if payload.lens_make:
        _sub(desc, "negpy", "CaptureLensMake", payload.lens_make)
    if payload.lens_model:
        _sub(desc, "negpy", "CaptureLensModel", payload.lens_model)
    if payload.focal_length_mm is not None:
        _sub(desc, "negpy", "CaptureFocalLength", _to_rational(payload.focal_length_mm))
    if payload.max_aperture is not None:
        _sub(desc, "negpy", "CaptureMaxAperture", _to_rational(payload.max_aperture))
    if payload.capture_exposure:
        _sub(desc, "negpy", "CaptureExposure", payload.capture_exposure)
    if payload.iso is not None:
        _sub(desc, "negpy", "CaptureFilmISO", str(payload.iso))
    if payload.film_stock:
        _sub(desc, "negpy", "CaptureFilmStock", payload.film_stock)
    if payload.film_manufacturer:
        _sub(desc, "negpy", "CaptureFilmManufacturer", payload.film_manufacturer)
    if payload.film_format:
        _sub(desc, "negpy", "CaptureFilmFormat", payload.film_format)
    if payload.film_color_type:
        _sub(desc, "negpy", "CaptureFilmColorType", payload.film_color_type)
    if payload.developer:
        _sub(desc, "negpy", "Developer", payload.developer)
    if payload.push_pull and payload.push_pull != "Normal":
        _sub(desc, "negpy", "PushPull", payload.push_pull)
    if payload.notes:
        _sub(desc, "negpy", "Notes", payload.notes)
    if payload.scan_method:
        _sub(desc, "negpy", "ScanMethod", payload.scan_method)
    if payload.capture_roll:
        _sub(desc, "negpy", "CaptureRoll", payload.capture_roll)
    if payload.capture_frame is not None:
        _sub(desc, "negpy", "CaptureFrame", str(payload.capture_frame))

    # Digitization rig, always from the source snapshot
    if payload.scan_camera_make:
        _sub(desc, "negpy", "ScanCameraMake", payload.scan_camera_make)
    if payload.scan_camera_model:
        _sub(desc, "negpy", "ScanCameraModel", payload.scan_camera_model)
    if payload.scan_lens_make:
        _sub(desc, "negpy", "ScanLensMake", payload.scan_lens_make)
    if payload.scan_lens_model:
        _sub(desc, "negpy", "ScanLensModel", payload.scan_lens_model)
    if payload.scan_focal_length_mm is not None:
        _sub(desc, "negpy", "ScanFocalLength", _to_rational(payload.scan_focal_length_mm))
    if payload.scan_aperture is not None:
        _sub(desc, "negpy", "ScanAperture", _to_rational(payload.scan_aperture))
    if payload.scan_exposure:
        _sub(desc, "negpy", "ScanExposure", payload.scan_exposure)
    if payload.scan_iso is not None:
        _sub(desc, "negpy", "ScanISO", str(payload.scan_iso))

    keywords: list[str] = []
    for val in (payload.film_stock, payload.film_manufacturer, payload.film_format, payload.film_color_type):
        if val and val not in keywords:
            keywords.append(val)
    if keywords:
        subject = ET.SubElement(desc, f"{{{_NS['dc']}}}subject")
        bag = ET.SubElement(subject, f"{{{_NS['rdf']}}}Bag")
        for kw in keywords:
            li = ET.SubElement(bag, f"{{{_NS['rdf']}}}li")
            li.text = kw

    body = ET.tostring(root, encoding="unicode", xml_declaration=False)
    if standalone:
        return f"{_XMP_BEGIN}\n{body}\n{_XMP_END}"
    return body


def build_xmp_bytes(payload: MetadataPayload, *, standalone: bool = True) -> bytes:
    return build_xmp_xml(payload, standalone=standalone).encode("utf-8")
