"""Exact Nikon output-profile identity and fail-closed tests."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO

import pytest
from PIL import ImageCms

from negpy.services.roll import nikon_icc


def test_embedded_profile_is_the_pinned_nikon_scan_rgb_profile() -> None:
    profile = nikon_icc.nikon_adobe_rgb_profile()

    assert len(profile) == 492
    assert int.from_bytes(profile[:4], "big") == len(profile)
    assert profile[12:16] == b"mntr"
    assert profile[16:20] == b"RGB "
    assert profile[20:24] == b"XYZ "
    assert profile[36:40] == b"acsp"
    assert hashlib.sha256(profile).hexdigest() == (
        "a8d0d753bd6129357cc2647435ce675e8637a679eb526fa180fba460874ce1d3"
    )
    parsed = ImageCms.ImageCmsProfile(BytesIO(profile))
    assert ImageCms.getProfileName(parsed).strip() == "Nikon Adobe RGB 4.0.0.3000"
    assert ImageCms.getProfileCopyright(parsed).strip() == (
        "Nikon Inc. & Nikon Corporation 2001"
    )
    assert nikon_icc.profile_receipt_binding() == {
        "bytes": 492,
        "name": "Nikon Adobe RGB 4.0.0.3000",
        "sha256": "a8d0d753bd6129357cc2647435ce675e8637a679eb526fa180fba460874ce1d3",
    }


def test_embedded_profile_one_byte_change_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = bytearray(nikon_icc.nikon_adobe_rgb_profile())
    profile[100] ^= 1
    monkeypatch.setattr(
        nikon_icc,
        "_NIKON_ADOBE_RGB_PROFILE_BASE64",
        base64.b64encode(profile).decode("ascii"),
    )
    nikon_icc.nikon_adobe_rgb_profile.cache_clear()

    with pytest.raises(nikon_icc.NikonICCProfileError, match="pinned SHA-256"):
        nikon_icc.nikon_adobe_rgb_profile()
