"""Package-safe Nikon Adobe RGB profile used by exact C-41 positives.

Nikon Scan's LS-5000 C-41 CMS-on TIFFs embed this exact 492-byte profile.
Keeping the tiny payload in source avoids a filesystem-relative runtime asset
that can disappear from a wheel or frozen desktop app. The decoded bytes are
validated before use; exact output must fail closed if packaging or source
corruption changes either their size or identity.
"""

from __future__ import annotations

import base64
import hashlib
from functools import lru_cache
from typing import Final


NIKON_ADOBE_RGB_PROFILE_NAME: Final = "Nikon Adobe RGB 4.0.0.3000"
NIKON_ADOBE_RGB_PROFILE_BYTES: Final = 492
NIKON_ADOBE_RGB_PROFILE_SHA256: Final = "a8d0d753bd6129357cc2647435ce675e8637a679eb526fa180fba460874ce1d3"

_NIKON_ADOBE_RGB_PROFILE_BASE64: Final = (
    "AAAB7E5LT04CIAAAbW50clJHQiBYWVogB88ADAAHABIAOwAWYWNzcEFQUEwAAAAA"
    "bm9uZQAAAAEAAAAAAAAAAAAAAAAAAPbWAAEAAAAA0y0AAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJZGVzYwAAAPAAAABN"
    "clhZWgAAAUAAAAAUZ1hZWgAAAVQAAAAUYlhZWgAAAWgAAAAUclRSQwAAAXwAAAAO"
    "Z1RSQwAAAYwAAAAOYlRSQwAAAZwAAAAOd3RwdAAAAawAAAAUY3BydAAAAcAAAAAs"
    "ZGVzYwAAAAAAAAAbTmlrb24gQWRvYmUgUkdCIDQuMC4wLjMwMDAAAAAAAAAAAAAA"
    "ABtOaWtvbiBBZG9iZSBSR0IgNC4wLjAuMzAwMAAAAABYWVogAAAAAAAAnBkAAE+m"
    "AAAE/FhZWiAAAAAAAAA0iwAAoCsAAA+VWFlaIAAAAAAAACYyAAAQLwAAvqBjdXJ2"
    "AAAAAAAAAAECMwAAY3VydgAAAAAAAAABAjMAAGN1cnYAAAAAAAAAAQIzAABYWVog"
    "AAAAAAAA81QAAQAAAAEWz3RleHQAAAAATmlrb24gSW5jLiAmIE5pa29uIENvcnBv"
    "cmF0aW9uIDIwMDEA"
)


class NikonICCProfileError(RuntimeError):
    """The source-embedded exact-output profile failed identity checks."""


@lru_cache(maxsize=1)
def nikon_adobe_rgb_profile() -> bytes:
    """Return the immutable, identity-checked NKAdobe ICC payload."""

    try:
        profile = base64.b64decode(_NIKON_ADOBE_RGB_PROFILE_BASE64, validate=True)
    except ValueError as error:
        raise NikonICCProfileError("embedded Nikon Adobe RGB profile is not valid base64") from error
    if len(profile) != NIKON_ADOBE_RGB_PROFILE_BYTES:
        raise NikonICCProfileError(f"embedded Nikon Adobe RGB profile is {len(profile)} bytes, expected {NIKON_ADOBE_RGB_PROFILE_BYTES}")
    digest = hashlib.sha256(profile).hexdigest()
    if digest != NIKON_ADOBE_RGB_PROFILE_SHA256:
        raise NikonICCProfileError("embedded Nikon Adobe RGB profile does not match its pinned SHA-256")
    return profile


def profile_receipt_binding() -> dict[str, str | int]:
    """Receipt fields for the profile after reusing the validation boundary."""

    nikon_adobe_rgb_profile()
    return {
        "name": NIKON_ADOBE_RGB_PROFILE_NAME,
        "bytes": NIKON_ADOBE_RGB_PROFILE_BYTES,
        "sha256": NIKON_ADOBE_RGB_PROFILE_SHA256,
    }
