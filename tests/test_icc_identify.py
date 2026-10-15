"""NegPy recognizes the profiles it embeds in its own exports."""

import pytest

from negpy.infrastructure.loaders import helpers


@pytest.mark.parametrize(
    "name,expected",
    [("AdobeCompat-v4", "ADOBE_RGB"), ("ProPhoto-v4", "PROPHOTO"), ("Rec2020-v4", "REC2020"), ("DisplayP3-v4", "P3_D65")],
)
def test_the_bundled_profiles_are_recognized(name, expected):
    import os

    from negpy.domain.models import ColorSpace

    path = os.path.join(os.path.dirname(__file__), "..", "icc", f"{name}.icc")
    with open(path, "rb") as fh:
        assert helpers.identify_color_space_from_icc(fh.read()) == ColorSpace[expected].value
