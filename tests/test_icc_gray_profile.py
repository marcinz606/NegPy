from negpy.domain.models import ColorSpace
from negpy.infrastructure.display.color_spaces import ColorSpaceRegistry
from negpy.infrastructure.display.icc_profile import is_gray_profile


def test_the_greyscale_profile_is_gray_and_srgb_is_not():
    assert is_gray_profile(ColorSpaceRegistry.get_icc_path(ColorSpace.GREYSCALE.value))
    assert not is_gray_profile(ColorSpaceRegistry.get_icc_path(ColorSpace.SRGB.value))


def test_a_missing_file_is_not_gray(tmp_path):
    assert not is_gray_profile(str(tmp_path / "gone.icc"))
