from negpy.desktop.view.sidebar.flatfield import _FILE_FILTER
from negpy.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS


def test_reference_picker_offers_every_supported_format():
    patterns = set(_FILE_FILTER.split(";;")[0].split("(")[1].rstrip(")").split())
    for ext in SUPPORTED_RAW_EXTENSIONS:
        assert f"*{ext}" in patterns
        assert f"*{ext.upper()}" in patterns
