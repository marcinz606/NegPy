"""Hash pins of the embedded lens maps on fixed inputs."""

from core_pin import digest, linux_pin

from negpy.features.lens.models import LensCorrections, LensMetadata
from negpy.features.lens.warps import PanasonicWarp, RectilinearWarp, SonyWarp

_BARREL = (1.0, -0.06, 0.012, -0.002, 0.0004, -0.0003)
_RED = (1.0006, -0.0598, 0.0119, -0.002, 0.0004, -0.0003)
_BLUE = (0.9993, -0.0603, 0.0121, -0.002, 0.0004, -0.0003)
_SHAPE = (120, 170, 3)


@linux_pin
def test_rectilinear_maps_pin():
    cases = [
        (RectilinearWarp((_BARREL,)), LensMetadata("DNG", fill_scale=0.93), LensCorrections(True, True), 0),
        (
            RectilinearWarp((_RED, _BARREL, _BLUE), (0.51, 0.49)),
            LensMetadata("DNG", active_area=(3, 5, 118, 166)),
            LensCorrections(True, True),
            2,
        ),
        (RectilinearWarp((_RED, _BARREL, _BLUE)), LensMetadata("DNG"), LensCorrections(False, True), 0),
    ]
    got = [warp.remap(lens, _SHAPE, 10, 50, channel, corrections) for warp, lens, corrections, channel in cases]
    assert digest(got) == "a56df0bc766a67736226cdd2870b26742d895152edc7a0713f6a0ee6f416511f"


@linux_pin
def test_panasonic_and_sony_maps_pin():
    lens = LensMetadata("x", fill_scale=0.97)
    pana = PanasonicWarp(0.9, 0.04, -0.01, 0.002).remap(lens, _SHAPE, 5, 40, 1)
    knots = tuple(float(v) for v in range(-60, 100, 10))
    sony = SonyWarp(knots, tuple(float(v) for v in range(16)), ()).remap(lens, _SHAPE, 5, 40, 0)
    assert digest(pana, sony) == "c42cfec4df0fc66865221cc2d7155440f5d66a4c791e21e7228e540bf1f00cb1"
