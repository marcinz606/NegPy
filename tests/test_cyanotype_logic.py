import numpy as np

from negpy.features.altprocess.models import Sensitizer
from negpy.features.cyanotype.logic import apply_cyanotype, sensitizer_constants
from negpy.kernel.image.logic import rgb_to_lab_working


def _ramp(d0: np.ndarray) -> np.ndarray:
    """(1, len(d0), 3) grey image whose columns sit at the given print densities."""
    lin = (10.0**-d0).astype(np.float32)
    return np.stack([lin] * 3, axis=-1)[None, :, :]


def _density(img: np.ndarray) -> np.ndarray:
    return -np.log10(np.clip(img.mean(axis=-1), 1e-6, 1.0))


def _cyano(img, **kw):
    kw.setdefault("enabled", True)
    return apply_cyanotype(img, **kw)


def _lab(img):
    return rgb_to_lab_working(img.astype(np.float32))


class TestTonalCurve:
    def test_disabled_returns_the_same_object(self):
        img = _ramp(np.linspace(0.0, 2.0, 16, dtype=np.float32))
        assert apply_cyanotype(img) is img

    def test_monotone_in_input_density(self):
        d0 = np.linspace(0.0, 3.0, 256, dtype=np.float32)
        out = _density(_cyano(_ramp(d0))[0])
        assert np.all(np.diff(out) >= -1e-5)

    def test_exposure_scale_sets_where_the_print_clips(self):
        """The scale is the negative density range the sensitiser prints: a short one
        runs out of scale early and blocks everything past it together."""
        d0 = np.linspace(0.0, 2.6, 512, dtype=np.float32)
        short = _density(_cyano(_ramp(d0), scale=1.0)[0])
        long = _density(_cyano(_ramp(d0), scale=2.4)[0])
        assert np.ptp(short[d0 > 1.2]) < 1e-4
        assert np.ptp(long[d0 > 1.2]) > 0.1

    def test_midtones_are_compressed_against_the_ends(self):
        """Cyanotype flattens the middle of the scale — the reason a correction curve
        for a cyanotype digital negative expands it again."""
        scale = 2.0
        d0 = np.linspace(0.0, scale, 401, dtype=np.float32)
        out = _density(_cyano(_ramp(d0), scale=scale)[0])
        grad = np.gradient(out, d0)
        mid = grad[180:220].mean()
        ends = np.concatenate([grad[40:80], grad[320:360]]).mean()
        assert 0.0 < mid < ends

    def test_more_exposure_darkens_the_whole_print(self):
        d0 = np.linspace(0.0, 1.5, 64, dtype=np.float32)
        base = _density(_cyano(_ramp(d0))[0])
        more = _density(_cyano(_ramp(d0), exposure=2.0)[0])
        assert np.all(more >= base - 1e-5)
        assert more.mean() > base.mean() + 0.05


class TestColour:
    def test_shadows_are_blue_and_highlights_carry_the_green_stain(self):
        d0 = np.linspace(0.0, 2.0, 256, dtype=np.float32)
        lab = _lab(_cyano(_ramp(d0))[0])
        shadow = lab[-1]
        highlight = lab[8]
        assert shadow[2] < -15.0, "Dmax must read as blue, not black"
        assert shadow[1] < 0.0
        assert highlight[2] > 0.0 and highlight[1] < 0.0, "residual sensitiser prints green"

    def test_new_sensitizer_goes_deeper_than_classic(self):
        d0 = np.linspace(0.0, 2.0, 64, dtype=np.float32)
        classic = _density(_cyano(_ramp(d0), sensitizer=Sensitizer.CLASSIC)[0])
        new = _density(_cyano(_ramp(d0), sensitizer=Sensitizer.NEW)[0])
        assert new.max() > classic.max() + 0.3
        assert sensitizer_constants(Sensitizer.NEW)["d_max"] > sensitizer_constants(Sensitizer.CLASSIC)["d_max"]

    def test_the_print_never_reaches_black(self):
        """No silver: Prussian blue tops out well short of a paper Dmax."""
        out = _cyano(_ramp(np.array([4.0], dtype=np.float32)), sensitizer=Sensitizer.NEW)
        assert _lab(out[0])[0, 0] > 15.0


class TestToning:
    def test_bleach_clears_the_highlights_first(self):
        d0 = np.linspace(0.0, 2.0, 256, dtype=np.float32)
        plain = _density(_cyano(_ramp(d0))[0])
        bleached = _density(_cyano(_ramp(d0), bleach=1.0)[0])
        assert np.all(bleached <= plain + 1e-5)
        assert bleached[32] < 0.03, "highlights wash back to bare paper"
        assert bleached[-1] > 0.10, "the deepest shadow keeps pigment"

    def test_tannin_turns_the_print_brown_and_a_little_deeper(self):
        d0 = np.linspace(0.0, 2.0, 64, dtype=np.float32)
        plain = _cyano(_ramp(d0))[0]
        toned = _cyano(_ramp(d0), tannin=1.0)[0]
        assert _lab(toned)[-1, 2] > 0.0, "iron tannate is warm, not blue"
        assert _lab(toned)[-1, 1] > _lab(plain)[-1, 1]
        assert _density(toned).max() > _density(plain).max()

    def test_partial_tannin_leaves_a_split(self):
        d0 = np.linspace(0.0, 2.0, 256, dtype=np.float32)
        lab = _lab(_cyano(_ramp(d0), bleach=0.4, tannin=0.4)[0])
        assert lab[16, 2] > lab[-1, 2], "highlights warmer than the shadows"
