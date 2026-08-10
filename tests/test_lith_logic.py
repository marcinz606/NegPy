import numpy as np

from negpy.features.exposure.papers import PAPER_PROFILES, resolve_paper
from negpy.features.lith.logic import LITH_CONSTANTS, apply_lith
from negpy.kernel.image.logic import rgb_to_lab_working

D_MAX = 2.0
FOMATONE = resolve_paper("foma_fomatone")


def _ramp(d0: np.ndarray) -> np.ndarray:
    """(1, len(d0), 3) grey image whose columns sit at the given print densities."""
    lin = (10.0**-d0).astype(np.float32)
    return np.stack([lin] * 3, axis=-1)[None, :, :]


def _density(img: np.ndarray) -> np.ndarray:
    return -np.log10(np.clip(img.mean(axis=-1), 1e-6, 1.0))


def _lith(img, **kw):
    kw.setdefault("enabled", True)
    return apply_lith(img, FOMATONE.lith_path, D_MAX, **kw)


class TestTonalCurve:
    def test_disabled_returns_the_same_object(self):
        img = _ramp(np.linspace(0.0, 2.0, 16, dtype=np.float32))
        assert apply_lith(img, FOMATONE.lith_path, D_MAX) is img

    def test_monotone_in_input_density(self):
        d0 = np.linspace(0.0, 2.5, 256, dtype=np.float32)
        out = _density(_lith(_ramp(d0), exposure=2.0)[0])
        assert np.all(np.diff(out) >= -1e-5)

    def test_two_branches_with_a_flat_band_between_them(self):
        """A long low-gamma foot, a near-vertical knee, then no separation at all."""
        d0 = np.linspace(0.0, 2.5, 512, dtype=np.float32)
        out = _density(_lith(_ramp(d0), exposure=2.0, abruptness=1.0)[0])
        grad = np.diff(out) / np.diff(d0)

        knee = int(np.argmax(grad))
        foot = grad[: knee // 2]
        band = grad[int(knee + 0.25 * (len(grad) - knee)) :]

        assert foot.max() < 0.6, "highlight branch should be low gamma"
        assert foot.mean() < 0.5, "and low over its whole length (published range 0.2-0.5)"
        assert grad[knee] > 10.0, "the knee should be a cliff, not a slope"
        assert band.max() < 0.05, "past the knee the lith band carries no tone"

    def test_over_exposure_veils_the_highlights_without_fogging_them(self):
        """Lith highlights carry tone, not bare paper: the default +2 stops puts
        a veil in them. Exposure 0 still prints clean white, so the slider keeps
        that end of the range."""
        white = _ramp(np.zeros(4, dtype=np.float32))
        assert _density(_lith(white, exposure=0.0)[0]).max() < 0.10
        veil = _density(_lith(white, exposure=2.0)[0]).max()
        assert 0.15 < veil < 0.35, f"paper white at +2 stops sits at {veil:.3f}"

    def test_highlights_keep_separation_at_the_default(self):
        """The failure this guards: too little veil and the whole highlight range
        collapses onto paper white and reads blown."""
        d0 = np.array([0.0, 0.1, 0.2, 0.3, 0.5], dtype=np.float32)
        spread = np.ptp(_density(_lith(_ramp(d0), exposure=2.0)[0]))
        assert spread > 0.15, f"highlight branch spans only {spread:.3f} D"

    def test_later_snatch_widens_the_black_band(self):
        d0 = np.linspace(0.0, 2.5, 256, dtype=np.float32)
        img = _ramp(d0)
        black = [(_density(_lith(img, exposure=2.0, snatch=s)[0]) > 1.9).sum() for s in (0.3, 0.55, 0.8)]
        assert black[0] < black[1] < black[2]

    def test_more_exposure_softens_the_gradation(self):
        d0 = np.linspace(0.0, 2.0, 256, dtype=np.float32)
        img = _ramp(d0)
        spans = [np.ptp(_density(_lith(img, exposure=e, snatch=0.9)[0])) for e in (0.0, 3.0)]
        assert spans[1] < spans[0]


class TestColour:
    def _lab(self, d0, paper=FOMATONE):
        out = apply_lith(_ramp(d0), paper.lith_path, D_MAX, enabled=True, exposure=2.0)
        return rgb_to_lab_working(out)[0]

    def test_hue_path_runs_peach_then_ochre_then_olive_then_neutral(self):
        dens = _density(_lith(_ramp(np.linspace(0.0, 2.5, 400, dtype=np.float32)), exposure=2.0)[0])
        lab = self._lab(np.linspace(0.0, 2.5, 400, dtype=np.float32))
        u = dens / D_MAX

        def at(target):
            return lab[int(np.argmin(np.abs(u - target)))]

        peach, olive, black = at(0.10), at(0.65), at(1.0)
        assert peach[1] > 5.0 and peach[2] > 10.0, "warm highlights"
        assert olive[1] < -2.0 and olive[2] > 5.0, "green transition zone"
        assert abs(black[1]) < 3.0 and abs(black[2]) < 3.0, "blacks come back to neutral"

    def test_papers_have_distinct_lith_character(self):
        d0 = np.linspace(0.0, 2.5, 200, dtype=np.float32)
        chroma = {}
        for key in ("foma_fomatone", "foma_fomabrom", "ilford_mg_rc"):
            lab = self._lab(d0, paper=resolve_paper(key))
            chroma[key] = float(np.hypot(lab[:, 1], lab[:, 2]).mean())
        # Ilford Multigrade resists lith; Fomatone is the colourful one.
        assert chroma["ilford_mg_rc"] < chroma["foma_fomabrom"] < chroma["foma_fomatone"]

    def test_every_bw_profile_declares_a_lith_path(self):
        for key, paper in PAPER_PROFILES.items():
            assert len(paper.lith_path) == len(LITH_CONSTANTS["path_u"]), key
