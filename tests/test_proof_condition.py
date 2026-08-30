"""The proof-condition controls.

The proof is a color table, not a pipeline stage: every setting here reaches the screen
only by changing the LUT `soft_proof_lut` bakes. So the tests assert on that table, which
is the artifact the canvas actually samples.
"""

import unittest

import numpy as np
from PIL import ImageCms

from negpy.domain.models import PROOF_INTENT_LABELS, ProofCondition, ProofIntent
from negpy.services.rendering.image_processor import GAMUT_WARNING_COLOR, ImageProcessor

# A printer-class CMYK profile, so the paper-simulation branch is the one under test.
# Bundled display profiles take the gamut-only branch and would exercise none of this.
_CMYK = "/usr/share/ghostscript/iccprofiles/default_cmyk.icc"
_SIZE = 17  # a coarse grid: these assertions are about which node moved, not about accuracy


def _have_cmyk() -> bool:
    try:
        return ImageProcessor._is_print_profile(ImageCms.getOpenProfile(_CMYK))
    except Exception:
        return False


def _lut(**kw):
    return ImageProcessor.soft_proof_lut("Adobe RGB", None, _CMYK, None, _SIZE, **kw)


class TestProofCondition(unittest.TestCase):
    def test_defaults_are_what_the_proof_did_before_it_had_controls(self):
        c = ProofCondition()
        self.assertEqual(c.intent, ProofIntent.RELATIVE_COLORIMETRIC.value)
        self.assertTrue(c.black_point)
        self.assertTrue(c.paper_white)
        self.assertFalse(c.ink_black)
        self.assertFalse(c.gamut_warning)

    def test_keeps_the_positions_the_old_pair_had(self):
        """`get_display_lut` and `get_gamut_lut` index the first two entries."""
        c = ProofCondition("in.icc", "out.icc")
        self.assertEqual((c[0], c[1]), ("in.icc", "out.icc"))

    def test_is_hashable_because_it_is_a_cache_key(self):
        self.assertEqual(hash(ProofCondition("a", "b")), hash(ProofCondition("a", "b")))
        self.assertNotEqual(ProofCondition("a", "b"), ProofCondition("a", "b", ProofIntent.PERCEPTUAL.value))

    def test_every_intent_has_a_label(self):
        self.assertEqual(set(PROOF_INTENT_LABELS), {i.value for i in ProofIntent})


class TestNonePresetBaseline(unittest.TestCase):
    """The None preset has to be a fixed point: applying it must land on a set-up the
    preset box then recognises as None, or picking it blanks the box it just filled."""

    def test_a_fresh_session_is_the_baseline(self):
        from negpy.desktop.session import AppState

        st = AppState()
        self.assertTrue(st.soft_proof_enabled)
        self.assertIsNone(st.proof_icc_path)
        self.assertEqual(st.proof_intent, ProofIntent.RELATIVE_COLORIMETRIC.value)
        for field in ("proof_black_point", "proof_paper_white", "proof_ink_black", "proof_gamut_warning"):
            self.assertFalse(getattr(st, field), field)


@unittest.skipUnless(_have_cmyk(), "no printer-class profile available")
class TestProofLut(unittest.TestCase):
    def setUp(self):
        self.base = _lut()

    def test_defaults_reproduce_the_unparameterised_call(self):
        """The no-regression case: an existing session must look exactly as it did."""
        plain = ImageProcessor.soft_proof_lut("Adobe RGB", None, _CMYK, None, _SIZE)
        np.testing.assert_array_equal(plain, self.base)

    def test_paper_white_off_restores_a_pure_white(self):
        white = self.base[-1, -1, -1]
        self.assertLess(float(white.max()), 0.99, "simulated paper white is dimmer than screen white")
        off = _lut(paper_white=False)[-1, -1, -1]
        self.assertGreater(float(off.min()), 0.99, "without the simulation white is screen white again")

    def test_paper_white_is_tinted_not_merely_dimmer(self):
        """A paper has a color cast, which is half of why a proof looks different."""
        white = self.base[-1, -1, -1]
        self.assertGreater(float(white.max() - white.min()), 0.005)

    def test_ink_black_moves_the_black_node(self):
        self.assertNotAlmostEqual(float(self.base[0, 0, 0].mean()), float(_lut(ink_black=True)[0, 0, 0].mean()), places=6)

    def test_ink_black_and_dropping_bpc_are_the_same_lever(self):
        np.testing.assert_allclose(_lut(ink_black=True), _lut(black_point=False), atol=1e-6)

    def test_each_intent_gives_its_own_table(self):
        tables = {i.value: _lut(intent=i.value) for i in ProofIntent}
        self.assertFalse(np.allclose(tables["relative"], tables["perceptual"]))

    def test_an_unknown_intent_falls_back_to_relative(self):
        np.testing.assert_array_equal(_lut(intent="nonsense"), _lut(intent="relative"))


@unittest.skipUnless(_have_cmyk(), "no printer-class profile available")
class TestGamutWarning(unittest.TestCase):
    def test_marks_exactly_the_nodes_the_gamut_mask_flags(self):
        warned = _lut(gamut_warning=True)
        plain = _lut()
        mask = ImageProcessor.gamut_lut("Adobe RGB", None, _CMYK, size=_SIZE)
        self.assertIsNotNone(mask)
        self.assertTrue(mask.any(), "the fixture profile must clip something for this to mean anything")
        np.testing.assert_allclose(warned[mask], float(GAMUT_WARNING_COLOR))
        np.testing.assert_array_equal(warned[~mask], plain[~mask])

    def test_off_by_default(self):
        np.testing.assert_array_equal(_lut(gamut_warning=False), _lut())


if __name__ == "__main__":
    unittest.main()
