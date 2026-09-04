import csv
import os
import unittest

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.exposure.processor import NormalizationProcessor, PhotometricProcessor
from negpy.features.process.models import ProcessMode

LG2 = np.log10(2.0)
MID_GREY_DENSITY = 0.74  # 18% reflector below a diffuse white
STRIP = np.round(np.arange(0.0, 2.01, 0.05), 2)  # scene densities of the embedded grey scale
PATCH = 8
GRID = 48
PORTRA_DMIN = (0.2, 0.6, 0.85)  # Status M, E-4050
ENVELOPE_A = 1.0  # reflection print (Buhr, US 5,528,339)
ENVELOPE_SLOPE = 0.35
TOL = 0.05
TAIL = 1.5 * LG2
TAIL_FRACTION = 0.03
TOE_ONSET = 1.75  # print density where the paper toe starts to roll the gradient off


def _c41_negative_densities(stops: np.ndarray, gamma: float = 0.55) -> np.ndarray:
    """
    C-41 densities for scene exposures (in stops around 18% gray), with a film
    toe: linear gamma above -2.5 stops, compressing below.
    """
    d = np.where(stops >= -2.5, gamma * stops * LG2, gamma * (-2.5) * LG2 + 0.32 * (stops + 2.5) * LG2)
    return np.where(stops >= -3.5, d, gamma * (-2.5) * LG2 - 0.32 * LG2 + 0.15 * (stops + 3.5) * LG2)


def _scene_densities(spread: float) -> tuple[np.ndarray, float]:
    """
    Patch grid of scene densities: log-uniform over `spread` about mid-grey (the
    meter's assumption that a scene integrates to grey), with a few specular and
    deep-shadow patches beyond it and the grey strip laid into the top row.
    Returns the grid and its deepest textural scene density.
    """
    rng = np.random.default_rng(7)
    below = above = spread / 2.0
    d = rng.uniform(MID_GREY_DENSITY - above, MID_GREY_DENSITY + below, size=(GRID, GRID))
    # Speculars and deep shadows: a few patches reaching TAIL beyond the textural range.
    tails = rng.random(d.shape) < TAIL_FRACTION
    d[tails] = rng.uniform(MID_GREY_DENSITY - above - TAIL, MID_GREY_DENSITY + below + TAIL, size=int(tails.sum()))
    d[0, :] = MID_GREY_DENSITY
    d[0, : len(STRIP)] = STRIP
    return d, MID_GREY_DENSITY + below


def _negative_scan(scene: np.ndarray) -> np.ndarray:
    stops = (MID_GREY_DENSITY - scene) / LG2
    above_base = _c41_negative_densities(stops) - _c41_negative_densities(np.array(-4.5))
    layers = [10.0 ** -(above_base + dmin) for dmin in PORTRA_DMIN]
    img = np.stack(layers, axis=-1).astype(np.float32)
    return np.repeat(np.repeat(img, PATCH, axis=0), PATCH, axis=1)


def _print_densities(spread: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the default Normalization + Print path; return (D_scene, D_print, deepest scene density)."""
    scene, deepest = _scene_densities(spread)
    img = _negative_scan(scene)
    config = WorkspaceConfig()
    ctx = PipelineContext(scale_factor=1.0, original_size=img.shape[:2], process_mode=ProcessMode.C41)
    norm = NormalizationProcessor(config.process).process(img, ctx)
    out = PhotometricProcessor(config.exposure).process(norm, ctx)
    t_b = 10.0 ** -float(EXPOSURE_CONSTANTS["d_max"])
    half = PATCH // 2
    lin = np.array([float(out[half, i * PATCH + half, 1]) for i in range(len(STRIP))])
    d_print = -np.log10(lin * (1.0 - t_b) + t_b)
    if os.environ.get("NEGPY_TONE_REPORT"):
        m = ctx.metrics
        print(
            f"spread={spread} range={m['norm_density_range']:.3f} textural={m['textural_range']:.3f} "
            f"anchor={m['metered_anchor']:.3f} slope={m['print_slopes'][1]:.3f}"
        )
    return STRIP, d_print, deepest


def _report(name: str, d_s: np.ndarray, d_p: np.ndarray, gamma: np.ndarray) -> None:
    path = os.environ.get("NEGPY_TONE_REPORT")
    if not path:
        return
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        for a, b, g in zip(d_s, d_p, gamma):
            w.writerow([name, f"{a:.2f}", f"{b:.3f}", f"{g:.3f}"])


class TestToneReproduction(unittest.TestCase):
    """
    Jones-diagram check of the default conversion against Kodak's preferred
    reproduction for reflection prints (Buhr, US 5,528,339): instantaneous
    system gamma between A and A + 0.35 * D_scene over scene densities 0.6 to
    1.45, rising monotonically toward the shadows (no mid-tone bell), and deep
    shadow reaching near paper black. The envelope is asserted for scenes whose
    range is at least the patent's 1.5 classification threshold; the flat scene
    only has to be bell-free. Mid-grey placement is not
    asserted: it depends on where a negative's bounds sit around
    `assumed_anchor`, an empirical prior this synthetic cannot supply.
    """

    SPREADS = {"flat": 1.2, "normal": 2.2, "contrasty": 2.8}
    ENVELOPE_MIN_SPREAD = 1.5

    @classmethod
    def setUpClass(cls):
        cls.runs = {}
        for name, spread in cls.SPREADS.items():
            d_s, d_p, deepest = _print_densities(spread)
            gamma = np.gradient(d_p, d_s)
            cls.runs[name] = (d_s, d_p, gamma, deepest)
            _report(name, d_s, d_p, gamma)

    def test_gamma_inside_preferred_envelope(self):
        for name, (d_s, _, gamma, deepest) in self.runs.items():
            if self.SPREADS[name] < self.ENVELOPE_MIN_SPREAD:
                continue
            with self.subTest(scene=name):
                band = (d_s >= 0.6) & (d_s <= min(1.45, deepest))
                lo = ENVELOPE_A - TOL
                hi = ENVELOPE_A + ENVELOPE_SLOPE * d_s[band] + TOL
                self.assertTrue(np.all(gamma[band] >= lo), f"gamma below {lo}: {gamma[band].round(3)}")
                self.assertTrue(np.all(gamma[band] <= hi), f"gamma above envelope: {gamma[band].round(3)} > {hi.round(3)}")

    def test_gamma_has_no_midtone_bell(self):
        for name, (d_s, d_p, gamma, deepest) in self.runs.items():
            with self.subTest(scene=name):
                band = (d_s >= 0.3) & (d_s <= min(1.45, deepest)) & (d_p < TOE_ONSET)
                drops = np.diff(gamma[band])
                self.assertGreaterEqual(float(drops.min()), -TOL, f"gamma falls toward shadows: {gamma[band].round(3)}")

    def test_textural_shadow_reaches_black(self):
        for name, (d_s, d_p, _, deepest) in self.runs.items():
            with self.subTest(scene=name):
                at_deepest = float(np.interp(deepest, d_s, d_p))
                self.assertGreaterEqual(at_deepest, EXPOSURE_CONSTANTS["shadow_reach_density"] - 0.1)

    def test_deep_shadow_reaches_near_paper_black(self):
        for name, (_, d_p, _, _) in self.runs.items():
            with self.subTest(scene=name):
                self.assertGreaterEqual(float(d_p[-1]), TOE_ONSET)
                self.assertLessEqual(float(d_p[-1]), float(EXPOSURE_CONSTANTS["d_max"]))


if __name__ == "__main__":
    unittest.main()
