"""Spectral crosstalk (dye unmix): applied to raw NEGATIVE densities before
bounds analysis and the stretch (Beer–Lambert domain), via ProcessConfig.
Replaces the old Lab-stage positive-domain op; legacy edits are migrated.
"""

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.features.exposure.normalization import resolve_crosstalk_matrix, unmix_log_image
from negpy.features.exposure.processor import NormalizationProcessor
from negpy.features.process.models import ProcessConfig, ProcessMode

_MATRIX = (1.0, -0.05, -0.02, -0.04, 1.0, -0.08, -0.01, -0.1, 1.0)


def _film(rng=None):
    """Synthetic negative: gray scene under a per-channel mask offset (in log space)."""
    rng = rng or np.random.default_rng(0)
    scene = rng.uniform(0.5, 2.0, (64, 64, 1)).astype(np.float32)  # density-ish, shared by channels
    mask = np.array([0.9, 0.55, 0.3], dtype=np.float32).reshape(1, 1, 3)  # orange-mask transmittance
    return np.clip(np.power(10.0, -scene) * mask, 1e-5, 1.0).astype(np.float32)


def _normalize(img, config):
    ctx = PipelineContext(scale_factor=1.0, original_size=img.shape[:2], process_mode=ProcessMode.C41)
    return NormalizationProcessor(config).process(img, ctx)


def test_strength_zero_is_identity():
    assert resolve_crosstalk_matrix(0.0, _MATRIX) is None
    img = _film()
    base = _normalize(img, ProcessConfig())
    off = _normalize(img, ProcessConfig(crosstalk_strength=0.0, crosstalk_matrix=_MATRIX))
    np.testing.assert_array_equal(base, off)


def test_matrix_rows_normalized():
    m = resolve_crosstalk_matrix(0.7, _MATRIX)
    assert m is not None
    np.testing.assert_allclose(m.sum(axis=1), 1.0, atol=1e-9)


def test_gray_film_stays_gray_under_unmix():
    """A gray scene under a pure per-channel mask offset: row-normalized rows sum
    to 1, so the shared scene term passes through, the mask offsets are re-derived
    by the per-channel stretch, and grays stay gray. Overall brightness may shift
    by the (mean − median) recombination term — bounded, not a cast."""
    img = _film()
    base = _normalize(img, ProcessConfig())
    unmixed = _normalize(img, ProcessConfig(crosstalk_strength=1.0, crosstalk_matrix=_MATRIX))
    # Gray preserved exactly: all channels of the unmixed result agree.
    np.testing.assert_allclose(unmixed[..., 0], unmixed[..., 1], atol=1e-4)
    np.testing.assert_allclose(unmixed[..., 0], unmixed[..., 2], atol=1e-4)
    # And the tone itself moves at most marginally vs the un-unmixed render.
    np.testing.assert_allclose(base, unmixed, atol=0.03)


def test_unmix_changes_coloured_content():
    rng = np.random.default_rng(1)
    img = np.clip(rng.uniform(0.01, 0.9, (64, 64, 3)), 1e-5, 1.0).astype(np.float32)
    base = _normalize(img, ProcessConfig())
    unmixed = _normalize(img, ProcessConfig(crosstalk_strength=1.0, crosstalk_matrix=_MATRIX))
    assert float(np.max(np.abs(base - unmixed))) > 1e-3


def test_legacy_lab_separation_migrates_to_process():
    """Old edits stored Lab color_separation (1.0–2.0) + crosstalk_matrix/profile
    under LabConfig; the flat namespace re-routes matrix/profile to ProcessConfig
    by field membership and from_flat_dict maps the slider to strength 0–1."""
    from negpy.domain.models import WorkspaceConfig

    d = WorkspaceConfig().to_dict()
    d.pop("crosstalk_strength", None)
    d["color_separation"] = 1.7
    d["crosstalk_matrix"] = list(_MATRIX)
    d["crosstalk_profile"] = "Portra 400"
    d["DEFAULT_MATRIX"] = list(_MATRIX)  # old LabConfig serialized field
    cfg = WorkspaceConfig.from_flat_dict(d)
    assert abs(cfg.process.crosstalk_strength - 0.7) < 1e-9
    assert cfg.process.crosstalk_matrix == _MATRIX
    assert cfg.process.crosstalk_profile == "Portra 400"
    # New-format round trip is stable (no double migration).
    again = WorkspaceConfig.from_flat_dict(cfg.to_dict())
    assert again.process.crosstalk_strength == cfg.process.crosstalk_strength


def test_unmix_log_image_matches_matrix():
    m = resolve_crosstalk_matrix(0.5, _MATRIX)
    log = np.random.default_rng(2).uniform(-3.0, 0.0, (8, 8, 3)).astype(np.float32)
    out = unmix_log_image(log, m)
    expected = np.einsum("hwc,kc->hwk", log, m.astype(np.float32))
    np.testing.assert_allclose(out, expected, atol=1e-6)
    assert unmix_log_image(log, None) is log
