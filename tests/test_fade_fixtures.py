"""Synthetic fade fixtures: a forward model and a scene, built independently of
`resolve_fade_matrix`, to validate that matrix and to make test images with known ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def forward_fade_matrix(ratio_g: float, ratio_b: float, ratio_r: float = 1.0) -> np.ndarray:
    """F = diag(ratio_r, ratio_r*ratio_g, ratio_r*ratio_b), applied to a scan's density. It is the
    identity at unit ratios. `ratio_g` and `ratio_b` are ratios to red, so the absolute survivals
    are `ratio_r` times each."""
    return np.diag([ratio_r, ratio_r * ratio_g, ratio_r * ratio_b])


def clean_slide(size: int = 256, seed: int = 0) -> np.ndarray:
    """An unfaded slide scan: a full-range luma gradient plus smooth per-channel chroma
    variation, not a flat gray wedge. Returns linear RGB in (0, 1]."""
    yy, xx = (np.mgrid[0:size, 0:size].astype(np.float64)) / size
    t = xx  # bright (left) to dark (right), full E-6 transfer range

    def field(seed_offset: int) -> np.ndarray:
        rng2 = np.random.default_rng(seed + seed_offset)
        f = np.zeros((size, size))
        for _ in range(4):
            fx, fy = rng2.uniform(0.5, 3.0, 2)
            phase = rng2.uniform(0, 2 * np.pi)
            amp = rng2.uniform(0.3, 1.0) / 4
            f += amp * np.sin(2 * np.pi * (fx * xx + fy * yy) + phase)
        return f

    span = 2.9  # TRANSFER_DENSITY_RANGE
    chroma_amp = 0.08
    log_r = -(t * span) + chroma_amp * field(1)
    log_g = -(t * span) + chroma_amp * field(2)
    log_b = -(t * span) + chroma_amp * field(3)
    image = np.stack([10.0**log_r, 10.0**log_g, 10.0**log_b], axis=-1)
    return np.clip(image, 1e-6, 1.0).astype(np.float64)


@dataclass
class Fixture:
    name: str
    ratio_g: float
    ratio_b: float
    stain: tuple  # additive offset in img_log units, applied after the matrix
    clean: np.ndarray
    faded_log: np.ndarray
    ratio_r: float = 1.0


def make_fixture(name: str, ratio_g: float, ratio_b: float, stain: tuple, seed: int = 0, ratio_r: float = 1.0) -> Fixture:
    clean = clean_slide(seed=seed)
    clean_log = np.log10(clean)
    forward = forward_fade_matrix(ratio_g, ratio_b, ratio_r=ratio_r)
    faded_log = clean_log @ forward.T + np.asarray(stain)
    return Fixture(name=name, ratio_g=ratio_g, ratio_b=ratio_b, stain=stain, clean=clean, faded_log=faded_log, ratio_r=ratio_r)


def fixture_set() -> list[Fixture]:
    z = (0.0, 0.0, 0.0)
    return [
        make_fixture("control (no fade, no stain)", 1.0, 1.0, z, seed=0),
        make_fixture("mild fade, no stain", 0.85, 0.9, z, seed=1),
        make_fixture("heavy fade, no stain", 0.35, 0.5, z, seed=2),
        make_fixture("mild fade, heavy stain", 0.85, 0.9, (0.15, 0.05, -0.10), seed=3),
        make_fixture("differential fade, with stain", 0.4, 0.95, (0.08, -0.03, 0.06), seed=4),
    ]


# --------------------------------------------------------------------------- invariants


def test_forward_matrix_is_identity_at_unity_ratios():
    assert np.allclose(forward_fade_matrix(1.0, 1.0), np.eye(3), atol=1e-12)


def test_control_fixture_matches_clean_reference():
    control = fixture_set()[0]
    assert np.allclose(control.faded_log, np.log10(control.clean), atol=1e-12)


def test_restoration_inverts_the_forward_matrix():
    from negpy.features.exposure.normalization import resolve_fade_matrix

    for ratio_r, ratio_g, ratio_b in ((1.0, 0.85, 0.9), (1.0, 0.35, 0.5), (0.25, 0.4, 0.95)):
        restore = resolve_fade_matrix(ratio_r, ratio_g, ratio_b)
        assert np.allclose(restore @ forward_fade_matrix(ratio_g, ratio_b, ratio_r=ratio_r), np.eye(3), atol=1e-10)


def test_ratio_r_below_one_recovers_density_a_pinned_reference_channel_cannot():
    """Pinning ratio_r at 1 restores the colour balance but leaves every channel at too low a
    density when red did fade; the true ratio_r restores it exactly."""
    true_ratio_r, ratio_g, ratio_b = 0.25, 0.75, 0.53
    fx = make_fixture("faded reference channel", ratio_g, ratio_b, (0.0, 0.0, 0.0), seed=7, ratio_r=true_ratio_r)
    clean_log = np.log10(fx.clean)

    forward = forward_fade_matrix(ratio_g, ratio_b, ratio_r=true_ratio_r)
    restore_full = np.linalg.inv(forward)
    restored_full_log = fx.faded_log @ restore_full.T
    err_full = np.max(np.abs(restored_full_log - clean_log))

    restore_pinned = np.linalg.inv(forward_fade_matrix(ratio_g, ratio_b, ratio_r=1.0))
    restored_pinned_log = fx.faded_log @ restore_pinned.T
    err_pinned = np.max(np.abs(restored_pinned_log - clean_log))

    assert err_full < 1e-10  # exact recovery given the true ratio_r
    assert err_pinned > 1.0  # pinning ratio_r = 1 leaves a real, large density error


def test_fixture_set_covers_stained_and_unstained_cases():
    names = [fx.name for fx in fixture_set()]
    assert any("stain" in n and "no stain" not in n for n in names)
    assert any("no stain" in n for n in names)
