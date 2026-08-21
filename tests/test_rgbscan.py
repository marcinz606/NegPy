import os
from dataclasses import replace

import numpy as np
import pytest

from negpy.domain.models import WorkspaceConfig
import cv2

from negpy.features.rgbscan.logic import (
    BLUE,
    GREEN,
    RED,
    estimate_shift,
    assemble_rgb,
    classify_channel,
    frame_affinity,
    capture_ordered,
    group_triplets,
    merge_rgb_triplet,
    probe_channel_means,
    rgbscan_token,
    triplet_affinity,
)
from negpy.features.rgbscan.models import RgbScanConfig

_SAMPLES = {"DSC00448.ARW": 0, "DSC00449.ARW": 1, "DSC00450.ARW": 2}  # file -> expected channel


def test_classify_channel_dominant():
    assert classify_channel([500, 80, 20]) == 0
    assert classify_channel([76, 710, 173]) == 1
    assert classify_channel([40, 360, 867]) == 2


def test_merge_picks_matching_channel():
    # Each "exposure" is dominant in its own channel; merge must pick the right one.
    red = np.zeros((2, 2, 3), np.float32)
    green = np.zeros((2, 2, 3), np.float32)
    blue = np.zeros((2, 2, 3), np.float32)
    red[..., 0] = 1.0
    green[..., 1] = 2.0
    blue[..., 2] = 3.0
    decode = {"r": red, "g": green, "b": blue}.__getitem__
    out = merge_rgb_triplet(decode, "r", "g", "b", align=False)
    assert np.all(out[..., 0] == 1.0)
    assert np.all(out[..., 1] == 2.0)
    assert np.all(out[..., 2] == 3.0)


def test_merge_rejects_shape_mismatch():
    decode = {"r": np.zeros((2, 2, 3)), "g": np.zeros((3, 2, 3)), "b": np.zeros((2, 2, 3))}.__getitem__
    with pytest.raises(ValueError):
        merge_rgb_triplet(decode, "r", "g", "b", align=False)


def _texture(h=128, w=128, seed=0):
    """Textured scene with enough edges for phase correlation to lock."""
    rng = np.random.default_rng(seed)
    base = rng.random((h, w), dtype=np.float32)
    return cv2.GaussianBlur(base, (0, 0), 1.5)


def _shift(img, dx, dy):
    m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def test_estimate_shift_recovers_subpixel_translation():
    ref = _texture()
    dx, dy = 2.5, -1.25
    mov = _shift(ref, dx, dy)
    est_dx, est_dy = estimate_shift(ref, mov)
    assert abs(est_dx - dx) < 0.3
    assert abs(est_dy - dy) < 0.3


def _scene_triplet(dx, dy):
    base = _texture()
    r = np.zeros((*base.shape, 3), np.float32)
    g = np.zeros_like(r)
    b = np.zeros_like(r)
    r[..., 0] = base
    g[..., 1] = _shift(base, dx, dy)  # green drifted relative to red
    b[..., 2] = _shift(base, -dx, dy)
    return base, r, g, b


def test_assemble_rgb_alignment_reduces_misregistration():
    base, r, g, b = _scene_triplet(3.0, 2.0)
    aligned = assemble_rgb(r, g, b, align=True)
    raw = assemble_rgb(r, g, b, align=False)
    # Ignore the warp border where REPLICATE/REFLECT differ; compare the interior.
    sl = (slice(8, -8), slice(8, -8))
    err_aligned = np.abs(aligned[..., 1][sl] - base[sl]).mean()
    err_raw = np.abs(raw[..., 1][sl] - base[sl]).mean()
    assert err_aligned < err_raw * 0.5


def test_assemble_rgb_no_align_is_plain_stack():
    base, r, g, b = _scene_triplet(3.0, 2.0)
    out = assemble_rgb(r, g, b, align=False)
    assert np.array_equal(out[..., 0], r[..., 0])
    assert np.array_equal(out[..., 1], g[..., 1])
    assert np.array_equal(out[..., 2], b[..., 2])


def test_align_skips_implausible_shift():
    # max_shift guard: a wildly shifted exposure is left untouched rather than warped.
    base = _texture()
    r = np.zeros((*base.shape, 3), np.float32)
    g = np.zeros_like(r)
    r[..., 0] = base
    g[..., 1] = _shift(base, 60.0, 0.0)  # > max_shift (0.02*128 -> floored to 16)
    out = assemble_rgb(r, g, g, align=True)
    assert np.array_equal(out[..., 1], g[..., 1])


def test_rgbscan_token_changes_with_align(tmp_path):
    g = tmp_path / "g.raw"
    b = tmp_path / "b.raw"
    g.write_bytes(b"g")
    b.write_bytes(b"b")
    on = rgbscan_token(RgbScanConfig(enabled=True, green_path=str(g), blue_path=str(b), align=True))
    off = rgbscan_token(RgbScanConfig(enabled=True, green_path=str(g), blue_path=str(b), align=False))
    assert on != off


def test_group_triplets_scrambled_order():
    # Capture order G, R, B — classification must place them correctly regardless.
    items = [("f1", 1), ("f2", 0), ("f3", 2), ("f4", 2), ("f5", 0), ("f6", 1)]
    triplets = group_triplets(items)
    assert len(triplets) == 2
    assert (triplets[0].red, triplets[0].green, triplets[0].blue) == ("f2", "f1", "f3")
    assert triplets[0].ok
    assert (triplets[1].red, triplets[1].green, triplets[1].blue) == ("f5", "f6", "f4")
    assert triplets[1].ok


def _scene_sig(seed, channel_gain=1.0, h=48, w=72):
    """A z-scored scene signature, as _scene_signature would produce.

    ``seed`` is the scene: two exposures of one frame share it. ``channel_gain`` stands
    in for a narrowband channel seeing the same scene at a different strength.
    """
    rng = np.random.default_rng(seed)
    scene = cv2.GaussianBlur(rng.random((h, w), dtype=np.float32), (0, 0), 1.5) * channel_gain
    gy, gx = np.gradient(scene)
    mag = np.hypot(gx, gy)
    mag = mag - mag.mean()
    sd = float(mag.std())
    return (mag / sd if sd > 0 else mag).astype(np.float32)


def _triplet_sigs(scene_seed):
    """One frame's three exposures: same scene, different channel strengths."""
    return {
        f"r{scene_seed}": _scene_sig(scene_seed, 1.0),
        f"g{scene_seed}": _scene_sig(scene_seed, 0.6),
        f"b{scene_seed}": _scene_sig(scene_seed, 0.3),
    }


def test_frame_affinity_separates_same_scene_from_different():
    same = frame_affinity(_scene_sig(1, 1.0), _scene_sig(1, 0.4))
    different = frame_affinity(_scene_sig(1), _scene_sig(2))
    assert same > 0.9
    assert different < 0.3


def test_frame_affinity_handles_missing_and_mismatched():
    assert frame_affinity(None, _scene_sig(1)) == 0.0
    assert frame_affinity(_scene_sig(1, h=48), _scene_sig(1, h=32)) == 0.0


def test_group_triplets_rejects_chunk_built_from_two_frames():
    """The off-by-one that a membership-only check cannot see: a shifted list still
    yields one of each channel, taken from two different frames."""
    sigs = {**_triplet_sigs(1), **_triplet_sigs(2)}
    # Chunk holds frame 1's red and green with frame 2's blue: one of each channel.
    items = [("r1", RED), ("g1", GREEN), ("b2", BLUE)]

    assert group_triplets(items)[0].ok, "membership alone accepts the mispairing"
    assert not group_triplets(items, sigs)[0].ok


def test_group_triplets_accepts_a_genuine_triplet():
    sigs = _triplet_sigs(7)
    items = [("r7", RED), ("g7", GREEN), ("b7", BLUE)]
    triplet = group_triplets(items, sigs)[0]
    assert triplet.ok
    assert (triplet.red, triplet.green, triplet.blue) == ("r7", "g7", "b7")


def test_group_triplets_keeps_membership_verdict_when_signatures_missing():
    """A file whose probe failed must not take the chunk down with it."""
    sigs = {"r1": _scene_sig(1), "g1": _scene_sig(1, 0.6)}  # no signature for b1
    items = [("r1", RED), ("g1", GREEN), ("b1", BLUE)]
    assert group_triplets(items, sigs)[0].ok


def test_triplet_affinity_is_the_weakest_pair():
    sigs = {**_triplet_sigs(3), "b9": _scene_sig(9)}
    good = group_triplets([("r3", RED), ("g3", GREEN), ("b3", BLUE)], sigs)[0]
    intruded = group_triplets([("r3", RED), ("g3", GREEN), ("b9", BLUE)], sigs)[0]
    assert triplet_affinity(good, sigs) > triplet_affinity(intruded, sigs)


def test_mutual_best_rejects_a_chunk_whose_partner_is_elsewhere():
    """The case a floor alone cannot catch: a chunk that scores respectably, while the
    member's real partner sits elsewhere in the folder and scores better."""
    sigs = {**_triplet_sigs(1), **_triplet_sigs(2)}
    # Frame 2's blue is a passable match for frame 1 -- but frame 1's own blue is better.
    sigs["b2"] = _scene_sig(1, 0.3) * 0.55 + _scene_sig(2, 0.3) * 0.45
    items = [("r1", RED), ("g1", GREEN), ("b2", BLUE), ("r2", RED), ("g2", GREEN), ("b1", BLUE)]

    intruded = group_triplets(items[:3], {k: sigs[k] for k in ("r1", "g1", "b2")})[0]
    assert intruded.ok, "on its own the chunk clears the floor"

    assert not group_triplets(items, sigs)[0].ok, "b1 is the better blue for frame 1"


def test_floor_still_rejects_when_the_pool_is_too_small_to_rank():
    """One candidate per channel makes every member trivially its own best match, so
    the floor is all that stands between three unrelated files and a triplet."""
    sigs = {"r1": _scene_sig(1), "g2": _scene_sig(2, 0.6), "b3": _scene_sig(3, 0.3)}
    items = [("r1", RED), ("g2", GREEN), ("b3", BLUE)]
    assert not group_triplets(items, sigs)[0].ok


def test_mutual_best_accepts_a_genuine_triplet_among_many():
    """A real triplet must survive both tests with other frames present to rank against."""
    sigs = {}
    for seed in (1, 2, 3, 4):
        sigs.update(_triplet_sigs(seed))
    items = [(f"{c}{seed}", ch) for seed in (1, 2, 3, 4) for c, ch in (("r", RED), ("g", GREEN), ("b", BLUE))]
    triplets = group_triplets(items, sigs)
    assert len(triplets) == 4
    assert all(t.ok for t in triplets)


def test_affinity_lookup_matches_pairwise():
    """The matrix shortcut and the pairwise path must agree."""
    from negpy.features.rgbscan.logic import _affinity_lookup

    sigs = {**_triplet_sigs(1), **_triplet_sigs(2)}
    paths = sorted(sigs)
    lookup = _affinity_lookup(paths, sigs)
    for a in paths:
        for b in paths:
            assert lookup(a, b) == pytest.approx(frame_affinity(sigs[a], sigs[b]), abs=1e-5)


def test_affinity_lookup_falls_back_on_mixed_shapes():
    """Two cameras in one folder have no common matrix; pairwise still works."""
    from negpy.features.rgbscan.logic import _affinity_lookup

    sigs = {"a": _scene_sig(1, h=48, w=72), "b": _scene_sig(1, h=32, w=48)}
    lookup = _affinity_lookup(sorted(sigs), sigs)
    assert lookup("a", "b") == 0.0
    assert lookup("a", "a") == pytest.approx(1.0, abs=1e-5)


def test_capture_ordered_sorts_by_the_clock_not_the_name():
    """The reported bug: a constant frame token puts the color word above the only
    part of the name that changes, so the roll sorts by color instead of by frame."""
    names = ["F1_Blue_39", "F1_Blue_49", "F1_Green_39", "F1_Green_49", "F1_Red_39", "F1_Red_49"]
    times = {
        "F1_Red_39": "2026-08-18 15:33:13",
        "F1_Green_39": "2026-08-18 15:33:16",
        "F1_Blue_39": "2026-08-18 15:33:20",
        "F1_Red_49": "2026-08-18 15:37:17",
        "F1_Green_49": "2026-08-18 15:37:22",
        "F1_Blue_49": "2026-08-18 15:37:27",
    }
    assert capture_ordered(names, times) == [
        "F1_Red_39",
        "F1_Green_39",
        "F1_Blue_39",
        "F1_Red_49",
        "F1_Green_49",
        "F1_Blue_49",
    ]


def test_capture_ordered_keeps_given_order_when_a_stamp_is_missing():
    """All or nothing: one undated file must not interleave two orderings."""
    names = ["b", "a", "c"]
    assert capture_ordered(names, {"a": "2026-01-01 00:00:01", "b": "2026-01-01 00:00:02"}) == names
    assert capture_ordered(names, {}) == names


def test_capture_ordered_breaks_ties_on_filename():
    """A whole-second stamp cannot separate a burst, so the order stays deterministic."""
    tied = {"c": "2026-01-01 00:00:01", "a": "2026-01-01 00:00:01", "b": "2026-01-01 00:00:00"}
    assert capture_ordered(["c", "a", "b"], tied) == ["b", "a", "c"]


def test_looks_narrowband_separates_trichrome_from_ordinary_scans():
    """The dialog asks for opposite things in the two cases, so the split has to hold.
    Values are the measured shape: white light leaves the channels close, a narrowband
    exposure puts one several times ahead."""
    from negpy.features.rgbscan.logic import looks_narrowband

    white_light = [[136.3, 231.1, 272.0], [134.9, 237.6, 282.9], [140.8, 245.3, 289.3]]
    trichrome = [[168.0, 20.4, 5.3], [35.4, 308.5, 90.0], [6.5, 56.6, 169.3]]
    assert not looks_narrowband(white_light)
    assert looks_narrowband(trichrome)
    assert not looks_narrowband([])


def test_looks_narrowband_tolerates_odd_frames():
    """A median, so one blank or unusual frame does not flip a whole folder."""
    from negpy.features.rgbscan.logic import looks_narrowband

    mostly_trichrome = [[168.0, 20.4, 5.3], [35.4, 308.5, 90.0], [6.5, 56.6, 169.3], [100.0, 99.0, 98.0]]
    assert looks_narrowband(mostly_trichrome)


def test_nothing_matched_message_offers_to_turn_the_mode_off():
    """An ordinary folder needs the mode off, not a lesson in triplet requirements."""
    from negpy.desktop.workers.render import rgb_nothing_matched_message

    title, body = rgb_nothing_matched_message({"loose": 36, "narrowband": False, "by_time": True})
    assert "Turn RGB Scan off" in body
    assert "one frame at a time" not in body
    assert title == "Nothing to assemble"


def test_nothing_matched_message_states_the_capture_requirement():
    """The constraint that positional chunking really imposes, and that a reader would
    otherwise get wrong: the three exposures have to be consecutive."""
    from negpy.desktop.workers.render import rgb_nothing_matched_message

    _, body = rgb_nothing_matched_message({"loose": 36, "narrowband": True, "by_time": True})
    assert "taken back to back before you move on to the next frame" in body
    assert "filenames do not matter" in body


def test_nothing_matched_message_does_not_promise_filenames_are_free_without_times():
    """Undated files are ordered by name, so the message must not say names are ignored."""
    from negpy.desktop.workers.render import rgb_nothing_matched_message

    _, body = rgb_nothing_matched_message({"loose": 36, "narrowband": True, "by_time": False})
    assert "filenames do not matter" not in body
    assert "sort into the order the shots were taken" in body


def test_grouping_notice_is_silent_on_a_clean_folder():
    from negpy.desktop.workers.render import rgb_grouping_notice

    assert rgb_grouping_notice(12, 0, 0, 0, True) == ""


def test_grouping_notice_names_the_reason():
    """The two failures need different things from the user, so the message separates
    them: a folder without whole triplets, versus shots that could not be ordered."""
    from negpy.desktop.workers.render import rgb_grouping_notice

    incomplete = rgb_grouping_notice(10, 6, 2, 0, True)
    assert "2 sets not one of each color" in incomplete
    assert "showing different frames" not in incomplete

    mismatched = rgb_grouping_notice(10, 6, 0, 2, True)
    assert "2 sets showing different frames" in mismatched
    assert "not one of each color" not in mismatched


def test_grouping_notice_reports_the_filename_fallback():
    """Undated files are grouped by name, which the user should not have to infer."""
    from negpy.desktop.workers.render import rgb_grouping_notice

    assert "state no capture time" in rgb_grouping_notice(0, 50, 16, 1, False)
    assert "capture time" not in rgb_grouping_notice(0, 50, 16, 1, True)


def test_grouping_notice_omits_the_count_when_nothing_assembled():
    from negpy.desktop.workers.render import rgb_grouping_notice

    assert rgb_grouping_notice(0, 6, 2, 0, True).startswith("RGB Scan: 6 files left separate")


def test_group_triplets_flags_bad_chunks():
    # Trailing short chunk and a chunk with a duplicate channel are flagged.
    items = [("f1", 0), ("f2", 0), ("f3", 2), ("f4", 1)]
    triplets = group_triplets(items)
    assert len(triplets) == 2
    assert not triplets[0].ok  # two reds, no green
    assert not triplets[1].ok  # only one file


def test_rgbscan_token_disabled():
    assert rgbscan_token(RgbScanConfig()) == ""
    assert rgbscan_token(RgbScanConfig(enabled=True)) == ""  # no paths


def test_rgbscan_token_changes_with_files(tmp_path):
    g = tmp_path / "g.raw"
    b = tmp_path / "b.raw"
    g.write_bytes(b"g")
    b.write_bytes(b"b")
    cfg = RgbScanConfig(enabled=True, green_path=str(g), blue_path=str(b))
    tok = rgbscan_token(cfg)
    assert tok.startswith("|rgb:")
    assert str(g) in tok and str(b) in tok


@pytest.mark.parametrize("fname,expected", _SAMPLES.items())
def test_real_sample_classification(fname, expected):
    """Narrowband ARW samples classify unambiguously by dominant channel (no demosaic)."""
    path = os.path.join("samples", fname)
    if not os.path.exists(path):
        pytest.skip(f"sample {fname} not present")
    assert classify_channel(probe_channel_means(path)) == expected


def test_preview_merge_pulls_green_blue_from_their_files():
    """Preview path must merge the triplet, not show the red exposure alone (color, not gray)."""
    if not all(os.path.exists(os.path.join("samples", f)) for f in _SAMPLES):
        pytest.skip("samples not present")
    from negpy.services.rendering.preview_manager import PreviewManager

    pm = PreviewManager()
    r, g, b = (os.path.join("samples", f) for f in _SAMPLES)
    merged, _, _ = pm.load_linear_preview_rgb(r, RgbScanConfig(enabled=True, green_path=g, blue_path=b), "Adobe RGB", use_camera_wb=True)
    red_only, _, _ = pm.load_linear_preview(r, "Adobe RGB", use_camera_wb=True)
    # Red-only has near-zero G/B (narrowband); the merge fills them from the other shots.
    assert merged[..., 1].mean() > red_only[..., 1].mean() * 3
    assert merged[..., 2].mean() > red_only[..., 2].mean() * 3
    # Red channel is unchanged (comes from the same red file).
    assert abs(float(merged[..., 0].mean()) - float(red_only[..., 0].mean())) < 1e-3


def test_attach_restored_triplets_rebuilds_asset(tmp_path):
    """Session restore must rebuild a triplet from saved green/blue paths, not re-classify."""
    from negpy.desktop.workers.render import AssetDiscoveryWorker

    r = tmp_path / "DSC1.raw"
    g = tmp_path / "DSC2.raw"
    b = tmp_path / "DSC3.raw"
    for f in (r, g, b):
        f.write_bytes(b"x")
    assets = [{"name": "DSC1.raw", "path": str(r), "hash": "h"}]
    triplets = {str(r): [str(g), str(b)]}
    out = AssetDiscoveryWorker()._attach_restored_triplets(assets, triplets)
    assert out[0]["green_path"] == str(g)
    assert out[0]["blue_path"] == str(b)
    assert out[0]["name"].endswith("(RGB)")


def test_config_roundtrip_preserves_rgbscan():
    cfg = WorkspaceConfig()
    cfg = type(cfg)(**{**cfg.__dict__, "rgbscan": RgbScanConfig(enabled=True, green_path="/g", blue_path="/b")})
    restored = WorkspaceConfig.from_flat_dict(cfg.to_dict())
    assert restored.rgbscan == cfg.rgbscan


def test_resolve_asset_rgbscan_injects_from_asset():
    """Batch export must use the asset dict's own green/blue, overriding a stale DB config."""
    from negpy.desktop.session import resolve_asset_rgbscan

    stale = replace(WorkspaceConfig(), rgbscan=RgbScanConfig(enabled=True, green_path="/old_g", blue_path="/old_b"))
    asset = {"path": "/r", "green_path": "/g", "blue_path": "/b"}
    out = resolve_asset_rgbscan(stale, asset)
    assert out.rgbscan == RgbScanConfig(enabled=True, green_path="/g", blue_path="/b", align=True)


def test_resolve_asset_rgbscan_honors_align():
    from negpy.desktop.session import resolve_asset_rgbscan

    asset = {"green_path": "/g", "blue_path": "/b", "align": False}
    out = resolve_asset_rgbscan(WorkspaceConfig(), asset)
    assert out.rgbscan.align is False


def test_resolve_asset_rgbscan_resets_when_not_triplet():
    """A non-triplet frame must not inherit a leaked/enabled triplet config."""
    from negpy.desktop.session import resolve_asset_rgbscan

    leaked = replace(WorkspaceConfig(), rgbscan=RgbScanConfig(enabled=True, green_path="/g", blue_path="/b"))
    out = resolve_asset_rgbscan(leaked, {"path": "/r"})
    assert out.rgbscan == RgbScanConfig()


def _with_bounds(config, **rgbscan_kwargs):
    return replace(
        config,
        rgbscan=RgbScanConfig(**rgbscan_kwargs) if rgbscan_kwargs else RgbScanConfig(),
        process=replace(config.process, local_floors=(-1.3, -1.4, -1.5), local_ceils=(-0.2, -0.3, -0.4)),
    )


def test_resolve_asset_rgbscan_drops_bounds_when_becoming_a_triplet():
    """The lone red exposure's bounds measured G/B on sensor leak alone; on the composite
    they invert both channels to black and leave a solid red frame."""
    from negpy.desktop.session import resolve_asset_rgbscan

    lone = _with_bounds(WorkspaceConfig())
    out = resolve_asset_rgbscan(lone, {"path": "/r", "green_path": "/g", "blue_path": "/b"})
    assert out.process.is_local_initialized is False


def test_resolve_asset_rgbscan_drops_bounds_when_leaving_a_triplet():
    from negpy.desktop.session import resolve_asset_rgbscan

    triplet = _with_bounds(WorkspaceConfig(), enabled=True, green_path="/g", blue_path="/b")
    out = resolve_asset_rgbscan(triplet, {"path": "/r"})
    assert out.process.is_local_initialized is False


def test_resolve_asset_rgbscan_drops_bounds_when_members_change():
    from negpy.desktop.session import resolve_asset_rgbscan

    triplet = _with_bounds(WorkspaceConfig(), enabled=True, green_path="/g", blue_path="/b")
    out = resolve_asset_rgbscan(triplet, {"path": "/r", "green_path": "/g2", "blue_path": "/b2"})
    assert out.process.is_local_initialized is False


def test_resolve_asset_rgbscan_keeps_bounds_on_unchanged_composition():
    """Reopening the same triplet must not throw away its analysis, and align is not a
    composition change."""
    from negpy.desktop.session import resolve_asset_rgbscan

    triplet = _with_bounds(WorkspaceConfig(), enabled=True, green_path="/g", blue_path="/b")
    out = resolve_asset_rgbscan(triplet, {"path": "/r", "green_path": "/g", "blue_path": "/b", "align": False})
    assert out.process.local_floors == triplet.process.local_floors
    assert out.process.local_ceils == triplet.process.local_ceils
    assert out.rgbscan.align is False


def test_resolve_asset_rgbscan_keeps_bounds_on_a_plain_frame():
    """The overwhelmingly common case: no triplet either side, nothing to invalidate."""
    from negpy.desktop.session import resolve_asset_rgbscan

    plain = _with_bounds(WorkspaceConfig())
    out = resolve_asset_rgbscan(plain, {"path": "/r"})
    assert out.process.local_floors == plain.process.local_floors


def test_resolve_asset_rgbscan_respects_locked_bounds():
    """lock_bounds is the user pinning the stretch by hand; invalidate_local_bounds no-ops."""
    from negpy.desktop.session import resolve_asset_rgbscan

    lone = _with_bounds(WorkspaceConfig())
    lone = replace(lone, process=replace(lone.process, lock_bounds=True))
    out = resolve_asset_rgbscan(lone, {"path": "/r", "green_path": "/g", "blue_path": "/b"})
    assert out.process.local_floors == lone.process.local_floors


def test_thumbnail_cache_key_namespaces_triplets():
    """Batch and rendered paths must derive the same triplet key so they share a cache slot."""
    from negpy.services.assets.thumbnails import thumbnail_cache_key

    assert thumbnail_cache_key("h", False).startswith("h")
    assert thumbnail_cache_key("h", True) != thumbnail_cache_key("h", False)
    assert "-rgb" in thumbnail_cache_key("h", True)


def test_thumbnail_decode_routes_triplet_to_merge(monkeypatch):
    """With green/blue paths, the thumbnail must merge the triplet, not decode red alone."""
    from negpy.services.assets import thumbnails

    calls = {}
    monkeypatch.setattr(thumbnails, "_decode_triplet_preview", lambda r, g, b: calls.setdefault("args", (r, g, b)))
    thumbnails.decode_source_image("r", "g", "b")
    assert calls["args"] == ("r", "g", "b")


def test_thumbnail_worker_namespaces_triplet_cache(monkeypatch):
    """A triplet caches under a distinct key so it never collides with the red file scanned plain."""
    from PIL import Image

    from negpy.services.assets import thumbnails

    saved: dict = {}

    class Store:
        def get_thumbnail(self, key):
            return saved.get(key)

        def save_thumbnail(self, key, img):
            saved[key] = img

    img = Image.new("RGB", (4, 4))
    monkeypatch.setattr(thumbnails, "decode_source_image", lambda *a, **k: img)
    monkeypatch.setattr(thumbnails, "prepare_thumbnail", lambda i, ts: i)

    store = Store()
    thumbnails.get_thumbnail_worker("r", "hash", store, 0, 0.5, "g", "b")
    triplet_key = thumbnails.thumbnail_cache_key("hash", True)
    assert triplet_key in saved and thumbnails.thumbnail_cache_key("hash", False) not in saved

    thumbnails.get_thumbnail_worker("r2", "hash2", store)
    assert thumbnails.thumbnail_cache_key("hash2", False) in saved


def test_triplet_ignores_stale_plain_hash_cache(monkeypatch):
    """A red-only thumb cached under the plain hash (pre-fix) must not be served for a triplet."""
    from PIL import Image

    from negpy.services.assets import thumbnails

    stale = Image.new("RGB", (4, 4), (255, 0, 0))
    merged = Image.new("RGB", (4, 4), (0, 255, 0))
    saved: dict = {thumbnails.thumbnail_cache_key("hash", False): stale}

    class Store:
        def get_thumbnail(self, key):
            return saved.get(key)

        def save_thumbnail(self, key, img):
            saved[key] = img

    monkeypatch.setattr(thumbnails, "decode_source_image", lambda *a, **k: merged)
    monkeypatch.setattr(thumbnails, "prepare_thumbnail", lambda i, ts: i)

    # The worker inverts the decoded negative, so identity is not the check — provenance is.
    out = thumbnails.get_thumbnail_worker("r", "hash", Store(), 0, 0.5, "g", "b")
    assert out is not stale
    assert saved[thumbnails.thumbnail_cache_key("hash", True)] is out
