import os
from dataclasses import replace

import numpy as np
import pytest

from negpy.domain.models import WorkspaceConfig
import cv2

from negpy.features.rgbscan.logic import (
    estimate_shift,
    assemble_rgb,
    classify_channel,
    group_triplets,
    merge_rgb_triplet,
    probe_channel_means,
    rgbscan_token,
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
