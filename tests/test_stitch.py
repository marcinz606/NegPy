import numpy as np
import pytest

import cv2

from negpy.features.stitch.logic import (
    StitchError,
    build_proxy,
    register_pair,
    register_parts,
    scale_affine,
    scale_transforms,
    stitch_composite,
)
from negpy.features.stitch.models import StitchConfig, stitch_hash, stitch_token

_H, _W = 900, 2000
_P0_W, _P1_W = 1200, 1280


def _scene(h=_H, w=_W, seed=7):
    """Textured scene with enough detail for SIFT to lock onto."""
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.random((h, w), dtype=np.float32), (0, 0), 1.5)
    grad = np.linspace(0.0, 0.3, w, dtype=np.float32)[None, :]
    mono = np.clip(0.1 + 0.6 * base + grad, 0.0, 1.0)
    scene = np.stack([mono, mono * 0.9, mono * 1.1], axis=-1)
    return np.clip(scene, 0.0, 1.0).astype(np.float32)


def _true_affine(tx=720.0, ty=6.0, angle_deg=0.4, scale=0.998):
    """Ground truth mapping part1 coords -> scene (== part0) coords."""
    a = np.deg2rad(angle_deg)
    c, s = scale * np.cos(a), scale * np.sin(a)
    return np.array([[c, -s, tx], [s, c, ty]], dtype=np.float64)


def _parts(scene=None):
    scene = _scene() if scene is None else scene
    part0 = scene[:, :_P0_W].copy()
    t = _true_affine()
    part1 = cv2.warpAffine(
        scene,
        t.astype(np.float32),
        (_P1_W, _H),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
    )
    return scene, part0, part1, t


def test_register_pair_recovers_known_affine():
    _, part0, part1, t = _parts()
    p0, s0 = build_proxy(part0)
    p1, s1 = build_proxy(part1)
    m_proxy, inliers = register_pair(p0, p1)
    assert inliers > 100
    m = scale_affine(m_proxy, s0, s1)
    ang_est = np.degrees(np.arctan2(m[1, 0], m[0, 0]))
    ang_true = np.degrees(np.arctan2(t[1, 0], t[0, 0]))
    assert abs(ang_est - ang_true) < 0.05
    assert abs(np.hypot(m[0, 0], m[1, 0]) - 0.998) < 0.002
    assert abs(m[0, 2] - t[0, 2]) < 2.0
    assert abs(m[1, 2] - t[1, 2]) < 2.0


def test_register_pair_unrelated_raises():
    rng = np.random.default_rng(1)
    a = (rng.random((400, 400)) * 255).astype(np.uint8)
    b = (rng.random((400, 400)) * 255).astype(np.uint8)
    with pytest.raises(StitchError):
        register_pair(a, b)


def _registered_config(part0, part1):
    transforms, canvas = register_parts([part0, part1])
    return StitchConfig(
        stitch_enabled=True,
        stitch_paths=("/p1",),
        stitch_transforms=tuple(tuple(float(v) for v in m.ravel()) for m in transforms),
        stitch_canvas=canvas,
        stitch_sizes=((part0.shape[1], part0.shape[0]), (part1.shape[1], part1.shape[0])),
    )


def test_register_parts_and_composite_seam():
    scene, part0, part1, _ = _parts()
    cfg = _registered_config(part0, part1)
    cw, ch = cfg.stitch_canvas
    assert abs(cw - _W) < 20
    assert abs(ch - _H) < 20

    rgb, ir = stitch_composite([part0, part1], [None, None], cfg)
    assert ir is None
    assert rgb.shape[:2] == (ch, cw)
    # Part0 is the reference: canvas offset is its (pure translation) transform.
    m0 = np.array(cfg.stitch_transforms[0], dtype=np.float64).reshape(2, 3)
    ox, oy = int(round(m0[0, 2])), int(round(m0[1, 2]))
    inner = (slice(oy + 20, oy + _H - 20), slice(ox + 20, ox + _W - 20))
    scene_inner = scene[20:-20, 20:-20]
    rms = float(np.sqrt(np.mean((rgb[inner] - scene_inner) ** 2)))
    assert rms < 0.02


def test_gain_compensation_removes_cast():
    scene, part0, part1, _ = _parts()
    cast = np.array([1.02, 0.98, 1.01], dtype=np.float32)
    cfg = _registered_config(part0, part1 * cast)
    rgb, _ = stitch_composite([part0, np.clip(part1 * cast, 0, 1)], [None, None], cfg)
    m0 = np.array(cfg.stitch_transforms[0], dtype=np.float64).reshape(2, 3)
    ox, oy = int(round(m0[0, 2])), int(round(m0[1, 2]))
    inner = (slice(oy + 20, oy + _H - 20), slice(ox + 20, ox + _W - 20))
    rms = float(np.sqrt(np.mean((rgb[inner] - scene[20:-20, 20:-20]) ** 2)))
    assert rms < 0.02


def test_ir_blended_only_when_all_present():
    _, part0, part1, t = _parts()
    cfg = _registered_config(part0, part1)

    rgb, ir = stitch_composite([part0, part1], [np.ones(part0.shape[:2], np.float32), None], cfg)
    assert ir is None

    ir0 = np.ones(part0.shape[:2], np.float32)
    ir1 = np.ones(part1.shape[:2], np.float32)
    # Shared defect (film dust, same scene spot in both parts) must survive the max-blend.
    sx, sy = 800, 450
    inv = cv2.invertAffineTransform(t)
    px, py = cv2.transform(np.array([[[sx, sy]]], np.float64), inv)[0, 0]
    ir0[sy - 4 : sy + 4, sx - 4 : sx + 4] = 0.1
    ir1[int(py) - 6 : int(py) + 6, int(px) - 6 : int(px) + 6] = 0.1
    # Single-part defect (sensor dust) inside the overlap is erased by the other
    # part's clean pixels under the max-blend.
    ir0[440:460, 1000:1020] = 0.1

    _, ir_out = stitch_composite([part0, part1], [ir0, ir1], cfg)
    assert ir_out is not None
    m0 = np.array(cfg.stitch_transforms[0], dtype=np.float64).reshape(2, 3)
    ox, oy = int(round(m0[0, 2])), int(round(m0[1, 2]))
    assert ir_out[oy + sy, ox + sx] < 0.5
    assert ir_out[oy + 450, ox + 1010] > 0.9
    # Canvas corners outside every part are clean (1.0) by convention.
    assert ir_out[0, -1] > 0.99


def test_scale_transforms_half_scale():
    _, part0, part1, _ = _parts()
    cfg = _registered_config(part0, part1)
    full, _ = stitch_composite([part0, part1], [None, None], cfg)

    halves = [cv2.resize(p, (p.shape[1] // 2, p.shape[0] // 2), interpolation=cv2.INTER_AREA) for p in (part0, part1)]
    transforms, (cw, ch) = scale_transforms(cfg, [(p.shape[1], p.shape[0]) for p in halves])
    assert abs(cw - cfg.stitch_canvas[0] / 2) <= 1
    assert abs(ch - cfg.stitch_canvas[1] / 2) <= 1

    half_cfg = StitchConfig(
        stitch_enabled=True,
        stitch_paths=cfg.stitch_paths,
        stitch_transforms=tuple(tuple(float(v) for v in m.ravel()) for m in transforms),
        stitch_canvas=(cw, ch),
        stitch_sizes=tuple((p.shape[1], p.shape[0]) for p in halves),
    )
    half = stitch_composite(halves, [None, None], half_cfg)[0]
    ref = cv2.resize(full, (cw, ch), interpolation=cv2.INTER_AREA)
    rms = float(np.sqrt(np.mean((half[20:-20, 20:-20] - ref[20:-20, 20:-20]) ** 2)))
    assert rms < 0.03


def test_stitch_hash_stable_and_order_sensitive():
    from negpy.services.assets.half_frame import base_hash

    h1 = stitch_hash(["aaa", "bbb"])
    assert h1 == stitch_hash(["aaa", "bbb"])
    assert h1 != stitch_hash(["bbb", "aaa"])
    assert h1.endswith("#stitch")
    assert base_hash(h1) == h1.split("#", 1)[0]


def test_workspace_config_roundtrip_preserves_stitch():
    import json
    from dataclasses import replace

    from negpy.domain.models import WorkspaceConfig

    stitch = StitchConfig(
        stitch_enabled=True,
        stitch_paths=("/p1.nef",),
        stitch_transforms=((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 50.5, 0.0, 1.0, 2.5)),
        stitch_canvas=(4000, 3000),
        stitch_sizes=((3000, 2000), (3000, 2000)),
    )
    cfg = replace(WorkspaceConfig(), stitch=stitch)
    # JSON round-trip turns tuples into lists; from_flat_dict must coerce back
    # or the frozen config loses hashability/equality.
    data = json.loads(json.dumps(cfg.to_dict()))
    restored = WorkspaceConfig.from_flat_dict(data)
    assert restored.stitch == stitch
    hash(restored.stitch)


def test_load_source_f32_stitches_composite(tmp_path):
    """Decode funnel must assemble the composite (translation-only fixture: exact content)."""
    import tifffile
    from dataclasses import replace

    from negpy.domain.models import WorkspaceConfig
    from negpy.services.rendering.image_processor import ImageProcessor

    rng = np.random.default_rng(3)
    scene = (rng.random((80, 120, 3)) * 60000).astype(np.uint16)
    p0 = tmp_path / "part0.tif"
    p1 = tmp_path / "part1.tif"
    tifffile.imwrite(str(p0), scene[:, :80])
    tifffile.imwrite(str(p1), scene[:, 40:])

    stitch = StitchConfig(
        stitch_enabled=True,
        stitch_paths=(str(p1),),
        stitch_transforms=((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 40.0, 0.0, 1.0, 0.0)),
        stitch_canvas=(120, 80),
        stitch_sizes=((80, 80), (80, 80)),
    )
    params = replace(WorkspaceConfig(), stitch=stitch)
    proc = ImageProcessor()
    f32, ir, _ = proc._load_source_f32(str(p0), params)
    assert ir is None
    assert f32.shape == (80, 120, 3)
    assert np.allclose(f32, scene.astype(np.float32) / 65535.0, atol=1e-3)

    again, _, _ = proc._load_source_f32(str(p0), params)
    assert again is f32  # decode cache hit
    moved = replace(
        params,
        stitch=replace(stitch, stitch_transforms=((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 41.0, 0.0, 1.0, 0.0))),
    )
    fresh, _, _ = proc._load_source_f32(str(p0), moved)
    assert fresh is not f32  # geometry change must invalidate the decode cache


def test_load_linear_preview_stitch(tmp_path):
    """Preview path must assemble the composite and cache it under the stitch identity."""
    import tifffile

    from negpy.services.rendering.preview_manager import PreviewManager

    rng = np.random.default_rng(4)
    scene = (rng.random((80, 120, 3)) * 60000).astype(np.uint16)
    p0 = tmp_path / "part0.tif"
    p1 = tmp_path / "part1.tif"
    tifffile.imwrite(str(p0), scene[:, :80])
    tifffile.imwrite(str(p1), scene[:, 40:])
    stitch = StitchConfig(
        stitch_enabled=True,
        stitch_paths=(str(p1),),
        stitch_transforms=((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 40.0, 0.0, 1.0, 0.0)),
        stitch_canvas=(120, 80),
        stitch_sizes=((80, 80), (80, 80)),
    )
    pm = PreviewManager()
    out, dims, meta = pm.load_linear_preview_stitch(str(p0), stitch, "Adobe RGB", use_camera_wb=False, file_hash="abc#stitch")
    assert out.shape == (80, 120, 3)
    assert dims == (80, 120)
    assert meta.get("ir_preview") is None
    assert np.allclose(out, scene.astype(np.float32) / 65535.0, atol=1e-3)

    again, _, _ = pm.load_linear_preview_stitch(str(p0), stitch, "Adobe RGB", use_camera_wb=False, file_hash="abc#stitch")
    assert again is out  # composite preview cache hit


def _worker(monkeypatch, buffers):
    from negpy.desktop.workers.stitch import StitchWorker

    worker = StitchWorker()
    monkeypatch.setattr(
        worker._processor,
        "_decode_oriented_f32",
        lambda path, params, fast_decode=False: (buffers[path], None, "Adobe RGB"),
    )
    return worker


def _worker_task(paths):
    from dataclasses import replace as _r

    from negpy.desktop.workers.stitch import StitchTask
    from negpy.domain.models import WorkspaceConfig

    files = tuple({"name": p.strip("/"), "path": p, "hash": f"h{i}"} for i, p in enumerate(paths))
    return StitchTask(files=files, params_by_path={p: _r(WorkspaceConfig()) for p in paths})


def test_stitch_worker_emits_registered_payload(monkeypatch):
    _, part0, part1, _ = _parts()
    worker = _worker(monkeypatch, {"/a.nef": part0, "/b.nef": part1})
    results, errors = [], []
    worker.registered.connect(results.append)
    worker.error.connect(errors.append)
    worker.run(_worker_task(["/a.nef", "/b.nef"]))
    assert not errors
    payload = results[0]
    assert [f["path"] for f in payload["files"]] == ["/a.nef", "/b.nef"]
    assert len(payload["transforms"]) == 2 and len(payload["transforms"][0]) == 6
    assert payload["sizes"] == ((_P0_W, _H), (_P1_W, _H))
    assert abs(payload["canvas"][0] - _W) < 20


def test_stitch_worker_error_on_unrelated(monkeypatch):
    rng = np.random.default_rng(2)
    a = np.clip(rng.random((300, 300, 3)), 0, 1).astype(np.float32)
    b = np.clip(rng.random((300, 300, 3)), 0, 1).astype(np.float32)
    worker = _worker(monkeypatch, {"/a.nef": a, "/b.nef": b})
    results, errors = [], []
    worker.registered.connect(results.append)
    worker.error.connect(errors.append)
    worker.run(_worker_task(["/a.nef", "/b.nef"]))
    assert not results
    assert errors and "overlap" in errors[0]


def test_stitch_worker_cancel_between_decodes(monkeypatch):
    _, part0, part1, _ = _parts()
    worker = _worker(monkeypatch, {"/a.nef": part0, "/b.nef": part1})

    real_decode = worker._processor._decode_oriented_f32

    def decode_then_cancel(path, params, fast_decode=False):
        worker._cancel.set()
        return real_decode(path, params, fast_decode)

    monkeypatch.setattr(worker._processor, "_decode_oriented_f32", decode_then_cancel)
    results, cancels = [], []
    worker.registered.connect(results.append)
    worker.cancelled.connect(lambda: cancels.append(True))
    worker.run(_worker_task(["/a.nef", "/b.nef"]))
    assert cancels and not results


def test_attach_restored_stitches_rebuilds_asset(tmp_path):
    """Session restore must rebuild a composite from the saved registration, not re-register."""
    from negpy.desktop.workers.render import AssetDiscoveryWorker

    p0 = tmp_path / "part0.nef"
    p1 = tmp_path / "part1.nef"
    for f in (p0, p1):
        f.write_bytes(b"x")
    assets = [{"name": "part0.nef", "path": str(p0), "hash": "h0"}]
    stitches = {
        str(p0): {
            "paths": [str(p1)],
            "transforms": [[1, 0, 0, 0, 1, 0], [1, 0, 40, 0, 1, 0]],
            "canvas": [120, 80],
            "sizes": [[80, 80], [80, 80]],
            "hash": "digest#stitch",
        }
    }
    out = AssetDiscoveryWorker()._attach_restored_stitches(assets, stitches)
    assert out[0]["hash"] == "digest#stitch"
    assert out[0]["stitch_paths"] == (str(p1),)
    assert out[0]["name"].endswith("(Stitch)")
    # A part gone missing on disk restores as a plain asset.
    stitches[str(p0)]["paths"] = [str(tmp_path / "gone.nef")]
    plain = AssetDiscoveryWorker()._attach_restored_stitches([dict(assets[0])], stitches)
    assert "stitch_paths" not in plain[0]


def test_resolve_asset_stitch_injects_and_resets():
    from dataclasses import replace

    from negpy.desktop.session import resolve_asset_stitch
    from negpy.domain.models import WorkspaceConfig

    asset = {
        "path": "/p0",
        "stitch_paths": ["/p1"],
        "stitch_transforms": [[1, 0, 0, 0, 1, 0], [1, 0, 40, 0, 1, 0]],
        "stitch_canvas": [120, 80],
        "stitch_sizes": [[80, 80], [80, 80]],
    }
    out = resolve_asset_stitch(WorkspaceConfig(), asset)
    assert out.stitch.stitch_enabled
    assert out.stitch.stitch_paths == ("/p1",)
    assert out.stitch.stitch_canvas == (120, 80)
    hash(out.stitch)  # lists from JSON/session must arrive as tuples

    leaked = replace(WorkspaceConfig(), stitch=out.stitch)
    reset = resolve_asset_stitch(leaked, {"path": "/other"})
    assert reset.stitch == StitchConfig()


def test_apply_stitch_replaces_parts_with_composite():
    from unittest.mock import MagicMock

    from negpy.desktop.session import DesktopSessionManager
    from negpy.infrastructure.storage.repository import StorageRepository

    repo = MagicMock(spec=StorageRepository)
    repo.load_file_settings.return_value = None
    repo.load_file_settings_by_path.return_value = None
    repo.get_global_setting.side_effect = lambda key, default=None: default
    repo.load_file_marks.return_value = {}
    repo.get_max_history_index.return_value = 0
    session = DesktopSessionManager(repo)
    session.state.uploaded_files = [
        {"name": "other.nef", "path": "/other", "hash": "hx"},
        {"name": "a.nef", "path": "/a", "hash": "ha"},
        {"name": "b.nef", "path": "/b", "hash": "hb"},
    ]
    composite = {
        "name": "a+b (Stitch)",
        "path": "/a",
        "hash": "digest#stitch",
        "stitch_paths": ("/b",),
        "stitch_transforms": ((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 40.0, 0.0, 1.0, 0.0)),
        "stitch_canvas": (120, 80),
        "stitch_sizes": ((80, 80), (80, 80)),
    }
    session.apply_stitch([1, 2], composite)
    files = session.state.uploaded_files
    assert [f["path"] for f in files] == ["/other", "/a"]
    assert files[1]["hash"] == "digest#stitch"
    assert session.state.selected_file_idx == 1
    saved = {c.args[0]: c.args[1] for c in repo.save_global_setting.call_args_list}
    assert "/a" in saved.get("session_stitches", {})


def test_stitch_real_panorama_samples():
    """End-to-end registration on the real camera-scan pair from discussion #555."""
    import os

    p0 = os.path.join("samples", "panorama", "Img427.nef")
    p1 = os.path.join("samples", "panorama", "Img428.nef")
    if not (os.path.exists(p0) and os.path.exists(p1)):
        pytest.skip("panorama samples not present")

    from negpy.domain.models import WorkspaceConfig
    from negpy.services.rendering.image_processor import ImageProcessor

    proc = ImageProcessor()
    params = WorkspaceConfig()
    parts = [proc._decode_oriented_f32(p, params, fast_decode=True)[0] for p in (p0, p1)]

    proxies = [build_proxy(p) for p in parts]
    _, inliers = register_pair(proxies[0][0], proxies[1][0])
    assert inliers > 1000

    transforms, (cw, ch) = register_parts(parts)
    h, w = parts[0].shape[:2]
    assert w <= cw < 2 * w + 20
    assert h <= ch < 2 * h + 20

    cfg = StitchConfig(
        stitch_enabled=True,
        stitch_paths=(p1,),
        stitch_transforms=tuple(tuple(float(v) for v in m.ravel()) for m in transforms),
        stitch_canvas=(cw, ch),
        stitch_sizes=tuple((p.shape[1], p.shape[0]) for p in parts),
    )
    rgb, ir = stitch_composite(parts, [None, None], cfg)
    assert rgb.shape == (ch, cw, 3)
    assert ir is None
    assert float(rgb.max()) <= 1.0 and float(rgb.min()) >= 0.0


def test_stitch_token_identity(tmp_path):
    assert stitch_token(StitchConfig()) == ""
    p1 = tmp_path / "a.nef"
    p1.write_bytes(b"x")
    base = dict(
        stitch_enabled=True,
        stitch_paths=(str(p1),),
        stitch_transforms=((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 50.0, 0.0, 1.0, 0.0)),
        stitch_canvas=(100, 100),
        stitch_sizes=((60, 100), (60, 100)),
    )
    tok = stitch_token(StitchConfig(**base))
    assert tok.startswith("|stitch:")
    moved = StitchConfig(**{**base, "stitch_transforms": ((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 60.0, 0.0, 1.0, 0.0))})
    assert stitch_token(moved) != tok
    # Missing part file -> inactive token (same convention as rgbscan_token).
    gone = StitchConfig(**{**base, "stitch_paths": (str(tmp_path / "missing.nef"),)})
    assert stitch_token(gone) == ""
