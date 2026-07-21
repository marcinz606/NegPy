import numpy as np
from negpy.services.rendering.image_processor import ImageProcessor
from negpy.domain.models import WorkspaceConfig


def test_image_service_buffer_to_pil_8bit() -> None:
    service = ImageProcessor()
    buffer = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    settings = WorkspaceConfig()

    img = service.buffer_to_pil(buffer, settings, bit_depth=8)
    assert img.mode == "RGB"
    assert img.size == (1, 1)
    assert img.getpixel((0, 0)) == (0, 127, 255)


def test_image_service_buffer_to_pil_16bit_bw() -> None:
    service = ImageProcessor()
    buffer = np.array([[0.0, 1.0]], dtype=np.float32)  # Single channel (grayscale)
    settings = WorkspaceConfig.from_flat_dict({"process_mode": "B&W"})

    img = service.buffer_to_pil(buffer, settings, bit_depth=16)
    # PIL uses 'I;16' for 16-bit single channel
    assert img.mode == "I;16"
    assert img.getpixel((1, 0)) == 65535


def test_image_service_bw_conversion() -> None:
    service = ImageProcessor()
    # 3-channel input but B&W mode
    buffer = np.zeros((10, 10, 3), dtype=np.float32)
    settings = WorkspaceConfig.from_flat_dict({"process_mode": "B&W"})

    img = service.buffer_to_pil(buffer, settings, bit_depth=8)
    assert img.mode == "L"


def test_image_service_bw_toned_keeps_color() -> None:
    """A toned or tinted B&W print is chromatic — buffer_to_pil must not
    collapse it to a luma plane (regression: gate checked only selenium/sepia,
    so blue/copper/gold-only toning silently rendered grey)."""
    from dataclasses import replace

    from negpy.features.toning.models import ToningConfig

    service = ImageProcessor()
    buffer = np.full((4, 4, 3), 0.5, dtype=np.float32)
    bw = WorkspaceConfig.from_flat_dict({"process_mode": "B&W"})

    for kw in (
        {"blue_strength": 1.0},
        {"copper_strength": 1.0},
        {"vanadium_strength": 1.0},
        {"gold_strength": 1.0},
        {"highlight_tint_strength": 0.5},
    ):
        settings = replace(bw, toning=ToningConfig(**kw))
        assert service.buffer_to_pil(buffer, settings, bit_depth=8).mode == "RGB", kw


def test_image_service_jit_conversions() -> None:
    from negpy.kernel.image.logic import uint16_to_float32, uint8_to_float32

    # Test uint16 to float32 JIT
    u16_arr = np.array([[[0, 32767, 65535]]], dtype=np.uint16)
    f32_res = uint16_to_float32(np.ascontiguousarray(u16_arr))
    assert f32_res.dtype == np.float32
    assert np.allclose(f32_res, [[[0.0, 32767 / 65535, 1.0]]])

    # Test uint8 to float32 JIT
    u8_arr = np.array([[[0, 127, 255]]], dtype=np.uint8)
    f32_res_u8 = uint8_to_float32(np.ascontiguousarray(u8_arr))
    assert f32_res_u8.dtype == np.float32
    assert np.allclose(f32_res_u8, [[[0.0, 127 / 255, 1.0]]])


def test_use_half_size_decode_rules(monkeypatch) -> None:
    import negpy.services.rendering.image_processor as ip

    class _Raw:
        pass

    monkeypatch.setattr(ip, "is_xtrans", lambda raw: False)
    assert ip._use_half_size_decode(_Raw(), linear_raw=False)
    assert ip._use_half_size_decode(_Raw(), linear_raw=True)

    # X-Trans + linear decode: half_size aliases the 6x6 CFA -> stay full-res.
    monkeypatch.setattr(ip, "is_xtrans", lambda raw: True)
    assert ip._use_half_size_decode(_Raw(), linear_raw=False)
    assert not ip._use_half_size_decode(_Raw(), linear_raw=True)

    monkeypatch.setattr(ip, "is_xtrans", lambda raw: False)
    wrapper = object.__new__(ip.NonStandardFileWrapper)
    assert not ip._use_half_size_decode(wrapper, linear_raw=False)


def _fake_decode_recorder(calls):
    def fake(file_path, linear_raw, fast=False):
        calls.append(fast)
        return np.zeros((4, 4, 3), dtype=np.uint16), {"orientation": 1, "color_space": "sRGB"}

    return fake


def test_load_source_f32_cache_key_separates_fast_decode(monkeypatch) -> None:
    service = ImageProcessor()
    calls: list = []
    monkeypatch.setattr(service, "_decode_sensor_rgb", _fake_decode_recorder(calls))
    cfg = WorkspaceConfig()

    service._load_source_f32("/nonexistent/a.raw", cfg, fast_decode=True)
    service._load_source_f32("/nonexistent/a.raw", cfg, fast_decode=True)
    assert calls == [True]  # second call is a cache hit

    # A full-res consumer (real export) must not reuse the half-size buffer.
    service._load_source_f32("/nonexistent/a.raw", cfg, fast_decode=False)
    assert calls == [True, False]


def test_load_source_f32_never_fast_decodes_rgbscan_triplets(monkeypatch, tmp_path) -> None:
    from dataclasses import replace

    from negpy.features.rgbscan.models import RgbScanConfig

    r, g, b = (tmp_path / n for n in ("r.raw", "g.raw", "b.raw"))
    for f in (r, g, b):
        f.write_bytes(b"x")

    service = ImageProcessor()
    calls: list = []
    monkeypatch.setattr(service, "_decode_sensor_rgb", _fake_decode_recorder(calls))
    cfg = replace(
        WorkspaceConfig(),
        rgbscan=RgbScanConfig(enabled=True, green_path=str(g), blue_path=str(b), align=False),
    )

    service._load_source_f32(str(r), cfg, fast_decode=True)
    assert calls and calls[0] is False


def test_augment_retouch_reuses_stats_across_threshold_changes(monkeypatch) -> None:
    from dataclasses import replace

    import negpy.services.rendering.image_processor as ip
    from negpy.features.retouch.models import RetouchConfig

    rng = np.random.default_rng(42)
    img = (np.full((160, 160, 3), 0.18) * (1.0 + rng.normal(0, 0.02, (160, 160, 3)))).astype(np.float32)
    img[80:83, 80:83] = 0.005

    calls = []
    real = ip.compute_dust_stats
    monkeypatch.setattr(ip, "compute_dust_stats", lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    service = ImageProcessor()
    for thr in (0.5, 0.6, 0.7):
        cfg = replace(WorkspaceConfig(), retouch=RetouchConfig(dust_remove=True, dust_threshold=thr, dust_size=4))
        service._augment_retouch(cfg, img, "same-source")
    assert len(calls) == 1, "stat maps must survive threshold-only changes"

    cfg = replace(WorkspaceConfig(), retouch=RetouchConfig(dust_remove=True, dust_threshold=0.7, dust_size=6))
    service._augment_retouch(cfg, img, "same-source")
    assert len(calls) == 2, "dust_size changes the blur windows and must recompute"


def test_ir_ratio_gain_downsamples_once_per_source(monkeypatch) -> None:
    """_ir_bake and _augment_retouch each call _ir_ratio_gain every render. The cache key is
    the source shape, not the downsampled one, so the second call resolves it without
    repaying the full-res erode+resize (~130ms on a 34MP scan)."""
    import negpy.services.rendering.image_processor as ip

    ir = np.full((200, 200), 0.9, dtype=np.float32)
    ir[150:154, 150:154] = 0.1
    img = np.full((200, 200, 3), 0.5, dtype=np.float32)
    img[150:154, 150:154] = 0.08

    calls: list = []
    real = ip.downsample_ir
    monkeypatch.setattr(ip, "downsample_ir", lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    service = ImageProcessor()
    first = service._ir_ratio_gain(ir, img, "s")
    assert len(calls) == 1
    second = service._ir_ratio_gain(ir, img, "s")
    assert len(calls) == 1, "a cache hit must not repay the downsample"
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
    assert first[2] == second[2] and first[3] == second[3]

    service._ir_ratio_gain(ir, img, "other-source")
    assert len(calls) == 2, "a new source must still recompute"


def test_ir_two_tier_bake() -> None:
    """Semi-transparent dust is fixed by division, the opaque core by the
    score-weighted fill — both inside the same bake, no strokes anywhere."""
    from dataclasses import replace

    from negpy.features.retouch.models import RetouchConfig

    h = w = 200
    ir = np.full((h, w), 0.9, dtype=np.float32)
    ir[40:44, 40:44] = 0.82 * 0.9  # semi-transparent (ratio ≈ 0.82 > cutoff 0.71)
    ir[150:154, 150:154] = 0.1  # opaque core (ratio ≈ 0.11 < cutoff)
    img = np.full((h, w, 3), 0.5, dtype=np.float32)
    img[40:44, 40:44] = 0.42
    img[150:154, 150:154] = 0.08

    service = ImageProcessor()
    cfg = replace(WorkspaceConfig(), retouch=RetouchConfig(ir_dust_remove=True, ir_attenuation=True))

    baked, corr_mask, degenerate, routed = service._ir_bake(img, ir, cfg, "s")
    assert not degenerate and routed is None
    assert corr_mask is not None
    assert baked[41, 41].mean() > img[41, 41].mean(), "semi-transparent speck not lifted by division"
    assert float(np.asarray(baked)[151, 151].min()) > 0.4, "opaque core not rebuilt by the fill"

    _, detected, _ = service._augment_retouch(cfg, baked, "s")
    assert detected is None, "no strokes for an IR-only config"


def test_ir_bake_fill_runs_without_attenuation() -> None:
    """ir_attenuation off is no longer a full escape hatch: the division gain is
    skipped, but the score-weighted fill still repairs opaque defects."""
    from dataclasses import replace

    from negpy.features.retouch.models import RetouchConfig

    ir = np.full((80, 80), 0.9, dtype=np.float32)
    ir[40:44, 40:44] = 0.1
    img = np.full((80, 80, 3), 0.5, dtype=np.float32)
    img[40:44, 40:44] = 0.06
    service = ImageProcessor()
    cfg = replace(WorkspaceConfig(), retouch=RetouchConfig(ir_dust_remove=True, ir_attenuation=False))
    baked, corr_mask, _, _ = service._ir_bake(img, ir, cfg, "s")
    assert corr_mask is not None and corr_mask[41, 41]
    assert float(np.asarray(baked)[41, 41].min()) > 0.4, "the fill must run even without division"
    off = np.ones((80, 80), dtype=bool)
    off[36:48, 36:48] = False
    assert np.array_equal(np.asarray(baked)[off], img[off]), "clean film untouched without the gain"


def test_run_pipeline_routes_wide_ir_blob_to_inpaint(monkeypatch) -> None:
    """A blob wider than the fill's reach is routed to the structure-following inpaint
    inside the pipeline, and its mask surfaces for the overlay wash."""
    from dataclasses import replace

    from negpy.features.retouch.models import RetouchConfig

    service = ImageProcessor()
    monkeypatch.setattr(service.engine_cpu, "process", lambda img, s, sh, ctx: img)  # identity → inspect the baked source
    h = w = 200
    ir = np.full((h, w), 0.9, dtype=np.float32)
    img = np.full((h, w, 3), 0.5, dtype=np.float32)
    ir[60:76, 60:76] = 0.1  # 16×16: interior radius ≥ the routing bar
    img[60:76, 60:76] = 0.06
    cfg = replace(WorkspaceConfig(), retouch=RetouchConfig(ir_dust_remove=True, ir_attenuation=False, ir_threshold=0.35))

    out, metrics = service.run_pipeline(img, cfg, "h", render_size_ref=512, prefer_gpu=False, readback_metrics=False, ir_buffer=ir)
    assert "hair_inpaint_masks" in metrics
    assert float(out[68, 68, 0]) > 0.4, "blob interior not inpainted out of the source"


def test_augment_retouch_cap_keeps_largest(monkeypatch) -> None:
    """Over budget, the largest detected regions survive."""
    from dataclasses import replace

    import negpy.services.rendering.image_processor as ip
    from negpy.features.retouch.models import RetouchConfig

    big = ([[0.5, 0.5]], 40.0, 0.1, 0.0, 0.0)
    small = ([[0.4, 0.4]], 5.0, 0.1, 0.0, 0.0)
    monkeypatch.setattr(ip, "detect_luma_regions", lambda *a, **k: ([small, big, small, big, big], None))

    img = np.full((160, 160, 3), 0.5, dtype=np.float32)
    manual = [([[0.1, 0.1]], 3.0, 0.0, 0.0)] * 510  # budget = 512 − 510 = 2
    cfg = replace(WorkspaceConfig(), retouch=RetouchConfig(dust_remove=True, manual_heal_strokes=manual))

    settings, _, _ = ImageProcessor()._augment_retouch(cfg, img, "s")
    survivors = settings.retouch.manual_heal_strokes[:2]
    assert all(s[1] == 40.0 for s in survivors), "cap dropped the largest instead of the head of the list"


def test_run_pipeline_skip_flatfield(monkeypatch) -> None:
    """The export/contact-sheet CPU fallback passes an already-flat-fielded buffer;
    skip_flatfield must prevent a second apply_flatfield (the latent double-apply)."""
    import negpy.services.rendering.image_processor as ip

    service = ImageProcessor()
    monkeypatch.setattr(service.engine_cpu, "process", lambda img, s, sh, ctx: img)
    calls = []
    real = ip.apply_flatfield
    monkeypatch.setattr(ip, "apply_flatfield", lambda img, ff: (calls.append(1), real(img, ff))[1])

    img = np.full((64, 64, 3), 0.5, dtype=np.float32)
    service.run_pipeline(img, WorkspaceConfig(), "h", render_size_ref=512, prefer_gpu=False, readback_metrics=False)
    assert len(calls) == 1
    calls.clear()
    service.run_pipeline(img, WorkspaceConfig(), "h", render_size_ref=512, prefer_gpu=False, readback_metrics=False, skip_flatfield=True)
    assert len(calls) == 0
