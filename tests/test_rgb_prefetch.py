import numpy as np

from negpy.features.rgbscan.models import RgbScanConfig
from negpy.infrastructure.loaders.memory import PreviewMemoryEstimate
from negpy.services.rendering import preview_manager as pm_mod
from negpy.services.rendering.prefetch_policy import MIN_RAM_RESERVE_BYTES
from negpy.services.rendering.preview_cache import PreviewBufferCache
from negpy.services.rendering.preview_manager import PreviewManager, estimate_rgb_triplet_preview_memory


def test_prefetch_linear_preview_rgb_warms_merged_triplet_cache(tmp_path, monkeypatch):
    red, green, blue = (tmp_path / f"{c}.arw" for c in "rgb")
    for p in (red, green, blue):
        p.write_bytes(b"raw")

    monkeypatch.setattr(pm_mod, "available_system_memory_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(
        pm_mod.loader_factory,
        "estimate_linear_preview_prefetch_memory",
        lambda *_a, **_k: PreviewMemoryEstimate(cached_bytes=1024, temporary_bytes=2048, source_dimensions=(8, 8)),
    )

    pm = PreviewManager()
    decode_calls = []

    def fake_decode(path, *_a, **_k):
        decode_calls.append(path)
        return np.ones((4, 4, 3), dtype=np.float32), (4, 4), {"color_space": "Adobe RGB"}

    pm.load_linear_preview = fake_decode
    rgbscan = RgbScanConfig(enabled=True, green_path=str(green), blue_path=str(blue), align=False)

    assert pm.prefetch_linear_preview_rgb(str(red), rgbscan, "Adobe RGB", use_camera_wb=False, file_hash="trip1")
    assert len(decode_calls) == 3
    assert pm.peek_linear_preview_rgb(rgbscan, "Adobe RGB", use_camera_wb=False, file_hash="trip1") is not None

    # Subsequent foreground open is an instant cache hit (0 additional decodes).
    pm.load_linear_preview_rgb(str(red), rgbscan, "Adobe RGB", use_camera_wb=False, file_hash="trip1")
    assert len(decode_calls) == 3


def test_concurrent_prefetch_budget_reservation_prevents_ram_overcommit(monkeypatch, tmp_path):
    """Concurrent prefetch threads share one atomic RAM/cache reservation instead of overcommitting."""
    import threading

    p1, p2 = tmp_path / "a.arw", tmp_path / "b.arw"
    p1.write_bytes(b"a")
    p2.write_bytes(b"b")
    per_task_ram = 64 * 1024 * 1024
    # Enough RAM for 1 prefetch above MIN_RAM_RESERVE_BYTES, NOT 2 simultaneously.
    monkeypatch.setattr(pm_mod, "available_system_memory_bytes", lambda: MIN_RAM_RESERVE_BYTES + int(per_task_ram * 1.5))
    monkeypatch.setattr(
        pm_mod.loader_factory,
        "estimate_linear_preview_prefetch_memory",
        lambda *_a, **_k: PreviewMemoryEstimate(
            cached_bytes=per_task_ram // 2, temporary_bytes=per_task_ram // 2, source_dimensions=(8, 8)
        ),
    )

    pm = PreviewManager()
    entered = threading.Event()
    release = threading.Event()

    def slow_decode(*_a, **_k):
        entered.set()
        release.wait(5.0)
        return np.ones((4, 4, 3), dtype=np.float32), (4, 4), {"color_space": "Adobe RGB"}

    pm.load_linear_preview = slow_decode
    results: list[bool] = []
    t1 = threading.Thread(
        target=lambda: results.append(pm.prefetch_linear_preview(str(p1), "Adobe RGB", use_camera_wb=False, file_hash="h1")),
        daemon=True,
    )
    t1.start()
    assert entered.wait(3.0)
    # While t1 holds the reservation, t2 must be rejected rather than overcommitting RAM.
    second = pm.prefetch_linear_preview(str(p2), "Adobe RGB", use_camera_wb=False, file_hash="h2")
    assert second is False
    release.set()
    t1.join(3.0)
    assert results == [True]
    # Once t1 releases its reservation, t2 is admitted.
    pm.load_linear_preview = lambda *_a, **_k: (np.ones((4, 4, 3), dtype=np.float32), (4, 4), {"color_space": "Adobe RGB"})
    assert pm.prefetch_linear_preview(str(p2), "Adobe RGB", use_camera_wb=False, file_hash="h2") is True


def test_foreground_rgb_preview_caches_only_merged_entry_and_estimates_realistic_budget(monkeypatch, tmp_path):
    """Foreground trichrome loads must not double-cache the standalone Red exposure, and
    triplet prefetch estimates must size only the merged preview rather than 3x full-sensor IR planes."""
    red, green, blue = (tmp_path / f"{c}.arw" for c in "rgb")
    for p in (red, green, blue):
        p.write_bytes(b"x" * 1024)

    pm = PreviewManager()
    file_hashes_seen: list = []

    def fake_decode(path, color_space=None, use_camera_wb=False, full_resolution=False, file_hash=None, **_kw):
        file_hashes_seen.append(file_hash)
        return np.ones((8, 12, 3), dtype=np.float32), (12, 8), {"color_space": "Adobe RGB"}

    pm.load_linear_preview = fake_decode
    rgbscan = RgbScanConfig(enabled=True, green_path=str(green), blue_path=str(blue), align=False)
    pm.load_linear_preview_rgb(str(red), rgbscan, "Adobe RGB", use_camera_wb=False, file_hash="trip-fg")

    assert file_hashes_seen == [None, None, None]
    assert pm._cache.usage().entries == 1

    monkeypatch.setattr(
        pm_mod.loader_factory,
        "estimate_linear_preview_prefetch_memory",
        lambda *_a, **_k: PreviewMemoryEstimate(
            cached_bytes=360 * 1024 * 1024,
            temporary_bytes=512 * 1024 * 1024,
            source_dimensions=(6000, 4000),
        ),
    )
    est = estimate_rgb_triplet_preview_memory(str(red), str(green), str(blue), 2000)
    # 2000x1333 * 24 B (RGB + 3-channel detect_preview) = ~64.0 MB merged cache entry, not 360 MB+.
    assert 60 * 1024 * 1024 < est.cached_bytes < 68 * 1024 * 1024
    # 3 * (6000*4000*7 = 168 MB) = ~504 MB transient decode on Auto half-size, not 2.6 GB.
    assert est.temporary_bytes < 540 * 1024 * 1024

    # Explicit non-Auto demosaic keeps full-resolution temporary budget per channel.
    est_aahd = estimate_rgb_triplet_preview_memory(str(red), str(green), str(blue), 2000, demosaic="AAHD")
    assert est_aahd.temporary_bytes == 3 * 512 * 1024 * 1024


def test_hq_rgb_neighbor_prefetch_warms_full_resolution_key_and_respects_hq_slot_cap(monkeypatch, tmp_path):
    """When HQ preview has >2 HQ buffer slots, prefetch_linear_preview_rgb(..., full_resolution=True)
    warms the exact full-res key that an HQ foreground load hits, without evicting protected HQ entries."""
    from dataclasses import replace as dc_replace

    red, green, blue = (tmp_path / f"{c}.arw" for c in "rgb")
    for p in (red, green, blue):
        p.write_bytes(b"x" * 1024)

    monkeypatch.setattr(pm_mod, "available_system_memory_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(
        pm_mod.loader_factory,
        "estimate_linear_preview_prefetch_memory",
        lambda *_a, **_k: PreviewMemoryEstimate(cached_bytes=2048, temporary_bytes=4096, source_dimensions=(8, 8)),
    )
    pm = PreviewManager()
    pm._cache = PreviewBufferCache(dc_replace(pm_mod.APP_CONFIG, preview_cache_max_entries=8, preview_cache_max_full_res_entries=3))
    calls: list[tuple[str, bool]] = []

    def fake_decode(path, color_space=None, use_camera_wb=False, full_resolution=False, file_hash=None, **_kw):
        calls.append((path, full_resolution))
        return np.ones((8, 8, 3), dtype=np.float32), (8, 8), {"color_space": "Adobe RGB"}

    pm.load_linear_preview = fake_decode
    rgbscan = RgbScanConfig(enabled=True, green_path=str(green), blue_path=str(blue), align=False)

    assert pm.prefetch_linear_preview_rgb(str(red), rgbscan, "Adobe RGB", use_camera_wb=False, file_hash="hq1", full_resolution=True)
    assert len(calls) == 3 and all(full is True for _p, full in calls)

    # Foreground HQ open hits the prefetched full_resolution=True entry (0 additional decodes).
    pm.load_linear_preview_rgb(str(red), rgbscan, "Adobe RGB", use_camera_wb=False, full_resolution=True, file_hash="hq1")
    assert len(calls) == 3

    # Non-TIFF RAWs (.raf/.cr3) with source_dimensions=None clamp phantom cap_edge^2 bytes instead of 8+ GiB.
    raf_r, raf_g, raf_b = (tmp_path / f"{c}.raf" for c in "rgb")
    for p in (raf_r, raf_g, raf_b):
        p.write_bytes(b"x" * 1024)
    monkeypatch.undo()
    est_raf = estimate_rgb_triplet_preview_memory(str(raf_r), str(raf_g), str(raf_b), 2000, full_resolution=True)
    assert est_raf.cached_bytes < 400 * 1024 * 1024
