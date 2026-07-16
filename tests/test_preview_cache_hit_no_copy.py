import numpy as np
from contextlib import contextmanager
from unittest.mock import patch, MagicMock
from negpy.services.rendering.preview_manager import PreviewManager
from negpy.services.rendering.preview_cache import PreviewCacheKey


def _make_loader_mock(metadata: dict | None = None):
    """Return a mock for loader_factory.get_loader() that produces a dummy ctx_mgr."""
    if metadata is None:
        metadata = {"color_space": "Adobe RGB"}

    @contextmanager
    def _ctx():
        yield MagicMock()  # raw object — never actually used on a cache hit

    return MagicMock(return_value=(_ctx(), metadata))


def test_load_linear_preview_cache_hit_no_copy():
    """load_linear_preview() on a warm cache should return the cached buffer directly (no copy)."""
    mgr = PreviewManager()
    buf = np.zeros((20, 20, 3), dtype=np.float32)
    key = PreviewCacheKey(
        file_hash="testhash",
        use_camera_wb=False,
        workspace_color_space="Adobe RGB",
        full_resolution=False,
    )
    mgr._cache.put(key, buf, (20, 20), {"color_space": "Adobe RGB"})

    mock_get_loader = _make_loader_mock()
    with patch("negpy.services.rendering.preview_manager.loader_factory.get_loader", mock_get_loader):
        result, dims, meta = mgr.load_linear_preview(
            "fake.nef",
            color_space="Adobe RGB",
            file_hash="testhash",
        )

    cached_buf = mgr._cache.get(key)[0]
    assert result is cached_buf, (
        "load_linear_preview cache hit should return the cached buffer directly, not a copy. result is a different object."
    )


def test_load_splash_and_linear_cache_hit_no_copy():
    """load_splash_and_linear() on a warm cache should return the cached buffer directly (no copy)."""
    mgr = PreviewManager()
    buf = np.zeros((20, 20, 3), dtype=np.float32)
    key = PreviewCacheKey(
        file_hash="testhash2",
        use_camera_wb=False,
        workspace_color_space="Adobe RGB",
        full_resolution=False,
    )
    mgr._cache.put(key, buf, (20, 20), {"color_space": "Adobe RGB"})

    mock_get_loader = _make_loader_mock()
    with patch("negpy.services.rendering.preview_manager.loader_factory.get_loader", mock_get_loader):
        splash, (result, dims, meta) = mgr.load_splash_and_linear(
            "fake.nef",
            color_space="Adobe RGB",
            file_hash="testhash2",
        )

    cached_buf = mgr._cache.get(key)[0]
    assert splash is None, "Splash should be None on a cache hit"
    assert result is cached_buf, (
        "load_splash_and_linear cache hit should return the cached buffer directly, not a copy. result is a different object."
    )


def test_warm_cache_hit_returns_same_buffer_object():
    """On a warm cache hit, load_linear_preview must return the cached buffer directly (no copy)."""
    mgr = PreviewManager()

    # Pre-populate cache
    stored_buf = np.zeros((20, 20, 3), dtype=np.float32)
    key = PreviewCacheKey(
        file_hash="ident_test",
        use_camera_wb=False,
        workspace_color_space="Adobe RGB",
        full_resolution=False,
    )
    mgr._cache.put(key, stored_buf, (20, 20), {"color_space": "Adobe RGB"})

    # put() stores the buffer by reference (read-only contract); get the reference directly
    cached_buf, _, _ = mgr._cache.get(key)

    # Now call load_linear_preview which should return from cache.
    # The loader is called before the cache check, so mock it to avoid a real file open.
    mock_get_loader = _make_loader_mock()
    with patch("negpy.services.rendering.preview_manager.loader_factory.get_loader", mock_get_loader):
        result_buf, dims, meta = mgr.load_linear_preview(
            "fake.nef",
            color_space="Adobe RGB",
            file_hash="ident_test",
        )

    assert result_buf is cached_buf, "Cache hit should return the exact same buffer object from cache — no copy."
