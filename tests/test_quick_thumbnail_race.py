from unittest.mock import patch

import numpy as np
from PIL import Image

from negpy.infrastructure.storage.local_asset_store import LocalAssetStore
from negpy.services.assets import thumbnails


def test_a_quick_thumbnail_never_replaces_a_rendered_one_saved_during_its_decode(tmp_path) -> None:
    store = LocalAssetStore(str(tmp_path / "cache"), str(tmp_path / "icc"))
    store.initialize()
    key = thumbnails.thumbnail_cache_key("h", False)
    rendered = Image.new("RGB", (8, 8), (200, 0, 0))

    def decode_while_a_render_lands(*_a, **_k):
        store.save_thumbnail(key, rendered, fingerprint="rendered-fp")
        return Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))

    with patch.object(thumbnails, "decode_bounded_source_preview", side_effect=decode_while_a_render_lands):
        thumbnails.get_thumbnail_worker("/x.tif", "h", asset_store=store)

    assert store.get_thumbnail_fingerprint(key) == "rendered-fp"
