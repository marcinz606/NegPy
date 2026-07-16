"""Per-frame memo of the last displayed render, for instant frame switching.

Navigating back to a frame whose edits haven't changed would re-run the whole
pipeline just to reproduce pixels that were already on screen — at HQ (full
resolution) that is seconds of spinner. The controller stores each frame's last
rendered display buffer here, keyed by everything that shaped it; on navigate-
back with a matching key the canvas is painted from the memo immediately and
the authoritative render refreshes metrics quietly in the background (so a
stale memo can only ever flash briefly, never persist).

Buffers are stored by reference under the same read-only contract as the
preview cache. GPU-texture results are never memoized (textures are destroyed
on navigation); in the default soft-proof path the displayed buffer is a CPU
array, which is what lands here.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional

from negpy.kernel.system.config import APP_CONFIG


class RenderMemo:
    """LRU of the last render per file: file_hash -> (memo_key, payload)."""

    def __init__(self, app_config: Any = None) -> None:
        self._app = app_config or APP_CONFIG
        self._entries: "OrderedDict[str, tuple[str, dict]]" = OrderedDict()

    def _budget(self) -> int:
        # One knob governs both HQ retention caches: the active frame plus the
        # one navigated from (see preview_cache_max_full_res_entries).
        return max(2, int(getattr(self._app, "preview_cache_max_full_res_entries", 2)))

    def store(self, file_hash: str, memo_key: str, payload: dict) -> None:
        if not file_hash or not memo_key:
            return
        self._entries.pop(file_hash, None)
        self._entries[file_hash] = (memo_key, payload)
        while len(self._entries) > self._budget():
            self._entries.popitem(last=False)

    def get(self, file_hash: str, memo_key: str) -> Optional[dict]:
        entry = self._entries.get(file_hash)
        if entry is None or entry[0] != memo_key:
            return None
        self._entries.move_to_end(file_hash)
        return entry[1]

    def rekey(self, file_hash: str, new_key: str) -> None:
        """Follow a render-neutral config change (e.g. measured bounds persisted
        after the render, with render=False): the stored pixels are still valid,
        only their identity moved."""
        entry = self._entries.get(file_hash)
        if entry is not None and new_key:
            self._entries[file_hash] = (new_key, entry[1])

    def invalidate(self, file_hash: str) -> None:
        self._entries.pop(file_hash, None)

    def clear(self) -> None:
        self._entries.clear()
