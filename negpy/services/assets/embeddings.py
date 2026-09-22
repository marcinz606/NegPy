"""Batch CLIP embedding generation for search by meaning: the same async/semaphore
shape as thumbnails.generate_batch_thumbnails, reusing get_thumbnail_worker's own
disk cache so indexing a session costs no second RAW decode -- a file whose thumbnail
was just generated is a cached JPEG read here, not a re-decode.

The controller only ever passes files with no cached vector for the active
MODEL_VERSION (a single bulk load_embeddings_for query, not a check per file here).
"""

import asyncio
import inspect
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger
from negpy.services.assets.semantic_model import MODEL_VERSION, ClipModel
from negpy.services.assets.thumbnails import get_thumbnail_worker

logger = get_logger(__name__)


async def generate_batch_embeddings(
    files: List[Dict[str, Any]],
    asset_store: Any,
    repo: Any,
    progress_callback: Optional[Any] = None,
    ready_callback: Optional[Any] = None,
    is_cancelled: Optional[Any] = None,
) -> Dict[str, np.ndarray]:
    """Embeds every file in `files`, persisting each vector as it lands.

    `ready_callback(file_hash, vector)` fires per file so the filmstrip's ranking can
    improve as the batch runs rather than waiting for the whole batch to finish.

    `is_cancelled()` is checked once per file, right before its (expensive) decode and
    embed step -- a library-wide batch can run long enough to need stopping mid-way. A
    file already past that check finishes normally; only files still queued are skipped,
    same granularity download_clip_model's own is_cancelled already uses.
    """
    model = ClipModel()
    semaphore = asyncio.Semaphore(APP_CONFIG.max_workers)
    completed = 0

    def _embed_one(f_info: Dict[str, Any]) -> Optional[np.ndarray]:
        thumb = get_thumbnail_worker(
            f_info["path"],
            f_info["hash"],
            asset_store,
            int(f_info.get("half") or 0),
            float(f_info.get("split_x") or 0.5),
            f_info.get("green_path") or "",
            f_info.get("blue_path") or "",
            tuple(f_info["crop_rect"]) if f_info.get("crop_rect") else None,
            float(f_info.get("gutter_thickness") or 0.0),
            str(f_info.get("process_mode") or ""),
        )
        if thumb is None:
            return None
        return model.embed_image(thumb)

    async def _worker(f_info: Dict[str, Any]) -> Tuple[str, Optional[np.ndarray]]:
        nonlocal completed
        async with semaphore:
            if is_cancelled is not None and is_cancelled():
                return f_info["hash"], None
            vector = await asyncio.to_thread(_embed_one, f_info)
            completed += 1
            if progress_callback:
                if inspect.iscoroutinefunction(progress_callback):
                    await progress_callback(completed, f_info["name"])
                else:
                    progress_callback(completed, f_info["name"])
            if vector is not None:
                repo.save_embedding(f_info["hash"], vector, MODEL_VERSION, f_info["path"])
                if ready_callback:
                    ready_callback(f_info["hash"], vector)
            return f_info["hash"], vector

    tasks = [_worker(f) for f in files]
    results = await asyncio.gather(*tasks)
    return {h: v for h, v in results if v is not None}
