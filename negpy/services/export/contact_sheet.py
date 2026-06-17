import math
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

MAX_TILES_PER_SHEET = 38

CELL_PX = 600  # long-edge of a single cell
GAP = 16  # gap between cells
MARGIN = 32  # black border around the grid


class ContactSheetService:
    """Composites rendered frames into darkroom-style contact sheets on black."""

    @staticmethod
    def grid_dims(n: int) -> Tuple[int, int]:
        """Square-ish (cols, rows) holding n frames."""
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        return cols, rows

    @staticmethod
    def build_sheets(
        tiles: List[np.ndarray],
        *,
        max_tiles: int = MAX_TILES_PER_SHEET,
        cell_px: int = CELL_PX,
        gap: int = GAP,
        margin: int = MARGIN,
    ) -> List[Image.Image]:
        """Paginate tiles (<=max_tiles per sheet) into grids on a black background."""
        sheets: List[Image.Image] = []
        for start in range(0, len(tiles), max_tiles):
            chunk = tiles[start : start + max_tiles]
            sheets.append(ContactSheetService._compose_sheet(chunk, cell_px, gap, margin))
        return sheets

    @staticmethod
    def _compose_sheet(tiles: List[np.ndarray], cell_px: int, gap: int, margin: int) -> Image.Image:
        cols, rows = ContactSheetService.grid_dims(len(tiles))

        sheet_w = margin * 2 + cols * cell_px + (cols - 1) * gap
        sheet_h = margin * 2 + rows * cell_px + (rows - 1) * gap
        canvas = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

        for idx, tile in enumerate(tiles):
            row, col = divmod(idx, cols)
            cell_x = margin + col * (cell_px + gap)
            cell_y = margin + row * (cell_px + gap)
            ContactSheetService._paste_centered(canvas, tile, cell_x, cell_y, cell_px)

        return Image.fromarray(canvas)

    @staticmethod
    def _paste_centered(canvas: np.ndarray, tile: np.ndarray, cell_x: int, cell_y: int, cell_px: int) -> None:
        """Resize tile to fit a cell_px square (keep aspect) and center it in the cell."""
        h, w = tile.shape[:2]
        scale = cell_px / max(h, w)
        tw = max(1, int(round(w * scale)))
        th = max(1, int(round(h * scale)))
        resized = cv2.resize(tile, (tw, th), interpolation=cv2.INTER_AREA)

        off_x = cell_x + (cell_px - tw) // 2
        off_y = cell_y + (cell_px - th) // 2
        canvas[off_y : off_y + th, off_x : off_x + tw] = resized
