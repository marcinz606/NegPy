import math
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

MAX_TILES_PER_SHEET = 38

CELL_PX = 600  # long-edge of a single cell
GUTTER = 16  # gap between cells
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
    def build_sheets(tiles: List[np.ndarray]) -> List[Image.Image]:
        """Paginate tiles (<=38 per sheet) into grids on a black background."""
        sheets: List[Image.Image] = []
        for start in range(0, len(tiles), MAX_TILES_PER_SHEET):
            chunk = tiles[start : start + MAX_TILES_PER_SHEET]
            sheets.append(ContactSheetService._compose_sheet(chunk))
        return sheets

    @staticmethod
    def _compose_sheet(tiles: List[np.ndarray]) -> Image.Image:
        cols, rows = ContactSheetService.grid_dims(len(tiles))

        sheet_w = MARGIN * 2 + cols * CELL_PX + (cols - 1) * GUTTER
        sheet_h = MARGIN * 2 + rows * CELL_PX + (rows - 1) * GUTTER
        canvas = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

        for idx, tile in enumerate(tiles):
            row, col = divmod(idx, cols)
            cell_x = MARGIN + col * (CELL_PX + GUTTER)
            cell_y = MARGIN + row * (CELL_PX + GUTTER)
            ContactSheetService._paste_centered(canvas, tile, cell_x, cell_y)

        return Image.fromarray(canvas)

    @staticmethod
    def _paste_centered(canvas: np.ndarray, tile: np.ndarray, cell_x: int, cell_y: int) -> None:
        """Resize tile to fit a CELL_PX square (keep aspect) and center it in the cell."""
        h, w = tile.shape[:2]
        scale = CELL_PX / max(h, w)
        tw = max(1, int(round(w * scale)))
        th = max(1, int(round(h * scale)))
        resized = cv2.resize(tile, (tw, th), interpolation=cv2.INTER_AREA)

        off_x = cell_x + (CELL_PX - tw) // 2
        off_y = cell_y + (CELL_PX - th) // 2
        canvas[off_y : off_y + th, off_x : off_x + tw] = resized
