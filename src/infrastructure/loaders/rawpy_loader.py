import rawpy
from typing import Any, ContextManager, Tuple
from src.domain.interfaces import IImageLoader


class RawpyLoader(IImageLoader):
    """
    Standard RAW loader (libraw).
    """

    def load(self, file_path: str) -> Tuple[ContextManager[Any], dict]:
        from typing import cast

        raw = rawpy.imread(file_path)

        # Map rawpy.sizes.flip to 0-3 CCW rotation scale
        # 0: 0°, 3: 180°, 5: 90° CCW (1), 6: 90° CW (3)
        flip_map = {0: 0, 3: 2, 5: 1, 6: 3}
        orientation = flip_map.get(raw.sizes.flip, 0)

        metadata = {
            "orientation": orientation,
            "raw_flip": raw.sizes.flip,
            "color_space": "Adobe RGB",
        }

        return cast(ContextManager[Any], raw), metadata
