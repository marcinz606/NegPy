import numpy as np

# Orange mask: blue is the densest dye layer at base, red the thinnest.
BASE_DENSITY = np.array([0.25, 0.55, 0.85], np.float32)
_HOLDER_T = 0.002
_DUST_T = 0.05
# Shapes past this many are gray, so the dye layers share the picture's structure.
_COLOR_SHAPES = 8


def film(h: int, w: int, seed: int = 0, shapes: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """A color negative in a black holder, as float32 (rgb, ir) transmittance.

    A clear-base rebate surrounds the picture; dust blocks RGB and IR at the same place. More *shapes* add texture.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32) / max(h, w, 1)
    scene = np.repeat((0.25 + 0.5 * x + 0.1 * np.sin(8 * y + seed))[..., None], 3, axis=-1)
    for i in range(shapes):
        cy, cx, r = rng.uniform(0, h / max(h, w)), rng.uniform(0, w / max(h, w)), rng.uniform(0.03, 0.15)
        tint = rng.uniform(-0.35, 0.35, 3) if i < _COLOR_SHAPES else rng.uniform(-0.35, 0.35)
        scene[(y - cy) ** 2 + (x - cx) ** 2 < r * r] += tint
    density = BASE_DENSITY + 1.6 * np.clip(scene, 0.0, 1.0)

    my, mx = h // 20, w // 20
    ry, rx = my + h // 12, mx + w // 20
    density[my : h - my, mx : w - mx][: ry - my] = BASE_DENSITY
    density[my : h - my, mx : w - mx][-(ry - my) :] = BASE_DENSITY
    density[my : h - my, mx : w - mx][:, : rx - mx] = BASE_DENSITY
    density[my : h - my, mx : w - mx][:, -(rx - mx) :] = BASE_DENSITY
    t = np.power(10.0, -density).astype(np.float32)
    ir = np.full((h, w), 0.9, np.float32)

    holder = np.ones((h, w), bool)
    holder[my : h - my, mx : w - mx] = False
    t[holder] = _HOLDER_T
    ir[holder] = _HOLDER_T

    for _ in range(12):
        cy, cx, r = rng.integers(0, max(h, 1)), rng.integers(0, max(w, 1)), rng.uniform(1.5, 5.0)
        speck = (np.arange(h)[:, None] - cy) ** 2 + (np.arange(w)[None, :] - cx) ** 2 < r * r
        t[speck] *= _DUST_T
        ir[speck] *= _DUST_T
    return t, ir


def negative(h: int, w: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """`film` as uint16 (rgb, ir), as a scanner returns it."""
    t, ir = film(h, w, seed)
    return (t * 65535).astype(np.uint16), (ir * 65535).astype(np.uint16)
