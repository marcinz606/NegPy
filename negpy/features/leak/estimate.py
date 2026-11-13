"""The leak under each shape, read from the film.

A leak is one light, so it adds exposure to the three layers in one ratio k, its color, times one
strength map: u_c = u_scene,c + A(s) F(t) k_c, with s along the shape and t across it. Under a
band, each cell's shadow spike is read against the clean film past that side's edge; the color
comes from lit rebate the band crosses, else from the readings themselves; the strength along the
band is the readings projected onto that color, and is capped along the band by the thinnest film
under it, so it never carves the picture cell by cell."""

from dataclasses import replace
from typing import List, NamedTuple, Optional, Sequence

import negpy_core
import numpy as np

from negpy.features.leak.film import FilmResponse, Rect, read_film
from negpy.features.leak.models import LeakColor, LeakKind, LeakShape
from negpy.features.leak.rebate import RebateSide, rebate_along, rebate_crossing, rebate_sides
from negpy.features.leak.shapes import band_falloff, band_frame, paint_weight, spot_frame

# A band's strength is read in cells _CELL_ALONG_PX long along its line, from the film within
# _NEAR_LINE of its width on each side, against a clean band _CLEAN_DEPTH widths deep past each edge.
_NEAR_LINE = 0.5
_CELL_ALONG_PX = 24
_CLEAN_DEPTH = 0.35
# A near spike this far under the clean one, in log1p(u), is a darker scene, not a reading.
_SCENE_STEP = 0.1
# A cell's shadow spike: the mode of its lowest _SPIKE_QUANTILE percent of log1p(u), in bins of
# _SPIKE_BIN, holding at least _SPIKE_MIN_SHARE of the cell; fewer than _SPIKE_MIN_PIXELS reads blank.
_SPIKE_QUANTILE = 25.0
_SPIKE_BIN = 0.04
_SPIKE_MIN_SHARE = 0.06
_SPIKE_MIN_PIXELS = 40
# The curve through a band's readings: its curvature penalty is 10^(_ENVELOPE_LOG_LAMBDA +
# _ENVELOPE_LOG_SPAN x Smoothness) in cells.
_ENVELOPE_LOG_LAMBDA = -1.0
_ENVELOPE_LOG_SPAN = 3.0
_ENVELOPE_ITERATIONS = 30
# Smoothness of every curve along a shape.
_SMOOTHNESS = 0.3
# Tukey's biweight tuning constant, in robust spreads of the readings about the curve.
_ROBUST_TUNING = 4.685
_RIDGE = 1e-6
# The share of film a cell needs to be read. The cap reads this percentile of the leak each pixel
# near a band's line allows, past a scratch's clear line, and is smoothed over _CAP_SMOOTH cells
# along the band in log.
CAP_FILM_FRACTION = 0.25
_CAP_PERCENTILE = 1.0
_CAP_NEAR = 0.5
_CAP_OPEN = 1e6
_CAP_SMOOTH = 1.0
# The local cap's grid, cells on the long side, smoothed over _LOCAL_CAP_SMOOTH cells in log.
_LOCAL_CAP_GRID = 48
_LOCAL_CAP_SMOOTH = 0.6
# A cap takes over from the field over this width in log.
_BLEND_SOFTNESS = 0.1
# Leak colors, as exposure per layer (red-, green-, blue-sensitive). Light through the base reaches
# the red-sensitive layer first and little gets past it; light on the emulsion side, through the lens
# or a gap at the gate, reaches the blue-sensitive layer first, and the yellow filter under it holds
# some back from the others. These ratios stand in only when the film shows no color of its own, and
# a color nothing reads is neutral. The color is read from at least _MIN_COLOR_CELLS cells.
_BACK_COLOR = np.array([1.0, 0.2, 0.05])
_FRONT_COLOR = np.array([0.5, 0.8, 1.0])
_NEUTRAL = np.array([1.0, 1.0, 1.0])
_MIN_COLOR_CELLS = 3
# A reading straying from the leak's color by more than this share of its size, or of _COLOR_FLOOR
# excess exposure for a small one, is scene, not leak.
_COLOR_STRAY = 0.5
_COLOR_FLOOR = 0.2
# Where a cell holds no shadow spike, its strength is fitted across the band instead: a straight
# scene ramp per layer plus the leak's color times its falloff, over _PROFILE_MIN_PIXELS or more,
# with _PROFILE_ITERATIONS robust passes.
_PROFILE_MIN_PIXELS = 60
_PROFILE_ITERATIONS = 3
# An edge leak is read in _EDGE_DEPTH_BINS bins in from the edge to its reach; each cell along the
# edge fits its strength and how far it reaches, from a _EDGE_DEPTH_BINS-th of the reach to all of it.
_EDGE_DEPTH_BINS = 12
# A spot is read in _SPOT_SECTORS sectors around its center.
_SPOT_SECTORS = 8
# A painted leak is read in _PAINT_CELL_PX cells against a ring of clean film _PAINT_RING_PX deep around
# it, whose shadow level spreads in over _PAINT_SPREAD cells; unread cells take the strength of
# those within _PAINT_SMOOTH cells.
_PAINT_CELL_PX = 16
_PAINT_RING_PX = 24
_PAINT_SPREAD = 3.0
_PAINT_SMOOTH = 1.0
# The strength map is the opening of the cells over _PAINT_OPEN cells, blurred over _PAINT_BLUR; its color
# is read in cells over _PAINT_STRONG of the strongest.
_PAINT_OPEN = 5
_PAINT_BLUR = 2.0
_PAINT_STRONG = 0.5
# A spot's color is read at this percentile of its core over its falloff, where that is within
# _SHADOW_LEAK of the leak its sectors read: shadows there.
_COLOR_FLOOR_PERCENTILE = 1.0
_SHADOW_LEAK = 1.5
# Light spreading in the base thins out exponentially: an edge leak fades as e^(-_EDGE_FADE x) over
# its reach x, eased to nothing at the reach. The soft shadow of a gap at the gate holds a shoulder
# by the line instead, e^(-(_EDGE_FADE x)^2 / 2).
_EDGE_FADE = 3.0
_EDGE_SHOULDER = 2.0
_FADE_MIN_CELLS = 3
_FADE_MIN_FIT = 0.5
# Along lit rebate, an edge leak is the rebate's level times one scale, reaching in by one reach
# times the level's share of its peak to a power of _PROFILE_POWERS; the fit sits under all but
# _PROFILE_QUANTILE of the readings, as scene only adds to them.
_PROFILE_POWERS = (0.0, 0.5, 1.0, 1.5, 2.0)
# The fade's shape is fitted to the readings over each cell's level, pooled at each depth by their
# _POOL_PERCENTILE across the cells, where _POOL_MIN_BINS depths hold _POOL_MIN_CELLS readings: a
# reading over _POOL_OVER of the level is scene, as a leak never rises inward. The shoulder must fit
# the pool _SHOULDER_MARGIN times better than the exponential, which a smooth gradient in the scene
# meets as well.
_POOL_MIN_BINS = 6
_POOL_MIN_CELLS = 3
_POOL_OVER = 1.25
_POOL_PERCENTILE = 25.0
_POOL_REACHES = 60
_SHOULDER_MARGIN = 0.5
# A leak with a steep tip: lit rebate that rises by _TIP_RATIO or more over the _TIP_NEAR of the long
# side next to the edge (1 px bins, at least _TIP_MIN_BINS, read at _TIP_PERCENTILE), in _TIP_CELLS of
# the cells. Its fade into the picture is fitted over _TIP_REACHES reaches from 1 px to the edge's.
_TIP_NEAR = 0.006
_TIP_MIN_BINS = 3
_TIP_RATIO = 2.0
_TIP_PERCENTILE = 25.0
_TIP_CELLS = 0.5
_TIP_REACHES = 40
# A tip's strength along the edge is read per 1 px from the rebate within _TIP_EDGE_PX of the edge,
# smoothed over _TIP_ALONG_BLUR px in log.
_TIP_EDGE_PX = 3
_TIP_ALONG_BLUR = 1.5
# A tip's scale is capped at this percentile of the local mean over its strength near the edge:
# its tongues slant across the 1 px columns it is read in, so the lowest ratios are shape, not subject.
_TIP_CAP_PERCENTILE = 10.0
_PROFILE_SCALES = 51
_PROFILE_QUANTILE = 0.25
# Rebate along an edge is lit by a leak when its strongest layer's median reads this much exposure.
REBATE_LIT = 0.5
# The picture right inside a lit edge carries at least this share of the rebate's light, and at
# most its inverse, when the leak crosses the frame line.
_CONTINUITY = 0.5
# The frame line is read _LINE_PX work pixels either side of it: the rebate's median outside, the
# picture's _LINE_PERCENTILE inside, from at least _LINE_MIN_PIXELS each. A leak's light at the line
# itself is carried at _LINE_CONTINUITY of the rebate's or more in each layer; only the scan's blur
# at the line stands between them.
_LINE_PX = 3
_LINE_CONTINUITY = 0.65
_LINE_PERCENTILE = 5.0
_LINE_MIN_PIXELS = 8
# A lit rebate is fog where fewer than _CROSSING_SHARE of its lit cells continue across the frame line.
# A leak crosses where at least that share of the read cells carry its light, from _CROSSING_MIN_CELLS
# cells or more.
_CROSSING_SHARE = 0.5
_CROSSING_MIN_CELLS = 3
# A crossing needs the rebate's light at the line, over the stretch, at _ABOVE_SCENE times the clean
# film's shadow level or more, and at most _LINE_OVER_STRIP times the strip's thin end: a gap's light
# lights the gap across, where picture past a crop set inside it has bare gap beyond.
_ABOVE_SCENE = 1.75
_LINE_OVER_STRIP = 10.0
# A crop set past the frame line holds rebate, up to _GAP_SEARCH of the long side in: film whose detail
# against a blur over _GAP_DETAIL_SIGMA stays under _GAP_TEXTURE times that of the rebate _LINE_PX to _GAP_OUTSIDE_PX
# outside, starting within _GAP_OUTSIDE_LEVEL of its level. It ends where the detail rises for _GAP_ROUGH_PX
# steps and the _GAP_PAST_PX past reads _GAP_RISE times the rebate's, at the sharpest step within
# _GAP_EDGE_PX; under _GAP_MIN_PX is none. Smooth film within _GAP_STEP_PX past a rise and _GAP_LEVEL times
# dimmer is still rebate. Its width along a side is a straight line, cells off it by more than _GAP_LINE_PX
# (or twice the median offset) set aside, and a field _GAP_CLEAR times its local mean clears it.
_GAP_SEARCH = 0.06
_GAP_DETAIL_SIGMA = 3.0
_GAP_TEXTURE = 1.5
_GAP_OUTSIDE_PX = 8
_GAP_LEVEL = 1.33
_GAP_OUTSIDE_LEVEL = 1.67
_GAP_ROUGH_PX = 2
_GAP_MIN_PX = 5
_GAP_PAST_PX = 10
_GAP_RISE = 2.0
_GAP_EDGE_PX = 5
_GAP_STEP_PX = 20
_GAP_CLEAR = 1.5
_GAP_LINE_PX = 3.0
# With no crossing, the picture past the rebate reads _GAP_LONE_RISE times its detail or more.
_GAP_LONE_RISE = 2.5
# The local mean the leak comes off: a blur just past grain at the work size.
_MEAN_SIGMA = 1.0
# Below this peak excess exposure the frame has no leak worth subtracting.
_MIN_LEAK = 1e-3


def spike_floor(values: np.ndarray) -> np.ndarray:
    """The level a cell's shadows pile up at, per channel, in excess exposure; NaN where no pile-up.

    A leak lifts every shadow thinner than it to about its own level, so the low end of the
    cell's log-exposure histogram spikes there: the mode of the lowest quartile. A cell whose low
    end is spread out holds no shadows to read and is left blank."""
    out = np.full(values.shape[-1], np.nan)
    if values.shape[0] < _SPIKE_MIN_PIXELS:
        return out
    logs = np.log1p(np.maximum(values, 0.0))
    for c in range(values.shape[-1]):
        v = logs[:, c]
        top = float(np.percentile(v, _SPIKE_QUANTILE))
        low = v[v <= top]
        bottom = float(low.min())
        if top - bottom < 2.0 * _SPIKE_BIN:
            out[c] = np.expm1(float(np.median(low)))
            continue
        hist, edges = np.histogram(low, bins=int(np.ceil((top - bottom) / _SPIKE_BIN)), range=(bottom, top))
        hist = np.convolve(hist, (1.0, 2.0, 1.0), mode="same") / 4.0
        k = int(np.argmax(hist))
        share = float(hist[max(0, k - 1) : k + 2].sum()) / v.size
        if share >= _SPIKE_MIN_SHARE:
            out[c] = np.expm1(0.5 * (edges[k] + edges[k + 1]))
    return out


def _floor_readings(
    u: np.ndarray, film: np.ndarray, inside: np.ndarray, side: np.ndarray, a_index: np.ndarray, weight: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """((n, 2, 3) readings of the strength along a band, per side, NaN where a cell has none,
    (n, 2) whether a side's scene steps darker toward the line). A reading is the shadow spike of
    the film near the line less that of the clean film past that side's edge, divided by the
    falloff there. A leak only adds, in every layer, so a near spike under the clean one in any layer
    by more than _SCENE_STEP in log is a darker scene, not a reading, and is left blank."""
    readings = np.full((n, 2, 3), np.nan)
    steps = np.full((n, 2), np.nan)
    for k, sign in enumerate((1.0, -1.0)):
        on_side = film & inside & (side * sign >= 0.0)
        near = on_side & (np.abs(side) < _NEAR_LINE)
        clean_zone = on_side & (np.abs(side) >= 1.0) & (np.abs(side) < 1.0 + _CLEAN_DEPTH)
        for i in range(n):
            cell = near & (a_index == i)
            if not cell.any():
                continue
            near_spike, clean_spike = spike_floor(u[cell]), spike_floor(u[clean_zone & (a_index == i)])
            drop = np.log1p(np.maximum(clean_spike, 0.0)) - np.log1p(np.maximum(near_spike, 0.0))
            if np.isfinite(drop).any():
                steps[i, k] = float(np.nanmax(drop) > _SCENE_STEP)
            if steps[i, k] != 1.0:
                readings[i, k] = (near_spike - clean_spike) / max(float(weight[cell].mean()), 1e-3)
    return readings, steps


def _sides_mean(values: np.ndarray) -> np.ndarray:
    """The mean over a band's two sides where read, NaN where neither is."""
    known = ~np.isnan(values)
    count = known.sum(axis=1)
    return np.where(count > 0, np.where(known, values, 0.0).sum(axis=1) / np.maximum(count, 1), np.nan)


def _band_color(band: LeakShape, readings: np.ndarray, sides: List[RebateSide], rect: Optional[Rect], shape) -> np.ndarray:
    """The leak's color, its peak layer at 1: lit rebate the band crosses holds no scene, so it shows
    the color itself; else the readings' common direction; else the preset for its side."""
    if band.color is LeakColor.FRONT:
        return _FRONT_COLOR.copy()
    if band.color is LeakColor.BACK:
        return _BACK_COLOR.copy()
    if rect is not None and sides:
        h, w = shape
        (x0, y0), (x1, y1) = band.points[:2]
        a, b = np.array([x0 * w, y0 * h]), np.array([x1 * w, y1 * h])
        d = (b - a) / max(float(np.hypot(*(b - a))), 1e-6)
        lift = rebate_crossing((a + b) / 2.0, d, rect, sides)
        if lift is not None and lift.size and float(lift.max()) > 0.0:
            return lift / float(lift.max())
    medians = [float(np.median(c[np.isfinite(c)])) if np.isfinite(c).any() else -np.inf for c in readings.T]
    ref = int(np.argmax(medians)) if np.isfinite(readings).any() else 0
    rows = readings[np.all(np.isfinite(readings), axis=1)]
    lead = rows[:, ref] if rows.size else rows
    if lead.size and (lead > 0.0).sum() >= _MIN_COLOR_CELLS:
        # Each layer's median ratio to the leak's strongest, over the cells where the leak reads
        # strongest: the scene across a band adds differences of its own color to some cells, and
        # the median sets them aside.
        strong = rows[lead >= float(np.median(lead[lead > 0.0]))]
        if strong.shape[0] >= _MIN_COLOR_CELLS:
            k = np.maximum(np.median(strong / strong[:, ref : ref + 1], axis=0), 0.0)
            return k / float(k.max())
    return _NEUTRAL.copy()


def _robust_curve(values: np.ndarray, seen: np.ndarray, smoothness: float) -> np.ndarray:
    """A smooth curve through (n,) readings with Tukey's biweights: the leak is smooth along its
    line, so a reading off its neighbors either way, a dark subject on the line or a bright one, is
    set aside rather than followed."""
    n = values.size
    if n < 3:
        return np.where(seen, values, 0.0)
    d = np.diff(np.eye(n), n=2, axis=0)
    penalty = (d.T @ d) * 10.0 ** (_ENVELOPE_LOG_LAMBDA + _ENVELOPE_LOG_SPAN * float(np.clip(smoothness, 0.0, 1.0)))
    y = np.nan_to_num(values)
    w = seen.astype(np.float64)
    z = y
    for _ in range(_ENVELOPE_ITERATIONS):
        # A light ridge keeps the fit defined where the weights leave fewer than two readings.
        z = np.linalg.solve(np.diag(w + _RIDGE) + penalty, w * y)
        r = (y - z)[seen]
        scale = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 1e-3)
        new_w = np.where(seen, np.clip(1.0 - ((y - z) / (_ROBUST_TUNING * scale)) ** 2, 0.0, 1.0) ** 2, 0.0)
        if np.allclose(new_w, w, atol=1e-4):
            break
        w = new_w
    return z


def _profile_strength(u: np.ndarray, side: np.ndarray, weight: np.ndarray, k: np.ndarray) -> Optional[float]:
    """The leak's strength across one slice of a band with no shadows to read: per layer a straight
    ramp of scene across the band plus the leak's color times its falloff, fitted with robust
    weights so an edge in the scene does not lift it."""
    m = u.shape[0]
    if m < _PROFILE_MIN_PIXELS:
        return None
    rows = np.zeros((3 * m, 7))
    for c in range(3):
        rows[c * m : (c + 1) * m, c] = 1.0
        rows[c * m : (c + 1) * m, 3 + c] = side
        rows[c * m : (c + 1) * m, 6] = k[c] * weight
    y = u.T.reshape(-1).astype(np.float64)
    w = np.ones_like(y)
    theta = None
    for _ in range(_PROFILE_ITERATIONS):
        theta = np.linalg.lstsq(rows * w[:, None], y * w, rcond=None)[0]
        r = y - rows @ theta
        scale = max(1.4826 * float(np.median(np.abs(r))), 1e-6)
        # Square roots of Tukey's biweights, as they scale the rows.
        w = np.clip(1.0 - (r / (_ROBUST_TUNING * scale)) ** 2, 0.0, 1.0)
    return float(theta[6]) if theta is not None else None


def _slice_cap(
    u: np.ndarray,
    film: np.ndarray,
    inside: np.ndarray,
    a_index: np.ndarray,
    weight: np.ndarray,
    k: np.ndarray,
    n: int,
    profile: Optional[np.ndarray] = None,
    percentile: float = _CAP_PERCENTILE,
) -> np.ndarray:
    """Along the band, the most strength the film near the line allows in the leak's strongest
    layer, which no leak can take below its base, smoothed in log along the band; a scale of
    *profile* where one is given. The weaker layers' share of the color is the least sure part of
    it, and the local cap guards them."""
    lead = int(np.argmax(k))
    cap = np.full(n, _CAP_OPEN)
    near = film & inside & (weight >= _CAP_NEAR)
    for i in range(n):
        sel = near & (a_index == i)
        if sel.sum() < _SPIKE_MIN_PIXELS:
            continue
        allowed = u[sel][:, lead] / (k[lead] * weight[sel] * (1.0 if profile is None else np.maximum(profile[sel], 1e-6)))
        cap[i] = max(float(np.percentile(allowed, percentile)), 0.0)
    known = cap < _CAP_OPEN
    if not known.any():
        return cap
    logs = np.log1p(np.where(known, cap, 0.0)).astype(np.float32)[:, None]
    mask = known.astype(np.float32)[:, None]
    blur = negpy_core.gaussian_blur(logs * mask, (1, 0), 0.1, _CAP_SMOOTH)[:, 0]
    norm = negpy_core.gaussian_blur(mask, (1, 0), 0.1, _CAP_SMOOTH)[:, 0]
    # Smoothing must never lift a cell past its own reading.
    return np.where(known, np.minimum(np.expm1(blur / np.maximum(norm, 1e-6)), cap), _CAP_OPEN)


def _local_cap(u: np.ndarray, film: np.ndarray) -> np.ndarray:
    """Per pixel and layer, the leak the thinnest film nearby allows: a low percentile per cell of
    the local mean the leak comes off, so grain dips do not hold it down,
    interpolated and smoothed in log over _LOCAL_CAP_SMOOTH cells, so it has no corner at a cell's
    center to show as a seam. A cell that is not film leaves the leak open. Where a shape's falloff
    reaches past its leak, this keeps a dark subject there from going to bare base."""
    h, w = u.shape[:2]
    gh, gw = (
        (max(3, round(_LOCAL_CAP_GRID * h / w)), _LOCAL_CAP_GRID) if w >= h else (_LOCAL_CAP_GRID, max(3, round(_LOCAL_CAP_GRID * w / h)))
    )
    rows = np.array_split(np.arange(h), gh)
    cols = np.array_split(np.arange(w), gw)
    cap = np.full((gh, gw, 3), _CAP_OPEN, dtype=np.float32)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            keep = film[r[0] : r[-1] + 1, c[0] : c[-1] + 1]
            if keep.mean() >= CAP_FILM_FRACTION:
                cap[i, j] = np.percentile(u[r[0] : r[-1] + 1, c[0] : c[-1] + 1][keep], _CAP_PERCENTILE, axis=0)
    log_cap = negpy_core.resize_linear(np.ascontiguousarray(np.log1p(np.maximum(cap, 0.0)), dtype=np.float32), (w, h))
    sigma = _LOCAL_CAP_SMOOTH * max(h, w) / float(_LOCAL_CAP_GRID)
    return np.expm1(negpy_core.gaussian_blur(log_cap, (0, 0), sigma)).astype(np.float32)


class ShapeReading(NamedTuple):
    color: np.ndarray
    field: np.ndarray
    # Whether the local cap holds the field; a field fitted pixel by pixel never reaches past its leak.
    held: bool = True
    # Rebate inside the crop, cleared whole and never held: it holds no scene.
    free: Optional[np.ndarray] = None


class EdgeRows(NamedTuple):
    """An edge leak read cell by cell along its edge: color, strength and reach as a share of the
    edge's, NaN where unread, and whether lit rebate the picture carries sets the strength."""

    color: np.ndarray
    amplitude: np.ndarray
    extent: np.ndarray
    supported: np.ndarray
    along: np.ndarray
    depth: np.ndarray
    length: float
    taper: float
    reach: float
    n: int
    a_index: np.ndarray
    inside: np.ndarray
    # Read from the rebate's profile, smooth along the edge already.
    profiled: bool = False
    # Fades from a shoulder by the line, as the shadow of a gap at the gate does.
    shoulder: bool = False
    # A steep tip's strength per 1 px along the edge and layer, read from the rebate at the edge.
    tip: Optional[np.ndarray] = None
    # Pixels of rebate between the edge's line and the frame line, inside a crop set past it.
    gap: int = 0


class _Context(NamedTuple):
    film: np.ndarray
    response: FilmResponse
    rect: Optional[Rect]
    sides: List[RebateSide]
    profiles: dict


def _in_color(cells: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Readings (..., 3) as strength along the color, NaN where a reading is not in the leak's color:
    a leak adds exposure in its own ratio, so a reading whose layers stray from it by more than
    _COLOR_STRAY of its size is a change in the scene, as sky against ground."""
    known = np.isfinite(cells)
    r = np.where(known, cells, 0.0)
    weight = np.where(known, k, 0.0)
    level = (r * weight).sum(axis=-1) / np.maximum((weight * weight).sum(axis=-1), 1e-12)
    stray = np.linalg.norm(np.where(known, r - level[..., None] * k, 0.0), axis=-1)
    size = np.linalg.norm(np.where(known, level[..., None] * k, 0.0), axis=-1)
    ok = known.any(axis=-1) & (stray <= _COLOR_STRAY * np.maximum(size, _COLOR_FLOOR))
    return np.where(ok, level, np.nan)


def _band_reading(u: np.ndarray, band: LeakShape, ctx: _Context) -> Optional[ShapeReading]:
    """The leak under one band, (h, w, 3) excess exposure per layer, and its color."""
    film, sides, rect, smoothness, falloff, strength = ctx.film, ctx.sides, ctx.rect, _SMOOTHNESS, band.falloff, band.strength
    shape = u.shape[:2]
    frame = band_frame(band, shape)
    if frame is None:
        return None
    along, side, length, taper = frame
    inside = (along >= 0.0) & (along <= length)
    if not (film & inside & (np.abs(side) < 1.0)).any():
        return None
    n = max(2, int(np.ceil(length / _CELL_ALONG_PX)))
    a_index = np.clip((along / length * n).astype(np.int64), 0, n - 1)
    weight = band_falloff(side, falloff)
    readings, steps = _floor_readings(u, film, inside, side, a_index, weight, n)
    k = _band_color(band, _sides_mean(readings), sides, rect, shape)
    cap = _slice_cap(u, film, inside, a_index, weight, k, n)
    amplitude = _sides_mean(_in_color(readings, k)[..., None])[:, 0]
    # Where the scene steps darker across the band on both sides, the clean film is no reference,
    # and the cap, the most leak the darkest film near the line allows, is the reading.
    stepped = np.isnan(amplitude) & np.all(np.isfinite(steps), axis=1) & np.all(steps, axis=1) & (cap < _CAP_OPEN)
    amplitude = np.where(stepped, cap, amplitude)
    across = film & inside & (np.abs(side) < 1.0 + _CLEAN_DEPTH)
    for i in np.flatnonzero(np.isnan(amplitude)):
        sel = across & (a_index == i)
        fitted = _profile_strength(u[sel], side[sel], weight[sel], k)
        if fitted is not None:
            amplitude[i] = fitted
    seen = np.isfinite(amplitude)
    if not seen.any():
        return None
    curve = np.maximum(_robust_curve(amplitude, seen, smoothness), 0.0)
    curve = soft_min(curve * strength, cap)
    position = np.clip(along / length * n - 0.5, 0.0, n - 1.0)
    per_pixel = np.interp(position.ravel(), np.arange(n), curve).reshape(along.shape)
    ends = _taper_ends(along, length, taper)
    field = (per_pixel * weight * ends).astype(np.float32)[..., None] * k.astype(np.float32)
    return ShapeReading(k, field)


def soft_min(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smaller of two leak fields, blended over _BLEND_SOFTNESS in log so the switch leaves no seam."""
    la, lb = np.log1p(np.maximum(a, 0.0)), np.log1p(np.maximum(b, 0.0))
    return np.maximum(np.expm1(-_BLEND_SOFTNESS * np.logaddexp(-la / _BLEND_SOFTNESS, -lb / _BLEND_SOFTNESS)), 0.0)


def _edge_fade(x: np.ndarray, shoulder: bool = False) -> np.ndarray:
    """An edge leak's share at depth x in its reach: full at the edge and outside it, fading
    exponentially inward, or from a *shoulder*, and eased to nothing at 1."""
    p = _EDGE_SHOULDER if shoulder else 1.0
    floor = np.exp(-(_EDGE_FADE**p) / p)
    inside = (np.exp(-((_EDGE_FADE * np.clip(x, 0.0, 1.0)) ** p) / p) - floor) / (1.0 - floor)
    return np.where(x <= 0.0, 1.0, np.where(x < 1.0, inside, 0.0))


def _tip_fade(x: np.ndarray) -> np.ndarray:
    """A steep tip's share at depth x in its reach: e^(-_EDGE_FADE x), the rate its rebate measures,
    eased to nothing at 2 rather than 1 so its tail stays."""
    floor = np.exp(-2.0 * _EDGE_FADE)
    inside = (np.exp(-_EDGE_FADE * np.clip(x, 0.0, 2.0)) - floor) / (1.0 - floor)
    return np.where(x <= 0.0, 1.0, np.where(x < 2.0, inside, 0.0))


def _gap_detail(u: np.ndarray) -> np.ndarray:
    """Per pixel, the film's detail against a blur: rebate holds no scene and reads smooth."""
    lu = np.log1p(np.maximum(u, 0.0)).mean(axis=-1).astype(np.float32)
    return np.abs(lu - negpy_core.gaussian_blur(lu, (0, 0), _GAP_DETAIL_SIGMA))


def _gap_width(u: np.ndarray, detail: np.ndarray, step: np.ndarray, search: int, open_end: bool = False) -> int:
    """Pixels of rebate in from a line, over film pixels *u* (m, 3) at whole-pixel depths *step*, 0
    when the line is the frame line. A crop set past the frame line holds rebate inside it: film as
    smooth as the rebate outside until the picture's texture starts, within *search*. Where differently
    lit stretches of rebate meet, the edge between them reads as texture too, but smooth film follows
    within _GAP_STEP_PX and it is dimmer, toward bare base; past the frame edge the picture is not, and
    the search runs *search* on from the dimmer stretch. With *open_end*, rebate that runs on past the
    search reaches it."""
    out = (step >= -_GAP_OUTSIDE_PX) & (step < -_LINE_PX)
    if out.sum() < _SPIKE_MIN_PIXELS:
        return 0
    outside = np.median(u[out], axis=0)
    shown = outside >= _CONTINUITY * REBATE_LIT
    if not shown.any():
        return 0
    span = _gap_span(search)
    tex = np.full(span, np.nan)
    lev = np.full((span, int(shown.sum())), np.nan)
    for d in range(span):
        row = step == d
        if row.sum() >= _LINE_MIN_PIXELS:
            tex[d] = float(np.median(detail[row]))
            lev[d] = np.median(u[row], axis=0)[shown]
    seen = np.flatnonzero(np.isfinite(tex[:search]))
    if seen.size == 0:
        return 0
    ratio = lev[seen[0]] / outside[shown]
    if tex[seen[0]] > _GAP_TEXTURE * float(np.median(detail[out])) or not np.all(
        (ratio >= 1.0 / _GAP_OUTSIDE_LEVEL) & (ratio <= _GAP_OUTSIDE_LEVEL)
    ):
        return 0
    level = lev[seen[0]]
    textures: list = []
    rough = 0
    d = int(seen[0])
    limit = search
    while d < limit:
        if not np.isfinite(tex[d]):
            d += 1
            continue
        smooth = _GAP_TEXTURE * float(np.median(textures)) if textures else np.inf
        rough = rough + 1 if tex[d] > smooth else 0
        if not rough:
            textures.append(float(tex[d]))
            level = lev[d]
            d += 1
            continue
        if rough < _GAP_ROUGH_PX:
            d += 1
            continue
        end = d - (rough - 1)
        resume = _gap_resume(tex, end, smooth)
        if resume is not None and np.all(lev[resume] <= level / _GAP_LEVEL):
            # Light-struck film beyond the gap widens the rebate: the search runs on from the dimmer stretch.
            rough, d, limit = 0, resume, min(resume + search, 2 * search)
            continue
        past = tex[end : end + _GAP_PAST_PX]
        past = past[np.isfinite(past)]
        # The picture starts where its texture does: a gap that merely ends in smoother film is scene.
        if end < _GAP_MIN_PX or past.size == 0 or float(np.median(past)) < _GAP_RISE * float(np.median(textures)):
            return 0
        # The scan blurs the gap's light over the frame edge, the sharpest step in the film there.
        return end + int(np.argmax(np.nan_to_num(tex[end : end + _GAP_EDGE_PX], nan=0.0))) + 1
    return limit if open_end and textures else 0


def _gap_span(search: int) -> int:
    """Depth a rebate read reaches: the search, run on from a dimmer stretch, and the picture past it."""
    return 2 * search + _GAP_STEP_PX + _GAP_PAST_PX


def _gap_resume(tex: np.ndarray, end: int, smooth: float) -> Optional[int]:
    """The first depth within _GAP_STEP_PX past *end* where _GAP_ROUGH_PX steps in a row read smooth."""
    for d in range(end + 1, min(end + _GAP_STEP_PX, tex.size - _GAP_ROUGH_PX)):
        run = tex[d : d + _GAP_ROUGH_PX]
        if np.all(np.isfinite(run)) and np.all(run <= smooth):
            return d
    return None


def _gap_inside(u: np.ndarray, film: np.ndarray, depth: np.ndarray, long_side: float) -> int:
    """Pixels of rebate between an edge's line and the picture's frame line (_gap_width)."""
    search = int(round(_GAP_SEARCH * long_side))
    band = film & (depth >= -_GAP_OUTSIDE_PX - 1) & (depth < _gap_span(search))
    if not band.any():
        return 0
    return _gap_width(u[band], _gap_detail(u)[band], np.rint(depth[band]).astype(np.int64), search)


def _edge_frame(u: np.ndarray, edge: LeakShape, film: np.ndarray) -> Optional[tuple]:
    """(along, depth from the edge's line, rebate inside the crop past it) for an edge."""
    frame = band_frame(edge, u.shape[:2])
    if frame is None:
        return None
    along, side, length, _ = frame
    reach = max(edge.width[0] * float(max(u.shape[:2])), 4.0)
    depth = np.where(side >= 0.0, side * reach, side)
    return along, depth, _gap_inside(u, film & (along >= 0.0) & (along <= length), depth, float(max(u.shape[:2])))


def _lone_gap(u: np.ndarray, film: np.ndarray, frame: tuple) -> int:
    """The rebate inside the crop past an edge's line where no leak crosses into the picture, 0 when it
    is not lit or not far smoother than the picture past it: with no crossing to vouch for the film
    outside the crop, a smooth stretch of picture there reads as rebate too."""
    along, depth, gap = frame
    if gap <= 0:
        return 0
    lead = u.max(axis=-1)
    strip = film & (along >= 0.0) & (depth >= 1.0) & (depth < gap - 1.0)
    past = film & (along >= 0.0) & (depth >= gap + _LINE_PX) & (depth < gap + _LINE_PX + _GAP_PAST_PX)
    if strip.sum() < _SPIKE_MIN_PIXELS or past.sum() < _SPIKE_MIN_PIXELS:
        return 0
    detail = _gap_detail(u)
    lit = float(np.median(lead[strip])) >= REBATE_LIT
    # The rebate's own detail at its lower quartile: an edge between stretches of it reads as texture.
    smooth = float(np.median(detail[past])) >= _GAP_LONE_RISE * float(np.percentile(detail[strip], 25))
    return gap if lit and smooth else 0


def _gap_reading(u: np.ndarray, edge: LeakShape, ctx: _Context) -> Optional[ShapeReading]:
    """The lit rebate inside the crop beside an edge whose picture reads no leak: rebate lights without
    a leak crossing the frame line, at the start of a roll or beside fog, and it is cleared all the same."""
    frame = _edge_frame(u, edge, ctx.film)
    if frame is None:
        return None
    strip = _gap_strip(u, frame[0], frame[1], _lone_gap(u, ctx.film, frame), ctx, edge.strength)
    return None if strip is None else ShapeReading(_NEUTRAL.copy(), np.zeros(u.shape, np.float32), free=strip)


def lone_gap(reading, edge: LeakShape) -> int:
    """Pixels of lit rebate inside a read frame's crop past an edge's line where no leak crosses
    (_lone_gap), for Auto."""
    u = np.maximum(reading.response.excess(reading.small), 0.0).astype(np.float32)
    frame = _edge_frame(u, edge, reading.film)
    return 0 if frame is None else _lone_gap(u, reading.film, frame)


def _edge_rows(u: np.ndarray, edge: LeakShape, ctx: _Context) -> Optional[EdgeRows]:
    """The leak in from a stretch of picture edge, read from the frame line (_gap_inside)."""
    frame = _edge_frame(u, edge, ctx.film)
    if frame is None:
        return None
    gap = frame[2]
    rows = _edge_rows_at(u, edge, ctx, gap)
    return None if rows is None else rows._replace(gap=gap)


def _edge_rows_at(u: np.ndarray, edge: LeakShape, ctx: _Context, gap: int) -> Optional[EdgeRows]:
    """The leak in from a stretch of picture edge, (h, w, 3) excess exposure per layer, and its color.
    Light through a gap spreads in the base and reaches further where it is stronger, so each cell
    along the edge has its own strength and reach: the leak is A(s) G(t / R(s)) k at depth t, never
    rising inward. Each cell's readings in from the edge are against the clean film past the reach.
    Lit rebate outside the edge is the leak at the edge, with no scene in it: there it sets the
    color and the strength, and the picture sets only the reach. None where the rebate is fogged."""
    film, rect = ctx.film, ctx.rect
    shape = u.shape[:2]
    h, w = shape
    frame = band_frame(edge, shape)
    if frame is None:
        return None
    along, side, length, taper = frame
    reach = max(edge.width[0] * float(max(shape)), 4.0)
    depth = np.where(side >= 0.0, side * reach, side)
    inside = (along >= 0.0) & (along <= length)
    depth = depth - gap
    n = max(2, int(np.ceil(length / _CELL_ALONG_PX)))
    a_index = np.clip((along / length * n).astype(np.int64), 0, n - 1)
    bins = _EDGE_DEPTH_BINS
    t_index = np.clip((depth / reach * bins).astype(np.int64), 0, bins - 1)
    within = film & inside & (depth >= 0.0) & (depth < reach)
    clean_zone = film & inside & (depth >= reach) & (depth < reach * (1.0 + _CLEAN_DEPTH))
    if not within.any():
        return None
    readings = np.full((n, bins, 3), np.nan)
    clean = np.full((n, 3), np.nan)
    for i in range(n):
        row = a_index == i
        clean_spike = clean[i] = spike_floor(u[clean_zone & row])
        for j in range(bins):
            cell = within & row & (t_index == j)
            if not cell.any():
                continue
            near_spike = spike_floor(u[cell])
            drop = np.log1p(np.maximum(clean_spike, 0.0)) - np.log1p(np.maximum(near_spike, 0.0))
            if np.isfinite(drop).any() and float(np.nanmax(drop)) > _SCENE_STEP:
                continue
            readings[i, j] = near_spike - clean_spike
    near_bins = max(1, bins // 4)
    line = _frame_line(u, film, inside, depth, a_index, n)
    k, level, crossing, fogged = _edge_rebate(edge, readings, line, clean, rect, ctx, shape, n, near_bins)
    lit = bool(np.isfinite(level).all() and float(level.max()) > 0.0)
    # A steep tip proves its own crossing, carried to the line tongue by tongue; read a few pixels
    # either side, its light falls too fast to look continuous.
    tip = _steep_tip(u, film & inside, along, depth, a_index, n, length, reach) if lit else None
    if tip is not None:
        amplitude, extent, profile, k = tip
        lead = int(np.argmax(k))
        return EdgeRows(k, amplitude[:, lead], extent, crossing, along, depth, length, taper, reach, n, a_index, inside, tip=profile)
    if fogged:
        return None
    cells = _in_color(readings, k)
    centers = (np.arange(bins) + 0.5) / bins
    if lit:
        amplitude, extent, shoulder = _fit_rebate_profile(cells, centers, level, crossing.any())
        return EdgeRows(
            k, amplitude, extent, crossing, along, depth, length, taper, reach, n, a_index, inside, profiled=True, shoulder=shoulder
        )
    supported = np.zeros(n, dtype=bool)
    amplitude, extent = np.full(n, np.nan), np.full(n, np.nan)
    # A cell with too few readings, as one over smooth sky, takes its strength and reach from the
    # cells around it through the curve along the edge: its own film cannot tell a leak's fade from
    # the scene's.
    for i in range(n):
        seen_j = np.isfinite(cells[i])
        if seen_j.sum() < 2:
            continue
        # Where the leak crosses the frame line it is read best in the rebate, which holds no scene,
        # and the picture sets only how far in it reaches.
        fit = _fit_depth_cells(cells[i, seen_j], centers[seen_j], level[i] if crossing[i] else None)
        if fit is not None:
            amplitude[i], extent[i] = fit
            supported[i] = bool(crossing[i])
    return EdgeRows(k, amplitude, extent, supported, along, depth, length, taper, reach, n, a_index, inside)


def _edge_reading(u: np.ndarray, edge: LeakShape, ctx: _Context) -> Optional[ShapeReading]:
    """The leak in from a stretch of picture edge, (h, w, 3) excess exposure per layer, and its color."""
    rows = _edge_rows(u, edge, ctx)
    if rows is None:
        return _gap_reading(u, edge, ctx)
    k, amplitude, extent = rows.color, rows.amplitude, rows.extent
    along, depth, length, taper, reach, n, a_index, inside = (
        rows.along,
        rows.depth,
        rows.length,
        rows.taper,
        rows.reach,
        rows.n,
        rows.a_index,
        rows.inside,
    )
    film, smoothness, strength = ctx.film, _SMOOTHNESS, edge.strength
    bins = _EDGE_DEPTH_BINS
    seen = np.isfinite(amplitude) & np.isfinite(extent)
    if not seen.any():
        return _gap_reading(u, edge, ctx)
    position = np.clip(along / length * n - 0.5, 0.0, n - 1.0).ravel()
    gap = rows.gap if rows.supported.any() else _lone_gap(u, film, (along, depth + rows.gap, rows.gap))
    strip = _gap_strip(u, along, depth + rows.gap, gap, ctx, strength)
    tip = rows.tip
    if tip is not None:
        return _tip_reading(u, rows, tip, ctx, strength, position)._replace(free=strip)
    if not rows.profiled:
        amplitude = np.maximum(_robust_curve(amplitude, seen, smoothness), 0.0)
        extent = np.clip(_robust_curve(extent, seen, smoothness), 1.0 / bins, 1.0)
    extent_px = np.interp(position, np.arange(n), extent).reshape(along.shape) * reach
    weight = _edge_fade(depth / extent_px, rows.shoulder)
    read = inside & (depth >= 0.0)
    if rows.supported.any():
        # Under a proven crossing, film the fade holds at half strength never reads bare base in the
        # leak's lead layer: bare film there is rebate or a sprocket hole's rim, which a crop set past the
        # frame line can hold.
        read &= u[..., int(np.argmax(k))] > _CONTINUITY * REBATE_LIT
    cap = _slice_cap(u, film, read, a_index, weight, k, n)
    amplitude = soft_min(amplitude * strength, cap)
    per_pixel = np.interp(position, np.arange(n), amplitude).reshape(along.shape)
    ends = _taper_ends(along, length, taper)
    field = (per_pixel * weight * ends).astype(np.float32)[..., None] * k.astype(np.float32)
    return ShapeReading(k, field, free=strip)


def _gap_strip(u: np.ndarray, along: np.ndarray, depth: np.ndarray, gap: int, ctx: _Context, strength: float) -> Optional[np.ndarray]:
    """The leak in the rebate across the frame line of the side the edge lies on, where a crop set
    past that line shows it: rebate holds no scene, so all its exposure is leak. The rebate reaches in
    to the picture's texture and out to the next frame's, each read per cell along the side and fitted
    as a straight line, as the frame line need not run square to the crop. The field stands
    _GAP_CLEAR over the local mean, past the removal's soft floor, to take the rebate to bare base."""
    if gap <= 0 or ctx.rect is None:
        return None
    y0, y1, x0, x1 = ctx.rect
    box = np.zeros(u.shape[:2], dtype=bool)
    box[y0:y1, x0:x1] = True
    search = int(round(_GAP_SEARCH * float(max(u.shape[:2]))))
    band = ctx.film & (np.abs(depth) < _gap_span(search))
    # Rows of the box along this side, extended across the line.
    along_box = along[box & band]
    if along_box.size == 0:
        return None
    first, last = float(along_box.min()), float(along_box.max())
    side = band & (along >= first) & (along <= last)
    cell = np.clip(np.floor((along - first) / _CELL_ALONG_PX).astype(np.int64), 0, None)
    n = int(cell[side].max()) + 1
    detail = _gap_detail(u)
    step = np.rint(depth).astype(np.int64)
    inner, outer = np.zeros(n), np.zeros(n)
    for i in range(n):
        sel = side & (cell == i)
        if sel.sum() < _SPIKE_MIN_PIXELS:
            continue
        inner[i] = _gap_width(u[sel], detail[sel], step[sel], search)
        if inner[i] > 0:
            # Outward, the rebate just inside the line is the reference: read with depth mirrored.
            outer[i] = _gap_width(u[sel], detail[sel], -step[sel] - 1, search, open_end=True)
    centers = (np.arange(n) + 0.5) * _CELL_ALONG_PX + first
    reach_in = _gap_line(centers, inner, gap)(along)
    reach_out = _gap_line(centers, outer, 0.0)(along)
    strip = side & (depth < reach_in) & (depth >= -np.maximum(reach_out, 0.0))
    if not strip.any():
        return None
    mean = negpy_core.gaussian_blur(np.maximum(u, 0.0).astype(np.float32), (0, 0), _MEAN_SIGMA)
    return mean * strip.astype(np.float32)[..., None] * np.float32(_GAP_CLEAR * min(strength, 1.0))


def _gap_line(centers: np.ndarray, widths: np.ndarray, fallback: float):
    """The width of the rebate along a side as a straight line through the cells that read it, with
    cells off it by more than _GAP_LINE_PX set aside; *fallback* where fewer than three cells read."""
    read = widths > 0
    if read.sum() < 3:
        return lambda along: np.full(along.shape, fallback)
    x, y = centers[read], widths[read]
    keep = np.ones(x.size, dtype=bool)
    for _ in range(3):
        b, a = np.polyfit(x[keep], y[keep], 1)
        keep = np.abs(y - (a + b * x)) <= max(_GAP_LINE_PX, 2.0 * float(np.median(np.abs(y - (a + b * x)))))
        if keep.sum() < 3:
            break
    return lambda along: a + b * along


def _taper_ends(along: np.ndarray, length: float, taper: float) -> np.ndarray:
    """A shape's share past the ends of its line, eased to nothing over *taper*."""
    beyond = np.maximum(np.maximum(-along, along - length), 0.0)
    ends = np.clip(1.0 - beyond / taper, 0.0, 1.0)
    return ends * ends * (3.0 - 2.0 * ends)


def _tip_reading(u: np.ndarray, rows: EdgeRows, tip: np.ndarray, ctx: _Context, strength: float, position: np.ndarray) -> ShapeReading:
    """A steep tip's field: each layer's strength along the edge from the rebate, one fade, and a cap
    on each layer's scale, so the tongues keep their shape. A layer near its shoulder reads its
    exposure short on both sides of the frame line alike, which one color across the layers cannot
    follow. Fitted pixel by pixel, it never reaches past its leak: the local cap does not hold it."""
    along, depth, n, reach = rows.along, rows.depth, rows.n, rows.reach
    seen = np.isfinite(rows.extent)
    extent = np.clip(_robust_curve(rows.extent, seen, _SMOOTHNESS), 1.0 / reach, 1.0)
    weight = _tip_fade(depth / (np.interp(position, np.arange(n), extent).reshape(along.shape) * reach))
    near = ctx.film & rows.inside & (depth >= 0.0)
    ends = _taper_ends(along, rows.length, rows.taper)
    mean = negpy_core.gaussian_blur(np.ascontiguousarray(u, dtype=np.float32), (0, 0), _MEAN_SIGMA)
    field = np.zeros(u.shape, np.float32)
    for c in range(3):
        level = np.interp(along, np.arange(tip.shape[0]) + 0.5, tip[:, c]).astype(np.float32)
        cap = _slice_cap(mean, ctx.film, near, rows.a_index, weight, np.eye(3)[c], n, level, _TIP_CAP_PERCENTILE)
        # A plain minimum: the cap is smooth along the edge already, and a soft one shaves a tip
        # many times its scene by a visible share.
        scale = np.minimum(strength, cap)
        field[..., c] = np.interp(position, np.arange(n), scale).reshape(along.shape) * level * weight * ends
    return ShapeReading(rows.color, field, held=False)


def _fit_depth_cells(cells: np.ndarray, centers: np.ndarray, fixed: Optional[float] = None) -> Optional[tuple]:
    """(strength, reach as a share of the edge's) that best fit one cell's readings in from the edge,
    at each reach in turn; the strength is *fixed* when given. A free fit stands only on
    _FADE_MIN_CELLS readings or more whose fade explains _FADE_MIN_FIT of their spread: scene that
    differs from the clean film past the reach reads about evenly with depth, a leak fades."""
    best = None
    for frac in np.arange(1, _EDGE_DEPTH_BINS + 1) / _EDGE_DEPTH_BINS:
        g = _edge_fade(centers / frac)
        gg = float(g @ g)
        if gg <= 0.0:
            continue
        level = fixed if fixed is not None else max(float(g @ cells) / gg, 0.0)
        err = float(np.sum((cells - level * g) ** 2))
        if best is None or err < best[0]:
            best = (err, level, float(frac))
    if best is None:
        return None
    if fixed is None:
        spread = float(np.sum((cells - cells.mean()) ** 2))
        if cells.size < _FADE_MIN_CELLS or best[0] > (1.0 - _FADE_MIN_FIT) * spread:
            return None
    return best[1], best[2]


def _depth_profile(value: np.ndarray, step: np.ndarray, bins: int) -> np.ndarray:
    """(bins,) _TIP_PERCENTILE of *value* in each 1 px depth step, NaN where empty."""
    out = np.full(bins, np.nan)
    for j in range(bins):
        v = value[step == j]
        if v.size:
            out[j] = float(np.percentile(v, _TIP_PERCENTILE))
    return out


def _steep_tip(
    u: np.ndarray, film: np.ndarray, along: np.ndarray, depth: np.ndarray, a_index: np.ndarray, n: int, length: float, reach: float
) -> Optional[tuple]:
    """(strength per cell and layer and reach per cell, NaN where unread; strength per 1 px along the
    edge and layer; color) for a leak whose lit rebate rises steeply away from the edge, None for any
    other. Light spreading in the base is continuous across the frame line, and the rebate holds no
    scene: carried to the line, it is the leak's strength there, tongue by tongue, where it is lit
    (REBATE_LIT) and the picture's first row carries at least _CONTINUITY of it. The fade inward is
    one light's, fitted in the lead layer to the picture 1 px at a time, under its readings, as scene
    only adds to them."""
    near = max(_TIP_MIN_BINS, round(_TIP_NEAR * float(max(u.shape[:2]))))
    deep = max(1, int(np.ceil(reach)))
    outside = film & (depth < 0.0) & (depth >= -near)
    within = film & (depth >= 0.0) & (depth < deep)
    edge_px = outside & (depth >= -_TIP_EDGE_PX)
    if not edge_px.any() or not within.any():
        return None
    k = np.maximum(np.median(u[edge_px], axis=0), 0.0)
    if not float(k.max()) > 0.0:
        return None
    k = k / float(k.max())
    lead = int(np.argmax(k))
    out_u, out_cell, out_step = u[outside], a_index[outside], np.minimum((-depth[outside]).astype(np.int64), near - 1)
    in_u, in_cell, in_step = u[within], a_index[within], np.minimum(depth[within].astype(np.int64), deep - 1)
    out_d = np.arange(near) + 0.5
    reaches = np.geomspace(1.0, reach, _TIP_REACHES)
    fades = _tip_fade((np.arange(deep) + 0.5)[None, :] / reaches[:, None])
    amplitude, extent, slopes = np.full((n, 3), np.nan), np.full(n, np.nan), np.full((n, 3), np.nan)
    for i in range(n):
        o, w = out_cell == i, in_cell == i
        for c in [lead] + [c for c in range(3) if c != lead]:
            rebate = _depth_profile(out_u[o, c], out_step[o], near)
            seen = np.isfinite(rebate) & (rebate > 0.0)
            if seen.sum() < _TIP_MIN_BINS:
                break
            logs = np.log(rebate[seen])
            slope, offset = np.polyfit(out_d[seen], logs, 1)
            if c == lead:
                # A tip keeps rising away from the edge; the scan's blur at the frame line only steps.
                err = float(np.sum((logs - (offset + slope * out_d[seen])) ** 2))
                spread = float(np.sum((logs - logs.mean()) ** 2))
                if slope * float(out_d[seen].max() - out_d[seen].min()) < np.log(_TIP_RATIO) or err > (1.0 - _FADE_MIN_FIT) * spread:
                    break
            slope = max(float(slope), 0.0)
            level = float(np.exp(offset)) if slope > 0.0 else float(np.exp(np.log(rebate[seen]).mean()))
            amplitude[i, c], slopes[i, c] = level, slope
            if c != lead:
                continue
            picture = _depth_profile(in_u[w, c], in_step[w], deep)
            read = np.isfinite(picture)
            # Lit at the line itself, and the picture's first row carries it across, as edge fog does not.
            if level < REBATE_LIT or not (read[0] and picture[0] >= _CONTINUITY * level):
                amplitude[i] = np.nan
                break
            err = picture[read][None, :] - level * fades[:, read]
            loss = np.where(err >= 0.0, _PROFILE_QUANTILE * err, (_PROFILE_QUANTILE - 1.0) * err).sum(axis=1)
            # Scene only lifts the picture's readings, so the fit under them can run long; the
            # rebate's own fade, with no scene in it, bounds it.
            extent[i] = min(float(reaches[int(np.argmin(loss))]), _EDGE_FADE / slope) / reach
    if np.isfinite(amplitude[:, lead]).mean() < _TIP_CELLS:
        return None
    # Each column's rebate by the edge, carried to the line by its cell's slope.
    size = max(1, int(np.ceil(length)))
    col = edge_px & (along >= 0.0) & (along < size)
    bins = along[col].astype(np.int64)
    count = np.bincount(bins, minlength=size).astype(np.float64)
    have = count > 0
    if not have.any():
        return None
    centers = np.arange(size) + 0.5
    cells = np.clip(centers / length * n - 0.5, 0.0, n - 1.0)
    out_px = np.interp(centers, centers[have], (np.bincount(bins, weights=-depth[col], minlength=size) / np.maximum(count, 1.0))[have])
    profile = np.zeros((size, 3), np.float32)
    for c in range(3):
        logs = np.bincount(bins, weights=np.log(np.maximum(u[col][:, c], 1e-6)), minlength=size) / np.maximum(count, 1.0)
        known = np.isfinite(slopes[:, c])
        slope = np.interp(cells, np.arange(n)[known], slopes[known, c]) if known.any() else np.zeros(size)
        carried = np.interp(centers, centers[have], logs[have]) - slope * out_px
        profile[:, c] = np.exp(negpy_core.gaussian_blur(carried.astype(np.float32)[:, None], (1, 0), 0.1, _TIP_ALONG_BLUR)[:, 0])
    return amplitude, extent, profile, k


def _fit_rebate_profile(cells: np.ndarray, centers: np.ndarray, level: np.ndarray, crossing: bool) -> tuple:
    """(strength, reach as a share of the edge's, whether it fades from a shoulder) per cell from the
    rebate's *level* along the edge: strength c level, reach r (level / its peak)^p, fitted over every
    (n, bins) reading at once. A gap's light is one source, so the picture follows the rebate's
    profile, and a stretch whose own shadows show nothing takes its shape from it. c is 1 where the
    leak *crossing* the frame line is proven, and at most 1 otherwise. A shoulder the pooled readings
    show (_shoulder_reach) sets r too."""
    share = np.maximum(level, 0.0) / float(level.max())
    seen = np.isfinite(cells)
    i_idx, j_idx = np.nonzero(seen)
    values = cells[seen]
    scales = np.array([1.0]) if crossing else np.linspace(0.0, 1.0, _PROFILE_SCALES)
    pooled = _shoulder_reach(cells, centers, level)
    shoulder = pooled is not None
    reaches = np.arange(1, _EDGE_DEPTH_BINS + 1) / _EDGE_DEPTH_BINS if pooled is None else np.array([pooled])
    best = (np.inf, 1.0 if crossing else 0.0, float(reaches[-1]), 0.0)
    if values.size:
        for p in _PROFILE_POWERS:
            for r in reaches:
                cell_reach = np.clip(r * share**p, 1.0 / _EDGE_DEPTH_BINS, 1.0)
                model = level[i_idx] * _edge_fade(centers[j_idx] / cell_reach[i_idx], shoulder)
                err = values[None, :] - scales[:, None] * model[None, :]
                loss = np.where(err >= 0.0, _PROFILE_QUANTILE * err, (_PROFILE_QUANTILE - 1.0) * err).sum(axis=1)
                at = int(np.argmin(loss))
                if loss[at] < best[0]:
                    best = (float(loss[at]), float(scales[at]), float(r), float(p))
    _, c, r, p = best
    return c * np.maximum(level, 0.0), np.clip(r * share**p, 1.0 / _EDGE_DEPTH_BINS, 1.0), shoulder


def _shoulder_reach(cells: np.ndarray, centers: np.ndarray, level: np.ndarray) -> Optional[float]:
    """The reach, as a share of the edge's, of a leak that fades from a shoulder; None for one that
    fades exponentially. The readings over each cell's level, pooled at each depth, are fitted by least
    squares at every reach under one free scale, as a reading is a shadow floor and the level a median.
    Pooled, the scene in one cell weighs little; a fit under every reading at once meets a shoulder's
    slow start and quick end with a short exponential."""
    shares = cells / np.maximum(level, 1e-6)[:, None]
    shares = np.where(shares <= _POOL_OVER, shares, np.nan)
    count = np.isfinite(shares).sum(axis=0)
    held = count >= _POOL_MIN_CELLS
    if held.sum() < _POOL_MIN_BINS:
        return None
    profile = np.nanpercentile(np.where(held[None, :], shares, 0.0), _POOL_PERCENTILE, axis=0)[held]
    weight = count[held].astype(np.float64)
    reaches = np.linspace(1.0 / _EDGE_DEPTH_BINS, 1.0, _POOL_REACHES)
    fits = {}
    for shoulder in (False, True):
        errs = []
        for r in reaches:
            fade = _edge_fade(centers[held] / r, shoulder)
            scale = float(np.sum(weight * profile * fade) / max(float(np.sum(weight * fade * fade)), 1e-12))
            errs.append(float(np.sum(weight * (profile - scale * fade) ** 2)))
        at = int(np.argmin(errs))
        fits[shoulder] = (errs[at], float(reaches[at]))
    return fits[True][1] if fits[True][0] < _SHOULDER_MARGIN * fits[False][0] else None


def _frame_line(u: np.ndarray, film: np.ndarray, inside: np.ndarray, depth: np.ndarray, a_index: np.ndarray, n: int) -> tuple:
    """((n, 3) rebate median just outside an edge, (n, 3) picture floor just inside it) per cell along
    the edge, NaN where too little film. The floor is the picture's thinnest film, which a leak lifts
    to its own level and a scene does not."""
    outside, floor = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    out_px = film & inside & (depth < 0.0) & (depth >= -_LINE_PX)
    in_px = film & inside & (depth >= 0.0) & (depth < _LINE_PX)
    for i in range(n):
        o, p = out_px & (a_index == i), in_px & (a_index == i)
        if o.sum() >= _LINE_MIN_PIXELS and p.sum() >= _LINE_MIN_PIXELS:
            outside[i] = np.median(u[o], axis=0)
            floor[i] = np.percentile(u[p], _LINE_PERCENTILE, axis=0)
    return outside, floor


def _edge_rebate(
    edge: LeakShape,
    readings: np.ndarray,
    line: tuple,
    clean: np.ndarray,
    rect: Optional[Rect],
    ctx: "_Context",
    shape,
    n: int,
    near_bins: int,
) -> tuple:
    """(color, rebate strength per cell, whether each cell's leak crosses the frame line, whether the
    rebate is fogged). Lit rebate holds no scene, so its glow is the leak's color, whichever side of
    the film the light came from. A leak's light is continuous across the frame line, where fog, a bath
    or an edge print stops: on Auto color, rebate whose light the picture's floor does not carry at the
    line is fogged, and the picture beside it holds no leak to read. The leak crosses into the picture
    in the cells with shadows by the edge that carry the rebate's light at the line, between
    _CONTINUITY of it and its inverse, where _CROSSING_SHARE of the read cells do, and where the
    rebate's light stands _ABOVE_SCENE over the shadows of *clean*, the film past the reach, along the
    stretch: a dimmer glow is as likely picture past a crop set inside it."""
    none = np.zeros(n, dtype=bool)
    unread = np.full(n, np.nan)
    if edge.color is not LeakColor.AUTO:
        k = _FRONT_COLOR.copy() if edge.color is LeakColor.FRONT else _BACK_COLOR.copy()
    else:
        k = _edge_color(edge, readings)
    if rect is None or edge.color is LeakColor.FRONT:
        return k, unread, none, False
    h, w = shape
    (x0, y0), (x1, y1) = edge.points[:2]
    rebate = rebate_along(np.array([x0 * w, y0 * h]), np.array([x1 * w, y1 * h]), rect, ctx.profiles, ctx.response, n)
    if rebate is None or float(np.median(rebate.max(axis=-1))) < REBATE_LIT:
        return k, unread, none, False
    if edge.color is LeakColor.AUTO:
        glow = np.median(rebate, axis=0)
        k = glow / max(float(glow.max()), 1e-12)
    outside, floor = line
    lead = int(np.argmax(k))
    lit = np.nan_to_num(outside[:, lead], nan=0.0) >= REBATE_LIT
    if edge.color is LeakColor.AUTO and lit.sum() >= _CROSSING_MIN_CELLS:
        # Read at the line, as the strength is: the strip reaches past the gap, where a weaker glow
        # holds edge print and reads redder.
        glow = np.median(outside[lit], axis=0)
        k = glow / max(float(glow.max()), 1e-12)
        lead = int(np.argmax(k))
        lit = np.nan_to_num(outside[:, lead], nan=0.0) >= REBATE_LIT
    level = (rebate @ k) / float(k @ k)
    read = lit & np.isfinite(floor[:, lead])
    if edge.color is LeakColor.AUTO and read.any():
        # Carried in every layer the rebate shows: fog beside other light in the picture is carried in
        # some layers and not in its own.
        shown = outside[read] >= _CONTINUITY * REBATE_LIT
        carried = np.where(shown, floor[read] >= _LINE_CONTINUITY * outside[read], True)
        if np.all(carried, axis=1).mean() < _CROSSING_SHARE:
            return k, level, none, True
    at_line = (np.nan_to_num(outside) @ k) / float(k @ k)
    known = lit & np.isfinite(outside[:, lead])
    if known.any() and float(np.median(at_line[known])) > _LINE_OVER_STRIP * max(float(np.median(level[known])), 1e-6):
        return k, level, none, False
    # The strip's thin end reaches past the gap to where the leak has faded; the rebate's light at the
    # line is the leak where it crosses.
    if known.sum() >= 2:
        cells = np.arange(n)
        level = np.interp(cells, cells[known], at_line[known])
    above = outside[:, lead] / np.maximum(clean[:, lead], 1e-6)
    if read.any() and np.isfinite(above[read]).any() and float(np.nanmedian(above[read])) < _ABOVE_SCENE:
        return k, level, none, False
    near = _in_color(readings[:, :near_bins], k)
    seen = np.isfinite(near).any(axis=1) & lit
    if seen.sum() < _CROSSING_MIN_CELLS:
        return k, level, none, False
    carried = np.where(seen, np.nanmax(np.where(np.isfinite(near), near, -np.inf), axis=1), np.nan)
    ratio = carried / np.maximum(at_line, 1e-6)
    crossing = seen & (ratio >= _CONTINUITY) & (ratio <= 1.0 / _CONTINUITY)
    if crossing.sum() < max(_CROSSING_MIN_CELLS, _CROSSING_SHARE * seen.sum()):
        return k, level, none, False
    return k, level, crossing, False


def _edge_color(edge: LeakShape, readings: np.ndarray) -> np.ndarray:
    """An edge leak's color from the picture: its readings nearest the edge, else the preset for
    its side."""
    if edge.color is LeakColor.FRONT:
        return _FRONT_COLOR.copy()
    if edge.color is LeakColor.BACK:
        return _BACK_COLOR.copy()
    nearest = readings[:, : max(1, readings.shape[1] // 3)].reshape(-1, 3)
    return _band_color(replace(edge, color=LeakColor.AUTO), nearest, [], None, (1, 1))


def _spot_reading(u: np.ndarray, spot: LeakShape, ctx: _Context) -> Optional[ShapeReading]:
    """The leak under a spot, (h, w, 3) excess exposure per layer, and its color: one strength times
    a falloff from its center to its edge, as light through a pinhole or a frame-number window
    spreads. In sectors around the center, the shadow spike of the film near the center is read
    against the clean ring past the edge, as a band's sides are; the sectors' median is the
    strength, and the darkest film near the center caps it."""
    frame = spot_frame(spot, u.shape[:2])
    if frame is None:
        return None
    rho, angle = frame
    film = ctx.film
    weight = np.where(rho < 1.0, band_falloff(np.clip(rho, 0.0, 1.0), spot.falloff), 0.0)
    sector = np.clip(((angle + np.pi) / (2.0 * np.pi) * _SPOT_SECTORS).astype(np.int64), 0, _SPOT_SECTORS - 1)
    near = film & (rho < _NEAR_LINE)
    ring = film & (rho >= 1.0) & (rho < 1.0 + _CLEAN_DEPTH)
    readings = np.full((_SPOT_SECTORS, 3), np.nan)
    stepped = np.zeros(_SPOT_SECTORS, dtype=bool)
    for i in range(_SPOT_SECTORS):
        cell = near & (sector == i)
        if not cell.any():
            continue
        near_spike, clean_spike = spike_floor(u[cell]), spike_floor(u[ring & (sector == i)])
        drop = np.log1p(np.maximum(clean_spike, 0.0)) - np.log1p(np.maximum(near_spike, 0.0))
        if np.isfinite(drop).any() and float(np.nanmax(drop)) > _SCENE_STEP:
            stepped[i] = True
            continue
        readings[i] = (near_spike - clean_spike) / max(float(weight[cell].mean()), 1e-3)
    k = _band_color(spot, readings, [], None, u.shape[:2])
    core = near & (weight >= _CAP_NEAR)
    if spot.color is LeakColor.AUTO and core.sum() >= _SPIKE_MIN_PIXELS:
        # Shadows under a leak pile up at the leak itself: where the core's darkest film reads no
        # more than the leak the sectors read, its layers are the leak's color.
        floor = np.percentile(u[core] / weight[core][:, None], _COLOR_FLOOR_PERCENTILE, axis=0)
        level = np.nanmedian(_in_color(readings, k)) if np.isfinite(readings).any() else np.nan
        if float(floor.max()) > 0.0 and (not np.isfinite(level) or float(floor.max()) <= _SHADOW_LEAK * level):
            k = np.maximum(floor, 0.0) / float(floor.max())
    lead = int(np.argmax(k))
    cap = _CAP_OPEN
    if core.sum() >= _SPIKE_MIN_PIXELS:
        cap = max(float(np.percentile(u[core][:, lead] / (k[lead] * weight[core]), _CAP_PERCENTILE)), 0.0)
    levels = _in_color(readings, k)
    if np.isfinite(levels).any():
        amplitude = max(float(np.nanmedian(levels)), 0.0)
    elif stepped.any() and cap < _CAP_OPEN:
        # A dark subject at the center with brighter film past the edge: its shadows are the leak.
        amplitude = cap
    else:
        return None
    amplitude = float(soft_min(np.array([amplitude * spot.strength]), np.array([cap]))[0])
    field = (amplitude * weight).astype(np.float32)[..., None] * k.astype(np.float32)
    return ShapeReading(k, field)


def _spread(values: np.ndarray, known: np.ndarray, sigma: float) -> np.ndarray:
    """Normalized convolution: *values* where *known*, carried into the cells around them."""
    weight = negpy_core.gaussian_blur(known.astype(np.float32), (0, 0), sigma)
    total = negpy_core.gaussian_blur(np.where(known, values, 0.0).astype(np.float32), (0, 0), sigma)
    return np.where(weight > 1e-4, total / np.maximum(weight, 1e-6), np.nan)


def _lower_envelope(cells: np.ndarray) -> np.ndarray:
    """A smooth map under (gh, gw) readings, NaN where unread: unread cells take those within
    _PAINT_SMOOTH cells, then an opening over _PAINT_OPEN cells drops what stands above its
    neighbors, and a blur over _PAINT_BLUR smooths it."""
    known = np.isfinite(cells)
    if not known.any():
        return np.zeros(cells.shape, np.float32)
    filled = _spread(np.maximum(np.where(known, cells, 0.0), 0.0), known, _PAINT_SMOOTH)
    filled = np.where(np.isfinite(filled), filled, 0.0).astype(np.float32)
    window = np.ones((_PAINT_OPEN, _PAINT_OPEN), np.uint8)
    return np.maximum(negpy_core.gaussian_blur(negpy_core.morphology_ex(filled, negpy_core.MORPH_OPEN, window), (0, 0), _PAINT_BLUR), 0.0)


def _paint_reading(u: np.ndarray, paint: LeakShape, ctx: _Context) -> Optional[ShapeReading]:
    """The leak under a painted region, (h, w, 3) excess exposure per layer, and its color: a smooth
    strength map times the region's weight, in one color, for a leak with no line or center, as
    a toy camera's blotches. In cells, the shadow spike inside is read against the scene's own
    shadow level, that of a ring of clean film around the region carried in; the strength is
    smoothed over the cells and capped by the darkest film inside."""
    shape = u.shape[:2]
    weight = paint_weight(paint, shape)
    if weight is None:
        return None
    film = ctx.film
    inside = film & (weight >= _CAP_NEAR)
    if inside.sum() < _SPIKE_MIN_PIXELS:
        return None
    cover = (weight > 0.0).astype(np.uint8)
    ring = film & (cover == 0) & (negpy_core.distance_transform(1 - cover, 2, 5) <= _PAINT_RING_PX)
    h, w = shape
    gh, gw = max(1, h // _PAINT_CELL_PX), max(1, w // _PAINT_CELL_PX)
    rows = np.minimum(np.arange(h) * gh // h, gh - 1)
    cols = np.minimum(np.arange(w) * gw // w, gw - 1)
    cell = rows[:, None] * gw + cols[None, :]
    near = np.full((gh * gw, 3), np.nan)
    clean = np.full((gh * gw, 3), np.nan)
    mean_w = np.zeros(gh * gw)
    for index in np.unique(cell[inside | ring]):
        here = cell == index
        if (inside & here).any():
            near[index] = spike_floor(u[inside & here])
            mean_w[index] = float(weight[inside & here].mean())
        if (ring & here).any():
            clean[index] = spike_floor(u[ring & here])
    near, clean, mean_w = near.reshape(gh, gw, 3), clean.reshape(gh, gw, 3), mean_w.reshape(gh, gw)
    scene = np.stack([_spread(clean[..., c], np.isfinite(clean[..., c]), _PAINT_SPREAD) for c in range(3)], axis=-1)
    drop = np.log1p(np.maximum(scene, 0.0)) - np.log1p(np.maximum(near, 0.0))
    stepped = np.nanmax(np.where(np.isfinite(drop), drop, -np.inf), axis=-1) > _SCENE_STEP
    readings = np.where(stepped[..., None], np.nan, (near - scene) / np.maximum(mean_w, 1e-3)[..., None])
    # A bright subject inside lifts a cell's reading over the ring's shadow level; a leak's strength is
    # smooth, so each layer's map is the lower envelope of its cells: an opening, then a blur.
    envelope = np.stack([_lower_envelope(readings[..., c]) for c in range(3)], axis=-1)
    if paint.color is not LeakColor.AUTO:
        k = _FRONT_COLOR.copy() if paint.color is LeakColor.FRONT else _BACK_COLOR.copy()
    else:
        # The region is painted wider than the leak: its color is read where the envelope is strong,
        # in the darkest film there when it holds shadows, else in the envelope's own layers.
        lead_map = envelope.max(axis=-1)
        strong_cells = lead_map >= _PAINT_STRONG * float(lead_map.max())
        k = np.median(envelope[strong_cells], axis=0)
        k = np.maximum(k, 0.0) / max(float(k.max()), 1e-12)
        strong = strong_cells.reshape(-1)[cell] & inside
        if strong.sum() >= _SPIKE_MIN_PIXELS:
            floor = np.percentile(u[strong] / weight[strong][:, None], _COLOR_FLOOR_PERCENTILE, axis=0)
            if float(floor.max()) > 0.0 and float(floor.max()) <= _SHADOW_LEAK * float(np.median(lead_map[strong_cells])):
                k = np.maximum(floor, 0.0) / float(floor.max())
    levels = _in_color(readings, k)
    if not np.isfinite(levels).any():
        return None
    amplitude = _lower_envelope(levels)
    lead = int(np.argmax(k))
    cap = np.full((gh, gw), _CAP_OPEN, dtype=np.float32)
    for index in np.unique(cell[inside]):
        here = inside & (cell == index)
        if here.sum() >= _SPIKE_MIN_PIXELS:
            cap.flat[index] = max(float(np.percentile(u[here][:, lead] / (k[lead] * weight[here]), _CAP_PERCENTILE)), 0.0)
    amplitude = np.where(np.isfinite(amplitude), amplitude, 0.0) * paint.strength
    amplitude = soft_min(amplitude, cap)
    per_pixel = negpy_core.resize_linear(amplitude.astype(np.float32), (w, h))
    field = (per_pixel * weight).astype(np.float32)[..., None] * k.astype(np.float32)
    return ShapeReading(k, field)


def _trimmed(read: ShapeReading, trim: float) -> ShapeReading:
    """*read* with the layer its color leads in taken *trim* more, the others as read."""
    scale = np.ones(3, dtype=np.float32)
    scale[int(np.argmax(read.color))] += trim
    color = np.maximum(read.color * scale, 0.0)
    return read._replace(color=color / max(float(color.max()), 1e-12), field=np.maximum(read.field * scale, 0.0))


class LeakEstimate(NamedTuple):
    response: FilmResponse
    field: np.ndarray
    mean: np.ndarray
    readings: List[ShapeReading]


def estimate_leak(
    img: np.ndarray,
    shapes: Sequence[LeakShape],
    geometry=None,
    dmax: float = 0.0,
    unmix: Optional[np.ndarray] = None,
    color_film: bool = True,
) -> Optional[LeakEstimate]:
    """The leak under the shapes, or None when the frame has no readable base or no leak under
    them. *field* and *mean* are at the work size (h, w, 3) in excess exposure per layer. Each shape
    is read, in order, on what the shapes before it left, and the fields add: two shapes over one
    leak remove it once. The local mean is the frame's excess exposure blurred past grain. A black-and-
    white film's one layer takes any light alike, so off *color_film* every shape reads its color off
    the film."""
    if not color_film:
        shapes = [replace(s, color=LeakColor.AUTO, trim=0.0) for s in shapes]
    reading = read_film(img, geometry, dmax, unmix)
    if reading is None:
        return None
    small, film, response = reading.small, reading.film, reading.response
    u = np.maximum(response.excess(small), 0.0).astype(np.float32)
    sides = rebate_sides(reading.profiles, response, reading.lines) if reading.rect is not None else []
    ctx = _Context(film, response, reading.rect, sides, reading.profiles)
    residual = u.copy()
    held = np.zeros_like(u)
    free = np.zeros_like(u)
    found: List[ShapeReading] = []
    for shape in shapes:
        if not shape.enabled or shape.strength <= 0.0:
            continue
        measure = {LeakKind.EDGE: _edge_reading, LeakKind.SPOT: _spot_reading, LeakKind.PAINT: _paint_reading}.get(
            shape.kind, _band_reading
        )
        read = measure(residual, shape, ctx)
        if read is None:
            continue
        if shape.trim:
            read = _trimmed(read, shape.trim)
        found.append(read)
        if read.held:
            held += read.field
        else:
            free += read.field
        if read.free is not None:
            free += read.free
        residual = np.maximum(residual - read.field - (0.0 if read.free is None else read.free), 0.0)
    if float((held + free).max()) < _MIN_LEAK:
        return None
    mean = negpy_core.gaussian_blur(np.ascontiguousarray(response.excess(small), dtype=np.float32), (0, 0), _MEAN_SIGMA)
    field = (soft_min(held, _local_cap(np.maximum(mean, 0.0), film)) + free).astype(np.float32)
    return LeakEstimate(response, field, mean, found)


def edge_rows(reading, edge: LeakShape) -> Optional[EdgeRows]:
    """An edge leak read cell by cell along its edge on a read frame, for Auto; None without a
    picture box or beside fogged rebate."""
    if reading.rect is None:
        return None
    u = np.maximum(reading.response.excess(reading.small), 0.0).astype(np.float32)
    sides = rebate_sides(reading.profiles, reading.response, reading.lines)
    ctx = _Context(reading.film, reading.response, reading.rect, sides, reading.profiles)
    return _edge_rows(u, edge, ctx)
