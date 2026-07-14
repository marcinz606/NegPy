"""Pure NumPy primitives for preview-driven roll-frame registration.

The routines in this module operate on decoded preview arrays.  Scanner I/O,
file formats, and device geometry deliberately live elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class FilmBaseModel:
    """Roll-specific colour and uniformity signature of unexposed film."""

    rgb: tuple[float, float, float]
    channel_scale: tuple[float, float, float]
    color_tolerance: float
    variation_scale: tuple[float, float, float]
    variation_limit: float


@dataclass(frozen=True)
class FilmBaseLearningConfig:
    """Roll-normalised thresholds used while learning film base."""

    uniform_candidate_percentile: float = 35.0
    cluster_radius: float = 0.15
    minimum_run_rows: int = 3
    minimum_preview_support: int = 2
    balanced_rows_per_preview: int = 8
    channel_residual_percentile: float = 90.0
    channel_residual_multiplier: float = 2.5
    minimum_global_scale_fraction: float = 0.01
    variation_mad_multiplier: float = 3.0


@dataclass(frozen=True)
class TargetEdge:
    """Target-side edge of an inter-frame film-base gap, in preview rows."""

    row: int
    confidence: str


@dataclass(frozen=True)
class TailDetectorConfig:
    """Dimensionless thresholds for the stopped-transport signature."""

    suffix_seed_diffs: int = 4
    floor_fraction: float = 0.25
    floor_mad_multiplier: float = 6.0
    burst_search_rows: int = 8
    burst_exclusion_rows: int = 2
    minimum_burst_ratio: float = 4.0
    minimum_burst_gap_rows: int = 3
    maximum_burst_gap_rows: int = 8
    maximum_stationarity_ratio: float = 1.6
    outer_agreement_rows: int = 1
    epsilon: float = 1e-6


@dataclass(frozen=True)
class TailObservation:
    """Direct observation of a transport stop and its repeated-row tail."""

    burst_row: int
    tail_start_row: int
    burst_ratio: float
    stationarity_ratio: float
    confidence: str


@dataclass(frozen=True)
class TailModel:
    """Robust linear transport-tail model in preview-row coordinates."""

    slope: float
    intercept: float
    inlier_positions: tuple[int, ...]

    def predict(self, position: int | float) -> float:
        return self.slope * float(position) + self.intercept


@dataclass(frozen=True)
class TailBoundary:
    """Direct or model-derived first repeated-row boundary for one preview."""

    tail_start_row: float
    source: str
    confidence: str
    observation: TailObservation | None = None


def _as_preview(preview: np.ndarray) -> np.ndarray:
    image = np.asarray(preview)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("preview must have shape (rows, columns, at least 3 channels)")
    if image.shape[0] < 3 or image.shape[1] < 4:
        raise ValueError("preview is too small to register")
    rgb = image[:, :, :3].astype(np.float64, copy=False)
    if not np.all(np.isfinite(rgb)):
        raise ValueError("preview contains non-finite values")
    return rgb


def _triple(values: np.ndarray) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


def _row_features(image: np.ndarray, channel_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    width = image.shape[1]
    inset = max(1, width // 8)
    plane = image[:, inset : width - inset]
    median_rgb = np.median(plane, axis=1)
    deviations = (plane - median_rgb[:, None, :]) / channel_scale
    variation = np.median(np.sqrt(np.sum(deviations * deviations, axis=2)), axis=1)
    return median_rgb, variation


def _runs(mask: np.ndarray, minimum_rows: int) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    return [(int(start), int(end)) for start, end in changes.reshape(-1, 2) if end - start >= minimum_rows]


def _film_base_mask(preview: np.ndarray, model: FilmBaseModel) -> tuple[np.ndarray, np.ndarray]:
    colour_scale = np.asarray(model.channel_scale, dtype=np.float64)
    variation_scale = np.asarray(model.variation_scale, dtype=np.float64)
    colours, variation = _row_features(_as_preview(preview), variation_scale)
    distance = np.linalg.norm((colours - np.asarray(model.rgb)) / colour_scale, axis=1)
    return (distance <= model.color_tolerance) & (variation <= model.variation_limit), distance


def _tail_candidate(image: np.ndarray, config: TailDetectorConfig) -> TailObservation | None:
    source = np.asarray(image)
    if source.ndim not in (2, 3) or source.shape[1] < 1:
        raise ValueError("tail image must have shape (rows, columns[, channels])")
    if source.shape[0] < config.suffix_seed_diffs + config.maximum_burst_gap_rows + 2:
        return None
    # Conversion must precede subtraction: unsigned scanner samples otherwise
    # wrap and turn a dark-going transport transient into a small difference.
    differences = np.abs(np.diff(source.astype(np.float32), axis=0))
    motion = np.mean(differences, axis=tuple(range(1, differences.ndim)))
    seed = motion[-config.suffix_seed_diffs :]
    floor = float(np.median(seed))
    mad = float(np.median(np.abs(seed - floor)))
    tolerance = max(config.floor_fraction * floor, config.floor_mad_multiplier * mad, config.epsilon)

    plateau = len(motion) - config.suffix_seed_diffs
    while plateau > 0 and abs(float(motion[plateau - 1]) - floor) <= tolerance:
        plateau -= 1
    suffix = motion[plateau:]
    if len(suffix) < config.suffix_seed_diffs:
        return None

    search_start = max(0, plateau - config.burst_search_rows)
    search_end = plateau - config.burst_exclusion_rows
    if search_end <= search_start:
        return None
    burst_row = search_start + int(np.argmax(motion[search_start:search_end]))
    floor_safe = max(floor, config.epsilon)
    burst_ratio = float(motion[burst_row]) / floor_safe
    suffix_low = max(float(np.percentile(suffix, 10)), config.epsilon)
    stationarity_ratio = float(np.percentile(suffix, 90)) / suffix_low
    burst_gap = plateau - burst_row
    if (
        burst_ratio < config.minimum_burst_ratio
        or not config.minimum_burst_gap_rows <= burst_gap <= config.maximum_burst_gap_rows
        or stationarity_ratio > config.maximum_stationarity_ratio
    ):
        return None
    return TailObservation(
        burst_row=burst_row,
        tail_start_row=plateau,
        burst_ratio=burst_ratio,
        stationarity_ratio=stationarity_ratio,
        confidence="medium",
    )


def observe_transport_tail(
    image: np.ndarray,
    config: TailDetectorConfig = TailDetectorConfig(),
) -> TailObservation | None:
    """Observe a stopped-transport tail, guarded by the image's outer bands."""

    full = _tail_candidate(image, config)
    if full is None:
        return None
    source = np.asarray(image)
    quarter = source.shape[1] // 4
    if quarter < 1:
        return full
    outer = np.concatenate((source[:, :quarter], source[:, -quarter:]), axis=1)
    outer_observation = _tail_candidate(outer, config)
    agrees = (
        outer_observation is not None
        and abs(outer_observation.burst_row - full.burst_row) <= config.outer_agreement_rows
        and abs(outer_observation.tail_start_row - full.tail_start_row) <= config.outer_agreement_rows
    )
    if not agrees:
        return full
    return TailObservation(
        burst_row=full.burst_row,
        tail_start_row=full.tail_start_row,
        burst_ratio=full.burst_ratio,
        stationarity_ratio=full.stationarity_ratio,
        confidence="high",
    )


def _theil_sen(points: Sequence[tuple[int, float]]) -> tuple[float, float]:
    slopes = [
        (row_b - row_a) / (position_b - position_a)
        for index, (position_a, row_a) in enumerate(points)
        for position_b, row_b in points[index + 1 :]
    ]
    slope = float(np.median(slopes))
    intercept = float(np.median([row - slope * position for position, row in points]))
    return slope, intercept


def fit_transport_tail(
    observations: Mapping[int, TailObservation],
    *,
    residual_tolerance_rows: float = 1.5,
) -> TailModel:
    """Fit a robust tail-start line and reject non-transport scene edges."""

    if residual_tolerance_rows < 0:
        raise ValueError("residual_tolerance_rows must be non-negative")
    points = sorted((int(position), float(observation.tail_start_row)) for position, observation in observations.items())
    if len(points) < 3:
        raise ValueError("at least three direct tail observations are required")
    if not all(np.isfinite(position) and np.isfinite(row) for position, row in points):
        raise ValueError("tail observations must be finite")

    slope, intercept = _theil_sen(points)
    inliers = [point for point in points if abs(point[1] - (slope * point[0] + intercept)) <= residual_tolerance_rows]
    if len(inliers) < 3:
        raise ValueError("tail observations do not support a consistent transport model")
    slope, intercept = _theil_sen(inliers)
    inlier_positions = tuple(position for position, row in points if abs(row - (slope * position + intercept)) <= residual_tolerance_rows)
    if len(inlier_positions) < 3:
        raise ValueError("tail observations do not support a consistent transport model")
    return TailModel(slope=slope, intercept=intercept, inlier_positions=inlier_positions)


def measure_transport_tails(
    previews: Mapping[int, np.ndarray],
    config: TailDetectorConfig = TailDetectorConfig(),
    *,
    residual_tolerance_rows: float = 1.5,
) -> dict[int, TailBoundary]:
    """Measure a roll jointly, inferring tails that lie outside short previews.

    Only full/outer-band agreements seed the robust line.  A medium direct
    observation can be used after the line independently confirms it.
    Predictions outside the acquired array remain explicit and are never
    clamped into a fabricated in-frame boundary.
    """

    if not previews:
        return {}
    observations = {int(position): observe_transport_tail(image, config) for position, image in previews.items()}
    direct = {
        position: observation
        for position, observation in observations.items()
        if observation is not None and observation.confidence == "high"
    }
    model = fit_transport_tail(direct, residual_tolerance_rows=residual_tolerance_rows)

    result: dict[int, TailBoundary] = {}
    for raw_position, image in previews.items():
        position = int(raw_position)
        observation = observations[position]
        prediction = model.predict(position)
        agrees_with_model = observation is not None and abs(observation.tail_start_row - prediction) <= residual_tolerance_rows
        if agrees_with_model and (observation.confidence == "medium" or position in model.inlier_positions):
            result[position] = TailBoundary(
                tail_start_row=float(observation.tail_start_row),
                source="direct",
                confidence=observation.confidence,
                observation=observation,
            )
            continue
        height = np.asarray(image).shape[0]
        last_observable_prediction = height - config.suffix_seed_diffs - 1
        if 0 <= prediction <= last_observable_prediction:
            raise ValueError(
                f"position {position}: predicted visible transport tail at row {prediction:.1f}, but no consistent stop was observed"
            )
        source = "outside" if prediction < 0 or prediction >= height else "inferred"
        result[position] = TailBoundary(
            tail_start_row=prediction,
            source=source,
            confidence="medium",
            observation=observation,
        )
    return result


def learn_film_base(
    previews: Sequence[np.ndarray],
    config: FilmBaseLearningConfig = FilmBaseLearningConfig(),
) -> FilmBaseModel:
    """Learn the repeated, spatially uniform film-base colour across a roll.

    The model is learned from cross-preview repetition rather than a fixed RGB
    value or an assumed leader length.  A smooth scene in one frame therefore
    cannot outweigh a colour that recurs in the gaps across the roll.
    """

    images = [_as_preview(preview) for preview in previews]
    if len(images) < 2:
        raise ValueError("at least two previews are required to learn film base")

    row_colours = [np.median(image[:, max(1, image.shape[1] // 8) : -max(1, image.shape[1] // 8)], axis=1) for image in images]
    all_colours = np.concatenate(row_colours, axis=0)
    channel_scale = np.percentile(all_colours, 95, axis=0) - np.percentile(all_colours, 5, axis=0)
    channel_scale = np.maximum(channel_scale, np.maximum(np.max(np.abs(all_colours), axis=0) * 1e-6, 1.0))

    features = [_row_features(image, channel_scale) for image in images]
    candidates: list[np.ndarray] = []
    for colours, variation in features:
        cutoff = float(np.percentile(variation, config.uniform_candidate_percentile))
        selected = colours[variation <= cutoff]
        candidates.append(selected if len(selected) else colours[[int(np.argmin(variation))]])

    anchors = np.concatenate(candidates, axis=0)
    # Film base can drift along a roll as the emulsion, lamp, or preview
    # exposure changes.  Work in roll-normalised colour units and cluster
    # broadly enough to retain that drift; spatial uniformity and support
    # across previews keep scene colours from winning.
    radius = config.cluster_radius
    best_anchor: np.ndarray | None = None
    transmission_floor = np.percentile(all_colours, 1, axis=0)
    transmission_span = np.maximum(
        np.percentile(all_colours, 99, axis=0) - transmission_floor,
        1.0,
    )
    best_score = (-1, -1, -1.0)
    for anchor in anchors:
        counts: list[int] = []
        for colours, variation in features:
            uniform_cutoff = float(np.percentile(variation, config.uniform_candidate_percentile))
            distance = np.linalg.norm((colours - anchor) / channel_scale, axis=1)
            # A transport tail at EOF can also be bright and uniform.  It is
            # not a target-side film-base gap because no image follows it, so
            # exclude suffix-only runs from anchor support.
            eligible_runs = [
                (start, end)
                for start, end in _runs(
                    (variation <= uniform_cutoff) & (distance <= radius),
                    config.minimum_run_rows,
                )
                if end < len(colours)
            ]
            counts.append(sum(end - start for start, end in eligible_runs))
        # C-41 film base is the roll's recurring least-dense material.  When
        # scene colours tie its cross-preview support, prefer the candidate
        # whose weakest channel has the highest roll-normalised transmission;
        # this uses film physics without assuming an orange-mask RGB value.
        normalised_transmission = (anchor - transmission_floor) / transmission_span
        score = (
            sum(count >= config.minimum_run_rows for count in counts),
            sum(min(count, config.balanced_rows_per_preview) for count in counts),
            float(np.min(normalised_transmission)),
        )
        if score > best_score:
            best_anchor = anchor
            best_score = score

    if best_anchor is None or best_score[0] < config.minimum_preview_support:
        raise ValueError("no repeated film-base colour found across previews")

    local_centres: list[np.ndarray] = []
    local_variation: list[np.ndarray] = []
    for colours, variation in features:
        uniform_cutoff = float(np.percentile(variation, config.uniform_candidate_percentile))
        distance = np.linalg.norm((colours - best_anchor) / channel_scale, axis=1)
        runs = _runs((variation <= uniform_cutoff) & (distance <= radius), config.minimum_run_rows)
        if not runs:
            continue
        # In a wide registration preview the gap precedes the target image,
        # but its length and starting row are deliberately unconstrained.
        start, end = runs[0]
        local_centres.append(np.median(colours[start:end], axis=0))
        local_variation.append(variation[start:end])

    if len(local_centres) < config.minimum_preview_support:
        raise ValueError("film base is not present in enough previews")
    centres = np.stack(local_centres)
    rgb = np.median(centres, axis=0)
    centre_residuals = np.abs(centres - rgb)
    colour_scale = np.maximum(
        np.percentile(centre_residuals, config.channel_residual_percentile, axis=0) * config.channel_residual_multiplier,
        np.maximum(channel_scale * config.minimum_global_scale_fraction, 1.0),
    )
    base_variation = np.concatenate(local_variation, axis=0)
    variation_median = float(np.median(base_variation))
    variation_mad = float(np.median(np.abs(base_variation - variation_median)))
    variation_limit = max(
        1e-6,
        variation_median + config.variation_mad_multiplier * 1.4826 * variation_mad,
    )
    return FilmBaseModel(
        rgb=_triple(rgb),
        channel_scale=_triple(colour_scale),
        color_tolerance=1.0,
        variation_scale=_triple(channel_scale),
        variation_limit=variation_limit,
    )


def measure_target_start(
    preview: np.ndarray,
    model: FilmBaseModel,
    *,
    minimum_run_rows: int = 3,
    expected_row: float | None = None,
    maximum_prediction_error: float | None = None,
    allow_short_prefix: bool = False,
) -> TargetEdge:
    """Find the target-side edge of the film-base gap in one wide preview.

    When a pitch or shifted-reference prior is available, ``expected_row``
    makes it the tie-breaker among visually plausible runs.  An optional
    error gate fails closed instead of accepting a distant scene look-alike.
    """

    if minimum_run_rows < 1:
        raise ValueError("minimum_run_rows must be positive")
    if maximum_prediction_error is not None and expected_row is None:
        raise ValueError("maximum_prediction_error requires expected_row")
    if expected_row is not None and not np.isfinite(expected_row):
        raise ValueError("expected_row must be finite")
    if maximum_prediction_error is not None and maximum_prediction_error < 0:
        raise ValueError("maximum_prediction_error must be non-negative")
    if type(allow_short_prefix) is not bool:
        raise ValueError("allow_short_prefix must be a boolean")
    if allow_short_prefix and expected_row is None:
        raise ValueError("allow_short_prefix requires expected_row")
    mask, distance = _film_base_mask(preview, model)
    candidates = [(start, end) for start, end in _runs(mask, minimum_run_rows) if end < len(mask)]
    if not candidates:
        prefix_end = 0
        while prefix_end < len(mask) and bool(mask[prefix_end]):
            prefix_end += 1
        if allow_short_prefix and 0 < prefix_end < minimum_run_rows and prefix_end < len(mask):
            # At the far transport boundary the nominal frame origin can clip
            # almost the entire inter-frame gap.  A short run attached to row
            # zero is still direct evidence; it remains medium-confidence and
            # must pass a second, registered preview before approval.
            return TargetEdge(row=prefix_end, confidence="medium")
        raise ValueError("preview contains no film-base run followed by image content")
    if expected_row is None:
        start, end = max(candidates, key=lambda run: (run[1] - run[0], -run[0]))
    else:
        if maximum_prediction_error is not None:
            candidates = [run for run in candidates if abs(run[1] - expected_row) <= maximum_prediction_error]
            if not candidates:
                raise ValueError("no film-base edge agrees with the supplied prediction")
        start, end = min(candidates, key=lambda run: (abs(run[1] - expected_row), -(run[1] - run[0])))
    post = distance[end : min(len(distance), end + minimum_run_rows)]
    confidence = "high" if end - start >= 2 * minimum_run_rows and np.all(post > model.color_tolerance) else "medium"
    return TargetEdge(row=end, confidence=confidence)


def measure_leading_margin(
    preview: np.ndarray,
    model: FilmBaseModel,
    *,
    minimum_run_rows: int = 3,
    max_initial_outlier_rows: int = 1,
) -> TargetEdge:
    """Measure the film-base prefix of an already aligned preview.

    Only a run attached to the beginning is eligible.  Film-base-coloured
    areas inside the photograph cannot silently become the alignment edge.
    """

    if minimum_run_rows < 1 or max_initial_outlier_rows < 0:
        raise ValueError("run length must be positive and outlier allowance non-negative")
    mask, _ = _film_base_mask(preview, model)
    for start, end in _runs(mask, minimum_run_rows):
        if start <= max_initial_outlier_rows:
            return TargetEdge(row=end, confidence="high" if start == 0 else "medium")
        break
    raise ValueError("aligned preview does not begin in film base")


def robust_pitch_prediction(
    observed: Mapping[int, int | float],
    *,
    frame: int,
    excluded_frames: Collection[int] = (),
) -> float:
    """Predict an edge with a Theil-Sen line through prior measurements."""

    excluded = {int(item) for item in excluded_frames}
    points = sorted((int(position), float(row)) for position, row in observed.items() if int(position) not in excluded)
    if len(points) < 3:
        raise ValueError("at least three included frames are required for a pitch prior")
    if not all(np.isfinite(position) and np.isfinite(row) for position, row in points):
        raise ValueError("pitch measurements must be finite")
    slope, intercept = _theil_sen(points)
    return slope * frame + intercept


def shifted_reference_prediction(
    reference: Mapping[int, int | float],
    observed: Mapping[int, int | float],
    *,
    frame: int,
    excluded_frames: Collection[int] = (),
) -> float:
    """Map a prior sweep to a refeed using the median shared-frame shift."""

    if frame not in reference:
        raise ValueError(f"frame {frame} is absent from the reference sweep")
    excluded = {int(item) for item in excluded_frames}
    shared = sorted((set(reference) & set(observed)) - excluded)
    if len(shared) < 2:
        raise ValueError("at least two included shared frames are required to align a reference sweep")
    shifts = np.asarray([float(observed[item]) - float(reference[item]) for item in shared])
    target = float(reference[frame])
    if not np.all(np.isfinite(shifts)) or not np.isfinite(target):
        raise ValueError("reference measurements must be finite")
    return target + float(np.median(shifts))
