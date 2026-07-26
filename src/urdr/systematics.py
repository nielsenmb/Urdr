"""Segment-dependent systematic injections and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .background import BackgroundConfig
from .benchmark import estimate_delta_nu
from .eacf import compute_eacf_map
from .models import ObservingWindow, TimeSeries
from .simulation import SimulationConfig, simulate_time_series

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SegmentSystematicConfig:
    """Affine systematic applied inside one time interval.

    Parameters
    ----------
    start_day
        Inclusive interval start on the time-series day axis.
    stop_day
        Exclusive interval stop on the time-series day axis.
    amplitude_scale
        Multiplicative scale applied about the global median flux.
    offset
        Additive flux offset inside the interval.
    drift_per_day
        Linear flux drift per day, centred on the interval midpoint.
    """

    start_day: float
    stop_day: float
    amplitude_scale: float = 1.0
    offset: float = 0.0
    drift_per_day: float = 0.0

    def __post_init__(self) -> None:
        """Validate the interval and affine transformation."""
        if not np.isfinite(self.start_day) or not np.isfinite(self.stop_day):
            raise ValueError("segment bounds must be finite")
        if self.stop_day <= self.start_day:
            raise ValueError("stop_day must be greater than start_day")
        if not np.isfinite(self.amplitude_scale) or self.amplitude_scale <= 0:
            raise ValueError("amplitude_scale must be finite and positive")
        if not np.isfinite(self.offset) or not np.isfinite(self.drift_per_day):
            raise ValueError("offset and drift_per_day must be finite")


@dataclass(frozen=True)
class SegmentDiagnostics:
    """Robust diagnostics comparing user-defined time segments.

    Parameters
    ----------
    maximum_scale_ratio
        Largest ratio between robust segment scales.
    maximum_location_shift
        Range of segment medians divided by the pooled robust scale.
    maximum_drift
        Largest absolute segment drift across its duration, divided by the
        pooled robust scale.
    segment_count
        Number of evaluated segments.
    """

    maximum_scale_ratio: float
    maximum_location_shift: float
    maximum_drift: float
    segment_count: int


@dataclass(frozen=True)
class SegmentVetoMetrics:
    """Performance of a calibrated segment-instability veto.

    Parameters
    ----------
    eacf_threshold
        Exact-window null threshold for the EACF statistic.
    instability_quantile_threshold
        Signal-calibrated percentile threshold for the combined diagnostic.
    raw_signal_detection_rate
        Oscillation detection rate before the segment veto.
    vetoed_signal_detection_rate
        Oscillation detection rate after the segment veto.
    raw_systematic_detection_rate
        Raw detection rate for each injected systematic class.
    vetoed_systematic_detection_rate
        Post-veto detection rate for each injected systematic class.
    target_false_positive_rate
        Requested clean-null false-positive rate.
    target_signal_retention
        Requested retention among EACF-detected signal injections.
    realizations
        Number of paired realisations per simulation class.
    """

    eacf_threshold: float
    instability_quantile_threshold: float
    raw_signal_detection_rate: float
    vetoed_signal_detection_rate: float
    raw_systematic_detection_rate: Mapping[str, float]
    vetoed_systematic_detection_rate: Mapping[str, float]
    target_false_positive_rate: float
    target_signal_retention: float
    realizations: int


def add_segment_systematics(
    series: TimeSeries,
    systematics: Sequence[SegmentSystematicConfig],
) -> TimeSeries:
    """Apply piecewise scale, offset, and drift changes to a time series.

    Parameters
    ----------
    series
        Base uniformly sampled time series.
    systematics
        Segment transformations. Overlapping transformations are applied in
        the supplied order.

    Returns
    -------
    TimeSeries
        Copy of ``series`` containing the requested transformations.
    """
    if not systematics:
        raise ValueError("at least one segment systematic is required")
    flux = np.asarray(series.flux, dtype=float).copy()
    global_median = float(np.median(flux[series.observed]))
    for config in systematics:
        select = (
            series.observed
            & (series.time >= config.start_day)
            & (series.time < config.stop_day)
        )
        if not np.any(select):
            raise ValueError("a systematic interval contains no observed cadences")
        midpoint = 0.5 * (config.start_day + config.stop_day)
        flux[select] = (
            global_median
            + config.amplitude_scale * (flux[select] - global_median)
            + config.offset
            + config.drift_per_day * (series.time[select] - midpoint)
        )
    flux[~series.observed] = np.nan
    return TimeSeries(series.time, flux, series.observed)


def segment_diagnostics(
    series: TimeSeries,
    segments_days: Sequence[tuple[float, float]],
) -> SegmentDiagnostics:
    """Measure robust scale, location, and drift changes between segments.

    Parameters
    ----------
    series
        Uniformly sampled time series with an explicit observing mask.
    segments_days
        Non-overlapping ``(start, stop)`` intervals on the time-series day axis.

    Returns
    -------
    SegmentDiagnostics
        Dimensionless robust measures of segment instability.
    """
    segments = _validate_segments(segments_days)
    scales = []
    locations = []
    drift_amplitudes = []
    for start, stop in segments:
        select = (
            series.observed & (series.time >= start) & (series.time < stop)
        )
        if np.count_nonzero(select) < 8:
            raise ValueError("each segment must contain at least eight observations")
        time = series.time[select]
        flux = series.flux[select]
        centred_time = time - np.mean(time)
        slope = float(
            np.sum(centred_time * (flux - np.mean(flux)))
            / np.sum(centred_time**2)
        )
        residual = flux - slope * centred_time
        locations.append(float(np.median(residual)))
        scales.append(max(_robust_scale(residual), np.finfo(float).eps))
        drift_amplitudes.append(abs(slope) * (stop - start))
    pooled_scale = max(float(np.median(scales)), np.finfo(float).eps)
    return SegmentDiagnostics(
        maximum_scale_ratio=float(np.max(scales) / np.min(scales)),
        maximum_location_shift=float(
            (np.max(locations) - np.min(locations)) / pooled_scale
        ),
        maximum_drift=float(np.max(drift_amplitudes) / pooled_scale),
        segment_count=len(segments),
    )


def benchmark_segment_veto(
    window: ObservingWindow,
    simulation: SimulationConfig,
    systematics: Mapping[str, Sequence[SegmentSystematicConfig]],
    segments_days: Sequence[tuple[float, float]],
    centre_frequencies_uhz: ArrayLike,
    filter_width_uhz: float,
    delta_nu_grid_uhz: ArrayLike,
    *,
    background: BackgroundConfig | None = None,
    realizations: int = 128,
    target_false_positive_rate: float = 0.01,
    target_signal_retention: float = 0.95,
    max_lag_seconds: float | None = None,
    seed: int = 0,
) -> SegmentVetoMetrics:
    """Benchmark a target-calibrated segment-instability veto.

    Diagnostic values are converted to empirical percentiles using oscillation
    injections. Their maximum percentile forms one combined instability score,
    so its final threshold directly controls signal retention without assuming
    that scale, offset, and drift diagnostics are independent.

    Parameters
    ----------
    window
        Exact observing window used for every simulation.
    simulation
        Noise, granulation, and injected-oscillation parameters.
    systematics
        Named piecewise systematic configurations to benchmark.
    segments_days
        Time intervals used by the segment diagnostics.
    centre_frequencies_uhz
        Trial EACF filter centres in microhertz.
    filter_width_uhz
        Full EACF filter width in microhertz.
    delta_nu_grid_uhz
        Trial large separations in microhertz.
    background
        Optional spectral-background treatment.
    realizations
        Number of paired realisations per simulation class.
    target_false_positive_rate
        Requested false-positive rate for the clean null.
    target_signal_retention
        Requested retention among EACF-detected signal injections.
    max_lag_seconds
        Optional maximum EACF lag.
    seed
        Root seed for deterministic paired simulations.

    Returns
    -------
    SegmentVetoMetrics
        Raw and vetoed detection rates and calibrated thresholds.
    """
    if realizations < 8:
        raise ValueError("at least eight realizations are required")
    if not systematics:
        raise ValueError("at least one systematic class is required")
    if not 0 < target_false_positive_rate < 1:
        raise ValueError("target_false_positive_rate must lie between zero and one")
    if not 0 < target_signal_retention <= 1:
        raise ValueError("target_signal_retention must lie in (0, 1]")
    segments = _validate_segments(segments_days)
    centres = np.atleast_1d(np.asarray(centre_frequencies_uhz, dtype=float))
    delta_nu_grid = np.atleast_1d(np.asarray(delta_nu_grid_uhz, dtype=float))
    seeds = np.random.SeedSequence(seed).spawn(realizations)

    null_scores = np.empty(realizations)
    signal_scores = np.empty(realizations)
    signal_diagnostics = np.empty((realizations, 3))
    systematic_scores = {
        name: np.empty(realizations) for name in systematics
    }
    systematic_diagnostics = {
        name: np.empty((realizations, 3)) for name in systematics
    }

    for index, child_seed in enumerate(seeds):
        base = simulate_time_series(
            window,
            simulation,
            np.random.default_rng(child_seed),
            include_oscillations=False,
        )
        signal = simulate_time_series(
            window,
            simulation,
            np.random.default_rng(child_seed),
            include_oscillations=True,
        )
        null_scores[index] = _eacf_score(
            base, centres, filter_width_uhz, delta_nu_grid, background,
            max_lag_seconds,
        )
        signal_scores[index] = _eacf_score(
            signal, centres, filter_width_uhz, delta_nu_grid, background,
            max_lag_seconds,
        )
        signal_diagnostics[index] = _diagnostic_array(
            segment_diagnostics(signal, segments)
        )
        for name, configuration in systematics.items():
            contaminated = add_segment_systematics(base, configuration)
            systematic_scores[name][index] = _eacf_score(
                contaminated, centres, filter_width_uhz, delta_nu_grid,
                background, max_lag_seconds,
            )
            systematic_diagnostics[name][index] = _diagnostic_array(
                segment_diagnostics(contaminated, segments)
            )

    eacf_threshold = _higher_quantile(
        null_scores, 1.0 - target_false_positive_rate
    )
    signal_detected = signal_scores >= eacf_threshold
    reference = (
        signal_diagnostics[signal_detected]
        if np.any(signal_detected)
        else signal_diagnostics
    )
    signal_instability = _percentile_score(signal_diagnostics, reference)
    calibration = (
        signal_instability[signal_detected]
        if np.any(signal_detected)
        else signal_instability
    )
    instability_threshold = _higher_quantile(
        calibration, target_signal_retention
    )

    raw_systematic = {}
    vetoed_systematic = {}
    for name in systematics:
        detected = systematic_scores[name] >= eacf_threshold
        instability = _percentile_score(
            systematic_diagnostics[name], reference
        )
        raw_systematic[name] = float(np.mean(detected))
        vetoed_systematic[name] = float(
            np.mean(detected & (instability <= instability_threshold))
        )

    return SegmentVetoMetrics(
        eacf_threshold=eacf_threshold,
        instability_quantile_threshold=instability_threshold,
        raw_signal_detection_rate=float(np.mean(signal_detected)),
        vetoed_signal_detection_rate=float(
            np.mean(signal_detected & (signal_instability <= instability_threshold))
        ),
        raw_systematic_detection_rate=raw_systematic,
        vetoed_systematic_detection_rate=vetoed_systematic,
        target_false_positive_rate=target_false_positive_rate,
        target_signal_retention=target_signal_retention,
        realizations=realizations,
    )


def _validate_segments(
    segments_days: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    segments = tuple((float(start), float(stop)) for start, stop in segments_days)
    if len(segments) < 2:
        raise ValueError("at least two diagnostic segments are required")
    if any(
        not np.isfinite(start)
        or not np.isfinite(stop)
        or stop <= start
        for start, stop in segments
    ):
        raise ValueError("segment bounds must be finite and ordered")
    ordered = sorted(segments)
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("diagnostic segments cannot overlap")
    return tuple(ordered)


def _robust_scale(values: FloatArray) -> float:
    median = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - median)))


def _diagnostic_array(diagnostics: SegmentDiagnostics) -> FloatArray:
    return np.asarray(
        [
            diagnostics.maximum_scale_ratio,
            diagnostics.maximum_location_shift,
            diagnostics.maximum_drift,
        ],
        dtype=float,
    )


def _percentile_score(values: FloatArray, reference: FloatArray) -> FloatArray:
    output = np.empty(values.shape[0], dtype=float)
    sorted_reference = np.sort(reference, axis=0)
    for index, row in enumerate(values):
        percentiles = [
            np.searchsorted(sorted_reference[:, column], value, side="right")
            / sorted_reference.shape[0]
            for column, value in enumerate(row)
        ]
        output[index] = max(percentiles)
    return output


def _eacf_score(
    series: TimeSeries,
    centres: FloatArray,
    filter_width_uhz: float,
    delta_nu_grid: FloatArray,
    background: BackgroundConfig | None,
    max_lag_seconds: float | None,
) -> float:
    result = compute_eacf_map(
        series,
        centres,
        filter_width_uhz,
        max_lag_seconds=max_lag_seconds,
        empirical_background=background,
    )
    return estimate_delta_nu(result, delta_nu_grid)[1]


def _higher_quantile(values: FloatArray, probability: float) -> float:
    try:
        return float(np.quantile(values, probability, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility
        return float(np.quantile(values, probability, interpolation="higher"))
