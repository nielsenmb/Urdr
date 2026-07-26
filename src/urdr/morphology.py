"""Morphology diagnostics for frequency-lag EACF maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .background import BackgroundConfig
from .benchmark import estimate_delta_nu
from .contaminants import CoherentSignalConfig, add_coherent_signal
from .eacf import EACFMap, compute_eacf_map
from .models import ObservingWindow, TimeSeries
from .simulation import SimulationConfig, simulate_time_series
from .systematics import SegmentSystematicConfig, add_segment_systematics

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MorphologyDiagnostics:
    """Summary of a candidate ridge in a frequency-lag EACF map.

    Parameters
    ----------
    peak_centre_uhz
        Filter centre containing the strongest primary-lag response.
    centre_offset
        Absolute offset from the expected ``numax``, divided by the predicted
        envelope width.
    ridge_width
        Width of the longest connected ridge above ``ridge_fraction`` of its
        maximum, divided by the predicted envelope width.
    ridge_fill
        Fraction of trial centres inside the predicted envelope that belong to
        the thresholded ridge.
    ridge_contrast
        Mean primary-lag response inside the predicted envelope divided by the
        mean response outside it. ``inf`` is returned when no flank is sampled.
    ridge_roughness
        Median absolute first difference of the primary-lag profile divided by
        its maximum.
    harmonic_support
        Maximum response near twice the primary seismic lag divided by the
        maximum primary-lag response.
    """

    peak_centre_uhz: float
    centre_offset: float
    ridge_width: float
    ridge_fill: float
    ridge_contrast: float
    ridge_roughness: float
    harmonic_support: float


@dataclass(frozen=True)
class MorphologyVetoMetrics:
    """Performance of a signal-calibrated EACF morphology check.

    Parameters
    ----------
    eacf_threshold
        Exact-window clean-null threshold for the raw EACF statistic.
    morphology_outlier_threshold
        Maximum calibrated morphology outlier score accepted as signal-like.
    raw_signal_detection_rate
        Oscillation detection rate before the morphology check.
    accepted_signal_detection_rate
        Oscillation detection rate after the morphology check.
    raw_contaminant_detection_rate
        Raw detection rate for every coherent or segment-systematic class.
    accepted_contaminant_detection_rate
        Post-check detection rate for every contaminant class.
    target_false_positive_rate
        Requested clean-null false-positive rate.
    target_signal_retention
        Requested retention among raw signal detections.
    realizations
        Number of paired realisations per simulation class.
    """

    eacf_threshold: float
    morphology_outlier_threshold: float
    raw_signal_detection_rate: float
    accepted_signal_detection_rate: float
    raw_contaminant_detection_rate: Mapping[str, float]
    accepted_contaminant_detection_rate: Mapping[str, float]
    target_false_positive_rate: float
    target_signal_retention: float
    realizations: int


def eacf_morphology(
    result: EACFMap,
    delta_nu_uhz: float,
    expected_numax_uhz: float,
    envelope_width_uhz: float,
    *,
    relative_lag_half_width: float = 0.05,
    ridge_fraction: float = 0.5,
) -> MorphologyDiagnostics:
    """Measure ridge morphology around a candidate seismic spacing.

    Parameters
    ----------
    result
        Frequency-lag EACF map.
    delta_nu_uhz
        Candidate large separation in microhertz.
    expected_numax_uhz
        Independent predicted envelope centre in microhertz.
    envelope_width_uhz
        Independent predicted full envelope width in microhertz.
    relative_lag_half_width
        Fractional half-width used around each expected ACF lag.
    ridge_fraction
        Fraction of the primary profile maximum defining ridge membership.

    Returns
    -------
    MorphologyDiagnostics
        Localisation, continuity, contrast, roughness, and harmonic summaries.

    Notes
    -----
    No diagnostic has a universal acceptance range. Use exact-window signal
    simulations, for example through :func:`benchmark_morphology_veto`, to
    calibrate their joint target-specific distribution.
    """
    if delta_nu_uhz <= 0 or expected_numax_uhz <= 0 or envelope_width_uhz <= 0:
        raise ValueError("frequency scales must be positive")
    if not 0 < relative_lag_half_width < 1:
        raise ValueError("relative_lag_half_width must lie between zero and one")
    if not 0 < ridge_fraction < 1:
        raise ValueError("ridge_fraction must lie between zero and one")

    primary_lag = 1e6 / float(delta_nu_uhz)
    if result.lags_seconds[-1] < 2.0 * primary_lag * (
        1.0 - relative_lag_half_width
    ):
        raise ValueError("lag grid does not cover the second seismic ACF peak")
    primary = _lag_profile(result, primary_lag, relative_lag_half_width)
    secondary = _lag_profile(result, 2.0 * primary_lag, relative_lag_half_width)
    peak_index = int(np.argmax(primary))
    peak = float(primary[peak_index])
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("candidate ridge has zero or non-finite EACF power")

    centres = result.centre_frequencies_uhz
    inside = np.abs(centres - expected_numax_uhz) <= envelope_width_uhz / 2.0
    if not np.any(inside):
        raise ValueError("no filter centre lies inside the predicted envelope")
    ridge = primary >= ridge_fraction * peak
    ridge_width = _longest_run_width(centres, ridge) / envelope_width_uhz
    flank = ~inside
    contrast = (
        float(
            np.mean(primary[inside])
            / max(np.mean(primary[flank]), np.finfo(float).eps)
        )
        if np.any(flank)
        else float("inf")
    )
    roughness = (
        float(np.median(np.abs(np.diff(primary))) / peak)
        if primary.size > 1
        else 0.0
    )
    return MorphologyDiagnostics(
        peak_centre_uhz=float(centres[peak_index]),
        centre_offset=float(
            abs(centres[peak_index] - expected_numax_uhz) / envelope_width_uhz
        ),
        ridge_width=float(ridge_width),
        ridge_fill=float(np.mean(ridge[inside])),
        ridge_contrast=contrast,
        ridge_roughness=roughness,
        harmonic_support=float(np.max(secondary) / peak),
    )


def benchmark_morphology_veto(
    window: ObservingWindow,
    simulation: SimulationConfig,
    centre_frequencies_uhz: ArrayLike,
    filter_width_uhz: float,
    delta_nu_grid_uhz: ArrayLike,
    *,
    coherent_contaminants: Mapping[str, CoherentSignalConfig] | None = None,
    segment_systematics: Mapping[
        str, Sequence[SegmentSystematicConfig]
    ] | None = None,
    background: BackgroundConfig | None = None,
    realizations: int = 128,
    target_false_positive_rate: float = 0.01,
    target_signal_retention: float = 0.95,
    max_lag_seconds: float | None = None,
    relative_lag_half_width: float = 0.05,
    ridge_fraction: float = 0.5,
    seed: int = 0,
) -> MorphologyVetoMetrics:
    """Benchmark morphology against clean, coherent, and non-stationary nulls.

    The raw EACF threshold is calibrated on clean exact-window null simulations.
    Morphology is then calibrated on the oscillation injections that pass that
    threshold. Each diagnostic is robustly centred and scaled using those
    injections; their maximum absolute deviation is calibrated jointly to the
    requested signal retention, without assuming that the diagnostics are
    independent.

    Parameters
    ----------
    window
        Exact observing window used by every simulation.
    simulation
        Noise, granulation, and injected-oscillation parameters.
    centre_frequencies_uhz
        Trial EACF filter centres in microhertz.
    filter_width_uhz
        Full EACF filter width in microhertz.
    delta_nu_grid_uhz
        Trial large separations in microhertz.
    coherent_contaminants
        Optional named coherent harmonic contaminants.
    segment_systematics
        Optional named piecewise variance, offset, or drift configurations.
    background
        Optional spectral-background treatment.
    realizations
        Number of paired realisations per simulation class.
    target_false_positive_rate
        Requested false-positive rate for the clean null.
    target_signal_retention
        Requested retention among raw EACF signal detections.
    max_lag_seconds
        Optional maximum EACF lag. It must include ``2 / delta_nu`` for the
        complete harmonic diagnostic.
    relative_lag_half_width
        Fractional half-width used around expected ACF lags.
    ridge_fraction
        Fraction of the primary profile maximum defining ridge membership.
    seed
        Root seed for deterministic paired simulations.

    Returns
    -------
    MorphologyVetoMetrics
        Raw and accepted rates with target-specific thresholds.
    """
    if realizations < 8:
        raise ValueError("at least eight realizations are required")
    if not 0 < target_false_positive_rate < 1:
        raise ValueError("target_false_positive_rate must lie between zero and one")
    if not 0 < target_signal_retention <= 1:
        raise ValueError("target_signal_retention must lie in (0, 1]")
    coherent = dict(coherent_contaminants or {})
    segmented = dict(segment_systematics or {})
    if not coherent and not segmented:
        raise ValueError("at least one contaminant class is required")

    centres = np.atleast_1d(np.asarray(centre_frequencies_uhz, dtype=float))
    delta_nu_grid = np.atleast_1d(np.asarray(delta_nu_grid_uhz, dtype=float))
    seeds = np.random.SeedSequence(seed).spawn(realizations)
    names = tuple(coherent) + tuple(segmented)
    if len(set(names)) != len(names):
        raise ValueError("contaminant class names must be unique")

    null_scores = np.empty(realizations)
    signal_scores = np.empty(realizations)
    signal_features = np.empty((realizations, 6))
    contaminant_scores = {name: np.empty(realizations) for name in names}
    contaminant_features = {
        name: np.empty((realizations, 6)) for name in names
    }

    for index, child_seed in enumerate(seeds):
        streams = child_seed.spawn(1 + len(coherent))
        base = simulate_time_series(
            window,
            simulation,
            np.random.default_rng(streams[0]),
            include_oscillations=False,
        )
        signal = simulate_time_series(
            window,
            simulation,
            np.random.default_rng(streams[0]),
            include_oscillations=True,
        )
        null_scores[index], _ = _score_and_features(
            base,
            centres,
            filter_width_uhz,
            delta_nu_grid,
            simulation,
            background,
            max_lag_seconds,
            relative_lag_half_width,
            ridge_fraction,
        )
        signal_scores[index], signal_features[index] = _score_and_features(
            signal,
            centres,
            filter_width_uhz,
            delta_nu_grid,
            simulation,
            background,
            max_lag_seconds,
            relative_lag_half_width,
            ridge_fraction,
        )
        for offset, (name, config) in enumerate(coherent.items(), start=1):
            candidate = add_coherent_signal(
                base, config, np.random.default_rng(streams[offset])
            )
            (
                contaminant_scores[name][index],
                contaminant_features[name][index],
            ) = _score_and_features(
                candidate,
                centres,
                filter_width_uhz,
                delta_nu_grid,
                simulation,
                background,
                max_lag_seconds,
                relative_lag_half_width,
                ridge_fraction,
            )
        for name, configs in segmented.items():
            candidate = add_segment_systematics(base, configs)
            (
                contaminant_scores[name][index],
                contaminant_features[name][index],
            ) = _score_and_features(
                candidate,
                centres,
                filter_width_uhz,
                delta_nu_grid,
                simulation,
                background,
                max_lag_seconds,
                relative_lag_half_width,
                ridge_fraction,
            )

    eacf_threshold = _higher_quantile(
        null_scores, 1.0 - target_false_positive_rate
    )
    signal_detected = signal_scores >= eacf_threshold
    reference = (
        signal_features[signal_detected]
        if np.any(signal_detected)
        else signal_features
    )
    signal_outlier = _outlier_scores(signal_features, reference)
    calibration = (
        signal_outlier[signal_detected] if np.any(signal_detected) else signal_outlier
    )
    morphology_threshold = _higher_quantile(
        calibration, target_signal_retention
    )

    raw_contaminant = {}
    accepted_contaminant = {}
    for name in names:
        detected = contaminant_scores[name] >= eacf_threshold
        outlier = _outlier_scores(contaminant_features[name], reference)
        raw_contaminant[name] = float(np.mean(detected))
        accepted_contaminant[name] = float(
            np.mean(detected & (outlier <= morphology_threshold))
        )

    return MorphologyVetoMetrics(
        eacf_threshold=eacf_threshold,
        morphology_outlier_threshold=morphology_threshold,
        raw_signal_detection_rate=float(np.mean(signal_detected)),
        accepted_signal_detection_rate=float(
            np.mean(signal_detected & (signal_outlier <= morphology_threshold))
        ),
        raw_contaminant_detection_rate=raw_contaminant,
        accepted_contaminant_detection_rate=accepted_contaminant,
        target_false_positive_rate=target_false_positive_rate,
        target_signal_retention=target_signal_retention,
        realizations=realizations,
    )


def _score_and_features(
    series: TimeSeries,
    centres: FloatArray,
    filter_width_uhz: float,
    delta_nu_grid: FloatArray,
    simulation: SimulationConfig,
    background: BackgroundConfig | None,
    max_lag_seconds: float | None,
    relative_lag_half_width: float,
    ridge_fraction: float,
) -> tuple[float, FloatArray]:
    result = compute_eacf_map(
        series,
        centres,
        filter_width_uhz,
        max_lag_seconds=max_lag_seconds,
        empirical_background=background,
    )
    delta_nu, score = estimate_delta_nu(
        result, delta_nu_grid, relative_lag_half_width
    )
    diagnostics = eacf_morphology(
        result,
        delta_nu,
        simulation.numax_uhz,
        simulation.envelope_width_uhz,
        relative_lag_half_width=relative_lag_half_width,
        ridge_fraction=ridge_fraction,
    )
    return score, _feature_array(diagnostics)


def _lag_profile(
    result: EACFMap,
    expected_lag: float,
    relative_half_width: float,
) -> FloatArray:
    select = (
        np.abs(result.lags_seconds - expected_lag)
        <= expected_lag * relative_half_width
    )
    if not np.any(select):
        nearest = int(np.argmin(np.abs(result.lags_seconds - expected_lag)))
        select = np.arange(result.lags_seconds.size) == nearest
    return np.max(result.values[:, select], axis=1)


def _longest_run_width(centres: FloatArray, selected: NDArray[np.bool_]) -> float:
    indices = np.flatnonzero(selected)
    if indices.size < 2:
        return 0.0
    split = np.flatnonzero(np.diff(indices) > 1) + 1
    runs = np.split(indices, split)
    return float(max(centres[run[-1]] - centres[run[0]] for run in runs))


def _feature_array(diagnostics: MorphologyDiagnostics) -> FloatArray:
    contrast = diagnostics.ridge_contrast
    if not np.isfinite(contrast):
        contrast = 1.0
    return np.asarray(
        [
            diagnostics.centre_offset,
            diagnostics.ridge_width,
            diagnostics.ridge_fill,
            contrast,
            diagnostics.ridge_roughness,
            diagnostics.harmonic_support,
        ],
        dtype=float,
    )


def _outlier_scores(values: FloatArray, reference: FloatArray) -> FloatArray:
    centre = np.median(reference, axis=0)
    scale = 1.4826 * np.median(np.abs(reference - centre), axis=0)
    standard_deviation = np.std(reference, axis=0)
    scale = np.where(scale > np.finfo(float).eps, scale, standard_deviation)
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    return np.max(np.abs(values - centre) / scale, axis=1)


def _higher_quantile(values: FloatArray, probability: float) -> float:
    try:
        return float(np.quantile(values, probability, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility
        return float(np.quantile(values, probability, interpolation="higher"))
