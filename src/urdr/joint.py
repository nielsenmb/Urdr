"""Joint simulation-calibrated inference for Urdr diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import beta

from .background import BackgroundConfig
from .benchmark import estimate_delta_nu
from .contaminants import (
    CoherenceDiagnostics,
    CoherentSignalConfig,
    add_coherent_signal,
    coherence_diagnostics,
)
from .eacf import compute_eacf_map
from .models import ObservingWindow, TimeSeries
from .morphology import MorphologyDiagnostics, eacf_morphology
from .simulation import SimulationConfig, simulate_time_series
from .systematics import (
    SegmentDiagnostics,
    SegmentSystematicConfig,
    add_segment_systematics,
    segment_diagnostics,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class JointDiagnostics:
    """All measurements used by the joint detector.

    Parameters
    ----------
    eacf_statistic
        Maximum EACF statistic over the trial large separations.
    delta_nu_uhz
        Large separation associated with the maximum EACF statistic.
    coherence
        Spectral-concentration diagnostics in the selected filter band.
    segments
        Robust between-segment stability diagnostics.
    morphology
        Frequency-lag ridge morphology diagnostics.
    """

    eacf_statistic: float
    delta_nu_uhz: float
    coherence: CoherenceDiagnostics
    segments: SegmentDiagnostics
    morphology: MorphologyDiagnostics


@dataclass(frozen=True)
class JointValidationMetrics:
    """Held-out performance of a calibrated joint detector.

    Parameters
    ----------
    true_positive_rate
        Fraction of held-out oscillation injections passing the joint threshold.
    false_positive_rate
        Fraction of held-out realisations for which any null or contaminant
        class passes the joint threshold.
    brier_score
        Mean squared error of held-out detection probabilities.
    signal_count
        Number of held-out signal injections.
    negative_realizations
        Number of held-out paired hard-negative realisations.
    """

    true_positive_rate: float
    false_positive_rate: float
    brier_score: float
    signal_count: int
    negative_realizations: int


@dataclass(frozen=True)
class DetectionResult:
    """Result returned by the joint Urdr detector.

    Parameters
    ----------
    detected
        Whether the held-out-calibrated false-alarm probability is below the
        requested target.
    detection_probability
        Logistic-calibrated probability from held-out simulations.
    false_alarm_probability
        Finite-simulation-corrected probability that any represented
        hard-negative class is at least as signal-like as the target.
    false_alarm_interval
        Equal-tailed 68 per cent beta-binomial interval on the false-alarm
        probability.
    joint_score
        Linear discriminant score before probability calibration.
    delta_nu_uhz
        Recovered large separation in microhertz.
    diagnostics
        Complete EACF, coherence, segment, and morphology measurements.
    diagnostic_flags
        Explanatory flags for diagnostic groups outside the central 95 per cent
        of held-out signal injections. Flags do not independently veto a target.
    """

    detected: bool
    detection_probability: float
    false_alarm_probability: float
    false_alarm_interval: tuple[float, float]
    joint_score: float
    delta_nu_uhz: float
    diagnostics: JointDiagnostics
    diagnostic_flags: tuple[str, ...]


@dataclass(frozen=True)
class JointDetector:
    """Target- and window-specific fitted joint detector.

    Instances are normally created by :func:`calibrate_joint_detector`.

    Parameters
    ----------
    centre_frequencies_uhz
        Trial EACF filter centres in microhertz.
    filter_width_uhz
        Full EACF filter width in microhertz.
    delta_nu_grid_uhz
        Trial large separations in microhertz.
    simulation
        Target-specific simulation parameters.
    segments_days
        Intervals used for segment diagnostics.
    background
        Optional spectral-background treatment.
    max_lag_seconds
        Optional maximum EACF lag.
    feature_centre, feature_scale
        Robust feature transformation learned from training simulations.
    discriminant_weights, discriminant_offset
        Regularised linear discriminant parameters.
    probability_slope, probability_offset
        Held-out logistic probability-calibration parameters.
    negative_validation_scores
        Per-realisaton maximum hard-negative scores used for false-alarm
        calibration.
    score_threshold
        Joint threshold at the requested false-positive rate.
    signal_feature_centre, signal_feature_scale
        Held-out signal reference used for explanatory diagnostic flags.
    flag_thresholds
        Signal-calibrated thresholds for coherence, segment, and morphology
        group deviations.
    target_false_positive_rate
        Requested false-positive rate.
    validation
        Held-out performance summary.
    """

    centre_frequencies_uhz: FloatArray
    filter_width_uhz: float
    delta_nu_grid_uhz: FloatArray
    simulation: SimulationConfig
    segments_days: tuple[tuple[float, float], ...]
    background: BackgroundConfig | None
    max_lag_seconds: float | None
    feature_centre: FloatArray
    feature_scale: FloatArray
    discriminant_weights: FloatArray
    discriminant_offset: float
    probability_slope: float
    probability_offset: float
    negative_validation_scores: FloatArray
    score_threshold: float
    signal_feature_centre: FloatArray
    signal_feature_scale: FloatArray
    flag_thresholds: FloatArray
    target_false_positive_rate: float
    validation: JointValidationMetrics

    def detect(self, series: TimeSeries) -> DetectionResult:
        """Run end-to-end joint inference for one time series.

        Parameters
        ----------
        series
            Target time series. Its cadence grid and observing mask should match
            those used to calibrate the detector.

        Returns
        -------
        DetectionResult
            Detection probability, false-alarm probability, recovered
            ``delta_nu``, uncertainty interval, diagnostics, and flags.
        """
        diagnostics, features = joint_diagnostics(
            series=series,
            simulation=self.simulation,
            centre_frequencies_uhz=self.centre_frequencies_uhz,
            filter_width_uhz=self.filter_width_uhz,
            delta_nu_grid_uhz=self.delta_nu_grid_uhz,
            segments_days=self.segments_days,
            background=self.background,
            max_lag_seconds=self.max_lag_seconds,
        )
        score = self._score(features)
        probability = _sigmoid(
            self.probability_offset + self.probability_slope * score
        )
        exceedances = int(
            np.count_nonzero(self.negative_validation_scores >= score)
        )
        total = self.negative_validation_scores.size
        false_alarm = float((exceedances + 1) / (total + 1))
        interval = _beta_interval(exceedances, total)
        flags = _diagnostic_flags(
            features,
            self.signal_feature_centre,
            self.signal_feature_scale,
            self.flag_thresholds,
        )
        return DetectionResult(
            detected=bool(false_alarm <= self.target_false_positive_rate),
            detection_probability=float(probability),
            false_alarm_probability=false_alarm,
            false_alarm_interval=interval,
            joint_score=score,
            delta_nu_uhz=diagnostics.delta_nu_uhz,
            diagnostics=diagnostics,
            diagnostic_flags=flags,
        )

    def _score(self, features: FloatArray) -> float:
        transformed = (features - self.feature_centre) / self.feature_scale
        return float(transformed @ self.discriminant_weights + self.discriminant_offset)


def joint_diagnostics(
    series: TimeSeries,
    simulation: SimulationConfig,
    centre_frequencies_uhz: ArrayLike,
    filter_width_uhz: float,
    delta_nu_grid_uhz: ArrayLike,
    segments_days: Sequence[tuple[float, float]],
    *,
    background: BackgroundConfig | None = None,
    max_lag_seconds: float | None = None,
) -> tuple[JointDiagnostics, FloatArray]:
    """Compute all joint features while evaluating the EACF map only once.

    Parameters
    ----------
    series
        Target or simulated time series.
    simulation
        Target-specific seismic parameters.
    centre_frequencies_uhz
        Trial EACF filter centres in microhertz.
    filter_width_uhz
        Full EACF filter width in microhertz.
    delta_nu_grid_uhz
        Trial large separations in microhertz.
    segments_days
        Intervals used for segment diagnostics.
    background
        Optional spectral-background treatment.
    max_lag_seconds
        Optional maximum EACF lag. The grid must cover the second seismic ACF
        peak for every trial separation.

    Returns
    -------
    diagnostics
        Structured measurements used by the joint detector.
    features
        Thirteen-element numerical feature vector.
    """
    centres = np.atleast_1d(np.asarray(centre_frequencies_uhz, dtype=float))
    delta_nu_grid = np.atleast_1d(np.asarray(delta_nu_grid_uhz, dtype=float))
    eacf_map = compute_eacf_map(
        series,
        centres,
        filter_width_uhz,
        max_lag_seconds=max_lag_seconds,
        empirical_background=background,
    )
    delta_nu, statistic = estimate_delta_nu(eacf_map, delta_nu_grid)
    morphology = eacf_morphology(
        eacf_map,
        delta_nu,
        simulation.numax_uhz,
        simulation.envelope_width_uhz,
    )
    coherence = coherence_diagnostics(
        series,
        morphology.peak_centre_uhz,
        filter_width_uhz,
        background,
    )
    segments = segment_diagnostics(series, segments_days)
    diagnostics = JointDiagnostics(
        eacf_statistic=statistic,
        delta_nu_uhz=delta_nu,
        coherence=coherence,
        segments=segments,
        morphology=morphology,
    )
    return diagnostics, _feature_array(diagnostics)


def calibrate_joint_detector(
    window: ObservingWindow,
    simulation: SimulationConfig,
    centre_frequencies_uhz: ArrayLike,
    filter_width_uhz: float,
    delta_nu_grid_uhz: ArrayLike,
    segments_days: Sequence[tuple[float, float]],
    *,
    coherent_contaminants: Mapping[str, CoherentSignalConfig] | None = None,
    segment_systematics: Mapping[
        str, Sequence[SegmentSystematicConfig]
    ] | None = None,
    background: BackgroundConfig | None = None,
    realizations: int = 512,
    validation_fraction: float = 0.25,
    target_false_positive_rate: float = 0.01,
    signal_prior_probability: float = 0.5,
    covariance_regularization: float = 0.1,
    max_lag_seconds: float | None = None,
    seed: int = 0,
) -> JointDetector:
    """Fit one detector to EACF, coherence, segment, and morphology features.

    Simulations are split by paired realisation before fitting. A regularised
    linear discriminant is learned on the training split. Its probability scale,
    false-alarm threshold, uncertainty, and diagnostic flags are calibrated
    only on held-out simulations. Clean nulls and all supplied contaminant
    classes form a single hard-negative hypothesis.

    Parameters
    ----------
    window
        Exact target observing window.
    simulation
        Noise, granulation, and seismic simulation parameters.
    centre_frequencies_uhz
        Trial EACF filter centres in microhertz.
    filter_width_uhz
        Full EACF filter width in microhertz.
    delta_nu_grid_uhz
        Trial large separations in microhertz.
    segments_days
        Intervals used for segment diagnostics.
    coherent_contaminants
        Optional named coherent hard-negative configurations.
    segment_systematics
        Optional named segment-systematic hard-negative configurations.
    background
        Optional spectral-background treatment.
    realizations
        Number of paired simulation realisations.
    validation_fraction
        Fraction of paired realisations reserved for calibration and validation.
    target_false_positive_rate
        Requested probability that any represented hard-negative class passes.
    signal_prior_probability
        Prior probability used by logistic probability calibration.
    covariance_regularization
        Shrinkage fraction towards a diagonal covariance matrix.
    max_lag_seconds
        Optional maximum EACF lag.
    seed
        Root seed for deterministic paired simulations.

    Returns
    -------
    JointDetector
        Fitted target-specific detector with held-out validation metrics.
    """
    _validate_calibration_inputs(
        realizations,
        validation_fraction,
        target_false_positive_rate,
        signal_prior_probability,
        covariance_regularization,
    )
    coherent = dict(coherent_contaminants or {})
    segmented = dict(segment_systematics or {})
    if not coherent and not segmented:
        raise ValueError("at least one contaminant class is required")
    names = tuple(coherent) + tuple(segmented)
    if len(set(names)) != len(names):
        raise ValueError("contaminant class names must be unique")

    centres = np.atleast_1d(np.asarray(centre_frequencies_uhz, dtype=float))
    delta_nu_grid = np.atleast_1d(np.asarray(delta_nu_grid_uhz, dtype=float))
    diagnostic_segments = tuple(
        (float(start), float(stop)) for start, stop in segments_days
    )
    validation_count = max(4, int(np.ceil(realizations * validation_fraction)))
    training_count = realizations - validation_count
    if training_count < 8:
        raise ValueError("calibration split leaves fewer than eight training cases")
    if 1.0 / (validation_count + 1) > target_false_positive_rate:
        raise ValueError(
            "validation split is too small to resolve the target false-positive "
            "rate; increase realizations or validation_fraction"
        )
    signal_features = np.empty((realizations, 13), dtype=float)
    negative_features = np.empty(
        (realizations, 1 + len(names), 13), dtype=float
    )
    seeds = np.random.SeedSequence(seed).spawn(realizations)

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
        signal_features[index] = _extract(
            signal,
            simulation,
            centres,
            filter_width_uhz,
            delta_nu_grid,
            diagnostic_segments,
            background,
            max_lag_seconds,
        )
        negative_features[index, 0] = _extract(
            base,
            simulation,
            centres,
            filter_width_uhz,
            delta_nu_grid,
            diagnostic_segments,
            background,
            max_lag_seconds,
        )
        column = 1
        for offset, config in enumerate(coherent.values(), start=1):
            candidate = add_coherent_signal(
                base, config, np.random.default_rng(streams[offset])
            )
            negative_features[index, column] = _extract(
                candidate,
                simulation,
                centres,
                filter_width_uhz,
                delta_nu_grid,
                diagnostic_segments,
                background,
                max_lag_seconds,
            )
            column += 1
        for configs in segmented.values():
            candidate = add_segment_systematics(base, configs)
            negative_features[index, column] = _extract(
                candidate,
                simulation,
                centres,
                filter_width_uhz,
                delta_nu_grid,
                diagnostic_segments,
                background,
                max_lag_seconds,
            )
            column += 1

    training_signal = signal_features[:training_count]
    training_negative = negative_features[:training_count].reshape(-1, 13)
    (
        feature_centre,
        feature_scale,
        weights,
        offset,
    ) = _fit_discriminant(
        training_signal, training_negative, covariance_regularization
    )

    validation_signal = signal_features[training_count:]
    validation_negative = negative_features[training_count:]
    signal_scores = _score_rows(
        validation_signal, feature_centre, feature_scale, weights, offset
    )
    negative_scores = _score_rows(
        validation_negative.reshape(-1, 13),
        feature_centre,
        feature_scale,
        weights,
        offset,
    ).reshape(validation_count, -1)
    maximum_negative_scores = np.max(negative_scores, axis=1)
    score_threshold = _conformal_threshold(
        maximum_negative_scores, target_false_positive_rate
    )
    probability_slope, probability_offset = _fit_probability_calibration(
        signal_scores,
        negative_scores.ravel(),
        signal_prior_probability,
    )
    signal_probabilities = _sigmoid_array(
        probability_offset + probability_slope * signal_scores
    )
    negative_probabilities = _sigmoid_array(
        probability_offset + probability_slope * negative_scores.ravel()
    )
    labels = np.concatenate(
        [np.ones(signal_probabilities.size), np.zeros(negative_probabilities.size)]
    )
    probabilities = np.concatenate(
        [signal_probabilities, negative_probabilities]
    )
    signal_centre, signal_scale = _robust_transform(validation_signal)
    flag_scores = _group_deviations(
        validation_signal, signal_centre, signal_scale
    )
    flag_thresholds = np.asarray(
        [_higher_quantile(flag_scores[:, i], 0.95) for i in range(3)]
    )
    signal_false_alarm = _false_alarm_probabilities(
        signal_scores, maximum_negative_scores
    )
    negative_false_alarm = _leave_one_out_false_alarm(
        maximum_negative_scores
    )
    validation = JointValidationMetrics(
        true_positive_rate=float(
            np.mean(signal_false_alarm <= target_false_positive_rate)
        ),
        false_positive_rate=float(
            np.mean(negative_false_alarm <= target_false_positive_rate)
        ),
        brier_score=float(np.mean((probabilities - labels) ** 2)),
        signal_count=validation_count,
        negative_realizations=validation_count,
    )
    return JointDetector(
        centre_frequencies_uhz=centres,
        filter_width_uhz=float(filter_width_uhz),
        delta_nu_grid_uhz=delta_nu_grid,
        simulation=simulation,
        segments_days=diagnostic_segments,
        background=background,
        max_lag_seconds=max_lag_seconds,
        feature_centre=feature_centre,
        feature_scale=feature_scale,
        discriminant_weights=weights,
        discriminant_offset=offset,
        probability_slope=probability_slope,
        probability_offset=probability_offset,
        negative_validation_scores=maximum_negative_scores,
        score_threshold=score_threshold,
        signal_feature_centre=signal_centre,
        signal_feature_scale=signal_scale,
        flag_thresholds=flag_thresholds,
        target_false_positive_rate=target_false_positive_rate,
        validation=validation,
    )


def _extract(
    series: TimeSeries,
    simulation: SimulationConfig,
    centres: FloatArray,
    filter_width_uhz: float,
    delta_nu_grid: FloatArray,
    segments: tuple[tuple[float, float], ...],
    background: BackgroundConfig | None,
    max_lag_seconds: float | None,
) -> FloatArray:
    return joint_diagnostics(
        series,
        simulation,
        centres,
        filter_width_uhz,
        delta_nu_grid,
        segments,
        background=background,
        max_lag_seconds=max_lag_seconds,
    )[1]


def _feature_array(diagnostics: JointDiagnostics) -> FloatArray:
    coherence = diagnostics.coherence
    segments = diagnostics.segments
    morphology = diagnostics.morphology
    contrast = (
        morphology.ridge_contrast
        if np.isfinite(morphology.ridge_contrast)
        else 1.0
    )
    return np.asarray(
        [
            np.log1p(diagnostics.eacf_statistic),
            coherence.maximum_bin_fraction,
            np.log1p(coherence.effective_bins),
            coherence.spectral_entropy,
            np.log1p(segments.maximum_scale_ratio),
            np.log1p(segments.maximum_location_shift),
            np.log1p(segments.maximum_drift),
            morphology.centre_offset,
            morphology.ridge_width,
            morphology.ridge_fill,
            np.log1p(contrast),
            morphology.ridge_roughness,
            morphology.harmonic_support,
        ],
        dtype=float,
    )


def _fit_discriminant(
    signal: FloatArray,
    negative: FloatArray,
    regularization: float,
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    combined = np.vstack([signal, negative])
    centre, scale = _robust_transform(combined)
    transformed_signal = (signal - centre) / scale
    transformed_negative = (negative - centre) / scale
    signal_mean = np.mean(transformed_signal, axis=0)
    negative_mean = np.mean(transformed_negative, axis=0)
    residuals = np.vstack(
        [transformed_signal - signal_mean, transformed_negative - negative_mean]
    )
    covariance = np.cov(residuals, rowvar=False)
    diagonal = np.diag(np.diag(covariance))
    covariance = (
        (1.0 - regularization) * covariance
        + regularization * diagonal
        + 1e-6 * np.eye(covariance.shape[0])
    )
    weights = np.linalg.solve(covariance, signal_mean - negative_mean)
    offset = float(-0.5 * (signal_mean + negative_mean) @ weights)
    return centre, scale, weights, offset


def _fit_probability_calibration(
    signal_scores: FloatArray,
    negative_scores: FloatArray,
    prior: float,
) -> tuple[float, float]:
    values = np.concatenate([signal_scores, negative_scores])
    labels = np.concatenate(
        [np.ones(signal_scores.size), np.zeros(negative_scores.size)]
    )
    weights = np.concatenate(
        [
            np.full(signal_scores.size, 0.5 / signal_scores.size),
            np.full(negative_scores.size, 0.5 / negative_scores.size),
        ]
    )
    design = np.column_stack([np.ones(values.size), values])
    coefficients = np.zeros(2, dtype=float)
    penalty = np.diag([1e-6, 1e-3])
    for _ in range(50):
        probability = _sigmoid_array(design @ coefficients)
        gradient = design.T @ (weights * (probability - labels))
        gradient += penalty @ coefficients
        curvature = weights * probability * (1.0 - probability)
        hessian = design.T @ (curvature[:, None] * design) + penalty
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    coefficients[0] += np.log(prior / (1.0 - prior))
    return float(max(coefficients[1], 0.0)), float(coefficients[0])


def _robust_transform(values: FloatArray) -> tuple[FloatArray, FloatArray]:
    centre = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - centre), axis=0)
    fallback = np.std(values, axis=0)
    scale = np.where(scale > np.finfo(float).eps, scale, fallback)
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    return centre, scale


def _score_rows(
    values: FloatArray,
    centre: FloatArray,
    scale: FloatArray,
    weights: FloatArray,
    offset: float,
) -> FloatArray:
    return np.asarray((values - centre) / scale @ weights + offset, dtype=float)


def _group_deviations(
    values: FloatArray,
    centre: FloatArray,
    scale: FloatArray,
) -> FloatArray:
    deviations = np.abs((values - centre) / scale)
    return np.column_stack(
        [
            np.max(deviations[:, 1:4], axis=1),
            np.max(deviations[:, 4:7], axis=1),
            np.max(deviations[:, 7:13], axis=1),
        ]
    )


def _diagnostic_flags(
    features: FloatArray,
    centre: FloatArray,
    scale: FloatArray,
    thresholds: FloatArray,
) -> tuple[str, ...]:
    scores = _group_deviations(features[None, :], centre, scale)[0]
    names = ("coherent_spectrum", "segment_instability", "atypical_morphology")
    return tuple(
        name
        for name, score, limit in zip(names, scores, thresholds)
        if score > limit
    )


def _beta_interval(exceedances: int, total: int) -> tuple[float, float]:
    lower = float(beta.ppf(0.16, exceedances + 1, total - exceedances + 1))
    upper = float(beta.ppf(0.84, exceedances + 1, total - exceedances + 1))
    return lower, upper


def _higher_quantile(values: FloatArray, probability: float) -> float:
    try:
        return float(np.quantile(values, probability, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility
        return float(np.quantile(values, probability, interpolation="higher"))


def _conformal_threshold(values: FloatArray, false_positive_rate: float) -> float:
    ordered = np.sort(values)
    rank = int(np.ceil((ordered.size + 1) * (1.0 - false_positive_rate)))
    return float(ordered[min(max(rank - 1, 0), ordered.size - 1)])


def _false_alarm_probabilities(
    scores: FloatArray,
    calibration: FloatArray,
) -> FloatArray:
    exceedances = np.sum(calibration[None, :] >= scores[:, None], axis=1)
    return np.asarray((exceedances + 1) / (calibration.size + 1), dtype=float)


def _leave_one_out_false_alarm(scores: FloatArray) -> FloatArray:
    output = np.empty(scores.size, dtype=float)
    for index, score in enumerate(scores):
        reference = np.delete(scores, index)
        output[index] = (
            np.count_nonzero(reference >= score) + 1
        ) / scores.size
    return output


def _sigmoid(value: float) -> float:
    return float(_sigmoid_array(np.asarray([value], dtype=float))[0])


def _sigmoid_array(values: FloatArray) -> FloatArray:
    clipped = np.clip(values, -40.0, 40.0)
    return np.asarray(1.0 / (1.0 + np.exp(-clipped)), dtype=float)


def _validate_calibration_inputs(
    realizations: int,
    validation_fraction: float,
    target_false_positive_rate: float,
    signal_prior_probability: float,
    covariance_regularization: float,
) -> None:
    if realizations < 16:
        raise ValueError("at least sixteen realizations are required")
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must lie between zero and 0.5")
    if not 0 < target_false_positive_rate < 1:
        raise ValueError("target_false_positive_rate must lie between zero and one")
    if not 0 < signal_prior_probability < 1:
        raise ValueError("signal_prior_probability must lie between zero and one")
    if not 0 <= covariance_regularization <= 1:
        raise ValueError("covariance_regularization must lie in [0, 1]")
