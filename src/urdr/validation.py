"""Broad held-out validation for Urdr detection methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .background import BackgroundConfig, EmpiricalBackgroundConfig
from .benchmark import estimate_delta_nu
from .contaminants import (
    CoherentSignalConfig,
    add_coherent_signal,
)
from .eacf import compute_eacf_map
from .joint import JointDetector, calibrate_joint_detector
from .models import ObservingWindow, TimeSeries
from .simulation import SimulationConfig, simulate_time_series
from .systematics import (
    SegmentSystematicConfig,
    add_segment_systematics,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ValidationCase:
    """One pre-registered target and observing-window validation cell.

    Parameters
    ----------
    name
        Unique human-readable case name.
    split
        Experimental split, normally ``"train"`` or ``"validation"``.
    window
        Exact cadence grid and observing mask.
    simulation
        Target-specific forward-model parameters.
    published_centres_uhz
        Broad filter-centre grid used by the published-style EACF arm.
    restricted_centres_uhz
        AsteroScale-informed filter-centre grid.
    filter_width_uhz
        Full EACF filter width in microhertz.
    delta_nu_grid_uhz
        Trial large separations in microhertz.
    segments_days
        Intervals used by the joint segment diagnostics.
    coherent_contaminants
        Named coherent hard-negative configurations.
    segment_systematics
        Named segment-dependent hard-negative configurations.
    background
        Background treatment used by the restricted and joint arms. The
        published-style arm always uses the legacy empirical treatment.
    max_lag_seconds
        Optional maximum EACF lag.
    """

    name: str
    split: str
    window: ObservingWindow
    simulation: SimulationConfig
    published_centres_uhz: FloatArray
    restricted_centres_uhz: FloatArray
    filter_width_uhz: float
    delta_nu_grid_uhz: FloatArray
    segments_days: tuple[tuple[float, float], ...]
    coherent_contaminants: Mapping[str, CoherentSignalConfig]
    segment_systematics: Mapping[str, Sequence[SegmentSystematicConfig]]
    background: BackgroundConfig | None = None
    max_lag_seconds: float | None = None

    def __post_init__(self) -> None:
        """Validate and normalise the validation-cell settings."""
        if not self.name or not self.split:
            raise ValueError("case name and split cannot be empty")
        if self.filter_width_uhz <= 0:
            raise ValueError("filter_width_uhz must be positive")
        for name in (
            "published_centres_uhz",
            "restricted_centres_uhz",
            "delta_nu_grid_uhz",
        ):
            values = np.atleast_1d(np.asarray(getattr(self, name), dtype=float))
            if values.ndim != 1 or values.size == 0 or np.any(values <= 0):
                raise ValueError(f"{name} must be a non-empty positive array")
            object.__setattr__(self, name, values)
        if not self.coherent_contaminants and not self.segment_systematics:
            raise ValueError("at least one contaminant class is required")


@dataclass(frozen=True)
class ValidationMetric:
    """Performance for one method, case, and simulation class.

    Parameters
    ----------
    case
        Validation-cell name.
    split
        Experimental split assigned to the cell.
    method
        Detection method name.
    simulation_class
        ``"signal"``, ``"null"``, or a named contaminant.
    sample_count
        Number of independent evaluation realisations.
    detection_rate
        Fraction classified as oscillation detections.
    delta_nu_recovery_rate
        Fraction recovering the injected separation within the tolerance.
        This is NaN for hard-negative classes.
    median_delta_nu_error
        Median absolute fractional separation error. This is NaN for
        hard-negative classes.
    brier_score
        Mean squared probability error for this class.
    """

    case: str
    split: str
    method: str
    simulation_class: str
    sample_count: int
    detection_rate: float
    delta_nu_recovery_rate: float
    median_delta_nu_error: float
    brier_score: float


@dataclass(frozen=True)
class ReliabilityBin:
    """One probability-reliability bin.

    Parameters
    ----------
    method
        Detection method name.
    split
        Experimental split included in the bin.
    lower, upper
        Inclusive lower and exclusive upper probability bounds.
    mean_probability
        Mean reported detection probability in the bin.
    observed_frequency
        Observed signal fraction in the bin.
    sample_count
        Number of predictions in the bin.
    """

    method: str
    split: str
    lower: float
    upper: float
    mean_probability: float
    observed_frequency: float
    sample_count: int


@dataclass(frozen=True)
class SyntheticValidation:
    """Results of a paired, broad synthetic validation experiment.

    Parameters
    ----------
    metrics
        Per-case, method, and simulation-class performance.
    reliability
        Probability reliability bins pooled within each split.
    target_false_positive_rate
        Requested detection false-positive rate.
    delta_nu_tolerance
        Fractional separation tolerance used for recovery metrics.
    calibration_realizations
        Number of target-specific calibration realisations.
    evaluation_realizations
        Number of independent evaluation realisations per class.
    """

    metrics: tuple[ValidationMetric, ...]
    reliability: tuple[ReliabilityBin, ...]
    target_false_positive_rate: float
    delta_nu_tolerance: float
    calibration_realizations: int
    evaluation_realizations: int

    def to_records(self) -> list[dict[str, float | int | str]]:
        """Return flat metric rows suitable for CSV or pandas.

        Returns
        -------
        list
            One serialisable dictionary per validation metric.
        """
        return [dict(vars(row)) for row in self.metrics]


def benchmark_synthetic_validation(
    cases: Sequence[ValidationCase],
    *,
    calibration_realizations: int = 512,
    evaluation_realizations: int = 256,
    validation_fraction: float = 0.25,
    target_false_positive_rate: float = 0.01,
    delta_nu_tolerance: float = 0.05,
    reliability_bins: int = 10,
    seed: int = 0,
) -> SyntheticValidation:
    """Compare published, restricted, and joint methods on held-out regimes.

    All methods see paired simulation realisations. Baseline false-alarm
    probabilities are conformal ranks against independent exact-window null
    simulations. The joint detector retains its held-out probability
    calibration. Evaluation realisations are independent of all calibration
    simulations.

    Parameters
    ----------
    cases
        Pre-registered target, signal-to-noise, window, and contaminant cells.
    calibration_realizations
        Number of simulations used for each target-specific calibration.
    evaluation_realizations
        Independent evaluation realisations per simulation class.
    validation_fraction
        Fraction reserved internally by the joint detector.
    target_false_positive_rate
        Detection threshold for all methods.
    delta_nu_tolerance
        Fractional tolerance defining successful separation recovery.
    reliability_bins
        Number of equal-width probability bins.
    seed
        Root seed for deterministic paired experiments.

    Returns
    -------
    SyntheticValidation
        Per-regime metrics and pooled reliability data.
    """
    if not cases:
        raise ValueError("at least one validation case is required")
    if calibration_realizations < 16 or evaluation_realizations < 4:
        raise ValueError("insufficient calibration or evaluation realisations")
    if not 0 < target_false_positive_rate < 1:
        raise ValueError("target_false_positive_rate must lie between zero and one")
    if not 0 < delta_nu_tolerance < 1:
        raise ValueError("delta_nu_tolerance must lie between zero and one")
    if reliability_bins < 2:
        raise ValueError("reliability_bins must be at least two")
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("validation case names must be unique")

    metrics: list[ValidationMetric] = []
    probability_rows: list[tuple[str, str, float, int]] = []
    case_seeds = np.random.SeedSequence(seed).spawn(len(cases))
    for case, case_seed in zip(cases, case_seeds, strict=True):
        calibration_seed, evaluation_seed = case_seed.spawn(2)
        baselines = _calibrate_baselines(
            case, calibration_realizations, calibration_seed
        )
        detector = calibrate_joint_detector(
            window=case.window,
            simulation=case.simulation,
            centre_frequencies_uhz=case.restricted_centres_uhz,
            filter_width_uhz=case.filter_width_uhz,
            delta_nu_grid_uhz=case.delta_nu_grid_uhz,
            segments_days=case.segments_days,
            coherent_contaminants=case.coherent_contaminants,
            segment_systematics=case.segment_systematics,
            background=case.background,
            realizations=calibration_realizations,
            validation_fraction=validation_fraction,
            target_false_positive_rate=target_false_positive_rate,
            max_lag_seconds=case.max_lag_seconds,
            seed=int(calibration_seed.generate_state(1)[0]),
        )
        class_results = _evaluate_case(
            case,
            baselines,
            detector,
            evaluation_realizations,
            target_false_positive_rate,
            evaluation_seed,
        )
        for method, classes in class_results.items():
            for class_name, predictions in classes.items():
                labels = predictions[:, 0]
                probabilities = predictions[:, 1]
                detections = predictions[:, 2]
                delta_nu = predictions[:, 3]
                is_signal = class_name == "signal"
                if is_signal:
                    errors = np.abs(
                        delta_nu - case.simulation.delta_nu_uhz
                    ) / case.simulation.delta_nu_uhz
                    recovery = float(np.mean(errors <= delta_nu_tolerance))
                    median_error = float(np.median(errors))
                else:
                    recovery = float("nan")
                    median_error = float("nan")
                metrics.append(
                    ValidationMetric(
                        case=case.name,
                        split=case.split,
                        method=method,
                        simulation_class=class_name,
                        sample_count=evaluation_realizations,
                        detection_rate=float(np.mean(detections)),
                        delta_nu_recovery_rate=recovery,
                        median_delta_nu_error=median_error,
                        brier_score=float(np.mean((probabilities - labels) ** 2)),
                    )
                )
                probability_rows.extend(
                    (method, case.split, float(probability), int(label))
                    for probability, label in zip(
                        probabilities, labels, strict=True
                    )
                )
    reliability = _reliability_table(probability_rows, reliability_bins)
    return SyntheticValidation(
        metrics=tuple(metrics),
        reliability=reliability,
        target_false_positive_rate=float(target_false_positive_rate),
        delta_nu_tolerance=float(delta_nu_tolerance),
        calibration_realizations=calibration_realizations,
        evaluation_realizations=evaluation_realizations,
    )


def _calibrate_baselines(
    case: ValidationCase,
    realizations: int,
    seed: np.random.SeedSequence,
) -> dict[str, tuple[FloatArray, FloatArray, BackgroundConfig | None]]:
    methods = {
        "published_eacf": (
            case.published_centres_uhz,
            EmpiricalBackgroundConfig(),
        ),
        "asteroscale_eacf": (
            case.restricted_centres_uhz,
            case.background,
        ),
    }
    child_seeds = seed.spawn(realizations)
    nulls = [
        simulate_time_series(
            case.window,
            case.simulation,
            np.random.default_rng(child),
            include_oscillations=False,
        )
        for child in child_seeds
    ]
    output = {}
    for name, (centres, background) in methods.items():
        scores = np.asarray(
            [
                _baseline_result(series, case, centres, background)[1]
                for series in nulls
            ]
        )
        output[name] = (np.asarray(centres), scores, background)
    return output


def _evaluate_case(
    case: ValidationCase,
    baselines: Mapping[
        str, tuple[FloatArray, FloatArray, BackgroundConfig | None]
    ],
    detector: JointDetector,
    realizations: int,
    target_false_positive_rate: float,
    seed: np.random.SeedSequence,
) -> dict[str, dict[str, FloatArray]]:
    class_names = (
        ("signal", "null")
        + tuple(case.coherent_contaminants)
        + tuple(case.segment_systematics)
    )
    output = {
        method: {
            name: np.empty((realizations, 4), dtype=float)
            for name in class_names
        }
        for method in (*baselines, "joint_urdr")
    }
    for index, child in enumerate(seed.spawn(realizations)):
        streams = child.spawn(2 + len(case.coherent_contaminants))
        base = simulate_time_series(
            case.window,
            case.simulation,
            np.random.default_rng(streams[0]),
            include_oscillations=False,
        )
        signal = simulate_time_series(
            case.window,
            case.simulation,
            np.random.default_rng(streams[0]),
            include_oscillations=True,
        )
        candidates: dict[str, tuple[TimeSeries, int]] = {
            "signal": (signal, 1),
            "null": (base, 0),
        }
        for offset, (name, config) in enumerate(
            case.coherent_contaminants.items(), start=1
        ):
            candidates[name] = (
                add_coherent_signal(
                    base, config, np.random.default_rng(streams[offset])
                ),
                0,
            )
        for name, configs in case.segment_systematics.items():
            candidates[name] = (add_segment_systematics(base, configs), 0)

        for class_name, (series, label) in candidates.items():
            for method, (centres, null_scores, background) in baselines.items():
                delta_nu, score = _baseline_result(
                    series, case, centres, background
                )
                fap = (np.count_nonzero(null_scores >= score) + 1) / (
                    null_scores.size + 1
                )
                output[method][class_name][index] = (
                    label, 1.0 - fap, fap <= target_false_positive_rate, delta_nu
                )
            result = detector.detect(series)
            output["joint_urdr"][class_name][index] = (
                label,
                result.detection_probability,
                result.detected,
                result.delta_nu_uhz,
            )
    return output


def _baseline_result(
    series: TimeSeries,
    case: ValidationCase,
    centres: FloatArray,
    background: BackgroundConfig | None,
) -> tuple[float, float]:
    eacf_map = compute_eacf_map(
        series,
        centres,
        case.filter_width_uhz,
        max_lag_seconds=case.max_lag_seconds,
        empirical_background=background,
    )
    return estimate_delta_nu(eacf_map, case.delta_nu_grid_uhz)


def _reliability_table(
    rows: Sequence[tuple[str, str, float, int]],
    bins: int,
) -> tuple[ReliabilityBin, ...]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    output = []
    for method, split in sorted({(row[0], row[1]) for row in rows}):
        selected = [row for row in rows if row[:2] == (method, split)]
        probabilities = np.asarray([row[2] for row in selected])
        labels = np.asarray([row[3] for row in selected])
        indices = np.minimum(
            np.searchsorted(edges, probabilities, side="right") - 1,
            bins - 1,
        )
        for index in range(bins):
            take = indices == index
            if not np.any(take):
                continue
            output.append(
                ReliabilityBin(
                    method=method,
                    split=split,
                    lower=float(edges[index]),
                    upper=float(edges[index + 1]),
                    mean_probability=float(np.mean(probabilities[take])),
                    observed_frequency=float(np.mean(labels[take])),
                    sample_count=int(np.count_nonzero(take)),
                )
            )
    return tuple(output)
