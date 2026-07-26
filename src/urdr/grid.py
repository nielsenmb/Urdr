"""Parameter-grid benchmarks for the empirical PSD background."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .background import EmpiricalBackgroundConfig
from .benchmark import BenchmarkMetrics, benchmark_background_treatments
from .models import ObservingWindow
from .simulation import SimulationConfig

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EmpiricalGridPoint:
    """One candidate empirical-background smoothing law.

    Parameters
    ----------
    a
        Scale coefficient in ``half_width = a * frequency**b``.
    b
        Power-law exponent in ``half_width = a * frequency**b``.
    """

    a: float
    b: float

    def __post_init__(self) -> None:
        """Validate the smoothing-law coefficients."""
        if self.a <= 0 or self.b <= 0:
            raise ValueError("a and b must be positive")

    @property
    def name(self) -> str:
        """Return a stable treatment name for tables and saved results."""
        return f"a={self.a:g},b={self.b:g}"


@dataclass(frozen=True)
class BackgroundBenchmarkCase:
    """One target and observing-window configuration in a grid benchmark.

    Parameters
    ----------
    name
        Unique human-readable case identifier.
    window
        Exact observing window for this case.
    simulation
        Target and time-series simulation parameters.
    centre_frequencies_uhz
        Trial EACF filter centres in microhertz.
    filter_width_uhz
        Full EACF filter width in microhertz.
    delta_nu_grid_uhz
        Trial large separations in microhertz.
    oscillation_amplitudes
        Signal amplitudes evaluated for this case.
    split
        Dataset split, normally ``"train"`` or ``"validation"``.
    """

    name: str
    window: ObservingWindow
    simulation: SimulationConfig
    centre_frequencies_uhz: FloatArray
    filter_width_uhz: float
    delta_nu_grid_uhz: FloatArray
    oscillation_amplitudes: FloatArray
    split: str = "train"

    def __post_init__(self) -> None:
        """Validate and normalise the benchmark-case settings."""
        if not self.name:
            raise ValueError("case name cannot be empty")
        if not self.split:
            raise ValueError("case split cannot be empty")
        if self.filter_width_uhz <= 0:
            raise ValueError("filter_width_uhz must be positive")
        for field_name in (
            "centre_frequencies_uhz",
            "delta_nu_grid_uhz",
            "oscillation_amplitudes",
        ):
            values = np.atleast_1d(np.asarray(getattr(self, field_name), dtype=float))
            if values.ndim != 1 or values.size == 0 or np.any(values <= 0):
                raise ValueError(f"{field_name} must be a non-empty positive array")
            object.__setattr__(self, field_name, values)


@dataclass(frozen=True)
class EmpiricalGridMetric:
    """Performance for one candidate, case, and signal amplitude.

    Parameters
    ----------
    case
        Benchmark-case name.
    split
        Dataset split assigned to the case.
    candidate
        Candidate smoothing-law coefficients.
    metrics
        Fixed-false-positive-rate detection and ``delta_nu`` metrics.
    """

    case: str
    split: str
    candidate: EmpiricalGridPoint
    metrics: BenchmarkMetrics


@dataclass(frozen=True)
class EmpiricalGridSummary:
    """Aggregate performance for one candidate over selected cases.

    Parameters
    ----------
    candidate
        Candidate smoothing-law coefficients.
    mean_true_positive_rate
        Mean signal-detection rate.
    mean_delta_nu_recovery_rate
        Mean fraction of successful ``delta_nu`` recoveries.
    median_delta_nu_error
        Median of the per-condition median fractional errors.
    maximum_false_positive_rate
        Largest measured false-positive rate across conditions.
    conditions
        Number of case-amplitude conditions in the aggregate.
    """

    candidate: EmpiricalGridPoint
    mean_true_positive_rate: float
    mean_delta_nu_recovery_rate: float
    median_delta_nu_error: float
    maximum_false_positive_rate: float
    conditions: int


@dataclass(frozen=True)
class EmpiricalGridBenchmark:
    """Results from a multi-target empirical-background grid benchmark.

    Parameters
    ----------
    metrics
        Per-case and per-amplitude benchmark metrics.
    target_false_positive_rate
        Requested false-positive rate used for every candidate.
    realizations
        Number of paired simulations per case and signal class.
    """

    metrics: tuple[EmpiricalGridMetric, ...]
    target_false_positive_rate: float
    realizations: int

    def summaries(self, split: str | None = None) -> tuple[EmpiricalGridSummary, ...]:
        """Aggregate detection and recovery metrics by candidate.

        Parameters
        ----------
        split
            Optional dataset split to include.

        Returns
        -------
        tuple
            Candidate summaries ordered by ``a`` and then ``b``.
        """
        selected = [
            row for row in self.metrics if split is None or row.split == split
        ]
        candidates = sorted(
            {row.candidate for row in selected},
            key=lambda item: (item.a, item.b),
        )
        output = []
        for candidate in candidates:
            rows = [row.metrics for row in selected if row.candidate == candidate]
            output.append(
                EmpiricalGridSummary(
                    candidate=candidate,
                    mean_true_positive_rate=float(
                        np.mean([row.true_positive_rate for row in rows])
                    ),
                    mean_delta_nu_recovery_rate=float(
                        np.mean([row.delta_nu_recovery_rate for row in rows])
                    ),
                    median_delta_nu_error=float(
                        np.median([row.median_delta_nu_error for row in rows])
                    ),
                    maximum_false_positive_rate=float(
                        np.max([row.false_positive_rate for row in rows])
                    ),
                    conditions=len(rows),
                )
            )
        return tuple(output)

    def pareto_front(
        self,
        split: str = "train",
    ) -> tuple[EmpiricalGridSummary, ...]:
        """Return candidates not dominated in detection and recovery rate.

        Parameters
        ----------
        split
            Dataset split on which to define the frontier.

        Returns
        -------
        tuple
            Non-dominated candidate summaries.

        Notes
        -----
        A candidate is dominated when another candidate has at least as high a
        mean true-positive rate and ``delta_nu`` recovery rate, with a strict
        improvement in at least one.
        """
        summaries = self.summaries(split)
        frontier = []
        for candidate in summaries:
            dominated = any(
                other != candidate
                and other.mean_true_positive_rate
                >= candidate.mean_true_positive_rate
                and other.mean_delta_nu_recovery_rate
                >= candidate.mean_delta_nu_recovery_rate
                and (
                    other.mean_true_positive_rate
                    > candidate.mean_true_positive_rate
                    or other.mean_delta_nu_recovery_rate
                    > candidate.mean_delta_nu_recovery_rate
                )
                for other in summaries
            )
            if not dominated:
                frontier.append(candidate)
        return tuple(frontier)

    def to_records(self) -> list[dict[str, float | int | str]]:
        """Return machine-readable records suitable for CSV or pandas.

        Returns
        -------
        list
            One flat dictionary per case, candidate, and signal amplitude.
        """
        return [
            {
                "case": row.case,
                "split": row.split,
                "a": row.candidate.a,
                "b": row.candidate.b,
                "oscillation_amplitude": row.metrics.oscillation_amplitude,
                "threshold": row.metrics.threshold,
                "false_positive_rate": row.metrics.false_positive_rate,
                "true_positive_rate": row.metrics.true_positive_rate,
                "delta_nu_recovery_rate": row.metrics.delta_nu_recovery_rate,
                "median_delta_nu_error": row.metrics.median_delta_nu_error,
                "realizations": self.realizations,
            }
            for row in self.metrics
        ]


def benchmark_empirical_grid(
    cases: Sequence[BackgroundBenchmarkCase],
    candidates: Iterable[EmpiricalGridPoint],
    *,
    anchors: int = 100,
    minimum_bins: int = 5,
    exclude_envelope: bool = True,
    realizations: int = 128,
    target_false_positive_rate: float = 0.01,
    delta_nu_tolerance: float = 0.05,
    max_lag_seconds: float | None = None,
    seed: int = 0,
) -> EmpiricalGridBenchmark:
    """Benchmark smoothing laws across targets, S/N, and observing windows.

    Each candidate receives its own exact-window null threshold at the requested
    false-positive rate. Common random numbers are used across candidates within
    every case. Independent child seeds are used between cases.

    Parameters
    ----------
    cases
        Target and observing-window benchmark cases.
    candidates
        Candidate ``a`` and ``b`` smoothing laws.
    anchors
        Number of log-spaced background anchors.
    minimum_bins
        Minimum periodogram bins contributing to an anchor.
    exclude_envelope
        Whether to exclude the predicted oscillation envelope.
    realizations
        Number of paired simulations per case and signal class.
    target_false_positive_rate
        Requested per-candidate false-positive rate.
    delta_nu_tolerance
        Fractional tolerance defining successful ``delta_nu`` recovery.
    max_lag_seconds
        Optional maximum EACF lag.
    seed
        Root seed for deterministic case-level seed streams.

    Returns
    -------
    EmpiricalGridBenchmark
        Per-condition metrics, aggregate helpers, and flat records.
    """
    case_list = tuple(cases)
    candidate_list = tuple(candidates)
    if not case_list:
        raise ValueError("at least one benchmark case is required")
    if not candidate_list:
        raise ValueError("at least one empirical candidate is required")
    if len({case.name for case in case_list}) != len(case_list):
        raise ValueError("benchmark case names must be unique")
    if len(set(candidate_list)) != len(candidate_list):
        raise ValueError("empirical candidates must be unique")

    rows: list[EmpiricalGridMetric] = []
    case_seeds = np.random.SeedSequence(seed).spawn(len(case_list))
    for case, case_seed in zip(case_list, case_seeds):
        treatments = {}
        for candidate in candidate_list:
            kwargs = {
                "a": candidate.a,
                "b": candidate.b,
                "anchors": anchors,
                "minimum_bins": minimum_bins,
            }
            if exclude_envelope:
                config = EmpiricalBackgroundConfig.excluding_envelope(
                    case.simulation.numax_uhz,
                    case.simulation.envelope_width_uhz,
                    **kwargs,
                )
            else:
                config = EmpiricalBackgroundConfig(**kwargs)
            treatments[candidate.name] = config

        result = benchmark_background_treatments(
            window=case.window,
            simulation=case.simulation,
            centre_frequencies_uhz=case.centre_frequencies_uhz,
            filter_width_uhz=case.filter_width_uhz,
            delta_nu_grid_uhz=case.delta_nu_grid_uhz,
            oscillation_amplitudes=case.oscillation_amplitudes,
            treatments=treatments,
            realizations=realizations,
            target_false_positive_rate=target_false_positive_rate,
            delta_nu_tolerance=delta_nu_tolerance,
            max_lag_seconds=max_lag_seconds,
            seed=int(case_seed.generate_state(1, dtype=np.uint32)[0]),
        )
        candidate_by_name = {item.name: item for item in candidate_list}
        rows.extend(
            EmpiricalGridMetric(
                case=case.name,
                split=case.split,
                candidate=candidate_by_name[metric.treatment],
                metrics=metric,
            )
            for metric in result.metrics
        )
    return EmpiricalGridBenchmark(
        metrics=tuple(rows),
        target_false_positive_rate=target_false_positive_rate,
        realizations=realizations,
    )


def make_observing_window(
    duration_days: float,
    cadence_seconds: float,
    *,
    gaps_days: Sequence[tuple[float, float]] = (),
    random_missing_fraction: float = 0.0,
    seed: int = 0,
) -> ObservingWindow:
    """Construct a reproducible uniform grid with explicit missing cadences.

    Parameters
    ----------
    duration_days
        Total observing duration in days.
    cadence_seconds
        Uniform cadence in seconds.
    gaps_days
        Half-open ``(start, stop)`` intervals to mark as missing, in days.
    random_missing_fraction
        Fraction of remaining cadences removed independently.
    seed
        Seed controlling random missing cadences.

    Returns
    -------
    ObservingWindow
        Uniform cadence grid and boolean observing mask.
    """
    if duration_days <= 0 or cadence_seconds <= 0:
        raise ValueError("duration and cadence must be positive")
    if not 0 <= random_missing_fraction < 1:
        raise ValueError("random_missing_fraction must lie in [0, 1)")
    cadence_days = cadence_seconds / 86400.0
    time = np.arange(0.0, duration_days, cadence_days)
    observed = np.ones(time.size, dtype=bool)
    for start, stop in gaps_days:
        if not 0 <= start < stop <= duration_days:
            raise ValueError("gaps must lie inside the observing duration")
        observed[(time >= start) & (time < stop)] = False
    if random_missing_fraction > 0:
        available = np.flatnonzero(observed)
        count = int(np.floor(random_missing_fraction * available.size))
        removed = np.random.default_rng(seed).choice(
            available, size=count, replace=False
        )
        observed[removed] = False
    if np.count_nonzero(observed) < 2:
        raise ValueError("observing window must retain at least two cadences")
    return ObservingWindow(time, observed)
