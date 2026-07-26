"""Reproducible benchmarks for EACF background treatments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .background import (
    BackgroundConfig,
    EmpiricalBackgroundConfig,
    HarveyBackgroundConfig,
)
from .eacf import EACFMap, compute_eacf_map
from .models import ObservingWindow, TimeSeries
from .simulation import SimulationConfig, simulate_time_series

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Performance of one background treatment at one signal amplitude."""

    treatment: str
    oscillation_amplitude: float
    threshold: float
    false_positive_rate: float
    true_positive_rate: float
    delta_nu_recovery_rate: float
    median_delta_nu_error: float


@dataclass(frozen=True)
class BackgroundBenchmark:
    """Collection of benchmark metrics and the common experiment settings."""

    metrics: tuple[BenchmarkMetrics, ...]
    target_false_positive_rate: float
    delta_nu_tolerance: float
    realizations: int

    def by_treatment(self, name: str) -> tuple[BenchmarkMetrics, ...]:
        """Return amplitude-ordered metrics for one treatment."""

        return tuple(
            row for row in self.metrics if row.treatment == name
        )


def default_background_treatments(
    numax_uhz: float,
    envelope_width_uhz: float,
) -> dict[str, BackgroundConfig | None]:
    """Return the four pre-registered background-comparison arms."""

    return {
        "none": None,
        "legacy_empirical": EmpiricalBackgroundConfig(),
        "target_aware_empirical": EmpiricalBackgroundConfig.excluding_envelope(
            numax_uhz, envelope_width_uhz
        ),
        "harvey_like": HarveyBackgroundConfig.excluding_envelope(
            numax_uhz, envelope_width_uhz
        ),
    }


def estimate_delta_nu(
    result: EACFMap,
    delta_nu_grid_uhz: ArrayLike,
    relative_half_width: float = 0.05,
) -> tuple[float, float]:
    """Return the trial ``delta_nu`` with the strongest EACF statistic."""

    grid = np.atleast_1d(np.asarray(delta_nu_grid_uhz, dtype=float))
    if grid.ndim != 1 or grid.size == 0 or np.any(grid <= 0):
        raise ValueError("delta_nu grid must be a non-empty positive 1D array")
    scores = np.asarray(
        [result.statistic(value, relative_half_width) for value in grid]
    )
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])


def benchmark_background_treatments(
    window: ObservingWindow,
    simulation: SimulationConfig,
    centre_frequencies_uhz: ArrayLike,
    filter_width_uhz: float,
    delta_nu_grid_uhz: ArrayLike,
    oscillation_amplitudes: ArrayLike,
    *,
    treatments: Mapping[str, BackgroundConfig | None] | None = None,
    realizations: int = 128,
    target_false_positive_rate: float = 0.01,
    delta_nu_tolerance: float = 0.05,
    max_lag_seconds: float | None = None,
    seed: int = 0,
) -> BackgroundBenchmark:
    """Compare background arms using paired exact-window simulations.

    A separate null threshold is estimated for each treatment. Signal
    realizations share random seeds across treatments and amplitudes, which
    reduces Monte Carlo variance in comparisons.
    """

    if realizations < 8:
        raise ValueError("at least eight realizations are required")
    if not 0 < target_false_positive_rate < 1:
        raise ValueError("target_false_positive_rate must lie between zero and one")
    if not 0 < delta_nu_tolerance < 1:
        raise ValueError("delta_nu_tolerance must lie between zero and one")
    amplitudes = np.atleast_1d(np.asarray(oscillation_amplitudes, dtype=float))
    if amplitudes.ndim != 1 or amplitudes.size == 0 or np.any(amplitudes <= 0):
        raise ValueError("oscillation amplitudes must be a non-empty positive array")
    centres = np.atleast_1d(np.asarray(centre_frequencies_uhz, dtype=float))
    delta_nu_grid = np.atleast_1d(np.asarray(delta_nu_grid_uhz, dtype=float))
    arms = dict(
        treatments
        or default_background_treatments(
            simulation.numax_uhz, simulation.envelope_width_uhz
        )
    )
    if not arms:
        raise ValueError("at least one background treatment is required")

    seeds = np.random.SeedSequence(seed).spawn(realizations)
    null_series = [
        simulate_time_series(
            window,
            simulation,
            np.random.default_rng(item),
            include_oscillations=False,
        )
        for item in seeds
    ]
    rows: list[BenchmarkMetrics] = []
    for name, background in arms.items():
        null_scores = np.asarray(
            [
                _score_series(
                    series,
                    centres,
                    filter_width_uhz,
                    delta_nu_grid,
                    background,
                    max_lag_seconds,
                )[1]
                for series in null_series
            ]
        )
        threshold = _higher_quantile(
            null_scores, 1.0 - target_false_positive_rate
        )
        measured_fpr = float(np.mean(null_scores >= threshold))

        for amplitude in amplitudes:
            estimates = np.empty(realizations)
            scores = np.empty(realizations)
            signal_config = replace(
                simulation, oscillation_amplitude=float(amplitude)
            )
            for index, item in enumerate(seeds):
                series = simulate_time_series(
                    window, signal_config, np.random.default_rng(item)
                )
                estimates[index], scores[index] = _score_series(
                    series,
                    centres,
                    filter_width_uhz,
                    delta_nu_grid,
                    background,
                    max_lag_seconds,
                )
            relative_error = np.abs(
                estimates / simulation.delta_nu_uhz - 1.0
            )
            detected = scores >= threshold
            rows.append(
                BenchmarkMetrics(
                    treatment=name,
                    oscillation_amplitude=float(amplitude),
                    threshold=threshold,
                    false_positive_rate=measured_fpr,
                    true_positive_rate=float(np.mean(detected)),
                    delta_nu_recovery_rate=float(
                        np.mean(relative_error <= delta_nu_tolerance)
                    ),
                    median_delta_nu_error=float(np.median(relative_error)),
                )
            )
    return BackgroundBenchmark(
        tuple(rows),
        target_false_positive_rate,
        delta_nu_tolerance,
        realizations,
    )


def _score_series(
    series: TimeSeries,
    centres: FloatArray,
    filter_width_uhz: float,
    delta_nu_grid: FloatArray,
    background: BackgroundConfig | None,
    max_lag_seconds: float | None,
) -> tuple[float, float]:
    result = compute_eacf_map(
        series,
        centres,
        filter_width_uhz,
        max_lag_seconds=max_lag_seconds,
        empirical_background=background,
    )
    return estimate_delta_nu(result, delta_nu_grid)


def _higher_quantile(values: FloatArray, probability: float) -> float:
    try:
        return float(np.quantile(values, probability, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility
        return float(np.quantile(values, probability, interpolation="higher"))
