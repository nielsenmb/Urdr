"""Target-specific Monte Carlo calibration of the EACF statistic."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .models import ObservingWindow, TimeSeries
from .simulation import SimulationConfig, simulate_time_series

FloatArray = NDArray[np.float64]
Statistic = Callable[[TimeSeries], float]


@dataclass(frozen=True)
class CalibrationResult:
    """Observed statistic and its target-specific simulated distributions."""

    observed_statistic: float
    null_statistics: FloatArray
    signal_statistics: FloatArray

    @property
    def false_alarm_probability(self) -> float:
        """Return finite-sample-corrected empirical false-alarm probability."""

        exceedances = np.count_nonzero(
            self.null_statistics >= self.observed_statistic
        )
        return float((exceedances + 1) / (self.null_statistics.size + 1))

    @property
    def detection_efficiency(self) -> float:
        """Return fraction of signal simulations exceeding the observation."""

        return float(np.mean(self.signal_statistics >= self.observed_statistic))


class SimulationCalibrator:
    """Calibrate one statistic using deterministic paired simulations."""

    def __init__(self, simulations: int = 128, seed: int = 0) -> None:
        if simulations < 8:
            raise ValueError("at least eight simulations are required")
        self.simulations = simulations
        self.seed = seed

    def calibrate(
        self,
        observed: TimeSeries,
        config: SimulationConfig,
        statistic: Statistic,
    ) -> CalibrationResult:
        """Evaluate observed, null, and signal statistics for one target."""

        observed_statistic = float(statistic(observed))
        null = np.empty(self.simulations, dtype=float)
        signal = np.empty(self.simulations, dtype=float)
        seeds = np.random.SeedSequence(self.seed).spawn(self.simulations)
        window = observed.window
        for index, seed in enumerate(seeds):
            null_rng = np.random.default_rng(seed)
            signal_rng = np.random.default_rng(seed)
            null[index] = statistic(
                simulate_time_series(
                    window, replace(config, oscillation_amplitude=0.0), null_rng
                )
            )
            signal[index] = statistic(
                simulate_time_series(window, config, signal_rng)
            )
        return CalibrationResult(observed_statistic, null, signal)

    def null_distribution(
        self,
        window: ObservingWindow,
        config: SimulationConfig,
        statistic: Statistic,
    ) -> FloatArray:
        """Generate only the window-aware empirical null distribution."""

        values = np.empty(self.simulations, dtype=float)
        seeds = np.random.SeedSequence(self.seed).spawn(self.simulations)
        null_config = replace(config, oscillation_amplitude=0.0)
        for index, seed in enumerate(seeds):
            values[index] = statistic(
                simulate_time_series(
                    window, null_config, np.random.default_rng(seed), False
                )
            )
        return values

