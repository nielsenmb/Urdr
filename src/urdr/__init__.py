"""Window-aware time-series EACF detection."""

from .calibration import CalibrationResult, SimulationCalibrator
from .background import (
    BackgroundConfig,
    EmpiricalBackgroundConfig,
    HarveyBackgroundConfig,
    estimate_background,
    estimate_empirical_background,
    estimate_harvey_background,
    whiten_spectrum,
)
from .benchmark import (
    BackgroundBenchmark,
    BenchmarkMetrics,
    benchmark_background_treatments,
    default_background_treatments,
    estimate_delta_nu,
)
from .eacf import EACFMap, compute_eacf, compute_eacf_map
from .models import AsteroScaleSamples, ObservingWindow, SearchRegion, TimeSeries
from .simulation import SimulationConfig, simulate_time_series

__all__ = [
    "AsteroScaleSamples",
    "BackgroundBenchmark",
    "BackgroundConfig",
    "BenchmarkMetrics",
    "CalibrationResult",
    "EACFMap",
    "EmpiricalBackgroundConfig",
    "HarveyBackgroundConfig",
    "ObservingWindow",
    "SearchRegion",
    "SimulationCalibrator",
    "SimulationConfig",
    "TimeSeries",
    "benchmark_background_treatments",
    "compute_eacf",
    "compute_eacf_map",
    "default_background_treatments",
    "estimate_background",
    "estimate_delta_nu",
    "estimate_empirical_background",
    "estimate_harvey_background",
    "simulate_time_series",
    "whiten_spectrum",
]

__version__ = "0.1.0"
