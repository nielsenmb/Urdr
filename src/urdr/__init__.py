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
from .contaminants import (
    CoherenceDiagnostics,
    CoherentSignalConfig,
    CoherentVetoMetrics,
    add_coherent_signal,
    benchmark_coherent_veto,
    coherence_diagnostics,
)
from .eacf import EACFMap, compute_eacf, compute_eacf_map
from .grid import (
    BackgroundBenchmarkCase,
    EmpiricalGridBenchmark,
    EmpiricalGridMetric,
    EmpiricalGridPoint,
    EmpiricalGridSummary,
    benchmark_empirical_grid,
    make_observing_window,
)
from .models import AsteroScaleSamples, ObservingWindow, SearchRegion, TimeSeries
from .simulation import SimulationConfig, simulate_time_series

__all__ = [
    "AsteroScaleSamples",
    "BackgroundBenchmark",
    "BackgroundBenchmarkCase",
    "BackgroundConfig",
    "BenchmarkMetrics",
    "CalibrationResult",
    "CoherenceDiagnostics",
    "CoherentSignalConfig",
    "CoherentVetoMetrics",
    "EACFMap",
    "EmpiricalBackgroundConfig",
    "EmpiricalGridBenchmark",
    "EmpiricalGridMetric",
    "EmpiricalGridPoint",
    "EmpiricalGridSummary",
    "HarveyBackgroundConfig",
    "ObservingWindow",
    "SearchRegion",
    "SimulationCalibrator",
    "SimulationConfig",
    "TimeSeries",
    "add_coherent_signal",
    "benchmark_background_treatments",
    "benchmark_coherent_veto",
    "benchmark_empirical_grid",
    "coherence_diagnostics",
    "compute_eacf",
    "compute_eacf_map",
    "default_background_treatments",
    "estimate_background",
    "estimate_delta_nu",
    "estimate_empirical_background",
    "estimate_harvey_background",
    "make_observing_window",
    "simulate_time_series",
    "whiten_spectrum",
]

__version__ = "0.1.0"
