"""Window-aware time-series EACF detection."""

from .calibration import CalibrationResult, SimulationCalibrator
from .background import (
    EmpiricalBackgroundConfig,
    estimate_empirical_background,
    whiten_spectrum,
)
from .eacf import EACFMap, compute_eacf, compute_eacf_map
from .models import AsteroScaleSamples, ObservingWindow, SearchRegion, TimeSeries
from .simulation import SimulationConfig, simulate_time_series

__all__ = [
    "AsteroScaleSamples",
    "CalibrationResult",
    "EACFMap",
    "EmpiricalBackgroundConfig",
    "ObservingWindow",
    "SearchRegion",
    "SimulationCalibrator",
    "SimulationConfig",
    "TimeSeries",
    "compute_eacf",
    "compute_eacf_map",
    "estimate_empirical_background",
    "simulate_time_series",
    "whiten_spectrum",
]

__version__ = "0.1.0"
