"""Window-aware time-series EACF detection."""

from .calibration import CalibrationResult, SimulationCalibrator
from .eacf import EACFMap, compute_eacf, compute_eacf_map
from .models import AsteroScaleSamples, ObservingWindow, SearchRegion, TimeSeries
from .simulation import SimulationConfig, simulate_time_series

__all__ = [
    "AsteroScaleSamples",
    "CalibrationResult",
    "EACFMap",
    "ObservingWindow",
    "SearchRegion",
    "SimulationCalibrator",
    "SimulationConfig",
    "TimeSeries",
    "compute_eacf",
    "compute_eacf_map",
    "simulate_time_series",
]

__version__ = "0.1.0"

