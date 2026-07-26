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
from .experiment import (
    ExperimentRun,
    SyntheticExperimentPlan,
    run_synthetic_experiment,
)
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
from .joint import (
    DetectionResult,
    JointDetector,
    JointDiagnostics,
    JointValidationMetrics,
    calibrate_joint_detector,
    joint_diagnostics,
)
from .morphology import (
    MorphologyDiagnostics,
    MorphologyVetoMetrics,
    benchmark_morphology_veto,
    eacf_morphology,
)
from .simulation import SimulationConfig, simulate_time_series
from .systematics import (
    SegmentDiagnostics,
    SegmentSystematicConfig,
    SegmentVetoMetrics,
    add_segment_systematics,
    benchmark_segment_veto,
    segment_diagnostics,
)
from .validation import (
    ReliabilityBin,
    SyntheticValidation,
    ValidationCase,
    ValidationMetric,
    benchmark_synthetic_validation,
)

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
    "DetectionResult",
    "EACFMap",
    "EmpiricalBackgroundConfig",
    "EmpiricalGridBenchmark",
    "EmpiricalGridMetric",
    "EmpiricalGridPoint",
    "EmpiricalGridSummary",
    "ExperimentRun",
    "HarveyBackgroundConfig",
    "JointDetector",
    "JointDiagnostics",
    "JointValidationMetrics",
    "MorphologyDiagnostics",
    "MorphologyVetoMetrics",
    "ObservingWindow",
    "ReliabilityBin",
    "SearchRegion",
    "SegmentDiagnostics",
    "SegmentSystematicConfig",
    "SegmentVetoMetrics",
    "SimulationCalibrator",
    "SimulationConfig",
    "SyntheticValidation",
    "SyntheticExperimentPlan",
    "TimeSeries",
    "ValidationCase",
    "ValidationMetric",
    "add_coherent_signal",
    "add_segment_systematics",
    "benchmark_background_treatments",
    "benchmark_coherent_veto",
    "benchmark_empirical_grid",
    "benchmark_morphology_veto",
    "benchmark_segment_veto",
    "benchmark_synthetic_validation",
    "calibrate_joint_detector",
    "coherence_diagnostics",
    "compute_eacf",
    "compute_eacf_map",
    "default_background_treatments",
    "estimate_background",
    "estimate_delta_nu",
    "estimate_empirical_background",
    "estimate_harvey_background",
    "eacf_morphology",
    "joint_diagnostics",
    "make_observing_window",
    "run_synthetic_experiment",
    "segment_diagnostics",
    "simulate_time_series",
    "whiten_spectrum",
]

__version__ = "0.1.0"
