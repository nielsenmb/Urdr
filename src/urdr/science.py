"""Pre-registered TESS-scale synthetic validation design."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .background import EmpiricalBackgroundConfig
from .contaminants import CoherentSignalConfig
from .experiment import SyntheticExperimentPlan
from .experiment import run_synthetic_experiment
from .grid import (
    BackgroundBenchmarkCase,
    EmpiricalGridBenchmark,
    EmpiricalGridPoint,
    make_observing_window,
)
from .simulation import SimulationConfig
from .systematics import SegmentSystematicConfig
from .validation import ReliabilityBin, ValidationCase, ValidationMetric


@dataclass(frozen=True)
class ScientificCaseMetadata:
    """Scientific labels attached to one pre-registered validation cell.

    Parameters
    ----------
    case
        Name of the corresponding validation case.
    regime
        Broad seismic regime: ``"red_giant"``, ``"subgiant"``, or
        ``"main_sequence"``.
    signal_level
        Relative injected oscillation level, normally ``"low"`` or ``"high"``.
    window_class
        Human-readable observing-window class.
    split
        Pre-registered training or validation split.
    """

    case: str
    regime: str
    signal_level: str
    window_class: str
    split: str


@dataclass(frozen=True)
class TessScientificGrid:
    """Frozen validation and empirical-background experiment inputs.

    Parameters
    ----------
    validation_plan
        Checkpointed three-method synthetic-validation plan.
    background_cases
        Matching cases for empirical ``a,b`` calibration.
    background_candidates
        Pre-registered smoothing-law candidates.
    metadata
        Scientific labels for each case.
    """

    validation_plan: SyntheticExperimentPlan
    background_cases: tuple[BackgroundBenchmarkCase, ...]
    background_candidates: tuple[EmpiricalGridPoint, ...]
    metadata: tuple[ScientificCaseMetadata, ...]

    def regime_by_case(self) -> dict[str, str]:
        """Return the seismic regime indexed by case name.

        Returns
        -------
        dict
            Mapping from validation-case name to seismic regime.
        """
        return {row.case: row.regime for row in self.metadata}


@dataclass(frozen=True)
class MethodAssessment:
    """Held-out performance summary for one detection method.

    Parameters
    ----------
    method
        Detection-method name.
    signal_detection_rate
        Mean held-out signal detection rate.
    delta_nu_recovery_rate
        Mean held-out large-separation recovery rate.
    worst_false_positive_rate
        Largest held-out null or contaminant detection rate.
    mean_brier_score
        Mean held-out Brier score across signal and hard-negative classes.
    expected_calibration_error
        Sample-weighted absolute reliability error.
    """

    method: str
    signal_detection_rate: float
    delta_nu_recovery_rate: float
    worst_false_positive_rate: float
    mean_brier_score: float
    expected_calibration_error: float


@dataclass(frozen=True)
class BackgroundCalibrationAssessment:
    """Training-to-validation stability of the empirical smoothing law.

    Parameters
    ----------
    training_pareto
        Candidate names on the training Pareto frontier.
    validation_pareto
        Candidate names on the held-out Pareto frontier.
    stable_candidates
        Candidates appearing on both frontiers.
    regime_pareto
        Pareto candidates within each seismic regime.
    supports_global_law
        Whether at least one candidate survives both global frontiers and all
        represented regime frontiers.
    """

    training_pareto: tuple[str, ...]
    validation_pareto: tuple[str, ...]
    stable_candidates: tuple[str, ...]
    regime_pareto: Mapping[str, tuple[str, ...]]
    supports_global_law: bool


@dataclass(frozen=True)
class _CaseDesign:
    name: str
    split: str
    regime: str
    signal_level: str
    window_class: str
    numax_uhz: float
    delta_nu_uhz: float
    envelope_width_uhz: float
    amplitude: float
    duration_days: float
    random_missing_fraction: float
    sector_gaps: bool
    seed: int


def make_tess_scientific_grid(
    *,
    calibration_realizations: int = 4096,
    evaluation_realizations: int = 2048,
    cadence_seconds: float = 120.0,
    seed: int = 20260726,
) -> TessScientificGrid:
    """Build the version-one targeted TESS synthetic experiment.

    The design is intentionally not a full Cartesian product. It covers the
    principal seismic regimes and observing-window classes while reserving
    difficult combinations for a genuinely held-out validation split.

    Parameters
    ----------
    calibration_realizations
        Target-specific calibration simulations per validation cell.
    evaluation_realizations
        Independent evaluation simulations per class and cell.
    cadence_seconds
        Uniform cadence used before applying exact observing masks.
    seed
        Root experiment seed.

    Returns
    -------
    TessScientificGrid
        Frozen validation plan, matching background cases, candidates, and
        scientific case labels.
    """
    if cadence_seconds <= 0:
        raise ValueError("cadence_seconds must be positive")
    designs = _tess_v1_designs()
    validation_cases = tuple(
        _make_validation_case(design, cadence_seconds) for design in designs
    )
    metadata = tuple(
        ScientificCaseMetadata(
            case=design.name,
            regime=design.regime,
            signal_level=design.signal_level,
            window_class=design.window_class,
            split=design.split,
        )
        for design in designs
    )
    background_cases = tuple(
        _make_background_case(case) for case in validation_cases
    )
    candidates = tuple(
        EmpiricalGridPoint(a, b)
        for a, b in (
            (0.40, 0.80),
            (0.40, 0.88),
            (0.40, 0.96),
            (0.66, 0.80),
            (0.66, 0.88),
            (0.66, 0.96),
            (0.90, 0.80),
            (0.90, 0.88),
            (0.90, 0.96),
        )
    )
    plan = SyntheticExperimentPlan(
        name="tess-synthetic-v1",
        cases=validation_cases,
        calibration_realizations=calibration_realizations,
        evaluation_realizations=evaluation_realizations,
        target_false_positive_rate=0.01,
        delta_nu_tolerance=0.05,
        reliability_bins=10,
        seed=seed,
    )
    return TessScientificGrid(plan, background_cases, candidates, metadata)


def assess_validation(
    metrics: Sequence[ValidationMetric],
    reliability: Sequence[ReliabilityBin],
    *,
    split: str = "validation",
) -> tuple[MethodAssessment, ...]:
    """Summarise discrimination and probability calibration separately.

    Parameters
    ----------
    metrics
        Per-case validation metrics.
    reliability
        Reliability bins, preferably pooled over the same cases.
    split
        Pre-registered split to assess.

    Returns
    -------
    tuple
        One held-out assessment per detection method.
    """
    selected = [row for row in metrics if row.split == split]
    if not selected:
        raise ValueError(f"no metrics found for split {split!r}")
    methods = sorted({row.method for row in selected})
    output = []
    for method in methods:
        rows = [row for row in selected if row.method == method]
        signal = [row for row in rows if row.simulation_class == "signal"]
        negatives = [row for row in rows if row.simulation_class != "signal"]
        bins = [
            row
            for row in reliability
            if row.method == method and row.split == split and row.sample_count > 0
        ]
        output.append(
            MethodAssessment(
                method=method,
                signal_detection_rate=_mean(signal, "detection_rate"),
                delta_nu_recovery_rate=_mean(
                    signal, "delta_nu_recovery_rate"
                ),
                worst_false_positive_rate=max(
                    (row.detection_rate for row in negatives),
                    default=float("nan"),
                ),
                mean_brier_score=_mean(rows, "brier_score"),
                expected_calibration_error=_calibration_error(bins),
            )
        )
    return tuple(output)


def assess_background_calibration(
    benchmark: EmpiricalGridBenchmark,
    regime_by_case: Mapping[str, str],
) -> BackgroundCalibrationAssessment:
    """Test whether a global empirical smoothing law survives held-out cells.

    A global law is supported only when at least one candidate is on both the
    training and validation Pareto frontiers and also remains non-dominated in
    every represented seismic regime. This is a conservative diagnostic, not a
    significance test.

    Parameters
    ----------
    benchmark
        Completed empirical-background grid benchmark.
    regime_by_case
        Mapping from benchmark case names to seismic regimes.

    Returns
    -------
    BackgroundCalibrationAssessment
        Global and regime-resolved Pareto stability.
    """
    missing = {row.case for row in benchmark.metrics} - set(regime_by_case)
    if missing:
        raise ValueError(f"missing regimes for cases: {sorted(missing)}")
    training = tuple(row.candidate.name for row in benchmark.pareto_front("train"))
    validation = tuple(
        row.candidate.name for row in benchmark.pareto_front("validation")
    )
    stable = tuple(sorted(set(training) & set(validation)))
    regimes = sorted({regime_by_case[row.case] for row in benchmark.metrics})
    regime_pareto = {
        regime: _pareto_names(
            [
                row
                for row in benchmark.metrics
                if regime_by_case[row.case] == regime
            ]
        )
        for regime in regimes
    }
    shared = set(stable)
    for names in regime_pareto.values():
        shared &= set(names)
    return BackgroundCalibrationAssessment(
        training_pareto=training,
        validation_pareto=validation,
        stable_candidates=stable,
        regime_pareto=regime_pareto,
        supports_global_law=bool(shared),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run or inspect the frozen TESS version-one experiment.

    Parameters
    ----------
    argv
        Optional command-line arguments. The process arguments are used when
        omitted.

    Returns
    -------
    int
        Zero after a successful inspection or run.
    """
    parser = argparse.ArgumentParser(
        description="Run the frozen Urdr TESS synthetic validation grid."
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("results/tess-synthetic-v1"),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--calibration-realizations", type=int, default=4096)
    parser.add_argument("--evaluation-realizations", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fingerprint and case names without running simulations.",
    )
    arguments = parser.parse_args(argv)
    grid = make_tess_scientific_grid(
        calibration_realizations=arguments.calibration_realizations,
        evaluation_realizations=arguments.evaluation_realizations,
        seed=arguments.seed,
    )
    print(f"fingerprint: {grid.validation_plan.fingerprint}")
    for row in grid.metadata:
        print(
            f"{row.split:10s} {row.regime:13s} "
            f"{row.signal_level:4s} {row.window_class:18s} {row.case}"
        )
    if arguments.dry_run:
        return 0
    run = run_synthetic_experiment(
        grid.validation_plan,
        arguments.output,
        workers=arguments.workers,
        resume=True,
    )
    print(f"completed cases: {run.completed_cases}")
    print(f"reused cases: {run.reused_cases}")
    print(f"metrics: {run.metrics_path}")
    print(f"reliability: {run.reliability_path}")
    return 0


def _tess_v1_designs() -> tuple[_CaseDesign, ...]:
    return (
        _CaseDesign(
            "train_rg_low_sector",
            "train",
            "red_giant",
            "low",
            "one_sector",
            100.0,
            10.0,
            40.0,
            0.16,
            27.4,
            0.02,
            False,
            11,
        ),
        _CaseDesign(
            "train_rg_high_multisector",
            "train",
            "red_giant",
            "high",
            "three_sector",
            180.0,
            15.0,
            65.0,
            0.45,
            82.2,
            0.03,
            True,
            12,
        ),
        _CaseDesign(
            "train_sg_low_multisector",
            "train",
            "subgiant",
            "low",
            "three_sector",
            700.0,
            40.0,
            250.0,
            0.16,
            82.2,
            0.03,
            True,
            13,
        ),
        _CaseDesign(
            "train_sg_high_sector",
            "train",
            "subgiant",
            "high",
            "one_sector",
            1000.0,
            55.0,
            350.0,
            0.45,
            27.4,
            0.02,
            False,
            14,
        ),
        _CaseDesign(
            "train_ms_low_sector",
            "train",
            "main_sequence",
            "low",
            "one_sector",
            2200.0,
            100.0,
            700.0,
            0.16,
            27.4,
            0.02,
            False,
            15,
        ),
        _CaseDesign(
            "validation_rg_low_sparse",
            "validation",
            "red_giant",
            "low",
            "sparse_sector",
            140.0,
            12.5,
            55.0,
            0.13,
            27.4,
            0.15,
            False,
            21,
        ),
        _CaseDesign(
            "validation_sg_high_sparse",
            "validation",
            "subgiant",
            "high",
            "sparse_multisector",
            850.0,
            47.0,
            300.0,
            0.40,
            82.2,
            0.15,
            True,
            22,
        ),
        _CaseDesign(
            "validation_ms_low_multisector",
            "validation",
            "main_sequence",
            "low",
            "three_sector",
            2600.0,
            115.0,
            800.0,
            0.13,
            82.2,
            0.04,
            True,
            23,
        ),
        _CaseDesign(
            "validation_ms_high_cvz",
            "validation",
            "main_sequence",
            "high",
            "cvz_like",
            3000.0,
            135.0,
            900.0,
            0.40,
            351.0,
            0.05,
            True,
            24,
        ),
    )


def _make_validation_case(
    design: _CaseDesign,
    cadence_seconds: float,
) -> ValidationCase:
    gaps = _sector_gaps(design.duration_days) if design.sector_gaps else ()
    window = make_observing_window(
        duration_days=design.duration_days,
        cadence_seconds=cadence_seconds,
        gaps_days=gaps,
        random_missing_fraction=design.random_missing_fraction,
        seed=design.seed,
    )
    simulation = SimulationConfig(
        white_noise_sigma=0.2,
        granulation_amplitude=0.12,
        granulation_timescale_days=max(0.03, 2.0 / design.numax_uhz),
        numax_uhz=design.numax_uhz,
        delta_nu_uhz=design.delta_nu_uhz,
        envelope_width_uhz=design.envelope_width_uhz,
        oscillation_amplitude=design.amplitude,
        mode_linewidth_uhz=max(0.15, 0.002 * design.numax_uhz),
    )
    half_width = 0.65 * design.envelope_width_uhz
    restricted = np.linspace(
        max(0.05 * design.numax_uhz, design.numax_uhz - half_width),
        design.numax_uhz + half_width,
        9,
    )
    published = np.linspace(
        max(1.0, 0.35 * design.numax_uhz),
        min(0.98 * 1e6 / (2.0 * cadence_seconds), 1.65 * design.numax_uhz),
        25,
    )
    segments = _segments(design.duration_days)
    contaminants = {
        "single_line": CoherentSignalConfig(design.numax_uhz, 0.5),
        "harmonic_comb": CoherentSignalConfig(
            design.numax_uhz / 3.0, 0.4, harmonics=3
        ),
    }
    systematics = {
        "variance_jump_moderate": (
            SegmentSystematicConfig(
                segments[-1][0], segments[-1][1], amplitude_scale=2.0
            ),
        ),
        "variance_jump_strong": (
            SegmentSystematicConfig(
                segments[-1][0], segments[-1][1], amplitude_scale=4.0
            ),
        ),
    }
    delta_nu_grid = np.linspace(
        0.8 * design.delta_nu_uhz,
        1.2 * design.delta_nu_uhz,
        41,
    )
    return ValidationCase(
        name=design.name,
        split=design.split,
        window=window,
        simulation=simulation,
        published_centres_uhz=published,
        restricted_centres_uhz=restricted,
        filter_width_uhz=design.envelope_width_uhz,
        delta_nu_grid_uhz=delta_nu_grid,
        segments_days=segments,
        coherent_contaminants=contaminants,
        segment_systematics=systematics,
        background=EmpiricalBackgroundConfig.excluding_envelope(
            design.numax_uhz, design.envelope_width_uhz
        ),
        max_lag_seconds=2.2e6 / float(np.min(delta_nu_grid)),
    )


def _make_background_case(case: ValidationCase) -> BackgroundBenchmarkCase:
    amplitude = case.simulation.oscillation_amplitude
    return BackgroundBenchmarkCase(
        name=case.name,
        split=case.split,
        window=case.window,
        simulation=case.simulation,
        centre_frequencies_uhz=case.restricted_centres_uhz,
        filter_width_uhz=case.filter_width_uhz,
        delta_nu_grid_uhz=case.delta_nu_grid_uhz,
        oscillation_amplitudes=np.asarray(
            [0.7 * amplitude, amplitude, 1.4 * amplitude]
        ),
    )


def _sector_gaps(duration_days: float) -> tuple[tuple[float, float], ...]:
    gaps = []
    boundary = 27.4
    while boundary + 0.35 < duration_days:
        gaps.append((boundary - 0.35, boundary + 0.35))
        boundary += 27.4
    return tuple(gaps)


def _segments(duration_days: float) -> tuple[tuple[float, float], ...]:
    count = max(2, int(np.ceil(duration_days / 27.4)))
    edges = np.linspace(0.0, duration_days, count + 1)
    return tuple(
        (float(start), float(stop))
        for start, stop in zip(edges[:-1], edges[1:])
    )


def _mean(rows: Sequence[object], field: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([getattr(row, field) for row in rows]))


def _calibration_error(rows: Sequence[ReliabilityBin]) -> float:
    if not rows:
        return float("nan")
    counts = np.asarray([row.sample_count for row in rows], dtype=float)
    errors = np.asarray(
        [abs(row.mean_probability - row.observed_frequency) for row in rows]
    )
    return float(np.average(errors, weights=counts))


def _pareto_names(rows: Sequence[object]) -> tuple[str, ...]:
    candidates = sorted(
        {row.candidate for row in rows}, key=lambda item: (item.a, item.b)
    )
    scores = {}
    for candidate in candidates:
        selected = [row.metrics for row in rows if row.candidate == candidate]
        scores[candidate] = (
            float(np.mean([row.true_positive_rate for row in selected])),
            float(np.mean([row.delta_nu_recovery_rate for row in selected])),
        )
    frontier = []
    for candidate, score in scores.items():
        dominated = any(
            other != candidate
            and other_score[0] >= score[0]
            and other_score[1] >= score[1]
            and other_score != score
            for other, other_score in scores.items()
        )
        if not dominated:
            frontier.append(candidate.name)
    return tuple(frontier)
