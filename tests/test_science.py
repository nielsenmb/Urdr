"""Tests for the pre-registered scientific validation design."""

import numpy as np

import urdr.science as science
from urdr import (
    BenchmarkMetrics,
    EmpiricalGridBenchmark,
    EmpiricalGridMetric,
    EmpiricalGridPoint,
    ReliabilityBin,
    ValidationMetric,
    assess_background_calibration,
    assess_validation,
    make_tess_scientific_grid,
)


def test_tess_grid_is_targeted_and_reproducible() -> None:
    """The frozen grid should cover all regimes without a Cartesian explosion."""
    first = make_tess_scientific_grid(
        calibration_realizations=16,
        evaluation_realizations=4,
        cadence_seconds=1800.0,
    )
    second = make_tess_scientific_grid(
        calibration_realizations=16,
        evaluation_realizations=4,
        cadence_seconds=1800.0,
    )

    assert first.validation_plan.fingerprint == second.validation_plan.fingerprint
    assert len(first.validation_plan.cases) == 9
    assert len(first.background_cases) == 9
    assert len(first.background_candidates) == 9
    assert {row.regime for row in first.metadata} == {
        "red_giant",
        "subgiant",
        "main_sequence",
    }
    assert {row.split for row in first.metadata} == {"train", "validation"}
    assert sum(row.window_class == "cvz_like" for row in first.metadata) == 1


def test_tess_v1_design_fingerprint_is_frozen() -> None:
    """Unreviewed changes to the production design should fail loudly."""
    grid = make_tess_scientific_grid(
        calibration_realizations=16,
        evaluation_realizations=4,
    )

    assert grid.validation_plan.fingerprint == (
        "27a39e86f6d850004946c447759a613b"
        "69442394b193ba9bb789f0063b12de67"
    )


def test_tess_grid_covers_second_peak_for_every_trial_separation() -> None:
    """Every registered map should cover the second peak at its smallest dnu."""
    grid = make_tess_scientific_grid(
        calibration_realizations=16,
        evaluation_realizations=4,
    )

    for case in grid.validation_plan.cases:
        required = 2e6 / float(np.min(case.delta_nu_grid_uhz))
        assert case.max_lag_seconds is not None
        assert case.max_lag_seconds >= required


def test_validation_assessment_keeps_calibration_separate() -> None:
    """Discrimination and reliability should remain separately reported."""
    metrics = (
        _validation_metric("signal", 0.8, 0.7, 0.12),
        _validation_metric("null", 0.02, np.nan, 0.04),
        _validation_metric("line", 0.10, np.nan, 0.08),
    )
    reliability = (
        ReliabilityBin(
            method="joint_urdr",
            split="validation",
            lower=0.0,
            upper=0.5,
            mean_probability=0.20,
            observed_frequency=0.10,
            sample_count=80,
        ),
        ReliabilityBin(
            method="joint_urdr",
            split="validation",
            lower=0.5,
            upper=1.0,
            mean_probability=0.80,
            observed_frequency=0.70,
            sample_count=20,
        ),
    )

    result = assess_validation(metrics, reliability)[0]

    assert result.signal_detection_rate == 0.8
    assert result.delta_nu_recovery_rate == 0.7
    assert result.worst_false_positive_rate == 0.10
    assert np.isclose(result.mean_brier_score, 0.08)
    assert np.isclose(result.expected_calibration_error, 0.10)


def test_background_assessment_requires_cross_regime_stability() -> None:
    """A global law should fail when no candidate works in every regime."""
    legacy = EmpiricalGridPoint(0.66, 0.88)
    broad = EmpiricalGridPoint(0.90, 0.96)
    rows = (
        _grid_metric("train_rg", "train", legacy, 0.9, 0.8),
        _grid_metric("train_rg", "train", broad, 0.7, 0.7),
        _grid_metric("validation_rg", "validation", legacy, 0.9, 0.8),
        _grid_metric("validation_rg", "validation", broad, 0.7, 0.7),
        _grid_metric("validation_ms", "validation", legacy, 0.4, 0.4),
        _grid_metric("validation_ms", "validation", broad, 0.9, 0.9),
    )
    benchmark = EmpiricalGridBenchmark(rows, 0.01, 64)

    assessment = assess_background_calibration(
        benchmark,
        {
            "train_rg": "red_giant",
            "validation_rg": "red_giant",
            "validation_ms": "main_sequence",
        },
    )

    assert legacy.name in assessment.training_pareto
    assert not assessment.supports_global_law
    assert assessment.regime_pareto["red_giant"] == (legacy.name,)
    assert assessment.regime_pareto["main_sequence"] == (broad.name,)


def test_scientific_grid_cli_dry_run(capsys) -> None:
    """Dry-run mode should expose the frozen design without starting work."""
    result = science.main(
        [
            "--dry-run",
            "--calibration-realizations",
            "16",
            "--evaluation-realizations",
            "4",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "fingerprint:" in output
    assert "validation_ms_high_cvz" in output


def _validation_metric(
    simulation_class: str,
    detection_rate: float,
    recovery_rate: float,
    brier_score: float,
) -> ValidationMetric:
    return ValidationMetric(
        case="validation_case",
        split="validation",
        method="joint_urdr",
        simulation_class=simulation_class,
        sample_count=100,
        detection_rate=detection_rate,
        delta_nu_recovery_rate=recovery_rate,
        median_delta_nu_error=0.02,
        brier_score=brier_score,
    )


def _grid_metric(
    case: str,
    split: str,
    candidate: EmpiricalGridPoint,
    true_positive_rate: float,
    recovery_rate: float,
) -> EmpiricalGridMetric:
    metrics = BenchmarkMetrics(
        treatment=candidate.name,
        oscillation_amplitude=0.2,
        threshold=1.0,
        false_positive_rate=0.01,
        true_positive_rate=true_positive_rate,
        delta_nu_recovery_rate=recovery_rate,
        median_delta_nu_error=0.02,
    )
    return EmpiricalGridMetric(case, split, candidate, metrics)
