"""Tests for checkpointed scientific experiment execution."""

import csv
import json

import numpy as np

import urdr.experiment as experiment
from urdr import (
    CoherentSignalConfig,
    ReliabilityBin,
    SegmentSystematicConfig,
    SimulationConfig,
    SyntheticExperimentPlan,
    SyntheticValidation,
    ValidationCase,
    ValidationMetric,
    make_observing_window,
    run_synthetic_experiment,
)


def _case(name: str = "validation_cell") -> ValidationCase:
    window = make_observing_window(
        duration_days=0.2,
        cadence_seconds=120.0,
        gaps_days=((0.09, 0.10),),
    )
    return ValidationCase(
        name=name,
        split="validation",
        window=window,
        simulation=SimulationConfig(
            white_noise_sigma=0.2,
            granulation_amplitude=0.1,
            numax_uhz=1000.0,
            delta_nu_uhz=100.0,
            envelope_width_uhz=400.0,
            oscillation_amplitude=1.8,
        ),
        published_centres_uhz=np.linspace(500.0, 1500.0, 5),
        restricted_centres_uhz=np.linspace(800.0, 1200.0, 3),
        filter_width_uhz=500.0,
        delta_nu_grid_uhz=np.array([90.0, 100.0, 110.0]),
        segments_days=((0.0, 0.1), (0.1, 0.2)),
        coherent_contaminants={
            "line": CoherentSignalConfig(1000.0, 0.8),
        },
        segment_systematics={
            "variance_jump": [
                SegmentSystematicConfig(0.1, 0.2, amplitude_scale=4.0)
            ],
        },
        max_lag_seconds=10_000.0,
    )


def _result(case_name: str) -> SyntheticValidation:
    return SyntheticValidation(
        metrics=(
            ValidationMetric(
                case=case_name,
                split="validation",
                method="joint_urdr",
                simulation_class="signal",
                sample_count=4,
                detection_rate=0.75,
                delta_nu_recovery_rate=0.5,
                median_delta_nu_error=0.02,
                brier_score=0.1,
            ),
        ),
        reliability=(
            ReliabilityBin(
                method="joint_urdr",
                split="validation",
                lower=0.5,
                upper=1.0,
                mean_probability=0.8,
                observed_frequency=0.75,
                sample_count=4,
            ),
        ),
        target_false_positive_rate=0.1,
        delta_nu_tolerance=0.05,
        calibration_realizations=16,
        evaluation_realizations=4,
    )


def test_experiment_manifest_fingerprints_exact_window() -> None:
    """The frozen design should change when the observing mask changes."""
    case = _case()
    changed_mask = case.window.observed.copy()
    changed_mask[10] = False
    changed = ValidationCase(
        **{
            **vars(case),
            "window": type(case.window)(case.window.time, changed_mask),
        }
    )
    first = SyntheticExperimentPlan("science", (case,), seed=42)
    second = SyntheticExperimentPlan("science", (changed,), seed=42)

    assert first.fingerprint != second.fingerprint
    assert first.to_manifest()["cases"][0]["window"]["duty_cycle"] > 0
    assert len(first.fingerprint) == 64


def test_experiment_writes_tables_and_resumes(monkeypatch, tmp_path) -> None:
    """A completed case should be loaded rather than simulated a second time."""
    case = _case()
    plan = SyntheticExperimentPlan(
        "science",
        (case,),
        calibration_realizations=16,
        evaluation_realizations=4,
        target_false_positive_rate=0.1,
        seed=42,
    )
    calls = []

    def fake_run(current_plan, current_case):
        calls.append((current_plan.name, current_case.name))
        return _result(current_case.name)

    monkeypatch.setattr(experiment, "_run_case", fake_run)
    first = run_synthetic_experiment(plan, tmp_path, workers=1)
    second = run_synthetic_experiment(plan, tmp_path, workers=1)

    assert calls == [("science", case.name)]
    assert first.completed_cases == second.completed_cases == 1
    assert first.reused_cases == 0
    assert second.reused_cases == 1
    assert json.loads(first.manifest_path.read_text())["fingerprint"] == (
        plan.fingerprint
    )
    with first.metrics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["case"] == case.name
    assert rows[0]["detection_rate"] == "0.75"


def test_experiment_refuses_changed_plan(monkeypatch, tmp_path) -> None:
    """An output directory cannot silently mix different pre-registrations."""
    case = _case()
    first = SyntheticExperimentPlan(
        "science",
        (case,),
        calibration_realizations=16,
        evaluation_realizations=4,
        target_false_positive_rate=0.1,
        seed=1,
    )
    second = SyntheticExperimentPlan(
        "science",
        (case,),
        calibration_realizations=16,
        evaluation_realizations=4,
        target_false_positive_rate=0.1,
        seed=2,
    )
    monkeypatch.setattr(
        experiment,
        "_run_case",
        lambda current_plan, current_case: _result(current_case.name),
    )
    run_synthetic_experiment(first, tmp_path)

    try:
        run_synthetic_experiment(second, tmp_path)
    except ValueError as error:
        assert "different experiment manifest" in str(error)
    else:
        raise AssertionError("a changed pre-registration should be rejected")


def test_case_seed_is_order_and_worker_independent() -> None:
    """Case seeds should depend only on the root seed and stable case name."""
    assert experiment._case_seed(42, "a") == experiment._case_seed(42, "a")
    assert experiment._case_seed(42, "a") != experiment._case_seed(42, "b")
    assert experiment._case_seed(42, "a") != experiment._case_seed(43, "a")


def test_reliability_bins_are_pooled_across_cases() -> None:
    """Saved reliability rows should pool matching bins using sample counts."""
    first = _result("first")
    second = SyntheticValidation(
        metrics=_result("second").metrics,
        reliability=(
            ReliabilityBin(
                method="joint_urdr",
                split="validation",
                lower=0.5,
                upper=1.0,
                mean_probability=0.6,
                observed_frequency=0.5,
                sample_count=2,
            ),
        ),
        target_false_positive_rate=0.1,
        delta_nu_tolerance=0.05,
        calibration_realizations=16,
        evaluation_realizations=4,
    )

    pooled = experiment._aggregate_reliability((first, second))

    assert len(pooled) == 1
    assert pooled[0].sample_count == 6
    assert np.isclose(pooled[0].mean_probability, (0.8 * 4 + 0.6 * 2) / 6)
    assert np.isclose(pooled[0].observed_frequency, (0.75 * 4 + 0.5 * 2) / 6)
