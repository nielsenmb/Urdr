"""Tests for broad synthetic validation."""

import numpy as np

from urdr import (
    CoherentSignalConfig,
    SegmentSystematicConfig,
    SimulationConfig,
    ValidationCase,
    benchmark_synthetic_validation,
    make_observing_window,
)


def _case() -> ValidationCase:
    window = make_observing_window(
        duration_days=0.8,
        cadence_seconds=120.0,
        gaps_days=((0.39, 0.41),),
    )
    return ValidationCase(
        name="held_out_window",
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
        published_centres_uhz=np.linspace(500.0, 1500.0, 11),
        restricted_centres_uhz=np.linspace(700.0, 1300.0, 7),
        filter_width_uhz=500.0,
        delta_nu_grid_uhz=np.array([90.0, 100.0, 110.0]),
        segments_days=((0.0, 0.4), (0.4, 0.8)),
        coherent_contaminants={
            "line": CoherentSignalConfig(1000.0, 0.8),
        },
        segment_systematics={
            "variance_jump": [
                SegmentSystematicConfig(0.4, 0.8, amplitude_scale=4.0)
            ],
        },
        max_lag_seconds=25_000.0,
    )


def test_validation_compares_all_methods_and_classes() -> None:
    """The matrix should preserve method, class, and reliability dimensions."""
    result = benchmark_synthetic_validation(
        [_case()],
        calibration_realizations=16,
        evaluation_realizations=4,
        validation_fraction=0.25,
        target_false_positive_rate=0.25,
        reliability_bins=4,
        seed=42,
    )

    methods = {row.method for row in result.metrics}
    classes = {row.simulation_class for row in result.metrics}
    assert methods == {"published_eacf", "asteroscale_eacf", "joint_urdr"}
    assert classes == {"signal", "null", "line", "variance_jump"}
    assert len(result.metrics) == 12
    assert result.reliability
    assert all(row.sample_count == 4 for row in result.metrics)
    assert all(
        np.isfinite(row.delta_nu_recovery_rate)
        for row in result.metrics
        if row.simulation_class == "signal"
    )
    assert all(
        np.isnan(row.delta_nu_recovery_rate)
        for row in result.metrics
        if row.simulation_class != "signal"
    )


def test_validation_rejects_duplicate_case_names() -> None:
    """Case names must remain unique in machine-readable result tables."""
    case = _case()
    try:
        benchmark_synthetic_validation(
            [case, case],
            calibration_realizations=16,
            evaluation_realizations=4,
            target_false_positive_rate=0.25,
        )
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate validation case names should fail")
