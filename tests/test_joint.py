"""Tests for the end-to-end joint inference layer."""

import numpy as np

from urdr import (
    CoherentSignalConfig,
    SegmentSystematicConfig,
    SimulationConfig,
    calibrate_joint_detector,
    joint_diagnostics,
    make_observing_window,
    simulate_time_series,
)


def _problem() -> dict:
    window = make_observing_window(
        duration_days=0.8,
        cadence_seconds=120.0,
        gaps_days=((0.39, 0.41),),
    )
    simulation = SimulationConfig(
        white_noise_sigma=0.2,
        granulation_amplitude=0.1,
        numax_uhz=1000.0,
        delta_nu_uhz=100.0,
        envelope_width_uhz=400.0,
        oscillation_amplitude=1.8,
    )
    return {
        "window": window,
        "simulation": simulation,
        "centre_frequencies_uhz": np.linspace(700.0, 1300.0, 7),
        "filter_width_uhz": 500.0,
        "delta_nu_grid_uhz": np.array([90.0, 100.0, 110.0]),
        "segments_days": ((0.0, 0.4), (0.4, 0.8)),
        "coherent_contaminants": {
            "line": CoherentSignalConfig(1000.0, 0.8),
        },
        "segment_systematics": {
            "variance_jump": [
                SegmentSystematicConfig(0.4, 0.8, amplitude_scale=4.0)
            ],
        },
        "realizations": 16,
        "validation_fraction": 0.25,
        "target_false_positive_rate": 0.25,
        "max_lag_seconds": 25_000.0,
        "seed": 42,
    }


def test_joint_diagnostics_returns_complete_feature_vector() -> None:
    """The shared extraction path should calculate all diagnostics once."""
    problem = _problem()
    series = simulate_time_series(
        problem["window"],
        problem["simulation"],
        np.random.default_rng(5),
        include_oscillations=True,
    )
    diagnostics, features = joint_diagnostics(
        series=series,
        simulation=problem["simulation"],
        centre_frequencies_uhz=problem["centre_frequencies_uhz"],
        filter_width_uhz=problem["filter_width_uhz"],
        delta_nu_grid_uhz=problem["delta_nu_grid_uhz"],
        segments_days=problem["segments_days"],
        max_lag_seconds=problem["max_lag_seconds"],
    )

    assert features.shape == (13,)
    assert np.all(np.isfinite(features))
    assert diagnostics.delta_nu_uhz in problem["delta_nu_grid_uhz"]
    assert diagnostics.eacf_statistic >= 0


def test_joint_detector_is_reproducible_and_returns_uncertainty() -> None:
    """Fixed paired simulations should reproduce the fitted joint detector."""
    problem = _problem()
    first = calibrate_joint_detector(**problem)
    second = calibrate_joint_detector(**problem)

    np.testing.assert_allclose(
        first.discriminant_weights, second.discriminant_weights
    )
    np.testing.assert_allclose(
        first.negative_validation_scores,
        second.negative_validation_scores,
    )
    assert first.validation == second.validation
    assert (
        first.validation.false_positive_rate
        <= problem["target_false_positive_rate"]
    )

    signal = simulate_time_series(
        problem["window"],
        problem["simulation"],
        np.random.default_rng(200),
        include_oscillations=True,
    )
    result = first.detect(signal)

    assert 0 <= result.detection_probability <= 1
    assert 0 < result.false_alarm_probability <= 1
    assert 0 <= result.false_alarm_interval[0] <= result.false_alarm_interval[1] <= 1
    assert result.delta_nu_uhz in problem["delta_nu_grid_uhz"]
    assert set(result.diagnostic_flags) <= {
        "coherent_spectrum",
        "segment_instability",
        "atypical_morphology",
    }


def test_joint_calibration_rejects_invalid_split() -> None:
    """A split leaving too little independent validation must fail."""
    problem = _problem()
    problem["validation_fraction"] = 0.5

    try:
        calibrate_joint_detector(**problem)
    except ValueError as error:
        assert "validation_fraction" in str(error)
    else:
        raise AssertionError("an invalid validation fraction should fail")


def test_joint_calibration_requires_false_alarm_resolution() -> None:
    """The held-out sample must resolve the requested false-positive rate."""
    problem = _problem()
    problem["target_false_positive_rate"] = 0.01

    try:
        calibrate_joint_detector(**problem)
    except ValueError as error:
        assert "too small to resolve" in str(error)
    else:
        raise AssertionError("an unresolved false-positive rate should fail")
