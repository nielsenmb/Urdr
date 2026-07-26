"""Tests for EACF ridge morphology and its calibrated benchmark."""

import numpy as np

from urdr import (
    CoherentSignalConfig,
    EACFMap,
    SegmentSystematicConfig,
    SimulationConfig,
    benchmark_morphology_veto,
    eacf_morphology,
    make_observing_window,
)


def test_eacf_morphology_recovers_localised_synthetic_ridge() -> None:
    """A constructed ridge should recover its centre and harmonic support."""
    centres = np.linspace(700.0, 1300.0, 13)
    lags = np.linspace(0.0, 25_000.0, 251)
    centre_profile = np.exp(-0.5 * ((centres - 1000.0) / 120.0) ** 2)
    primary = np.exp(-0.5 * ((lags - 10_000.0) / 300.0) ** 2)
    secondary = 0.4 * np.exp(-0.5 * ((lags - 20_000.0) / 300.0) ** 2)
    values = centre_profile[:, None] * (primary + secondary) + 1e-4
    result = EACFMap(centres, lags, values)

    diagnostics = eacf_morphology(
        result,
        delta_nu_uhz=100.0,
        expected_numax_uhz=1000.0,
        envelope_width_uhz=400.0,
    )

    assert diagnostics.peak_centre_uhz == 1000.0
    assert diagnostics.centre_offset == 0.0
    assert 0 < diagnostics.ridge_width <= 1
    assert diagnostics.ridge_contrast > 1
    assert 0.35 < diagnostics.harmonic_support < 0.45


def test_morphology_requires_a_sampled_envelope() -> None:
    """Diagnostics should reject a prior envelope outside the trial centres."""
    result = EACFMap(
        np.array([800.0, 900.0]),
        np.array([0.0, 10_000.0, 20_000.0]),
        np.ones((2, 3)),
    )

    try:
        eacf_morphology(result, 100.0, 1500.0, 100.0)
    except ValueError as error:
        assert "predicted envelope" in str(error)
    else:
        raise AssertionError("an unsampled envelope should fail")


def test_morphology_benchmark_is_reproducible() -> None:
    """Paired hard-negative benchmarks should reproduce rates and thresholds."""
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
        oscillation_amplitude=1.5,
    )
    kwargs = dict(
        window=window,
        simulation=simulation,
        centre_frequencies_uhz=np.linspace(700.0, 1300.0, 7),
        filter_width_uhz=500.0,
        delta_nu_grid_uhz=np.array([90.0, 100.0, 110.0]),
        coherent_contaminants={
            "line": CoherentSignalConfig(1000.0, 0.8),
        },
        segment_systematics={
            "variance_jump": [
                SegmentSystematicConfig(0.4, 0.8, amplitude_scale=4.0)
            ],
        },
        realizations=8,
        target_false_positive_rate=0.25,
        target_signal_retention=0.9,
        max_lag_seconds=25_000.0,
        seed=42,
    )

    first = benchmark_morphology_veto(**kwargs)
    second = benchmark_morphology_veto(**kwargs)

    assert first == second
    assert set(first.raw_contaminant_detection_rate) == {
        "line",
        "variance_jump",
    }
    assert 0 <= first.accepted_signal_detection_rate <= 1
    assert (
        first.accepted_signal_detection_rate
        <= first.raw_signal_detection_rate
    )
    for name, raw_rate in first.raw_contaminant_detection_rate.items():
        assert first.accepted_contaminant_detection_rate[name] <= raw_rate
