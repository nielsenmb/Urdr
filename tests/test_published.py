"""Tests for the published Lomb--Scargle EACF workflow."""

import numpy as np
import pytest

from urdr import (
    DeltaNuScaling,
    SimulationConfig,
    calibrate_published_eacf,
    compute_published_eacf_map,
    envelope_width_uhz,
    make_observing_window,
    simulate_time_series,
)


def _series(*, oscillation_amplitude=0.5, missing_fraction=0.03, seed=12):
    window = make_observing_window(
        duration_days=6.0,
        cadence_seconds=600.0,
        random_missing_fraction=missing_fraction,
        seed=seed,
    )
    config = SimulationConfig(
        white_noise_sigma=0.2,
        granulation_amplitude=0.05,
        granulation_timescale_days=0.1,
        numax_uhz=500.0,
        delta_nu_uhz=32.0,
        envelope_width_uhz=160.0,
        oscillation_amplitude=oscillation_amplitude,
        mode_linewidth_uhz=2.0,
    )
    return simulate_time_series(window, config, np.random.default_rng(seed)), config


def test_delta_nu_scaling_uses_solar_reference():
    """The physical relation passes through its solar reference point."""
    scaling = DeltaNuScaling()
    assert scaling.predict([scaling.solar_numax_uhz])[0] == pytest.approx(
        scaling.solar_delta_nu_uhz
    )


def test_envelope_width_matches_published_relation():
    """Default Hanning widths follow Eq. 1 for cool stars."""
    assert envelope_width_uhz([1000.0])[0] == pytest.approx(0.66 * 1000**0.88)


def test_published_map_is_normalized_and_physically_masked():
    """Complex-modulus rows start at unity and contain the banana mask."""
    series, _ = _series()
    centres = np.array([450.0, 500.0, 550.0])
    result = compute_published_eacf_map(series, centres)

    assert result.values.shape == result.physical_mask.shape
    assert result.values.shape[0] == centres.size
    assert np.allclose(result.values[:, 0], 1.0)
    assert np.all(result.physical_mask.sum(axis=1) > 0)
    assert np.all(np.isfinite(result.collapsed()))


def test_published_map_recovers_injected_seismic_scale():
    """The strongest allowed lag is close to the injected large separation."""
    series, _ = _series(oscillation_amplitude=1.0, missing_fraction=0.0)
    result = compute_published_eacf_map(
        series,
        np.linspace(450.0, 550.0, 5),
        filter_widths_uhz=200.0,
    )
    assert result.best_delta_nu_uhz == pytest.approx(32.0, rel=0.2)


def test_exact_window_calibration_is_reproducible():
    """Target-specific null maxima are deterministic for a fixed seed."""
    series, config = _series(oscillation_amplitude=0.4)
    kwargs = dict(
        simulations=8,
        seed=34,
        filter_widths_uhz=180.0,
        max_lag_seconds=60_000.0,
    )
    first = calibrate_published_eacf(series, config, [475.0, 500.0], **kwargs)
    second = calibrate_published_eacf(series, config, [475.0, 500.0], **kwargs)

    assert np.array_equal(first.null_statistics, second.null_statistics)
    assert 0.0 <= first.detection_merit < 1.0
    assert first.false_alarm_probability >= 1.0 / 9.0


def test_filter_centres_must_be_positive():
    """Invalid trial frequencies fail before spectrum calculation."""
    series, _ = _series()
    with pytest.raises(ValueError, match="filter centres"):
        compute_published_eacf_map(series, [0.0])
