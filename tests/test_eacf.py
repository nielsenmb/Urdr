import numpy as np

import urdr.eacf as eacf
from urdr import (
    EmpiricalBackgroundConfig,
    SimulationConfig,
    TimeSeries,
    compute_eacf_map,
)


def test_fft_autocorrelation_matches_direct_linear_result() -> None:
    """FFT correlation should preserve the original linear statistic."""
    rng = np.random.default_rng(123)
    values = rng.normal(size=257)

    expected = np.correlate(values, values, mode="full")[values.size - 1 :]
    actual = eacf._nonnegative_autocorrelation(values)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_fft_autocorrelation_rejects_invalid_input() -> None:
    """The internal FFT helper should fail clearly for malformed arrays."""
    for values in (np.array([]), np.ones((2, 2))):
        try:
            eacf._nonnegative_autocorrelation(values)
        except ValueError as error:
            assert "non-empty 1D" in str(error)
        else:
            raise AssertionError("invalid autocorrelation input should fail")
from urdr.simulation import simulate_time_series


def _template(size: int = 4096) -> TimeSeries:
    time = np.arange(size) * (120.0 / 86400.0)
    observed = np.ones(size, dtype=bool)
    observed[900:1020] = False
    return TimeSeries(time, np.zeros(size), observed)


def test_injected_comb_strengthens_expected_lag() -> None:
    template = _template()
    config = SimulationConfig(
        white_noise_sigma=0.25,
        numax_uhz=1000.0,
        delta_nu_uhz=100.0,
        envelope_width_uhz=350.0,
        oscillation_amplitude=2.0,
        mode_linewidth_uhz=1.5,
    )
    signal = simulate_time_series(
        template.window, config, np.random.default_rng(12)
    )
    noise = simulate_time_series(
        template.window, config, np.random.default_rng(12), False
    )
    centres = np.linspace(850.0, 1150.0, 5)
    signal_map = compute_eacf_map(signal, centres, 500.0, max_lag_seconds=20000)
    noise_map = compute_eacf_map(noise, centres, 500.0, max_lag_seconds=20000)
    assert signal_map.statistic(100.0, 0.1) > noise_map.statistic(100.0, 0.1)


def test_eacf_map_shape() -> None:
    template = _template(512)
    rng = np.random.default_rng(2)
    series = TimeSeries(
        template.time,
        rng.normal(size=template.time.size),
        template.observed,
    )
    result = compute_eacf_map(series, [800.0, 1000.0], 400.0, 5000.0)
    assert result.values.shape == (2, result.lags_seconds.size)


def test_eacf_accepts_target_aware_empirical_background() -> None:
    template = _template(1024)
    config = SimulationConfig(
        white_noise_sigma=0.2,
        granulation_amplitude=1.0,
        granulation_timescale_days=0.2,
        numax_uhz=1000.0,
        delta_nu_uhz=100.0,
        envelope_width_uhz=350.0,
        oscillation_amplitude=1.0,
    )
    series = simulate_time_series(
        template.window, config, np.random.default_rng(6)
    )
    background = EmpiricalBackgroundConfig.excluding_envelope(1000.0, 350.0)
    result = compute_eacf_map(
        series,
        [900.0, 1000.0, 1100.0],
        500.0,
        max_lag_seconds=20_000.0,
        empirical_background=background,
    )
    assert np.all(np.isfinite(result.values))
