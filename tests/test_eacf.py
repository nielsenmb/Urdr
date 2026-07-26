import numpy as np

from urdr import SimulationConfig, TimeSeries, compute_eacf_map
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

