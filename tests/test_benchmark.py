import numpy as np

from urdr import (
    BackgroundBenchmark,
    EmpiricalBackgroundConfig,
    SimulationConfig,
    TimeSeries,
    benchmark_background_treatments,
)
from urdr.benchmark import estimate_delta_nu
from urdr.eacf import EACFMap


def test_estimate_delta_nu_selects_strongest_trial() -> None:
    lags = 1e6 / np.array([120.0, 100.0, 80.0])
    values = np.array([[0.1, 0.8, 0.2]])
    result = EACFMap(np.array([1000.0]), lags, values)

    estimate, score = estimate_delta_nu(
        result, [80.0, 100.0, 120.0], relative_half_width=0.01
    )

    assert estimate == 100.0
    assert score == 0.8


def test_benchmark_is_reproducible_and_reports_separate_metrics() -> None:
    size = 256
    time = np.arange(size) * (120.0 / 86400.0)
    observed = np.ones(size, dtype=bool)
    observed[80:100] = False
    window = TimeSeries(time, np.zeros(size), observed).window
    config = SimulationConfig(
        white_noise_sigma=0.3,
        numax_uhz=1000.0,
        delta_nu_uhz=100.0,
        envelope_width_uhz=300.0,
        oscillation_amplitude=1.0,
    )
    kwargs = dict(
        window=window,
        simulation=config,
        centre_frequencies_uhz=[900.0, 1000.0, 1100.0],
        filter_width_uhz=500.0,
        delta_nu_grid_uhz=[90.0, 100.0, 110.0],
        oscillation_amplitudes=[0.3, 1.0],
        treatments={
            "none": None,
            "legacy": EmpiricalBackgroundConfig(anchors=12, minimum_bins=2),
        },
        realizations=8,
        target_false_positive_rate=0.25,
        max_lag_seconds=15_000.0,
        seed=42,
    )

    first = benchmark_background_treatments(**kwargs)
    second = benchmark_background_treatments(**kwargs)

    assert isinstance(first, BackgroundBenchmark)
    assert first == second
    assert len(first.metrics) == 4
    assert all(0 <= row.true_positive_rate <= 1 for row in first.metrics)
    assert all(0 <= row.delta_nu_recovery_rate <= 1 for row in first.metrics)
