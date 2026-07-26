import numpy as np

from urdr import SimulationCalibrator, SimulationConfig, TimeSeries
from urdr.simulation import simulate_time_series


def test_calibration_is_reproducible_and_window_aware() -> None:
    time = np.arange(256) / 48
    observed = np.ones(256, dtype=bool)
    observed[40:70] = False
    template = TimeSeries(time, np.zeros(256), observed)
    config = SimulationConfig(oscillation_amplitude=0.5)
    target = simulate_time_series(
        template.window, config, np.random.default_rng(10)
    )

    def variance(series: TimeSeries) -> float:
        return float(np.var(series.flux[series.observed]))

    calibrator = SimulationCalibrator(simulations=8, seed=42)
    first = calibrator.calibrate(target, config, variance)
    second = calibrator.calibrate(target, config, variance)
    np.testing.assert_array_equal(first.null_statistics, second.null_statistics)
    np.testing.assert_array_equal(first.signal_statistics, second.signal_statistics)
    assert 0 < first.false_alarm_probability <= 1

