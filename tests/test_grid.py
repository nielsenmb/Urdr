"""Tests for multi-condition empirical-background calibration."""

import numpy as np

from urdr import (
    BackgroundBenchmarkCase,
    EmpiricalGridBenchmark,
    EmpiricalGridPoint,
    SimulationConfig,
    benchmark_empirical_grid,
    make_observing_window,
)


def _case(name: str, split: str = "train") -> BackgroundBenchmarkCase:
    """Return a compact grid-benchmark case."""
    window = make_observing_window(
        0.4,
        120.0,
        gaps_days=((0.12, 0.15),),
        random_missing_fraction=0.02,
        seed=4,
    )
    simulation = SimulationConfig(
        white_noise_sigma=0.3,
        granulation_amplitude=0.2,
        numax_uhz=1000.0,
        delta_nu_uhz=100.0,
        envelope_width_uhz=300.0,
        oscillation_amplitude=1.0,
    )
    return BackgroundBenchmarkCase(
        name=name,
        split=split,
        window=window,
        simulation=simulation,
        centre_frequencies_uhz=np.array([900.0, 1000.0, 1100.0]),
        filter_width_uhz=500.0,
        delta_nu_grid_uhz=np.array([90.0, 100.0, 110.0]),
        oscillation_amplitudes=np.array([0.4, 1.0]),
    )


def test_make_observing_window_is_reproducible() -> None:
    """Gap intervals and random missing cadences should be deterministic."""
    first = make_observing_window(
        1.0, 120.0, gaps_days=((0.2, 0.3),), random_missing_fraction=0.1, seed=8
    )
    second = make_observing_window(
        1.0, 120.0, gaps_days=((0.2, 0.3),), random_missing_fraction=0.1, seed=8
    )

    np.testing.assert_array_equal(first.observed, second.observed)
    assert first.duty_cycle < 0.9


def test_empirical_grid_is_reproducible_and_exports_records() -> None:
    """The grid should preserve paired seeds and expose flat results."""
    candidates = [EmpiricalGridPoint(0.5, 0.8), EmpiricalGridPoint(0.66, 0.88)]
    kwargs = dict(
        cases=[_case("training"), _case("held_out", "validation")],
        candidates=candidates,
        anchors=12,
        minimum_bins=2,
        realizations=8,
        target_false_positive_rate=0.25,
        max_lag_seconds=15_000.0,
        seed=42,
    )

    first = benchmark_empirical_grid(**kwargs)
    second = benchmark_empirical_grid(**kwargs)

    assert isinstance(first, EmpiricalGridBenchmark)
    assert first == second
    assert len(first.metrics) == 8
    assert len(first.to_records()) == 8
    assert {row["split"] for row in first.to_records()} == {
        "train",
        "validation",
    }
    assert first.summaries("train")[0].conditions == 2


def test_pareto_front_removes_dominated_candidate() -> None:
    """A candidate worse on both primary metrics should be excluded."""
    from urdr import BenchmarkMetrics, EmpiricalGridMetric

    strong = EmpiricalGridPoint(0.5, 0.8)
    weak = EmpiricalGridPoint(1.0, 1.0)

    def metric(name: str, tpr: float, recovery: float) -> BenchmarkMetrics:
        return BenchmarkMetrics(name, 1.0, 2.0, 0.01, tpr, recovery, 0.1)

    result = EmpiricalGridBenchmark(
        metrics=(
            EmpiricalGridMetric("case", "train", strong, metric(strong.name, 0.8, 0.9)),
            EmpiricalGridMetric("case", "train", weak, metric(weak.name, 0.7, 0.8)),
        ),
        target_false_positive_rate=0.01,
        realizations=128,
    )

    assert tuple(item.candidate for item in result.pareto_front()) == (strong,)
